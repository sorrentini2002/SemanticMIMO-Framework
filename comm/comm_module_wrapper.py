# ============================================================
# comm/comm_module_wrapper.py
# ============================================================
# Advanced adapter that makes CommModule compatible with the
# nn.Sequential pipeline in proposal.py, while fully activating
# the advanced features of CommModule:
#   - selection_scores  →  power / stream / mode allocation
#   - last_info         →  per-forward channel statistics
#   - reconfigure()     →  eval-time SNR sweep
#
# Score alignment contract (Componente 3 of impl. plan):
#   - Store_Class_Token_Attn_Wrapper saves class_token_attention
#     of shape [B, N] where position 0 is the CLS token row.
#   - CommModule expects selection_scores of shape [B, N_patches]
#     or [B, N_tokens] (it handles both via _resolve_power_scores).
#   - We pass the full [B, N] attention tensor: CommModule's
#     internal helpers strip the CLS position as needed.
#   - If scores are unavailable or have mismatched shape we fall
#     back to None → CommModule uses uniform allocation.
# ============================================================

import torch
import torch.nn as nn
from .comm_module import CommModule


class CommModuleWrapper(nn.Module):
    """
    Full-featured wrapper around CommModule for split-learning.

    Responsibilities:
      1. Maintains nn.Sequential compatibility (forward returns Tensor).
      2. Reads CLS attention scores from the upstream compressor block
         and forwards them to CommModule as selection_scores.
      3. Saves per-forward channel statistics in self.last_info.
      4. Exposes reconfigure() for eval-time SNR sweeps.

    Args:
        input_dim (int): Embedding dimension D (e.g. 192 for DeiT-tiny).
        config    (dict): Full config dict passed to CommModule.
                          Must contain the 'comm' top-level key.
    """

    def __init__(self, input_dim: int, config: dict):
        super().__init__()

        # --- Build the inner CommModule ---
        self.comm = CommModule(input_dim=input_dim, config=config)

        # --- Score source: set by proposal.build_model() ---
        # Holds a reference to the Compress_Batches block so we can
        # read its stored class-token attention on every forward pass.
        self._score_source = None   # type: ignore[assignment]

        # --- Last forward statistics ---
        # Updated on every forward call; accessible for logging.
        self.last_info: dict = {}

        # --- Channel eval-only mode ---
        # When True, the radio channel is bypassed during training
        # (self.training=True) and active during eval. This allows
        # training a clean model while validating against noisy conditions.
        self._channel_eval_only = False
        # Cache the original config value so we can restore it.
        self._cfg_use_channel = self.comm.use_channel
        
        # --- Semantic Waterfilling Toggle ---
        # When True (default), selection scores are extracted and sent to the
        # CommModule. When False, we pretend no scores exist (returns None),
        # which causes CommModule to use uniform allocation and no assignment.
        self.semantic_waterfilling = True

    # ----------------------------------------------------------
    # Score wiring (called once from proposal.build_model)
    # ----------------------------------------------------------

    def set_score_source(self, compressor_module) -> None:
        """
        Register the upstream compressor block as the score source.

        This enables importance-based power/stream/mode allocation
        without changing the nn.Sequential API.

        Args:
            compressor_module: The Compress_Batches_and_Select_Tokens_Block_Wrapper
                               instance stored in proposal.model.compressor_module.
                               Must have .block.attn.class_token_attention.
        """
        self._score_source = compressor_module

    # ----------------------------------------------------------
    # Score extraction helpers
    # ----------------------------------------------------------

    def _get_selection_scores(self, x: torch.Tensor):
        """
        Extract class-token attention scores from the score source.

        Returns [B, N] tensor or None if scores are unavailable,
        incompatible shape, or semantic_waterfilling is disabled.
        """
        if not self.semantic_waterfilling or self._score_source is None:
            return None

        bsz, n_tokens, _ = x.shape
        attn = None

        # Prefer ADC-aligned scores whenever shape matches the current tensor.
        if hasattr(self._score_source, "last_adc_scores"):
            adc_scores = self._score_source.last_adc_scores
            if (
                torch.is_tensor(adc_scores)
                and adc_scores.dim() == 2
                and adc_scores.shape[0] == bsz
                and adc_scores.shape[1] == n_tokens
            ):
                attn = adc_scores

        # Fallback to raw class-token attention.
        if attn is None:
            try:
                attn = self._score_source.block.attn.class_token_attention
            except AttributeError:
                # Score source does not expose class_token_attention — skip.
                return None

        if attn is None or not torch.is_tensor(attn):
            return None

        # Detach and move to the same device as x (attention may have
        # been computed before a device transfer).
        scores = attn.detach().to(device=x.device, dtype=x.dtype)

        # ── Score alignment contract ──────────────────────────────────
        # class_token_attention has shape [B, N] where:
        #   position 0  →  CLS  attention row  (NOT a patch score)
        #   positions 1…N-1  →  patch token scores
        #
        # CommModule._resolve_power_scores() expects selection_scores
        # of shape [B, N_patches] = [B, N-1].  When selection_indices
        # is absent it simply does:
        #       scores[:, :n_patches]   (n_patches = n_tokens - 1)
        # If we passed the full [B, N] tensor, that slice would be
        #   [CLS_score, patch_1, …, patch_{N-2}]   ← WRONG (off-by-one)
        #
        # Fix: strip the CLS position and return only patch scores.
        patch_scores = scores[:, 1:]   # [B, N-1]  — aligned to patch tokens

        if patch_scores.shape[1] == 0:
            return None   # Edge case: only CLS token present

        return patch_scores   # [B, N-1] — correctly aligned to patch tokens

    # ----------------------------------------------------------
    # Forward
    # ----------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Feature tensor [B, N, D] from the client backbone.
        Returns:
            out (Tensor): Reconstructed features [B, N, D].
                          Shape is identical to input.
        """
        # --- Channel eval-only toggle ---
        # Temporarily override use_channel based on train/eval phase.
        if self._channel_eval_only:
            self.comm.use_channel = (not self.training) and self._cfg_use_channel
        # (If _channel_eval_only is False, use_channel stays as configured.)

        # --- Read importance scores for advanced allocation ---
        selection_scores = self._get_selection_scores(x)

        # --- Mandatory Pre-Channel Power Normalization (Il Tetto di Cristallo) ---
        # Costringe fisicamente l'energia media del batch/tensore compresso a 1.0.
        # Disinnesca il reward hacking: se l'Encoder gonfia la magnitudo, viene schiacciato giù proporzionalmente.
        rms = torch.sqrt(torch.mean(x ** 2, dim=(1, 2), keepdim=True))
        x = x / (rms + 1e-9)

        # --- Run the full CommModule pipeline ---
        # CommModule returns (output_tensor, stats_dict).
        out, info = self.comm(x, selection_scores=selection_scores)
        
        # =====================================================================
        # DIAGNOSTICS PHASE 3: Payload Integrity (Channel Impact)
        # =====================================================================
        if hasattr(self._score_source, "diagnostic_stats") and self.training:
            stats = self._score_source.diagnostic_stats
            stats["payload_x_norm"].append(x.norm().item())
            stats["payload_out_norm"].append(out.norm().item())
            # Verifichiamo la discrepanza creata solo dal canale (MSE L2 puro) Prima del padding
            stats["payload_diff_norm"].append((x - out).norm().item())
        # =====================================================================

        # --- Persist stats for external logging / SNR sweep ---
        self.last_info = info

        return out

    # ----------------------------------------------------------
    # Utilities
    # ----------------------------------------------------------

    def get_last_info(self) -> dict:
        """
        Safe accessor for the statistics dict of the last forward pass.

        Returns an empty dict if forward has not been called yet.
        Useful in training loops:
            ch_stats = model.channel.get_last_info()
            wandb.log(ch_stats)
        """
        return dict(self.last_info)
        
    def reconfigure(self, config_update: dict):
        """Passes the configuration dictates (like SNR) down to the internal CommModule."""
        if hasattr(self.comm, "reconfigure"):
            self.comm.reconfigure(config_update)

    def set_channel_eval_only(self, enabled: bool) -> None:
        """
        Toggle channel eval-only mode.

        When enabled:
          - Training:   channel is bypassed (identity pass-through)
          - Evaluation: channel is active (per original config)

        This allows training a clean model while validating it
        against a noisy channel in a single run.

        Args:
            enabled: True to activate eval-only channel mode.
        """
        self._channel_eval_only = enabled
        # If disabling, restore the original config value immediately.
        if not enabled:
            self.comm.use_channel = self._cfg_use_channel

    def set_semantic_waterfilling(self, enabled: bool) -> None:
        """Dynamically enable or disable semantic waterfilling."""
        self.semantic_waterfilling = enabled

    def reconfigure(self, config_snippet: dict) -> None:
        """
        Partial in-place reconfiguration of the channel.

        Designed for eval-time SNR sweeps without re-instantiating
        the full model. Example:
            wrapper.reconfigure({'channel': {'snr_db': 5.0}})

        Args:
            config_snippet (dict): Sub-dict of the 'comm' config to merge.
                                   Only keys present will be overwritten.
        """
        self.comm.reconfigure(config_snippet)
