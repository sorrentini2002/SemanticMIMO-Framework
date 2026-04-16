# ============================================================
# methods/gumbel_method.py
# ============================================================
# Split-learning method based on Gumbel-Softmax token selection.
#
# This file consolidates the logic previously scattered across:
#   - methods/gumbel/gumbel.py     (Gumbel sampling & ST masks)
#   - methods/gumbel/core.py       (score-based selection & diversity)
#   - methods/gumbel/schedules.py  (tau annealing schedules)
#   - methods/gumbel/utils.py      (gather_tokens helper)
#
# The wrapper class (Gumbel_Token_Selection_Block_Wrapper) mirrors
# the interface of proposal.py's Compress_Batches_and_Select_Tokens_Block_Wrapper,
# making it a drop-in replacement within the SemanticMIMO framework.
#
# Key integration points:
#   - self.last_adc_scores : [B, N_selected] scores for MIMO waterfilling
#   - self._model_ref      : back-reference to the outer `model` class
#   - compress_labels()    : batch-merged soft labels (same as proposal.py)
#   - clean_validation bypass via self._model_ref.clean_validation
#
# Supported channel types (same as proposal.py):
#   - Gaussian_Noise_Analogic_Channel
#   - MyMIMOChannel
#   - CommModuleWrapper (full CommModule pipeline)
# ============================================================

import math
import logging
import weakref
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import VisionTransformer
from comm.comm_module_wrapper import CommModuleWrapper

logger = logging.getLogger(__name__)


# ============================================================

from .gumbel.gumbel import sample_gumbel_topk, compute_gumbel_mc_scores
from .gumbel.utils import gather_tokens
from .gumbel.schedules import compute_tau

# ============================================================
# BLOCK WRAPPER — Gumbel token selection at the split point
# ============================================================

class Gumbel_Token_Selection_Block_Wrapper(nn.Module):
    """
    Wraps a single ViT transformer block and performs Gumbel-Softmax
    token selection after the block's normal forward pass.

    This wrapper follows the same interface contract as
    ``Compress_Batches_and_Select_Tokens_Block_Wrapper`` in proposal.py:

    - Stores ``last_adc_scores`` (shape [B, N_selected]) for MIMO
      semantic waterfilling via CommModuleWrapper.
    - Stores ``_model_ref`` (back-reference to the outer model) so
      that the ``clean_validation`` flag can be read at eval time.
    - Exposes ``compress_labels()`` so main.py label handling is compatible.
    """

    def __init__(self,
                 block: nn.Module,
                 method_cfg: dict):
        """
        Args:
            block: Original ViT Block (timm) to wrap.
            method_cfg: Configurazione Hydra passata esplicitamente.
        """
        super().__init__()

        self.block = block

        # ----- Mandatory interface variables (SemanticMIMO contract) -----
        self.last_adc_scores = None          # [B, N_selected] 
        object.__setattr__(self, '_model_ref', None)

        # ==========================================
        # ESTRAZIONE PARAMETRI DA HYDRA DICT (CFG)
        # ==========================================
        
        # Regole Architetturali Generali
        self.compression_enabled = method_cfg.get('compression_enabled', True)
        self.token_compression = method_cfg.get('token_compression', 1.0)
        
        # Hyper-parametri core Gumbel (Mapping per mantenere le tue logiche)
        self.tau_max = method_cfg.get('tau_start', 2.0)
        self.tau_min = method_cfg.get('tau_end', 0.1)
        self.anneal_steps = method_cfg.get('steps', 10000)
        self.anneal_mode = method_cfg.get('schedule', 'linear')
        self.hard = method_cfg.get('hard', True)
        self.straight_through = method_cfg.get('straight_through', True)
        
        # Flag di Regolarizzazione (per calcolo loss out-of-band)
        self.entropy_reg_enabled = method_cfg.get('entropy_reg_enabled', False)
        self.entropy_reg_weight = method_cfg.get('entropy_reg_weight', 0.1)
        self.cov_reg_enabled = method_cfg.get('cov_reg_enabled', False)
        self.cov_reg_weight = method_cfg.get('cov_reg_weight', 0.5)
        self.cov_reg_margin = method_cfg.get('cov_reg_margin', 0.3)
        self.cov_reg_max_tokens = method_cfg.get('cov_reg_max_tokens', 64)
        
        # Setup per Valutazione / Inferenza
        self.eval_k = method_cfg.get('eval_k', None)  # None = use token_compression
        self.gumbel_mc_enabled = method_cfg.get('gumbel_mc_enabled', False)
        self.gumbel_mc_tau = method_cfg.get('gumbel_mc_tau', 0.5)
        self.gumbel_mc_samples = method_cfg.get('gumbel_mc_samples', 16)
        self.gumbel_mc_aggregate = method_cfg.get('gumbel_mc_aggregate', 'mean')
        
        # Warmup strategy
        self.warmup_steps = method_cfg.get('warmup_steps', 1000)
        # Costruisce dizionario dinamico per eventuali metodi Diversify
        self.diversify_cfg = {
            'enabled': method_cfg.get('diversify_enabled', False),
            'lambda': method_cfg.get('diversify_lambda', 0.2),
            'metric': method_cfg.get('diversify_metric', 'cosine')
        }

        # Tracking variables
        self.n_new_tokens = 0
        self._global_step = 0

        # Intermediates for regularization loss (populated during gumbel_compress)
        self._last_patch_scores = None   # [B, num_patches] soft probabilities
        self._last_tokens_sel = None     # [B, 1+n_alpha, D] selected tokens
        self._last_full_patches = None   # [B, num_patches, D] full patch embeddings for covariance reg

    # ------------------------------------------------------------------
    # Step management
    # ------------------------------------------------------------------

    def register_step(self, step: int):
        """Update the internal global step counter (used for tau annealing)."""
        self._global_step = step

    @property
    def current_tau(self) -> float:
        """Current Gumbel-Softmax temperature based on the annealing schedule."""
        # Annealing only starts after warmup
        if self._global_step < self.warmup_steps:
            return self.tau_max
            
        effective_step = self._global_step - self.warmup_steps
        return compute_tau(
            effective_step,
            self.tau_max,
            self.tau_min,
            self.anneal_steps,
            self.anneal_mode,
        )

    # ------------------------------------------------------------------
    # Gumbel token selection  (replaces merge_batches_and_select_tokens)
    # ------------------------------------------------------------------

    def gumbel_compress(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform Gumbel-Softmax token selection on the output of the
        transformer block.

        Steps:
            1.  Compute CLS-row attention scores from the stored attention.
            2.  Determine n_alpha (number of patch tokens to keep).
            3.  Apply Gumbel top-k sampling (train) or MC-averaged
                deterministic top-k (eval with MC enabled).
            4.  Build self.last_adc_scores with shape [B, 1 + n_alpha]:
                - First position: CLS dummy score = 1.0
                - Remaining positions: patch scores of selected tokens.

        Args:
            x: [B, N, D] — tokens after the block forward pass.

        Returns:
            x_sel: [B, 1 + n_alpha, D] — selected tokens (CLS + top patches).
        """
        B, N, D = x.shape
        num_patches = N - 1
        device = x.device

        if self.training:
            self._last_full_patches = x[:, 1:, :]

        # ---- Determine number of patch tokens to keep ----
        # Bug 5 fix: use eval_k at inference time if configured
        if not self.training and self.eval_k is not None:
            n_alpha = max(1, min(self.eval_k, num_patches))
        else:
            n_alpha = max(1, int(self.token_compression * num_patches))
        self.n_new_tokens = 1 + n_alpha  # CLS + selected patches

        # ---- Retrieve CLS-row attention LOGITS [B, N] ----
        # Using raw logits before softmax ensures Gumbel sampling runs in logits space.
        if hasattr(self.block.attn, 'class_token_logits') and self.block.attn.class_token_logits is not None:
            cls_attn = self.block.attn.class_token_logits
        else:
            # Fallback path: convert probabilities to logits to keep train/eval consistent.
            cls_attn = torch.log(self.block.attn.class_token_attention.clamp(min=1e-10))
            
        # ---- Bug 4 fix: MC-averaged eval vs Gumbel train ----
        if not self.training and self.gumbel_mc_enabled:
            # MC-averaged deterministic top-k for robust evaluation
            patch_scores_raw = cls_attn[:, 1:]  # [B, num_patches]

            # Compute MC-averaged probabilities (outputs probabilities!)
            patch_scores = compute_gumbel_mc_scores(
                patch_scores_raw,
                num_samples=self.gumbel_mc_samples,
                tau=self.gumbel_mc_tau,
                aggregate=self.gumbel_mc_aggregate,
            )

            # Deterministic top-k on MC-averaged scores
            _, topk_relative_indices = torch.topk(
                patch_scores, k=n_alpha, dim=1, sorted=False
            )
            topk_indices = topk_relative_indices + 1  # global indices
            topk_indices, sort_order = torch.sort(topk_indices, dim=1)

            # Gather selected tokens (CLS + top patches)
            cls_indices = torch.zeros((B, 1), dtype=torch.long, device=device)
            indices_sel = torch.cat([cls_indices, topk_indices], dim=1)
            tokens_sel = gather_tokens(x, indices_sel)

            # Selected patch scores for MIMO waterfilling
            topk_relative_sorted = torch.gather(
                topk_relative_indices, 1, sort_order
            )
            selected_patch_probs = torch.gather(
                patch_scores, 1, topk_relative_sorted
            )
        else:
            # Gumbel-Softmax sampling (training path)
            cls_attn_3d = cls_attn.unsqueeze(1)  # [B, 1, N]
            tau = self.current_tau
            tokens_sel, indices_sel, patch_scores, gs_tau = sample_gumbel_topk(
                tokens=x,
                attn=cls_attn_3d,
                n_alpha=n_alpha,
                tau=tau,
                hard=self.hard,
                straight_through=self.straight_through,
                generator=None
            )
            # patch_scores : [B, num_patches]

            # Store intermediates for regularization loss (Bug 3)
            self._last_patch_scores = patch_scores
            self._last_tokens_sel = tokens_sel

            # Selected patch scores (these are UNNORMALIZED LOGITS)
            selected_patch_indices = indices_sel[:, 1:] - 1  # [B, n_alpha], 0-based
            selected_patch_logits = torch.gather(
                patch_scores, 1, selected_patch_indices
            )
            # Normalize to probabilities [0, 1] for the MIMO channel
            selected_patch_probs = torch.sigmoid(selected_patch_logits / tau)

        # ---- Build last_adc_scores for MIMO waterfilling ----
        # Using probs ensures cls_dummy=1.0 is always the maximum power token!
        cls_dummy = torch.ones(
            (B, 1), dtype=selected_patch_probs.dtype, device=device
        )
        self.last_adc_scores = torch.cat(
            [cls_dummy, selected_patch_probs], dim=1
        )  # [B, 1 + n_alpha]

        # Bug 4 Fix: Energy Normalization
        # Scale selected tokens to preserve the total average energy of the original signal
        # Since we transmit fewer tokens, we can boost their power
        scale_factor = math.sqrt(N / self.n_new_tokens)
        tokens_sel = tokens_sel * scale_factor

        return tokens_sel

    # ------------------------------------------------------------------
    # Regularization losses  (Bug 3 fix)
    # ------------------------------------------------------------------

    def compute_reg_loss(self) -> torch.Tensor:
        """
        Compute regularization losses from the last forward pass.

        Returns:
            reg_loss: scalar tensor (0.0 if no regularization is enabled
                      or no intermediates are available).
        """
        # Only compute during training when intermediates are available
        if self._last_patch_scores is None:
            return torch.tensor(0.0)

        device = self._last_patch_scores.device
        reg_loss = torch.tensor(0.0, device=device)
        
        # Soft selection probabilities (same approach as original Scardapane code)
        tau = self.current_tau
        p = torch.sigmoid(self._last_patch_scores / tau)  # [B, num_patches]

        # ---- Entropy regularization ----
        # In the original code, this minimizes negative entropy over the mean batch
        # prob, effectively maximizing the global entropy and preventing mode collapse.
        if self.entropy_reg_enabled:
            p_mean = p.mean(dim=0)  # [num_patches] marginal probability
            ent_reg = -torch.sum(p_mean * torch.log(p_mean + 1e-8))
            reg_loss = reg_loss + self.entropy_reg_weight * ent_reg

        # ---- Covariance regularization ----
        # Penalizes high correlation among ALL tokens based on their probability of being selected.
        if self.cov_reg_enabled and self._last_full_patches is not None:
            patch_embeds = self._last_full_patches  # [B, num_patches, D]
            B_size, N_minus_1, D = patch_embeds.shape
            
            p_sub = p
            
            # Subsample if too many tokens (memory)
            if N_minus_1 > self.cov_reg_max_tokens:
                idx = torch.randperm(N_minus_1, device=device)[:self.cov_reg_max_tokens]
                patch_embeds = patch_embeds[:, idx, :]
                p_sub = p[:, idx]

            # Normalize embeddings for cosine similarity
            patch_embeds_norm = torch.nn.functional.normalize(patch_embeds, p=2, dim=-1)
            
            # Compute similarity matrix [B, S, S]
            sim_matrix = torch.bmm(patch_embeds_norm, patch_embeds_norm.transpose(1, 2))
            
            # Mask out diagonal (self-similarity)
            eye = torch.eye(sim_matrix.shape[-1], device=device).unsqueeze(0).expand(B_size, -1, -1)
            sim_matrix = sim_matrix * (1.0 - eye)
            
            # Apply margin max(0, cos - margin)
            penalty = torch.clamp(sim_matrix - self.cov_reg_margin, min=0.0)
            
            # Weight by soft-selection probs: w_i * w_j * penalty
            w = p_sub / (p_sub.sum(dim=-1, keepdim=True) + 1e-8)  # normalized weights
            w_matrix = torch.bmm(w.unsqueeze(-1), w.unsqueeze(1))  # [B, S, S]
            
            cov_loss_batch = torch.sum(w_matrix * penalty, dim=(1, 2))
            reg_loss = reg_loss + self.cov_reg_weight * cov_loss_batch.mean()

        return reg_loss


    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass:
            1. Run the original transformer block (attention + MLP).
            2. Check for clean_validation bypass.
            3. If not bypassed, apply Gumbel token selection (and
               optional batch merging).

        Args:
            x: [B, N, D]

        Returns:
            [B_out, N_out, D] where B_out / N_out depend on
            compression settings.
        """
        # --- Original block forward (attention + MLP) ---
        x = x + self.block.drop_path1(self.block.ls1(self.block.attn(self.block.norm1(x))))
        x = x + self.block.drop_path2(self.block.ls2(self.block.mlp(self.block.norm2(x))))

        # --- Bug 2 fix: auto-increment global step for tau annealing ---
        if self.training:
            self._global_step += 1

        # --- Clean validation bypass (Rule 4) ---
        clean_val = False
        if not self.training and self._model_ref is not None:
            clean_val = getattr(self._model_ref, 'clean_validation', False)

        # Apply compression only if warmup is over
        if self.compression_enabled and not clean_val and self._global_step >= self.warmup_steps:
            # Apply Gumbel token compression
            x = self.gumbel_compress(x)

        # If clean_val is True, compression_enabled is False, or warmup is active, x passes through unmodified (no compression)
        return x

    # ------------------------------------------------------------------
    # compress_labels  (for main.py compatibility)
    # ------------------------------------------------------------------

    def compress_labels(self, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
        """
        Merge labels to match predictions.
        Since Gumbel natively does not compress batches, we simply return
        the one-hot encoded labels.

        Args:
            labels:      [B] — integer class labels.
            num_classes: int — total number of classes.

        Returns:
            new_labels: [B, num_classes] (standard one-hot).
        """
        return F.one_hot(labels, num_classes=num_classes).float()


# ============================================================
# Store_Class_Token_Attn_Wrapper  (same as proposal.py)
# ============================================================

class Store_Class_Token_Attn_Wrapper(nn.Module):
    """
    Thin wrapper around a timm Attention module that stores the
    CLS-row attention scores (averaged across heads) after every
    forward pass.

    Attribute ``class_token_attention`` has shape [B, N] and is
    consumed by the Gumbel block wrapper to rank patch tokens.
    """

    def __init__(self, attn):
        super().__init__()
        self.attn = attn
        self.class_token_attention = None
        self.class_token_logits = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        # QKV projection
        qkv = (self.attn.qkv(x)
               .reshape(B, N, 3, self.attn.num_heads, self.attn.head_dim)
               .permute(2, 0, 3, 1, 4))
        q, k, v = qkv.unbind(0)
        q, k = self.attn.q_norm(q), self.attn.k_norm(k)

        # Scaled dot-product
        q = q * self.attn.scale
        attn_logits = q @ k.transpose(-2, -1)
        
        # Store pre-softmax CLS-row logits averaged across heads → [B, N]
        self.class_token_logits = attn_logits[:, :, 0, :].mean(dim=1)

        attn = attn_logits.softmax(dim=-1)

        # Store CLS-row attention averaged across heads  →  [B, N]
        self.class_token_attention = attn[:, :, 0, :].mean(dim=1)

        # Normal attention output
        attn = self.attn.attn_drop(attn)
        attn_output = attn @ v
        x = attn_output.transpose(1, 2).reshape(B, N, C)
        x = self.attn.proj(x)
        x = self.attn.proj_drop(x)
        return x


# ============================================================
# OUTER MODEL  — mirrors proposal.py's `model` class
# ============================================================

class model(nn.Module):
    """
    Top-level split-learning model that uses Gumbel-Softmax token
    selection at the split point.

    Constructor signature is identical to proposal.py so that Hydra
    can instantiate it transparently.
    """

    def __init__(self,
                 model: VisionTransformer,
                 channel,
                 split_index,
                 method_cfg: dict,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.method_cfg = method_cfg
        
        # ---- Arch Flags ----
        compression_enabled = method_cfg.get('compression_enabled', True)
        desired_compression = method_cfg.get('desired_compression', None)
        token_compression = method_cfg.get('token_compression', 1.0)
        self.channel_eval_only = method_cfg.get('channel_eval_only', False)
        self.semantic_waterfilling = method_cfg.get('semantic_waterfilling', True)

        # ---- Resolve compression rates ----
        if not compression_enabled:
            self.compression_ratio = 1.0
        else:
            if desired_compression is not None:
                assert token_compression is None or token_compression == 1.0, \
                    "When desired_compression is set, token_compression should be None"
                self.compression_ratio = desired_compression
                self.method_cfg['token_compression'] = desired_compression
            else:
                if token_compression is None:
                    token_compression = 1.0
                self.compression_ratio = token_compression
                self.method_cfg['token_compression'] = token_compression

        # Will be assigned inside build_model
        self.compressor_module = None
        self.clean_validation = False

        # ---- Build model ----
        self.model = self.build_model(
            model, channel, split_index,
            self.method_cfg
        )

        # Store channel reference
        self.channel = channel

        # Communication cost tracker (same as proposal.py)
        self.communication = 0

        # Method name
        self.name = "GumbelMethod"

        # Regularization loss from last forward pass (Bug 3)
        self.last_reg_loss = None

    # ------------------------------------------------------------------
    # build_model
    # ------------------------------------------------------------------

    def build_model(self,
                    model: VisionTransformer,
                    channel,
                    split_index: int,
                    method_cfg: dict):
        """
        Assemble the split-learning pipeline.
        """

        # --- Wrap attention to expose CLS scores ---
        model.blocks[split_index - 1].attn = Store_Class_Token_Attn_Wrapper(
            model.blocks[split_index - 1].attn
        )

        # --- Wrap the block with Gumbel token selection ---
        model.blocks[split_index - 1] = Gumbel_Token_Selection_Block_Wrapper(
            block=model.blocks[split_index - 1],
            method_cfg=method_cfg
        )
        self.compressor_module = model.blocks[split_index - 1]

        # Wire the back-reference so the wrapper can read clean_validation.
        # Use object.__setattr__ to avoid nn.Module registering the parent
        # as a sub-module (which would cause infinite recursion).
        object.__setattr__(self.compressor_module, '_model_ref', weakref.proxy(self))

        # --- Split into client / server blocks ---
        blocks_before = model.blocks[:split_index]    # client
        blocks_after  = model.blocks[split_index:]    # server

        # --- Insert channel ---
        model.blocks = nn.Sequential(*blocks_before, channel, *blocks_after)

        # --- Wire scores → CommModuleWrapper ---
        if isinstance(channel, CommModuleWrapper):
            channel.set_score_source(self.compressor_module)
            # Apply eval-only channel mode if requested
            if hasattr(channel, "set_channel_eval_only"):
                channel.set_channel_eval_only(self.channel_eval_only)
            # Toggle semantic waterfilling
            if hasattr(channel, "set_semantic_waterfilling"):
                channel.set_semantic_waterfilling(self.semantic_waterfilling)
            
            # Disable bottleneck if compression is disabled
            compression_enabled = method_cfg.get('compression_enabled', True)
            if not compression_enabled and hasattr(channel, "comm"):
                channel.comm.use_bottleneck = False

        return model

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]
        if self.training:
            self.communication += self.compression_ratio * batch_size
        output = self.model.forward(x)

        # Bug 3 fix: compute and store regularization loss
        if self.training and self.compressor_module is not None:
            self.last_reg_loss = self.compressor_module.compute_reg_loss()
        else:
            self.last_reg_loss = None

        return output
