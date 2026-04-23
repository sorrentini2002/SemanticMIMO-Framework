# ============================================================
# methods/gumbel_method.py
# ============================================================
# Split-learning method based on Gumbel-Softmax token selection.
#
# CHANGELOG (Curriculum Learning Extension):
#   - Logit Scaling Dinamico:   _compute_logit_scale() + register_epoch()
#   - Entropy Bottleneck:       max(0, H_actual - H_target(epoch)) loss
#   - Stability Bonus (EMA):    _selection_freq_ema applied to logits
#   - Weight decay group tag:   SCORE_HEAD flag for main.py router
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

from .gumbel.gumbel import sample_gumbel_topk
from .gumbel.schedules import compute_tau


# ============================================================
# BLOCK WRAPPER — Gumbel token selection at the split point
# ============================================================

class Gumbel_Token_Selection_Block_Wrapper(nn.Module):
    """
    Wraps a single ViT transformer block and performs Gumbel-Softmax
    token selection after the block's normal forward pass.

    NEW: Curriculum learning through three mechanisms:
      1. Logit Scaling Dinamico  — alpha_scale ramps from logit_scale_start
         to logit_scale_end over the full training, letting tau have a real
         effect only once logits have "matured" beyond noise.
      2. Entropy Bottleneck      — replaces the old batch-diversity loss.
         Loss = entropy_bottleneck_weight * max(0, H_actual - H_target(epoch)).
         H_target decreases from entropy_target_start to entropy_target_end,
         acting as a soft ceiling that follows the training curriculum.
      3. Stability Bonus (EMA)   — patches that are consistently selected
         across batches receive a small logit bonus, reinforcing systematic
         choices without freezing the distribution.
    """

    def __init__(self,
                 block: nn.Module,
                 method_cfg: dict):
        super().__init__()

        self.block = block

        # ----- Mandatory interface variables (SemanticMIMO contract) -----
        self.last_adc_scores = None
        object.__setattr__(self, '_model_ref', None)

        # ==========================================
        # ARCHITECTURAL PARAMETERS (from Hydra dict)
        # ==========================================
        self.compression_enabled = method_cfg.get('compression_enabled', True)
        self.token_compression    = method_cfg.get('token_compression', 1.0)

        self.tau_max      = method_cfg.get('tau_start', 2.0)
        self.tau_min      = method_cfg.get('tau_end', 0.1)
        self.anneal_steps = method_cfg.get('steps', 10000)
        self.anneal_mode  = method_cfg.get('schedule', 'linear')
        self.hard             = method_cfg.get('hard', True)
        self.straight_through = method_cfg.get('straight_through', True)

        self.entropy_reg_enabled = method_cfg.get('entropy_reg_enabled', False)
        self.cov_reg_enabled     = method_cfg.get('cov_reg_enabled', False)

        self.eval_k            = method_cfg.get('eval_k', 32)
        self.gumbel_mc_enabled = method_cfg.get('gumbel_mc_enabled', False)
        self.gumbel_mc_tau     = method_cfg.get('gumbel_mc_tau', 0.5)
        self.diversify_cfg = {
            'enabled': method_cfg.get('diversify_enabled', False),
            'lambda':  method_cfg.get('diversify_lambda', 0.2),
            'metric':  method_cfg.get('diversify_metric', 'cosine')
        }

        # ==========================================
        # CURRICULUM LEARNING PARAMETERS (NEW)
        # ==========================================

        # --- 1. Logit Scaling Dinamico ---
        # alpha_scale = logit_scale_start → logit_scale_end over training.
        # Mode: 'linear', 'cosine', 'exp'
        # With alpha_scale small at start, tau annealing has no effect (all
        # logits ≈ 0 → uniform softmax regardless of tau).  As logits mature,
        # tau starts to matter and the distribution sharpens at a controlled pace.
        self.logit_scale_start = method_cfg.get('logit_scale_start', 0.1)
        self.logit_scale_end   = method_cfg.get('logit_scale_end', 1.0)
        self.logit_scale_mode  = method_cfg.get('logit_scale_mode', 'cosine')

        # --- 2. Entropy Bottleneck ---
        # Loss = entropy_bottleneck_weight * max(0, H_actual - H_target(epoch))
        # H_target decreases linearly from entropy_target_start to entropy_target_end.
        # This creates a soft "ceiling" on entropy that follows the curriculum.
        # When H_actual < H_target the term is zero (no penalty for being sharp).
        self.entropy_bottleneck_enabled = method_cfg.get('entropy_bottleneck_enabled', True)
        self.entropy_target_start   = method_cfg.get('entropy_target_start', 5.2)
        self.entropy_target_end     = method_cfg.get('entropy_target_end', 2.0)
        self.entropy_bottleneck_weight = method_cfg.get('entropy_bottleneck_weight', 0.05)

        # --- 3. Stability Bonus (EMA of selection frequencies) ---
        # After each forward, we update an EMA of which patches were selected.
        # On the NEXT forward these frequencies are added (scaled) to the logits,
        # rewarding patches that are systematically useful across batches.
        self.stability_bonus_enabled   = method_cfg.get('stability_bonus_enabled', False)
        self.stability_bonus_ema_decay = method_cfg.get('stability_bonus_ema_decay', 0.97)
        self.stability_bonus_weight    = method_cfg.get('stability_bonus_weight', 0.3)
        # EMA buffer: shape [num_patches] — initialised lazily on first forward
        self._selection_freq_ema: torch.Tensor | None = None

        # --- Epoch / total-epoch tracking (set by register_epoch) ---
        self._current_epoch = 0
        self._total_epochs  = 1   # updated by training_schedule via register_epoch()

        # ==========================================
        # STEP-LEVEL TRACKING (unchanged)
        # ==========================================
        self.n_new_tokens  = 0
        self._global_step  = 0

        # ==========================================
        # SIMPLICIAL INTERACTION GRAPH (unchanged)
        # ==========================================
        if hasattr(block, 'norm1'):
            embed_dim = block.norm1.weight.shape[0]
        else:
            embed_dim = block.mlp.fc1.in_features

        self.w_u   = nn.Linear(embed_dim, embed_dim)
        self.w_tri = nn.Linear(embed_dim, embed_dim)
        nn.init.xavier_uniform_(self.w_u.weight)
        nn.init.xavier_uniform_(self.w_tri.weight)

        self.beta       = nn.Parameter(torch.tensor(method_cfg.get('beta_init', 1.0)))
        self.gamma      = nn.Parameter(torch.tensor(method_cfg.get('gamma_init', 1.0)))
        self.gate_param = nn.Parameter(torch.tensor(method_cfg.get('gate_init', -1.0)))

        self.branch_norm_weight = nn.Parameter(torch.ones(1))
        self.branch_norm_bias   = nn.Parameter(torch.zeros(1))

        logger.info(
            f"[GumbelHead] Init: beta={self.beta.item():.2f}, gamma={self.gamma.item():.2f}, "
            f"gate={self.gate_param.item():.2f} | "
            f"logit_scale {self.logit_scale_start}→{self.logit_scale_end} ({self.logit_scale_mode}) | "
            f"entropy_bottleneck={'ON' if self.entropy_bottleneck_enabled else 'OFF'} "
            f"H_target {self.entropy_target_start}→{self.entropy_target_end} | "
            f"stability_bonus={'ON' if self.stability_bonus_enabled else 'OFF'}"
        )

        # Diagnostic stats dict (unchanged keys + new ones)
        self.diagnostic_stats = {
            "tau": [], "logits_std": [], "logits_mean": [], "logits_max": [], "logits_min": [],
            "entropy": [], "entropy_target": [], "logit_alpha_scale": [],
            "grad_score_head": [], "grad_backbone": [],
            "payload_x_norm": [], "payload_out_norm": [], "payload_diff_norm": [],
            "y_tri_raw_mean": [], "y_tri_raw_std": [],
            "y_tri_norm_mean": [], "y_tri_norm_std": [],
            "beta": [], "gamma": [], "gate_sigmoid": [],
            "base_entropy": [], "batch_entropy": [], "p_max": [],
            "interaction_norm_check": [],
            # NEW curriculum keys
            "stability_ema_max": [], "stability_ema_std": [],
        }
        print(f"\n[DEBUG] Gumbel Head Initialized. Keys in stats: {list(self.diagnostic_stats.keys())}")

    # ------------------------------------------------------------------
    # Step / epoch management
    # ------------------------------------------------------------------

    def register_step(self, step: int):
        """Update the internal global step counter (used for tau annealing)."""
        self._global_step = step

    def register_epoch(self, epoch: int, total_epochs: int):
        """
        Update epoch-level counters used by the curriculum mechanisms:
          - logit alpha scaling
          - entropy target H_target(epoch)
        Called once per epoch from training_schedule() in main.py.
        """
        self._current_epoch = epoch
        self._total_epochs  = max(1, total_epochs)

    @property
    def current_tau(self) -> float:
        return compute_tau(
            self._global_step, self.tau_max, self.tau_min,
            self.anneal_steps, self.anneal_mode,
        )

    # ------------------------------------------------------------------
    # Curriculum helper methods (NEW)
    # ------------------------------------------------------------------

    def _curriculum_progress(self) -> float:
        """Fraction of training completed: 0.0 at epoch 1, 1.0 at final epoch."""
        return (self._current_epoch - 1) / max(1, self._total_epochs - 1)

    def _compute_logit_scale(self) -> float:
        """
        Compute the current alpha_scale for logit scaling.

        The scale is small at the start (logits ≈ 0 → uniform softmax → random
        selection) and increases as training progresses (logits differentiate →
        tau annealing starts to matter → selection sharpens).

        Returns a float in [logit_scale_start, logit_scale_end].
        """
        t = self._curriculum_progress()          # 0 → 1
        s, e = self.logit_scale_start, self.logit_scale_end

        if self.logit_scale_mode == 'linear':
            return s + (e - s) * t

        elif self.logit_scale_mode == 'cosine':
            # Starts at s, ends at e; uses a reversed cosine (slow start, fast middle)
            cosine = 0.5 * (1.0 - math.cos(math.pi * t))   # 0→1 (slow at edges)
            return s + (e - s) * cosine

        elif self.logit_scale_mode == 'exp':
            # Exponential: s * (e/s)^t  — very slow growth early, fast later
            if s <= 0:
                return e * t  # fallback for bad config
            return s * (e / s) ** t

        return e  # fallback: full scale

    def _compute_entropy_target(self) -> float:
        """
        H_target(epoch): soft ceiling on entropy that decreases linearly.

        At epoch 1  → H_target = entropy_target_start  (≈ log(N_patches))
        At final ep → H_target = entropy_target_end    (≈ 2.0)

        The bottleneck loss max(0, H_actual - H_target) is zero whenever the
        distribution is sharp enough; it only fires to slow down OVER-uniformity.
        """
        t = self._curriculum_progress()
        return (self.entropy_target_start
                + (self.entropy_target_end - self.entropy_target_start) * t)

    # ------------------------------------------------------------------
    # Gumbel token selection
    # ------------------------------------------------------------------

    def gumbel_compress(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform Gumbel-Softmax token selection on the output of the
        transformer block, with curriculum learning extensions.
        """
        B, N, D = x.shape
        num_patches = N - 1
        device = x.device

        # --- Budget ---
        target_n_alpha = max(1, int(self.token_compression * num_patches))
        if self.training:
            min_k = min(8, target_n_alpha)
            max_k = min(num_patches, max(64, target_n_alpha * 2))
            n_alpha = torch.randint(min_k, max_k + 1, (1,)).item()
        else:
            n_alpha = target_n_alpha
        self.n_new_tokens = 1 + n_alpha

        # --- Base attention scores from CLS row ---
        cls_attention  = self.block.attn.class_token_attention
        base_patch_scores = cls_attention[:, 1:]   # [B, N-1]

        # ==========================================
        # 1. SIMPLICIAL SCORING (unchanged)
        # ==========================================
        cls_token    = x[:, 0:1, :]    # [B, 1, D]
        patch_tokens = x[:, 1:, :]     # [B, N-1, D]

        m_cls     = self.w_u(cls_token)       # [B, 1, D]
        patch_tri = self.w_tri(patch_tokens)  # [B, N-1, D]

        y_tri_raw  = torch.norm(m_cls * patch_tri, p=2, dim=-1)   # [B, N-1]
        y_tri_norm = F.layer_norm(y_tri_raw, y_tri_raw.shape[-1:])
        y_tri      = y_tri_norm * self.branch_norm_weight + self.branch_norm_bias

        x_std       = base_patch_scores
        raw_logits  = (self.beta * x_std) + (torch.sigmoid(self.gate_param) * self.gamma * y_tri)

        # ==========================================
        # 2. LOGIT SCALING DINAMICO (NEW)
        # ==========================================
        # During training, scale logits by a curriculum ramp so the network
        # starts with near-uniform logits (random selection) and only develops
        # discriminative power once the backbone has learned useful features.
        # At eval time, we use full scale (alpha_scale = 1.0).
        if self.training:
            alpha_scale = self._compute_logit_scale()
        else:
            alpha_scale = 1.0

        final_logits = raw_logits * alpha_scale   # [B, N-1]

        # ==========================================
        # 3. STABILITY BONUS — apply EMA from PREVIOUS batches (NEW)
        # ==========================================
        # The EMA is updated AFTER selection (below), so here we use the
        # bonus computed during the previous forward.  This avoids a chicken-
        # and-egg problem while still reinforcing systematic selections.
        if self.stability_bonus_enabled and self._selection_freq_ema is not None:
            ema = self._selection_freq_ema.to(device=device, dtype=final_logits.dtype)
            if ema.shape[0] == num_patches:
                # Center the EMA so neutral patches get 0 bonus, not a uniform lift
                ema_centered = ema - ema.mean()
                final_logits = final_logits + self.stability_bonus_weight * ema_centered.unsqueeze(0)

        # Soft probabilities (for entropy computation and ADC scores)
        patch_scores_probs = F.softmax(final_logits, dim=-1)   # [B, N-1]

        # ==========================================
        # 4. ENTROPY BOTTLENECK LOSS (NEW — replaces batch entropy maximization)
        # ==========================================
        # Loss_ent = entropy_bottleneck_weight * max(0, H_actual - H_target(epoch))
        #
        # Semantics:
        #   • When H_actual > H_target  → distribution is too uniform for this
        #     stage of training → push it down gently.
        #   • When H_actual < H_target  → distribution has legitimately sharpened
        #     → loss is zero, no interference with classification gradient.
        #
        # H_target decreases from entropy_target_start to entropy_target_end so
        # the "allowed" uniformity shrinks progressively with the curriculum.
        if self.entropy_bottleneck_enabled and self.training:
            # Instance-level entropy, averaged over batch
            H_actual = -(patch_scores_probs * torch.log(patch_scores_probs + 1e-9)).sum(dim=-1).mean()
            H_target_val = self._compute_entropy_target()
            # max(0, H_actual - H_target): penalise only over-uniformity
            entropy_ceiling_loss = torch.clamp(H_actual - H_target_val, min=0.0)
            self.entropy_reg_loss = self.entropy_bottleneck_weight * entropy_ceiling_loss
            # Expose scalar for diagnostics
            self._last_H_actual  = H_actual.item()
            self._last_H_target  = H_target_val
        else:
            # Fallback: keep the old batch-diversity term (or zero when not training)
            p_mean = patch_scores_probs.mean(dim=0)
            batch_entropy = -torch.sum(p_mean * torch.log(p_mean + 1e-9))
            self.entropy_reg_loss = batch_entropy
            self._last_H_actual  = batch_entropy.item()
            self._last_H_target  = float('nan')

        # ==========================================
        # DIAGNOSTICS PHASE 1 & 2 (extended)
        # ==========================================
        if hasattr(self, "diagnostic_stats") and self.training:
            tau = self.current_tau
            self.diagnostic_stats["tau"].append(tau)
            self.diagnostic_stats["logits_std"].append(final_logits.std().item())
            self.diagnostic_stats["logits_mean"].append(final_logits.mean().item())
            self.diagnostic_stats["logits_max"].append(final_logits.max().item())
            self.diagnostic_stats["logits_min"].append(final_logits.min().item())
            self.diagnostic_stats["p_max"].append(patch_scores_probs.max().item())

            H_inst = -(patch_scores_probs * torch.log(patch_scores_probs + 1e-9)).sum(dim=-1).mean()
            self.diagnostic_stats["entropy"].append(H_inst.item())
            self.diagnostic_stats["entropy_target"].append(self._last_H_target)
            self.diagnostic_stats["logit_alpha_scale"].append(alpha_scale)

            # Batch diversity entropy (kept for monitoring even when not used as loss)
            p_mean_diag = patch_scores_probs.mean(dim=0)
            batch_ent_val = -torch.sum(p_mean_diag * torch.log(p_mean_diag + 1e-9))
            self.diagnostic_stats["batch_entropy"].append(batch_ent_val.item())

            p_base = base_patch_scores / (base_patch_scores.sum(dim=-1, keepdim=True) + 1e-9)
            ent_base = -torch.sum(p_base * torch.log(p_base + 1e-9), dim=-1).mean()
            self.diagnostic_stats["base_entropy"].append(ent_base.item())

            self.diagnostic_stats["y_tri_raw_mean"].append(y_tri_raw.mean().item())
            self.diagnostic_stats["y_tri_raw_std"].append(y_tri_raw.std().item())
            self.diagnostic_stats["y_tri_norm_mean"].append(y_tri_norm.mean().item())
            self.diagnostic_stats["y_tri_norm_std"].append(y_tri_norm.std().item())
            self.diagnostic_stats["beta"].append(self.beta.item())
            self.diagnostic_stats["gamma"].append(self.gamma.item())
            self.diagnostic_stats["gate_sigmoid"].append(torch.sigmoid(self.gate_param).item())
            self.diagnostic_stats["interaction_norm_check"].append(y_tri_norm.std().item())

            if self._selection_freq_ema is not None:
                ema_buf = self._selection_freq_ema
                self.diagnostic_stats["stability_ema_max"].append(ema_buf.max().item())
                self.diagnostic_stats["stability_ema_std"].append(ema_buf.std().item())

            if final_logits.requires_grad:
                final_logits.register_hook(
                    lambda g: self.diagnostic_stats["grad_score_head"].append(g.norm().item())
                )
            if x.requires_grad:
                x.register_hook(
                    lambda g: self.diagnostic_stats["grad_backbone"].append(g.norm().item())
                )

        # ==========================================
        # 5. GUMBEL SELECTION (unchanged logic)
        # ==========================================
        tau = self.current_tau

        tokens_sel, indices_sel, patch_scores, gs_tau = sample_gumbel_topk(
            tokens=x,
            scores=final_logits,
            n_alpha=n_alpha,
            tau=tau,
            hard=self.hard,
            straight_through=self.straight_through,
            generator=None,
        )

        # ==========================================
        # 6. UPDATE STABILITY EMA (NEW)
        # ==========================================
        # Must happen AFTER selection so the EMA reflects actual selections.
        # We build a hard binary mask [B, N-1] from indices_sel.
        if self.stability_bonus_enabled and self.training:
            with torch.no_grad():
                relative_sel = indices_sel[:, 1:] - 1              # [B, n_alpha] 0-based
                hard_mask = torch.zeros(B, num_patches, device=device)
                hard_mask.scatter_(1, relative_sel.clamp(0, num_patches - 1), 1.0)
                batch_freq = hard_mask.mean(dim=0)                  # [N-1]

                if self._selection_freq_ema is None or self._selection_freq_ema.shape[0] != num_patches:
                    self._selection_freq_ema = batch_freq.cpu()
                else:
                    d = self.stability_bonus_ema_decay
                    self._selection_freq_ema = (
                        d * self._selection_freq_ema + (1.0 - d) * batch_freq.cpu()
                    )

        # ==========================================
        # 7. BUILD last_adc_scores (unchanged)
        # ==========================================
        selected_patch_indices = indices_sel[:, 1:] - 1
        selected_patch_scores  = torch.gather(patch_scores_probs, 1, selected_patch_indices)
        cls_dummy = torch.ones((B, 1), dtype=selected_patch_scores.dtype, device=device)
        self.last_adc_scores  = torch.cat([cls_dummy, selected_patch_scores], dim=1)
        self.last_indices_sel = indices_sel
        self.last_original_N  = N

        return tokens_sel

    # ------------------------------------------------------------------
    # forward (unchanged)
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.block.drop_path1(self.block.ls1(self.block.attn(self.block.norm1(x))))
        x = x + self.block.drop_path2(self.block.ls2(self.block.mlp(self.block.norm2(x))))

        clean_val = False
        if not self.training and self._model_ref is not None:
            clean_val = getattr(self._model_ref, 'clean_validation', False)

        if self.compression_enabled and not clean_val:
            x = self.gumbel_compress(x)

        return x

    # ------------------------------------------------------------------
    # compress_labels  (unchanged)
    # ------------------------------------------------------------------

    def compress_labels(self, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
        return F.one_hot(labels, num_classes=num_classes).float()


# ============================================================
# Store_Class_Token_Attn_Wrapper  (unchanged)
# ============================================================

class Store_Class_Token_Attn_Wrapper(nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.attn = attn
        self.class_token_attention = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        qkv = (self.attn.qkv(x)
               .reshape(B, N, 3, self.attn.num_heads, self.attn.head_dim)
               .permute(2, 0, 3, 1, 4))
        q, k, v = qkv.unbind(0)
        q, k = self.attn.q_norm(q), self.attn.k_norm(k)

        q    = q * self.attn.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)

        self.class_token_attention = attn[:, :, 0, :].mean(dim=1)   # [B, N]

        attn = self.attn.attn_drop(attn)
        attn_output = attn @ v
        x = attn_output.transpose(1, 2).reshape(B, N, C)
        x = self.attn.proj(x)
        x = self.attn.proj_drop(x)
        return x


# ============================================================
# OUTER MODEL  (minimal changes: exposes register_epoch)
# ============================================================

class model(nn.Module):
    def __init__(self,
                 model: VisionTransformer,
                 channel,
                 split_index,
                 method_cfg: dict,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.method_cfg = method_cfg

        compression_enabled = method_cfg.get('compression_enabled', True)
        desired_compression = method_cfg.get('desired_compression', None)
        token_compression   = method_cfg.get('token_compression', 1.0)
        self.channel_eval_only    = method_cfg.get('channel_eval_only', False)
        self.semantic_waterfilling = method_cfg.get('semantic_waterfilling', True)

        if not compression_enabled:
            self.compression_ratio = 1.0
        else:
            if desired_compression is not None:
                assert token_compression is None or token_compression == 1.0
                self.compression_ratio = desired_compression
                self.method_cfg['token_compression'] = desired_compression
            else:
                if token_compression is None:
                    token_compression = 1.0
                self.compression_ratio = token_compression
                self.method_cfg['token_compression'] = token_compression

        self.compressor_module = None
        self.clean_validation  = False

        self.model = self.build_model(model, channel, split_index, self.method_cfg)
        self.channel = channel
        self.communication = 0
        self.name = "GumbelMethod"

    # ------------------------------------------------------------------
    # build_model (unchanged)
    # ------------------------------------------------------------------

    def build_model(self, model, channel, split_index, method_cfg):
        model.blocks[split_index - 1].attn = Store_Class_Token_Attn_Wrapper(
            model.blocks[split_index - 1].attn
        )
        model.blocks[split_index - 1] = Gumbel_Token_Selection_Block_Wrapper(
            block=model.blocks[split_index - 1],
            method_cfg=method_cfg
        )
        self.compressor_module = model.blocks[split_index - 1]

        object.__setattr__(self.compressor_module, '_model_ref', weakref.proxy(self))

        blocks_before = model.blocks[:split_index]
        blocks_after  = model.blocks[split_index:]
        model.blocks  = nn.Sequential(*blocks_before, channel, *blocks_after)

        if isinstance(channel, CommModuleWrapper):
            channel.set_score_source(self.compressor_module)
            if hasattr(channel, "set_channel_eval_only"):
                channel.set_channel_eval_only(self.channel_eval_only)
            if hasattr(channel, "set_semantic_waterfilling"):
                channel.set_semantic_waterfilling(self.semantic_waterfilling)
            compression_enabled = method_cfg.get('compression_enabled', True)
            if not compression_enabled and hasattr(channel, "comm"):
                channel.comm.use_bottleneck = False

        return model

    # ------------------------------------------------------------------
    # forward (unchanged)
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]
        if self.training:
            self.communication += self.compression_ratio * batch_size
        return self.model.forward(x)
