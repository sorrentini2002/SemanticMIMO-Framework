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
from methods.token_utils import gather_tokens, ClassTokenAttentionTrackerWrapper

logger = logging.getLogger(__name__)


# ============================================================
# HELPER FUNCTIONS (from former methods/gumbel/ modules)
# ============================================================

# gather_tokens is now imported from methods.token_utils



def compute_tau(step, tau_max=1.0, tau_min=0.3, num_steps=10000, anneal_mode="linear"):
    """
    Compute Gumbel-Softmax temperature tau based on annealing schedule.
    
    Args:
        step: int, current global step
        tau_max: float, initial temperature
        tau_min: float, minimum temperature
        num_steps: int, duration of annealing (in steps)
        anneal_mode: str, "linear", "cosine", or "exp"
    
    Returns:
        float: current tau
    """
    if step >= num_steps:
        return tau_min
        
    t = step / float(num_steps) # 0 to 1
    
    if anneal_mode == 'linear':
        # Linear decay from tau_max to tau_min
        return tau_max - t * (tau_max - tau_min)
        
    elif anneal_mode == 'cosine':
        # Cosine decay
        # Starts at tau_max, ends at tau_min
        cosine_decay = 0.5 * (1 + math.cos(math.pi * t))
        return tau_min + (tau_max - tau_min) * cosine_decay
        
    elif anneal_mode == 'exp':
        # Exponential decay
        # tau = tau_max * (tau_min / tau_max)^t
        return tau_max * (tau_min / tau_max)**t
        
    else:
        # constant (or default fallback)
        return tau_max


def compute_gumbel_mc_scores(scores, num_samples=16, tau=0.5, aggregate="mean", generator=None):
    """
    Compute Monte-Carlo averaged token probabilities via repeated Gumbel perturbations.
    
    Args:
        scores: [B, N-1] - Unnormalized log-probabilities.
        num_samples: int - Number of MC samples.
        tau: float - Gumbel temperature for eval.
        aggregate: str - "mean" or "median" aggregation.
        generator: torch.Generator for reproducibility.
        
    Returns:
        p_mean: [B, N-1] - Aggregated probabilities.
    """
    B, num_patches = scores.shape
    device = scores.device
    
    # Generate Gumbel noise for num_samples at once.
    # shape: [num_samples, B, num_patches]
    eps = 1e-20
    if generator is None:
        u = torch.rand((num_samples, B, num_patches), device=device)
    else:
        u = torch.rand((num_samples, B, num_patches), device='cpu', generator=generator).to(device)
        
    gumbel_noise = -torch.log(-torch.log(u + eps) + eps)
    
    # Expand scores to [num_samples, B, num_patches]
    scores_expanded = scores.unsqueeze(0).expand(num_samples, -1, -1)
    
    # 1. Removal of Logarithm (Linear Sampling)
    # 2. Decay Noise Amplitude: physical amplitude of noise decays with temperature
    noisy_scores = scores_expanded + (gumbel_noise * tau)
    
    # 3. Exponential Amplification via 'Double Softmax'
    p_soft = F.softmax(noisy_scores / tau, dim=-1)
    
    # Aggregate
    if aggregate == "median":
        p_agg, _ = p_soft.median(dim=0)
    else:
        p_agg = p_soft.mean(dim=0)
        
    return p_agg


def sample_gumbel_from_scores(scores, n_alpha, tau=1.0, hard=True, straight_through=True, generator=None):
    """
    Perform Gumbel-Softmax sampling on given scores.
    
    Args:
        scores: [B, N-1] - Unnormalized log-probabilities (or scores) for patch tokens.
        n_alpha: int - Number of tokens to select.
        tau: float - Gumbel temperature.
        hard: bool - Hard selection.
        straight_through: bool - ST estimation.
        generator: torch.Generator.
        
    Returns:
        indices_sel_patches: [B, n_alpha] (Relative indices 0..N-2)
        m: [B, N-1] (Straight-through mask, or None)
        gs_tau: float (Used tau)
    """
    B, num_patches = scores.shape
    
    # 1. Gumbel Noise
    eps = 1e-20
    if generator is None:
         u = torch.rand_like(scores)
    else:
         u = torch.rand(scores.shape, device='cpu', generator=generator).to(scores.device)
         
    gumbel_noise = -torch.log(-torch.log(u + eps) + eps)
    
    # 2. Add noise with Decaying Amplitude
    # 1. Removal of Logarithm: treat attention probabilities directly as linear inputs
    noisy_scores = scores + (gumbel_noise * tau)
    
    # 3. Soft Probabilities - Exponential Amplification via 'Double Softmax'
    # By omitting the logarithm, dividing the already-softmaxed linear scores by tau
    # inside this second softmax converts micro-differences into explosive peaks.
    m_soft = F.softmax(noisy_scores / tau, dim=-1)
    
    # 4. Hard Selection
    _, topk_relative_indices = torch.topk(noisy_scores, k=n_alpha, dim=1, sorted=False)
    
    # 5. Straight-Through Mask
    # Pure mathematical magnitude: we do NOT artificially scale the gradient by n_alpha,
    # allowing the raw JSCC physics and true probabilities to drive the backpropagation.
    m = None
    if straight_through:
        m_hard = torch.zeros_like(scores)
        m_hard.scatter_(1, topk_relative_indices, 1.0)
        m = m_hard + (m_soft - m_soft.detach())
        
    return topk_relative_indices, m, tau


def sample_gumbel_topk(tokens, attn=None, n_alpha=1, tau=1.0, hard=True, straight_through=True, generator=None, scores=None):
    """
    Select top-k tokens based on CLS attention scores or explicit patch scores, with Gumbel noise.
    
    Args:
        scores: [B, N-1] explicit scores for each patch token. If provided, attn is ignored.
    """
    B, N, D = tokens.shape
    
    # Handle already averaged attention
    if attn is not None:
        if attn.dim() == 4:
            attn_mean = attn.mean(dim=1)
        else:
            attn_mean = attn
        
    # Validation
    if n_alpha >= N - 1:
        indices = torch.arange(N, device=tokens.device).unsqueeze(0).expand(B, -1)
        return tokens, indices, None, tau
        
    if n_alpha <= 0:
        indices = torch.zeros((B, 1), dtype=torch.long, device=tokens.device)
        return gather_tokens(tokens, indices), indices, None, tau
        
    # Get scores
    if scores is not None:
        patch_scores = scores
    elif attn is not None:
        cls_scores = attn_mean[:, 0, :]
        patch_scores = cls_scores[:, 1:] # [B, N-1]
    else:
        raise ValueError("Must provide either attn or scores to sample_gumbel_topk")
    
    # Use extracted function
    topk_relative_indices, m, gs_tau = sample_gumbel_from_scores(
        patch_scores, n_alpha, tau, hard, straight_through, generator
    )
    
    # Convert to global indices
    topk_indices = topk_relative_indices + 1
    topk_indices, _ = torch.sort(topk_indices, dim=1)
    
    # Add CLS
    cls_indices = torch.zeros((B, 1), dtype=torch.long, device=tokens.device)
    indices_sel = torch.cat([cls_indices, topk_indices], dim=1)

    # Gather
    tokens_sel = gather_tokens(tokens, indices_sel)
    
    # Apply Straight-Through
    if straight_through and m is not None:
        sorted_relative_indices = topk_indices - 1
        m_gathered = torch.gather(m, 1, sorted_relative_indices)
        m_cls = torch.ones((B, 1), device=tokens.device, dtype=tokens.dtype)
        m_final = torch.cat([m_cls, m_gathered], dim=1).unsqueeze(-1)
        tokens_sel = tokens_sel * m_final
        
    return tokens_sel, indices_sel, patch_scores, gs_tau


# ============================================================
# BLOCK WRAPPER — Gumbel token selection at the split point
# ============================================================

class GumbelTokenSelectorBlockWrapper(nn.Module):
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
    # Gumbel token selection (ADDITIVE HYBRID STRATEGY)
    # ------------------------------------------------------------------


    def gumbel_compress(self, x: torch.Tensor) -> torch.Tensor:
        """
        COOPERATIVE JSCC STRATEGY (Gate * Gamma).
        
        Formula: z = LN(a_cls) + sigmoid(g) * gamma * LN(y_raw)
        
        Where:
        - a_cls = Native DeiT attention scores from the class token [B, N-1]
        - y_raw = || W_u(X_cls) ⊙ W_tri(X_patch) ||_2  [Projected geometric interaction]
        - LN(·) = Layer Normalization applied independently to BOTH branches 
                  to ensure a perfectly balanced numeric initialization.
        - sigmoid(g) ∈ (0, 1) = Soft-gate to control channel noise filtration.
        - gamma = Learnable contrast multiplier to break the variance unit ceiling.
        - beta = Bypassed (implicitly fixed to 1.0) to avoid scaling race conditions.
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

        # --- Extract Tokens & Native Attention ---
        cls_token    = x[:, 0:1, :]    # [B, 1, D]
        patch_tokens = x[:, 1:, :]     # [B, N-1, D]

        # Extract native a_cls from the block's attention registry
        cls_attention = self.block.attn.class_token_attention
        base_patch_scores = cls_attention[:, 1:]  # [B, N-1] (a_cls)

        # ==========================================================
        # BRANCH 1: SOURCE-AWARE SEMANTIC ANCHOR (Fixed at scale 1.0)
        # ==========================================================
        a_cls_norm = F.layer_norm(base_patch_scores, base_patch_scores.shape[-1:])  # [B, N-1]

        # ==========================================================
        # BRANCH 2: CHANNEL-AWARE GEOMETRIC PROJECTION (Gate * Gamma)
        # ==========================================================
        # Step 1: Compute learned projections via W_u and W_tri
        m_cls     = self.w_u(cls_token)       # [B, 1, D]
        patch_tri = self.w_tri(patch_tokens)  # [B, N-1, D]

        # Step 2: Hadamard product (element-wise multiply with broadcasting)
        hadamard_product = m_cls * patch_tri  # [B, N-1, D]

        # Step 3: L2 norm across feature dimension
        y_raw = torch.norm(hadamard_product, p=2, dim=-1)  # [B, N-1]

        # Step 4: LayerNorm to stabilize geometric variance before contrast modulation
        y_norm = F.layer_norm(y_raw, y_raw.shape[-1:])  # [B, N-1]

        # Step 5: Soft-Gating AND Contrast Expansion (Cooperative Gating)
        gate_gain = torch.sigmoid(self.gate_param)  # Scalar in (0, 1)
        
        # Dual parameter chain: sigmoid(g) * gamma * LN(y)
        branch_geometric = gate_gain * self.gamma * y_norm  # [B, N-1]

        # ==========================================================
        # ADDITIVE FUSION (The Regulated Duel)
        # ==========================================================
        raw_logits = a_cls_norm + branch_geometric  # [B, N-1]

        # ==========================================================
        # DIAGNOSTICS: Extract statistics BEFORE curriculum scaling
        # ==========================================================
        y_tri_raw_mean = y_raw.mean().item()
        y_tri_raw_std = y_raw.std().item()
        gate_sigmoid_val = gate_gain.item()
        
        # ==========================================
        # 2. LOGIT SCALING DINAMICO (curriculum, kept)
        # ==========================================
        if self.training:
            alpha_scale = self._compute_logit_scale()
        else:
            alpha_scale = 1.0

        final_logits = raw_logits * alpha_scale  # [B, N-1]

        # ==========================================
        # 3. STABILITY BONUS — apply EMA from PREVIOUS batches
        # ==========================================
        if self.stability_bonus_enabled and self._selection_freq_ema is not None:
            ema = self._selection_freq_ema.to(device=device, dtype=final_logits.dtype)
            if ema.shape[0] == num_patches:
                ema_centered = ema - ema.mean()
                final_logits = final_logits + self.stability_bonus_weight * ema_centered.unsqueeze(0)

        # Soft probabilities (for entropy computation and ADC scores)
        patch_scores_probs = F.softmax(final_logits, dim=-1)   # [B, N-1]

        # ==========================================
        # 4. ENTROPY BOTTLENECK LOSS
        # ==========================================
        if self.entropy_bottleneck_enabled and self.training:
            H_actual = -(patch_scores_probs * torch.log(patch_scores_probs + 1e-9)).sum(dim=-1).mean()
            H_target_val = self._compute_entropy_target()
            entropy_ceiling_loss = torch.clamp(H_actual - H_target_val, min=0.0)
            self.entropy_reg_loss = self.entropy_bottleneck_weight * entropy_ceiling_loss
            self._last_H_actual  = H_actual.item()
            self._last_H_target  = H_target_val
        else:
            p_mean = patch_scores_probs.mean(dim=0)
            batch_entropy = -torch.sum(p_mean * torch.log(p_mean + 1e-9))
            self.entropy_reg_loss = batch_entropy
            self._last_H_actual  = batch_entropy.item()
            self._last_H_target  = float('nan')

        # ==========================================
        # DIAGNOSTICS PHASE (with proper key population)
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

            p_mean_diag = patch_scores_probs.mean(dim=0)
            batch_ent_val = -torch.sum(p_mean_diag * torch.log(p_mean_diag + 1e-9))
            self.diagnostic_stats["batch_entropy"].append(batch_ent_val.item())

            # Base attention extraction for telemetry reference
            cls_attention = self.block.attn.class_token_attention
            base_patch_scores = cls_attention[:, 1:]
            p_base = base_patch_scores / (base_patch_scores.sum(dim=-1, keepdim=True) + 1e-9)
            ent_base = -torch.sum(p_base * torch.log(p_base + 1e-9), dim=-1).mean()
            self.diagnostic_stats["base_entropy"].append(ent_base.item())

            # ✓ SPEC COMPLIANCE: Populate diagnostic telemetry keys safely
            self.diagnostic_stats["y_tri_raw_mean"].append(y_tri_raw_mean)
            self.diagnostic_stats["y_tri_raw_std"].append(y_tri_raw_std)
            self.diagnostic_stats["y_tri_norm_mean"].append(y_norm.mean().item())
            self.diagnostic_stats["y_tri_norm_std"].append(y_norm.std().item())
            self.diagnostic_stats["gate_sigmoid"].append(gate_sigmoid_val)
            self.diagnostic_stats["beta"].append(self.beta.item())   # Inactive parameter logged safely
            self.diagnostic_stats["gamma"].append(self.gamma.item()) # ACTIVE parameter logged dynamically
            self.diagnostic_stats["interaction_norm_check"].append(y_norm.std().item())

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
        # 6. UPDATE STABILITY EMA
        # ==========================================
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
        # 7. BUILD last_adc_scores — Semantic Waterfilling Scores
        # ==========================================
        selected_patch_indices = indices_sel[:, 1:] - 1
        selected_logits = torch.gather(final_logits, 1, selected_patch_indices)  # [B, n_alpha]
        selected_scores = F.softmax(selected_logits, dim=-1)                     # [B, n_alpha]
        cls_score = selected_scores.max(dim=1, keepdim=True).values + 0.01       # [B, 1]
        self.last_adc_scores  = torch.cat([cls_score, selected_scores], dim=1)   # [B, 1+n_alpha]
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


# ClassTokenAttentionTrackerWrapper is now imported from methods.token_utils




# ============================================================
# ServerAttentionEntropyTracker — Server-side attention entropy capture
# ============================================================

class ServerAttentionEntropyTracker(nn.Module):
    """
    Wraps a timm Attention module on the server side to capture
    the per-forward Shannon entropy of the softmax attention matrix.

    The wrapper replicates the full Attention forward pass verbatim
    so that gradients and outputs are identical to the unwrapped module.
    After each forward the scalar mean entropy (averaged over batch,
    heads, and query positions) is stored in ``self.last_entropy``.

    Metric definition:
        For attention matrix A ∈ R^{B×H×N×N} (post-softmax):
            H_q = -Σ_k A[b,h,q,k] * log(A[b,h,q,k] + ε)   [per query]
            scalar = mean over b, h, q
        Unit: nats.  Higher entropy → more uniform / noise-dominated attention.

    Note:
        - The entropy is computed inside torch.no_grad() to avoid adding
          any gradient computation overhead.
        - This wrapper is ONLY attached to server-side blocks (post-channel).
          Client-side blocks remain unwrapped.
    """

    _ENTROPY_EPS: float = 1e-9  # prevent log(0)

    def __init__(self, attn: nn.Module):
        super().__init__()
        self.attn = attn
        self.last_entropy: float = 0.0  # updated every forward

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        B, N, C = x.shape

        # ── Replicate timm Attention.forward exactly ──────────────────
        qkv = (
            self.attn.qkv(x)
            .reshape(B, N, 3, self.attn.num_heads, self.attn.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.attn.q_norm(q), self.attn.k_norm(k)

        q = q * self.attn.scale
        attn_logits = q @ k.transpose(-2, -1)       # [B, H, N, N]
        if attn_mask is not None:
            attn_logits = attn_logits + attn_mask
        attn_weights = attn_logits.softmax(dim=-1)  # [B, H, N, N]

        # ── Capture entropy (no-grad) ──────────────────────────────────
        with torch.no_grad():
            log_a = torch.log(attn_weights + self._ENTROPY_EPS)
            entropy_per_query = -(attn_weights * log_a).sum(dim=-1)  # [B, H, N]
            self.last_entropy = float(entropy_per_query.mean().item())

        # ── Complete the forward pass ──────────────────────────────────
        attn_weights = self.attn.attn_drop(attn_weights)
        attn_out = attn_weights @ v                              # [B, H, N, head_dim]
        x = attn_out.transpose(1, 2).reshape(B, N, C)
        x = self.attn.proj(x)
        x = self.attn.proj_drop(x)
        return x

class GumbelSplitLearningModel(nn.Module):
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
        # Initialise before build_model so the reference is available
        # if build_model needs it; populated inside build_model.
        self._server_attn_wrappers: list = []  # list[ServerAttentionEntropyTracker]

        self.model = self.build_model(model, channel, split_index, self.method_cfg)
        self.channel = channel
        self.communication = 0
        self.name = "GumbelMethod"

    # ------------------------------------------------------------------
    # build_model (unchanged)
    # ------------------------------------------------------------------

    def build_model(self, model, channel, split_index, method_cfg):
        model.blocks[split_index - 1].attn = ClassTokenAttentionTrackerWrapper(
            model.blocks[split_index - 1].attn
        )
        model.blocks[split_index - 1] = GumbelTokenSelectorBlockWrapper(
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

        # ── Install ServerAttentionEntropyTracker on server-side blocks ────────
        # Server blocks are those AFTER the channel in model.blocks.
        # model.blocks = [client_blocks..., channel, server_blocks...]
        # channel is at index split_index (0-based after rebuild).
        entropy_wrappers = []
        for block in blocks_after:
            if hasattr(block, 'attn'):
                block.attn = ServerAttentionEntropyTracker(block.attn)
                entropy_wrappers.append(block.attn)
        # Store reference on self for aggregation in forward()
        self._server_attn_wrappers = entropy_wrappers

        return model

    # ------------------------------------------------------------------
    # forward (unchanged)
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]
        if self.training:
            self.communication += self.compression_ratio * batch_size

        out = self.model.forward(x)

        # ── Aggregate server-side attention entropy ───────────────────
        # After the forward pass, all ServerAttentionEntropyTracker instances have
        # stored their per-block entropy in .last_entropy.  Compute the
        # global mean across all server blocks (single scalar per forward).
        if self._server_attn_wrappers:
            entropies = [w.last_entropy for w in self._server_attn_wrappers]
            avg_entropy = sum(entropies) / len(entropies)
            # Write into channel's last_info so training/eval loops can log it.
            if hasattr(self.channel, "last_info") and isinstance(self.channel.last_info, dict):
                self.channel.last_info["server_attn_entropy"] = avg_entropy

        return out
