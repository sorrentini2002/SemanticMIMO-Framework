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

from .gumbel.gumbel import sample_gumbel_topk
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
        self.cov_reg_enabled = method_cfg.get('cov_reg_enabled', False)
        
        # Setup per Valutazione / Inferenza
        self.eval_k = method_cfg.get('eval_k', 32)
        self.gumbel_mc_enabled = method_cfg.get('gumbel_mc_enabled', False)
        self.gumbel_mc_tau = method_cfg.get('gumbel_mc_tau', 0.5)
        # Costruisce dizionario dinamico per eventuali metodi Diversify
        self.diversify_cfg = {
            'enabled': method_cfg.get('diversify_enabled', False),
            'lambda': method_cfg.get('diversify_lambda', 0.2),
            'metric': method_cfg.get('diversify_metric', 'cosine')
        }

        # Tracking variables
        self.n_new_tokens = 0
        self._global_step = 0

        # ==========================================
        # SIMPLICIAL INTERACTING GRAPH (Branca a bypass dei Gradienti)
        # ==========================================
        # Estrarre embed_dim in modo sicuro saltando eventuali wrapper:
        if hasattr(block, 'norm1'):
            embed_dim = block.norm1.weight.shape[0]
        else:
            embed_dim = block.mlp.fc1.in_features  # Fallback
            
        # Proiezione per estrarre M_cls (Vettore marginale di contesto)
        self.w_u = nn.Linear(embed_dim, embed_dim)
        # Livello di interazione per computare il Triangolo Simpliciale
        self.w_tri = nn.Linear(embed_dim, embed_dim)
        # LayerNorm per stabilità locale prima del prodotto
        self.norm_u = nn.LayerNorm(embed_dim)
        self.norm_tri = nn.LayerNorm(embed_dim)
        # Parametro di Forza Bruta inizializzato da config per controllare la polarizzazione dei logits
        self.gamma = nn.Parameter(torch.tensor(method_cfg.get('gamma_init', 5.0)))
        self.beta = nn.Parameter(torch.tensor(method_cfg.get('beta_init', 0.0)))

        # Inizializzatore dizionario diagnostico
        self.diagnostic_stats = {
            "tau": [], "logits_std": [], "logits_mean": [], "entropy": [],
            "grad_score_head": [], "grad_backbone": [],
            "payload_x_norm": [], "payload_out_norm": [], "payload_diff_norm": []
        }

    # ------------------------------------------------------------------
    # Step management
    # ------------------------------------------------------------------

    def register_step(self, step: int):
        """Update the internal global step counter (used for tau annealing)."""
        self._global_step = step

    @property
    def current_tau(self) -> float:
        """Current Gumbel-Softmax temperature based on the annealing schedule."""
        return compute_tau(
            self._global_step,
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
            3.  Apply Gumbel top-k sampling.
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

        # Number of patch tokens to keep
        target_n_alpha = max(1, int(self.token_compression * num_patches))
        
        # --- Multi-Budget Training: Vaccinazione Stocastica ---
        # Durante il training, estraiamo un numero random di token per evitare il "Sequence Length Shock"
        if self.training:
            min_k = min(8, target_n_alpha)
            max_k = min(num_patches, max(64, target_n_alpha * 2))
            n_alpha = torch.randint(min_k, max_k + 1, (1,)).item()
        else:
            n_alpha = target_n_alpha
            
        self.n_new_tokens = 1 + n_alpha  # CLS + selected patches

        # ---- Ritorno all'Attenzione CLS (La 'Gabbia') ----
        # I punteggi derivano dai pesi di attenzione originali del ViT: 
        # sono già confinati in [0, 1] perché in uscita da un softmax.
        cls_attention = self.block.attn.class_token_attention
        base_patch_scores = cls_attention[:, 1:] # [B, N-1]

        # ==========================================
        # 1. SOSTITUZIONE DELLA METRICA (Simplicial Scoring)
        # ==========================================
        # Estraiamo M_cls (il contesto marginale del token di classificazione)
        cls_token = x[:, 0:1, :] # [B, 1, D]
        patch_tokens = x[:, 1:, :] # [B, N-1, D]
        
        # Calcoliamo le proiezioni per l'interazione
        # [B, 1, D]
        m_cls = self.norm_u(self.w_u(cls_token))
        # [B, N-1, D]
        patch_tri = self.norm_tri(self.w_tri(patch_tokens))
        
        # Interazione: calcoliamo il peso simpliciale s_j^{tri}
        # Multi_Head attention interaction approssimata combinando la magnitudo L2
        # per "rigonfiare" artificialmente l'attenzione piatta.
        
        # Magnitudo dell'interazione [B, N-1]
        interaction_strength = torch.norm(m_cls * patch_tri, p=2, dim=-1)
        
        # Interazione base (senza gamma prepensato, applicato post-standardizzazione)
        simplicial_scores = base_patch_scores * (1.0 + interaction_strength)
        
        # ==========================================
        # 1. Eliminazione totale L1 Norm / Logaritmo
        # 2. Implementazione Z-Score Standardization + Trasformazione Affine
        # ==========================================
        # Forniamo Energie pure / Raw Logits
        sm_mean = simplicial_scores.mean(dim=-1, keepdim=True)
        sm_std = simplicial_scores.std(dim=-1, keepdim=True)
        # Forza media 0 e varianza 1 lungo la dimensione spaziale (token)
        standardized_scores = (simplicial_scores - sm_mean) / (sm_std + 1e-9)

        # Trasformazione Affine Apprendibile: Ripristina la capacità di gridare del segnale
        final_logits = self.gamma * standardized_scores + self.beta

        # Per il p_mean della regularization/entropy calcoliamo un softmax esplorativo
        patch_scores_probs = F.softmax(final_logits, dim=-1)

        # ---- Gumbel select directly from imported module ----
        tau = self.current_tau

        # -- Pre-Channel Batch Entropy Regularization (Spatial Diversity) --
        # 1. Passaggio alla Batch Entropy: calcoliamo la distribuzione media sul batch
        # per evitare l'Index Collapse, permettendo però alla rete di polarizzarsi 
        # (sharpness alta) sulla singola istanza/immagine individualmente.
        p_mean = patch_scores_probs.mean(dim=0)
        entropy_reg_loss = -torch.sum(p_mean * torch.log(p_mean + 1e-9))
        self.entropy_reg_loss = entropy_reg_loss

        # =====================================================================
        # DIAGNOSTICS PHASE 1 & 2: Sharpness, Temperature, & STE Gradient Audit
        # =====================================================================
        if hasattr(self, "diagnostic_stats") and self.training:
            # 1. Log Temperature and Sharpness
            self.diagnostic_stats["tau"].append(tau)
            self.diagnostic_stats["logits_std"].append(final_logits.std().item())
            self.diagnostic_stats["logits_mean"].append(final_logits.mean().item())
            
            # L'entropia matematica H = -sum(p * log p)
            entropy_val = -torch.sum(patch_scores_probs * torch.log(patch_scores_probs + 1e-9), dim=-1).mean()
            self.diagnostic_stats["entropy"].append(entropy_val.item())

            # 2. Registrazione Hooks per Audit del Gradiente STE
            if final_logits.requires_grad:
                final_logits.register_hook(lambda g: self.diagnostic_stats["grad_score_head"].append(g.norm().item()))
            if x.requires_grad:
                x.register_hook(lambda g: self.diagnostic_stats["grad_backbone"].append(g.norm().item()))
        # =====================================================================

        tokens_sel, indices_sel, patch_scores, gs_tau = sample_gumbel_topk(
            tokens=x,
            scores=final_logits,  # PASS AFFINE TRANSFORMED LOGITS
            n_alpha=n_alpha,
            tau=tau,
            hard=self.hard,
            straight_through=self.straight_through,
            generator=None,
        )
        # tokens_sel : [B, 1 + n_alpha, D]
        # patch_scores : [B, num_patches]

        # ---- Build last_adc_scores for MIMO waterfilling ----
        # Shape must be [B, 1 + n_alpha].
        # Position 0  → CLS dummy score = 1.0
        # Positions 1…n_alpha → patch scores of the *selected* patches.
        #
        # indices_sel[:, 1:] are global indices (1..N-1) of the kept patches.
        # patch_scores indices are 0..N-2.  So we need: patch_scores[:, idx-1].
        selected_patch_indices = indices_sel[:, 1:] - 1                    # [B, n_alpha], 0-based
        selected_patch_scores = torch.gather(patch_scores_probs, 1, selected_patch_indices)  # [B, n_alpha]

        cls_dummy = torch.ones((B, 1), dtype=selected_patch_scores.dtype, device=device)
        self.last_adc_scores = torch.cat([cls_dummy, selected_patch_scores], dim=1)  # [B, 1 + n_alpha]
        
        # --- Store indices for Server Spatial Reconstruction ---
        self.last_indices_sel = indices_sel
        self.last_original_N = N

        # NOTE: tokens_sel are already masked by sample_gumbel_topk (STE logic inside gumbel.py)

        return tokens_sel



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

        # --- Clean validation bypass (Rule 4) ---
        clean_val = False
        if not self.training and self._model_ref is not None:
            clean_val = getattr(self._model_ref, 'clean_validation', False)

        if self.compression_enabled and not clean_val:
            # Apply Gumbel token compression
            x = self.gumbel_compress(x)

        # If clean_val is True or compression_enabled is False, x passes through unmodified (no compression)
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
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)

        # Store CLS-row attention averaged across heads  →  [B, N]
        # NOTE: Do NOT detach here — gradient must flow through scores
        # to the QKV weights for Gumbel-Softmax STE to learn a selection policy.
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
        return self.model.forward(x)
