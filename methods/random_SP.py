import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import weakref
from timm.models import VisionTransformer
from comm.comm_module_wrapper import CommModuleWrapper


def select_random(tokens, n_alpha, generator=None, selection_cfg=None, pre_selected_indices=None):
    """
    Select n_alpha random patch tokens + CLS.
    
    Args:
        tokens: [B, N, D] (includes CLS at index 0)
        n_alpha: int (Number of patch tokens to keep)
        generator: torch.Generator for reproducibility.
        selection_cfg: dict or None.
        pre_selected_indices: [B, m] or None.
        
    Returns:
        tokens_sel: [B, 1 + n_alpha, D]
        indices_sel: [B, 1 + n_alpha]
    """
    if selection_cfg is None:
        selection_cfg = {}

    B, N, D = tokens.shape
    num_patches = N - 1
    
    # Generate random scores for patches
    # [B, N-1]
    # NOTE: We generate on CPU first to avoid device mismatch with generator (often on CPU)
    # and then move to the target device. This is safer for MPS/reproducibility.
    scores = torch.rand(B, num_patches, generator=generator, device='cpu').to(tokens.device)
    
    return select_by_scores(tokens, scores, n_alpha, selection_cfg, pre_selected_indices=pre_selected_indices)


def select_by_scores(tokens, scores, n_alpha, selection_cfg, embeddings=None, pre_selected_indices=None):
    """
    General selection function that supports:
    1. 'keep_cls_neighbors': Force keeping top-m tokens by CLS attention (if provided in pre_selected_indices).
    2. 'diversify': Greedy selection with diversity penalty.
    3. Standard Top-K.

    Args:
        tokens: [B, N, D] - Token embeddings (used for gather).
        scores: [B, N-1] - Scores for patches (index 0 is CLS, so scores align with indices 1..N-1).
        n_alpha: int - Target number of patch tokens to keep.
        selection_cfg: dict - Configuration dict.
        embeddings: [B, N-1, D'] - Embeddings used for diversity calculation (patches only). 
                                   If None and diversity is on, tokens[:, 1:, :] will be used.
        pre_selected_indices: [B, m] - Indices (global, 1..N-1) relative to 'tokens' that MUST be selected first.
                                       Usually these are the 'keep_cls_neighbors' candidates.

    Returns:
        tokens_sel: [B, 1 + n_alpha, D]
        indices_sel: [B, 1 + n_alpha]
        final_scores: [B, N-1] or whatever scores were used.
    """
    B, num_patches = scores.shape
    N = num_patches + 1
    device = tokens.device

    # Handle edge case: keep all
    if n_alpha >= num_patches:
        indices = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
        return tokens, indices, scores

    # Handle edge case: keep none (only CLS)
    if n_alpha <= 0:
        indices = torch.zeros((B, 1), dtype=torch.long, device=device)
        return tokens[:, :1, :], indices, scores

    # --- Diversity Config ---
    diversify_cfg = selection_cfg.get("diversify", {})
    use_diversity = diversify_cfg.get("enabled", False)
    
    # Store selected indices (global 1..N-1)
    final_indices_patches = None

    if not use_diversity:
        # --- Standard Top-K (with forced inclusion) ---
        if pre_selected_indices is not None and pre_selected_indices.shape[1] > 0:
            m = pre_selected_indices.shape[1]
            if m >= n_alpha:
                # If forced >= requested, take top n_alpha from forced
                final_indices_patches = pre_selected_indices[:, :n_alpha]
            else:
                remaining_k = n_alpha - m
                
                # Mask out selected in scores (-inf)
                scores_masked = scores.clone()
                # pre_selected_indices are 1..N-1. score indices are 0..N-2.
                score_indices = pre_selected_indices - 1
                
                src = torch.tensor(float('-inf'), device=device).expand_as(score_indices)
                scores_masked.scatter_(1, score_indices, src)
                
                # Top-k on remaining
                _, top_rem_idx = torch.topk(scores_masked, remaining_k, dim=1)
                
                # Convert back to global
                top_rem_idx = top_rem_idx + 1
                
                # Combine
                final_indices_patches = torch.cat([pre_selected_indices, top_rem_idx], dim=1)
        else:
            # Plain top-k
            _, top_idx = torch.topk(scores, n_alpha, dim=1)
            final_indices_patches = top_idx + 1

    else:
        # --- Diversity-Aware Greedy Selection ---
        lambda_val = diversify_cfg.get("lambda", 0.2)
        metric = diversify_cfg.get("metric", "cosine") # "cosine" or "l2"
        
        # Prepare embeddings
        if embeddings is None:
            embeddings = tokens[:, 1:, :] # [B, N-1, D]
        
        if metric == "cosine":
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        # Mask for scores to avoid re-selecting
        curr_scores = scores.clone()
        
        final_indices_list = []
        current_cluster_embs = None # [B, k_curr, D]
        start_k = 0
        
        # 1. Handle Pre-selected
        if pre_selected_indices is not None and pre_selected_indices.shape[1] > 0:
            m = pre_selected_indices.shape[1]
            if m > n_alpha:
                 pre_selected_indices = pre_selected_indices[:, :n_alpha]
                 m = n_alpha
            
            # Mask scores
            loc_idx = pre_selected_indices - 1
            src_inf = torch.tensor(float('-inf'), device=device).expand_as(loc_idx)
            curr_scores.scatter_(1, loc_idx, src_inf)
            
            # Init cluster embs
            idx_expanded = loc_idx.unsqueeze(-1).expand(-1, -1, embeddings.size(-1))
            current_cluster_embs = torch.gather(embeddings, 1, idx_expanded)
            
            final_indices_list.append(pre_selected_indices)
            start_k = m
        else:
            # Pick first strictly by score
            _, first_idx = torch.topk(curr_scores, 1, dim=1) # [B, 1] relative
            
            src_inf = torch.tensor(float('-inf'), device=device).expand_as(first_idx)
            curr_scores.scatter_(1, first_idx, src_inf)
            
            idx_expanded = first_idx.unsqueeze(-1).expand(-1, -1, embeddings.size(-1))
            current_cluster_embs = torch.gather(embeddings, 1, idx_expanded)
            
            final_indices_list.append(first_idx + 1)
            start_k = 1
            
        # 2. Greedy Loop
        # We need to pick n_alpha - start_k more
        for k in range(start_k, n_alpha):
            # Similarity Penalty
            # [B, N-1, k_curr]
            if metric == "cosine":
                sim_matrix = torch.bmm(embeddings, current_cluster_embs.transpose(1, 2))
                max_sim, _ = sim_matrix.max(dim=2) # [B, N-1]
                penalty = lambda_val * max_sim
            else:
                # L2 distance metric: maximize score - lambda * (-min_dist) = score + lambda * min_dist
                # Actually user said: s[j] - lambda * max_{k} sim(u_j, u_k)
                # If metric is L2, 'sim' is subjective. Usually implies -dist.
                # But let's assume 'sim' means L2 distance (which is dissim).
                # Then we want to maximize score + lambda * min_dist to current set.
                # However, common diversity usage is maximizing distance.
                # "maximize s[j] - lambda * max_sim" -> if max_sim is high (close), we penalize.
                # If L2 is distance, we want to maximize score + lambda * dist?
                # Let's interpret "metric: l2" as using L2 distance as the *inverse* of similarity.
                # Or just use Negative L2 as similarity.
                # sim = - ||u_j - u_k||
                # Then - lambda * max(sim) = - lambda * max(-dist) = - lambda * (-min_dist) = + lambda * min_dist.
                # So we maximize score + lambda * distance_to_closest_selected.
                
                # Compute dists: [B, N-1, k_curr]
                # dist = sqrt(|x|^2 + |y|^2 - 2xy)
                # For stability, use squared distance usually? But sticking to sim interpretation.
                # Let's use simple expansion for dist.
                
                # Expand dims
                # emb: [B, N-1, 1, D]
                # cluster: [B, 1, k, D]
                dist = torch.cdist(embeddings, current_cluster_embs, p=2) # [B, N-1, k]
                
                # min dist to set
                min_dist, _ = dist.min(dim=2) # [B, N-1]
                
                # Objective: score + lambda * min_dist
                # (Equivalent to score - lambda * (-min_dist))
                # So penalty = - lambda * min_dist
                penalty = - lambda_val * min_dist

            objective = curr_scores - penalty
            
            # Pick best
            _, best_idx = torch.topk(objective, 1, dim=1) # [B, 1]
            
            # Update
            src_inf = torch.tensor(float('-inf'), device=device).expand_as(best_idx)
            curr_scores.scatter_(1, best_idx, src_inf)
            
            idx_expanded = best_idx.unsqueeze(-1).expand(-1, -1, embeddings.size(-1))
            new_emb = torch.gather(embeddings, 1, idx_expanded)
            
            current_cluster_embs = torch.cat([current_cluster_embs, new_emb], dim=1)
            final_indices_list.append(best_idx + 1)
            
        final_indices_patches = torch.cat(final_indices_list, dim=1)

    # Sort indices
    # We sort to prevent arbitrary permutation effects on position embeddings relative to compute
    # (though in ViT it's set permutation invariant after positions added, sorting is cleaner)
    final_indices_patches, _ = torch.sort(final_indices_patches, dim=1)
    
    # Prepend CLS
    cls_idx = torch.zeros((B, 1), dtype=torch.long, device=device)
    final_indices = torch.cat([cls_idx, final_indices_patches], dim=1)
    
    tokens_sel = gather_tokens(tokens, final_indices)
    
    return tokens_sel, final_indices, scores


# =============================================================================
# MODEL INTERFACE CLASSES
# =============================================================================

class Store_Class_Token_Attn_Wrapper(nn.Module):
    """
    Wraps a timm attention module and stores CLS-row attention scores.
    """
    def __init__(self, attn):
        super().__init__()
        self.attn = attn
        self.class_token_attention = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, n_tokens, channels = x.shape
        qkv = (
            self.attn.qkv(x)
            .reshape(bsz, n_tokens, 3, self.attn.num_heads, self.attn.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.attn.q_norm(q), self.attn.k_norm(k)
        q = q * self.attn.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        # Shape [B, N]: mean over heads of CLS row attention.
        self.class_token_attention = attn[:, :, 0, :].mean(dim=1)
        attn = self.attn.attn_drop(attn)
        attn_output = attn @ v
        x = attn_output.transpose(1, 2).reshape(bsz, n_tokens, channels)
        x = self.attn.proj(x)
        x = self.attn.proj_drop(x)
        return x


class Random_SP_Block_Wrapper(nn.Module):
    """
    Selects random patch tokens with optional diversity and CLS-neighbor support.
    """
    def __init__(self, block, method_cfg):
        super().__init__()
        self.block = block
        self.method_cfg = method_cfg
        self.compression_enabled = method_cfg.get("compression_enabled", True)
        self.token_compression = method_cfg.get("token_compression", 1.0)
        self.eval_k = method_cfg.get("eval_k", None)
        self.keep_cls_neighbors = method_cfg.get("keep_cls_neighbors", 0)

        # Interface variables
        self.last_adc_scores = None
        self.n_new_tokens = 0
        self.generator = None
        object.__setattr__(self, "_model_ref", None)

    def _select(self, x: torch.Tensor) -> torch.Tensor:
        bsz, n_tokens, dim = x.shape
        num_patches = n_tokens - 1
        device = x.device

        if num_patches <= 0:
            self.n_new_tokens = 1
            self.last_adc_scores = torch.ones((bsz, 1), device=device, dtype=x.dtype)
            return x

        # Compression rate
        if not self.training and self.eval_k is not None:
             n_alpha = max(1, min(int(self.eval_k), num_patches))
        else:
             n_alpha = max(1, min(int(self.token_compression * num_patches), num_patches))
        
        self.n_new_tokens = 1 + n_alpha

        # Handle pre-selected indices (top-m CLS neighbors)
        pre_selected_indices = None
        if self.keep_cls_neighbors > 0:
            cls_attn = self.block.attn.class_token_attention
            if cls_attn is not None:
                patch_attn = cls_attn[:, 1:]
                m = min(self.keep_cls_neighbors, n_alpha, num_patches)
                _, pre_selected_indices = torch.topk(patch_attn, m, dim=1)
                pre_selected_indices = pre_selected_indices + 1 # global 1..N-1

        # Perform selection using local functions
        tokens_sel, indices_sel, _ = select_random(
            tokens=x,
            n_alpha=n_alpha,
            generator=self.generator,
            selection_cfg=self.method_cfg,
            pre_selected_indices=pre_selected_indices
        )

        # Build last_adc_scores for waterfilling using CLS attention
        cls_attn = self.block.attn.class_token_attention
        if cls_attn is not None:
            selected_patch_indices = indices_sel[:, 1:] - 1
            patch_attn = cls_attn[:, 1:]
            selected_patch_scores = torch.gather(patch_attn, 1, selected_patch_indices)
            cls_dummy = torch.ones((bsz, 1), device=device, dtype=x.dtype)
            self.last_adc_scores = torch.cat([cls_dummy, selected_patch_scores], dim=1)
        else:
            self.last_adc_scores = torch.ones((bsz, 1 + n_alpha), device=device, dtype=x.dtype)

        # Store for spatial reconstruction
        self.last_indices_sel = indices_sel
        self.last_original_N = n_tokens

        return tokens_sel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normal block pass
        x = x + self.block.drop_path1(self.block.ls1(self.block.attn(self.block.norm1(x))))
        x = x + self.block.drop_path2(self.block.ls2(self.block.mlp(self.block.norm2(x))))

        clean_val = False
        if not self.training and self._model_ref is not None:
            clean_val = getattr(self._model_ref, "clean_validation", False)

        if self.compression_enabled and not clean_val:
            x = self._select(x)

        return x

    def compress_labels(self, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
        return F.one_hot(labels, num_classes=num_classes).float()


class model(nn.Module):
    """
    Split-learning model using Random-SP token selection.
    """
    def __init__(self, model: VisionTransformer, channel, split_index, method_cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.method_cfg = method_cfg
        
        compression_enabled = method_cfg.get("compression_enabled", True)
        desired_compression = method_cfg.get("desired_compression", None)
        token_compression = method_cfg.get("token_compression", 1.0)
        self.channel_eval_only = method_cfg.get("channel_eval_only", False)
        self.semantic_waterfilling = method_cfg.get("semantic_waterfilling", True)

        if not compression_enabled:
            self.compression_ratio = 1.0
        else:
            if desired_compression is not None:
                self.compression_ratio = desired_compression
                self.method_cfg["token_compression"] = desired_compression
            else:
                self.compression_ratio = token_compression if token_compression is not None else 1.0
                self.method_cfg["token_compression"] = self.compression_ratio

        self.compressor_module = None
        self.clean_validation = False
        self.model = self.build_model(model, channel, split_index, self.method_cfg)
        self.channel = channel
        self.communication = 0
        self.name = "RandomSPMethod"

    def build_model(self, model: VisionTransformer, channel, split_index: int, method_cfg: dict):
        # Wrap attention to capture CLS scores
        model.blocks[split_index - 1].attn = Store_Class_Token_Attn_Wrapper(
            model.blocks[split_index - 1].attn
        )

        # Wrap block with selection logic
        model.blocks[split_index - 1] = Random_SP_Block_Wrapper(
            block=model.blocks[split_index - 1],
            method_cfg=method_cfg
        )
        self.compressor_module = model.blocks[split_index - 1]

        # Connect back-reference
        object.__setattr__(self.compressor_module, "_model_ref", weakref.proxy(self))

        # Split model
        blocks_before = model.blocks[:split_index]
        blocks_after = model.blocks[split_index:]
        model.blocks = nn.Sequential(*blocks_before, channel, *blocks_after)

        # Configure channel
        if isinstance(channel, CommModuleWrapper):
            channel.set_score_source(self.compressor_module)
            if hasattr(channel, "set_channel_eval_only"):
                channel.set_channel_eval_only(self.channel_eval_only)
            if hasattr(channel, "set_semantic_waterfilling"):
                channel.set_semantic_waterfilling(self.semantic_waterfilling)

        return model

    def forward(self, x: torch.Tensor):
        if self.training:
            self.communication += self.compression_ratio * x.shape[0]
        return self.model.forward(x)

def gather_tokens(tokens, indices):
    """
    Gather tokens based on indices. [B, N, D] -> [B, K, D]
    """
    B, N, D = tokens.shape
    indices_expanded = indices.unsqueeze(-1).expand(-1, -1, D)
    return torch.gather(tokens, 1, indices_expanded)
