
import torch
import torch.nn.functional as F
from .utils import gather_tokens
import logging

logger = logging.getLogger(__name__)

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
