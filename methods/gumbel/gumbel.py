import torch
import torch.nn.functional as F
from .utils import gather_tokens

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
    
    # Broadcast scores to [num_samples, B, num_patches]
    scores_expanded = scores.unsqueeze(0).expand(num_samples, -1, -1)
    
    # Canonical Gumbel-Softmax: softmax((logits + gumbel) / tau)
    tau = max(float(tau), 1e-6)
    noisy_scores = scores_expanded + gumbel_noise
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
        scores: [B, N-1] - Logits for patch tokens.
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
    
    # 2. Add noise in logits space
    tau = max(float(tau), 1e-6)
    noisy_scores = scores + gumbel_noise
    
    # 3. Soft Probabilities
    # m_soft approximates softmax((scores + gumbel) / tau)
    m_soft = F.softmax(noisy_scores / tau, dim=-1)
    
    # 4. Hard Selection
    _, topk_relative_indices = torch.topk(noisy_scores, k=n_alpha, dim=1, sorted=False)
    
    # 5. Straight-Through Mask
    m = None
    if straight_through:
        m_hard = torch.zeros_like(scores)
        m_hard.scatter_(1, topk_relative_indices, 1.0)
        m = m_hard + (m_soft * n_alpha - (m_soft * n_alpha).detach())
        
    return topk_relative_indices, m, tau

def sample_gumbel_topk(tokens, attn, n_alpha, tau=1.0, hard=True, straight_through=True, generator=None):
    """
    Select top-k tokens based on CLS attention scores with Gumbel noise.
    """
    B, N, D = tokens.shape
    
    # Handle already averaged attention
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
        
    # Get scores (CLS attention)
    cls_scores = attn_mean[:, 0, :]
    patch_scores = cls_scores[:, 1:] # [B, N-1]
    
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
