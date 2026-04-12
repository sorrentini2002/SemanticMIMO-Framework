
import torch

def gather_tokens(tokens, indices):
    """
    Gather tokens based on indices.
    
    Args:
        tokens: [B, N, D]
        indices: [B, K] where K is the number of tokens to keep.
        
    Returns:
        gathered_tokens: [B, K, D]
    """
    B, N, D = tokens.shape
    B_idx, K = indices.shape
    
    # Expand indices to gather along the feature dimension
    # indices: [B, K] -> [B, K, D]
    indices_expanded = indices.unsqueeze(-1).expand(-1, -1, D)
    
    # Gather
    gathered_tokens = torch.gather(tokens, 1, indices_expanded)
    
    return gathered_tokens
