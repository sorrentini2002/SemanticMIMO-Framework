import torch
import torch.nn as nn

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
    
    # Expand indices to gather along the feature dimension
    # indices: [B, K] -> [B, K, D]
    indices_expanded = indices.unsqueeze(-1).expand(-1, -1, D)
    
    # Gather
    gathered_tokens = torch.gather(tokens, 1, indices_expanded)
    
    return gathered_tokens


class ClassTokenAttentionTrackerWrapper(nn.Module):
    """
    Wraps a timm attention module and stores CLS-row attention scores.
    """
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
