import torch.nn as nn

class Bottleneck(nn.Module):
    """
    Bottleneck module for dimensionality reduction.
    Maps input dimension (d) to output dimension (o) and back.
    """
    def __init__(self, in_dim, out_dim=None):
        super().__init__()
        self.in_dim = in_dim
        # If out_dim is None or equal to in_dim, this might be identity,
        # but typically the CommModule handles the identity bypassing.
        # Here we assume if this utilized, we do the projection.
        self.out_dim = out_dim if out_dim is not None else in_dim
        
        self.compressor = nn.Linear(in_dim, self.out_dim)
        self.decompressor = nn.Linear(self.out_dim, in_dim)
        
    def forward(self, x):
        """
        Returns:
            compressed: [B, N, O]
            reconstructed: [B, N, D]
        """
        z = self.compressor(x)
        x_hat = self.decompressor(z)
        return z, x_hat
