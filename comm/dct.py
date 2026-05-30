# ============================================================
# comm/dct.py
# ============================================================
# DCT-II / IDCT helpers for spatial diversity transmission.
#
# Design contract:
#   - All matrices are built with torch.no_grad() and .detach()
#     → fixed orthonormal basis, no learnable parameters,
#       no gradient instability from DCT construction.
#   - Matrices are cached per (k, device_str, dtype) to avoid
#     recomputation on every forward pass.
#   - All operations are PyTorch-differentiable w.r.t. the
#     *signal* (gradients flow through matmul), but NOT through
#     the DCT basis itself (basis is detached).
# ============================================================

import math
from typing import Tuple

import torch

# Module-level cache: (k, device_str, dtype_str) → [k, k] Tensor
_dct_cache: dict = {}


# ------------------------------------------------------------------
# Matrix construction
# ------------------------------------------------------------------

def build_dct_matrix(k: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Build the k×k orthonormal (unitary) DCT-II matrix.

    Definition:
        D[0, j] = sqrt(1/k)                                     for all j
        D[i, j] = sqrt(2/k) * cos(π (2j+1) i / (2k))          for i > 0

    Properties:
        D @ D^T = I   (orthogonal — energy-preserving round-trip)
        IDCT = D^T    (trivially, since D is real orthogonal)

    The matrix is built in float64 for numerical accuracy and then
    cast to the requested dtype.  It is detached from the autograd
    graph so that no gradient flows through the basis construction.

    Returns:
        Tensor of shape [k, k], detached, on the requested device.
    """
    if k <= 0:
        raise ValueError(f"build_dct_matrix: k must be > 0, got {k}")

    cache_key = (k, str(device), str(dtype))
    if cache_key in _dct_cache:
        return _dct_cache[cache_key]

    with torch.no_grad():
        j = torch.arange(k, dtype=torch.float64)  # [k]
        i = torch.arange(k, dtype=torch.float64)  # [k]

        # Outer product: angle[i, j] = π (2j+1) i / (2k)
        angle = math.pi * (2.0 * j.unsqueeze(0) + 1.0) * i.unsqueeze(1) / (2.0 * k)
        D = torch.cos(angle)  # [k, k]

        # Orthonormal scaling
        D[0, :] = D[0, :] * math.sqrt(1.0 / k)
        D[1:, :] = D[1:, :] * math.sqrt(2.0 / k)

        D = D.to(device=device, dtype=dtype).detach()

    _dct_cache[cache_key] = D
    return D


def clear_dct_cache() -> None:
    """Clear the module-level DCT matrix cache (useful in tests)."""
    _dct_cache.clear()


# ------------------------------------------------------------------
# Forward / inverse transforms
# ------------------------------------------------------------------

def apply_dct_spatial(signal: torch.Tensor, k_active: int) -> torch.Tensor:
    """
    Apply DCT-II along the spatial (mode/antenna) dimension.

    Args:
        signal:   [B, k_active, T]  — signal in mode/antenna domain.
        k_active: number of active spatial modes (must equal signal.shape[1]).

    Returns:
        [B, k_active, T]  — signal in DCT domain.

    Gradients flow through the matmul w.r.t. ``signal``.
    The DCT matrix itself is detached and carries no gradient.

    Mathematical detail:
        For each time slot t, the k_active-dimensional column vector is
        multiplied by the DCT matrix:
            y[:, :, t] = D @ x[:, :, t]
        Implemented efficiently as a batched matmul:
            Y = D @ X   where X is [k_active, T] per batch element.
    """
    if k_active <= 0:
        raise ValueError(f"apply_dct_spatial: k_active must be > 0, got {k_active}")
    if signal.shape[1] != k_active:
        raise ValueError(
            f"apply_dct_spatial: signal.shape[1]={signal.shape[1]} "
            f"does not match k_active={k_active}"
        )

    D = build_dct_matrix(k_active, device=signal.device, dtype=signal.dtype)
    # D: [k, k],  signal: [B, k, T]
    # Result: [B, k, T]  via broadcasting: (k,k) @ (B,k,T) with einsum
    return torch.einsum("ij,bjt->bit", D, signal)


def apply_idct_spatial(signal: torch.Tensor, n_out: int) -> torch.Tensor:
    """
    Apply inverse DCT (IDCT = DCT^T for unitary DCT-II) along the spatial dimension.

    Args:
        signal: [B, k_active, T]  — signal in DCT domain.
        n_out:  number of output spatial dimensions.
                Must equal signal.shape[1] (no up-sampling — the pruning
                decision that reduced k_active is irreversible at the receiver).

    Returns:
        [B, n_out, T]  — signal back in mode/antenna domain.

    Note:
        When k_active == n_tx, this is a perfect inversion.
        When k_active < n_tx, the pruned rows are zero-padded *before*
        this call (done in CommModule._forward_dct_spatial), so n_out
        will equal k_active here; the spatial expansion is deferred to
        the V @ s_mode projection step.
    """
    k_active = signal.shape[1]
    if k_active <= 0:
        raise ValueError(f"apply_idct_spatial: signal must have at least 1 spatial dim")
    if n_out != k_active:
        raise ValueError(
            f"apply_idct_spatial: n_out={n_out} must equal signal.shape[1]={k_active}. "
            f"Zero-pad pruned rows before calling this function."
        )

    D = build_dct_matrix(k_active, device=signal.device, dtype=signal.dtype)
    # IDCT = D^T @ Y
    return torch.einsum("ji,bjt->bit", D, signal)  # D^T is D transposed → "ji"
