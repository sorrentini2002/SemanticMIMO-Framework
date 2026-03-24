# ============================================================
# comm/__init__.py — Public API of the comm package
# ============================================================
# Import order matters: mimo and bottleneck have no internal
# cross-dependencies, so they are loaded first.
# ============================================================

"""Local communication package for split-learning channels."""

# --- Core physical-layer building blocks ---
from .bottleneck import Bottleneck
from .mimo import (
    MIMOAWGNChannel,
    pack_tokens_to_mimo_symbols,
    unpack_mimo_symbols_to_tokens,
)

# --- Legacy / simple channel classes ---
from .communication import (
    Gaussian_Noise_Analogic_Channel,
    Identity,
)

# --- Advanced orchestrator (bottleneck + channel + power alloc) ---
from .comm_module import CommModule

# --- nn.Sequential-compatible wrapper around CommModule ---
from .comm_module_wrapper import CommModuleWrapper
