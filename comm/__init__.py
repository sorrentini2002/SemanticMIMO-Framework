# ============================================================
# comm/__init__.py
# ============================================================
# Public API for the comm package.
# ============================================================

from .comm_module import CommModule
from .comm_module_wrapper import CommModuleWrapper
from .dct import build_dct_matrix, apply_dct_spatial, apply_idct_spatial, clear_dct_cache

__all__ = [
    "CommModule",
    "CommModuleWrapper",
    # DCT Spatial Diversity helpers
    "build_dct_matrix",
    "apply_dct_spatial",
    "apply_idct_spatial",
    "clear_dct_cache",
]
