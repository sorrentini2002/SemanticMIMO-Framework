# Changelog — `mimo.py`

Detailed analysis document of all modifications made in the updated version compared to the original version.

---

## 1. Imports

### Original version
```python
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
```

### Updated version
```python
from collections.abc import Mapping
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn

try:
    from omegaconf import ListConfig
except Exception:  # pragma: no cover
    ListConfig = ()
```

**What changed:** Two new imports have been added:

- **`Mapping`** from `collections.abc`: used in the new `_get_diagonal_random_cfg` method to robustly and idiomatically verify that the `diagonal.random` configuration is a key-value structure, compatible with any mapping implementation (not just `dict`).

- **`ListConfig` from `omegaconf`**: added via a `try/except` block for optional compatibility. If `omegaconf` is not installed, `ListConfig` is assigned to `()` (empty tuple), so `isinstance(..., ListConfig)` checks never raise exceptions. This enables native support for Hydra/OmegaConf configurations without making them a required dependency.

---

## 2. Method `_resolve_snr` — Support for `ListConfig`

### Original version
```python
def _resolve_snr(self, x: torch.Tensor, generator: Optional[torch.Generator]) -> float:
    if self.training and self.train_mode == "sampled":
        if isinstance(self.snr_db, (list, tuple)):
            ...
    if isinstance(self.snr_db, (list, tuple)):
        return float(max(self.snr_db))
    return float(self.snr_db)
```

### Updated version
```python
def _resolve_snr(self, x: torch.Tensor, generator: Optional[torch.Generator]) -> float:
    is_snr_sequence = isinstance(self.snr_db, (list, tuple, ListConfig))
    if self.training and self.train_mode == "sampled":
        if is_snr_sequence:
            ...
    if is_snr_sequence:
        return float(max(self.snr_db))
    return float(self.snr_db)
```

**What changed:** The `isinstance` check is executed **only once** at the beginning of the method, the result is saved in the `is_snr_sequence` variable and reused in the two subsequent branches. This brings two improvements:

1. **Support for `ListConfig`**: the original version only recognized `list` and `tuple` as SNR sequences. The updated version also includes `ListConfig` from OmegaConf, the type produced by Hydra when a list is specified in the YAML configuration. Without this modification, passing `snr_db` as a Hydra list would have caused the value to be treated as a scalar, silently applying the maximum SNR during training.

2. **Reduction of duplication**: the `isinstance` call is no longer repeated twice with the same argument.

---

## 3. New method `_get_diagonal_random_cfg`

### Original version
Absent. Reading the `diagonal.random` configuration happened inline and duplicated in three separate places in the code (`_validate_diagonal_config`, `sample_diagonal_gains`, `forward`), each with a slight variant:

```python
# In _validate_diagonal_config (old)
random_cfg = self.diagonal_cfg.get("random", {}) or {}
if not isinstance(random_cfg, dict):
    raise ValueError("comm.channel.diagonal.random must be a mapping when provided.")

# In sample_diagonal_gains (old)
random_cfg = self.diagonal_cfg.get("random", {}) or {}

# In forward (old)
random_cfg = self.diagonal_cfg.get("random", {}) or {}
```

### Updated version
```python
def _get_diagonal_random_cfg(self) -> dict:
    random_cfg = self.diagonal_cfg.get("random", {})
    if random_cfg is None:
        return {}
    if isinstance(random_cfg, Mapping):
        return dict(random_cfg)
    raise ValueError("comm.channel.diagonal.random must be a mapping when provided.")
```

**What changed:** All the logic for safe access to the `random` key of the diagonal configuration has been centralized in a single helper method. The method:

- Returns `{}` if the key doesn't exist or has value `None` (explicitly handles `None` rather than relying on the `or {}` operator)
- Accepts any type that implements `Mapping` (not just `dict`), including OmegaConf/Hydra structures
- Converts the result to pure `dict`, making calling code independent of the concrete type of the configuration
- Raises `ValueError` with descriptive message if the value is not a mapping

This eliminates code duplication and ensures consistent behavior across all call sites.

---

## 4. Method `_validate_diagonal_config` — Use of `_get_diagonal_random_cfg`

### Original version
```python
def _validate_diagonal_config(self) -> None:
    random_cfg = self.diagonal_cfg.get("random", {}) or {}
    if not isinstance(random_cfg, dict):
        raise ValueError("comm.channel.diagonal.random must be a mapping when provided.")
    random_enabled = bool(random_cfg.get("enabled", False))
    ...
```

### Updated version
```python
def _validate_diagonal_config(self) -> None:
    random_cfg = self._get_diagonal_random_cfg()
    random_enabled = bool(random_cfg.get("enabled", False))
    ...
```

**What changed:** The inline config reading and manual `isinstance(random_cfg, dict)` check have been replaced with a call to `_get_diagonal_random_cfg()`. The method is now more concise and delegated, removing duplicated validation logic.

---

## 5. Method `sample_diagonal_gains` — Use of `_get_diagonal_random_cfg`

### Original version
```python
def sample_diagonal_gains(self, bsz, *, device, generator=None, dtype=torch.float32):
    ...
    random_cfg = self.diagonal_cfg.get("random", {}) or {}
    random_enabled = bool(random_cfg.get("enabled", False))
    ...
```

### Updated version
```python
def sample_diagonal_gains(self, bsz, *, device, generator=None, dtype=torch.float32):
    ...
    random_cfg = self._get_diagonal_random_cfg()
    random_enabled = bool(random_cfg.get("enabled", False))
    ...
```

**What changed:** Inline config reading is replaced with the helper method. Functional behavior remains identical, but now automatically benefits from `Mapping` support and centralized `None` handling.

---

## 6. Method `forward` — Diagonal statistics: use of `_get_diagonal_random_cfg`

### Original version
```python
if self.fading == "diagonal":
    ...
    random_cfg = self.diagonal_cfg.get("random", {}) or {}
    if not bool(random_cfg.get("enabled", False)):
        stats["mimo_diagonal_gains"] = [float(v) for v in diag_gains[0].tolist()]
```

### Updated version
```python
if self.fading == "diagonal":
    ...
    random_cfg = self._get_diagonal_random_cfg()
    if not bool(random_cfg.get("enabled", False)):
        stats["mimo_diagonal_gains"] = [float(v) for v in diag_gains[0].tolist()]
```

**What changed:** The last inline read site of the `diagonal.random` configuration has been replaced with a call to the helper method. Behavior is unchanged: fixed gains are logged in statistics only if random gains are disabled.

---

## Summary of changes

| Area | Type of change |
|---|---|
| Imports | Added `Mapping` and `ListConfig` (optional with fallback to `()`) |
| `_resolve_snr` | Extended `isinstance` to `ListConfig`; check deduplicated into `is_snr_sequence` variable |
| `_get_diagonal_random_cfg` | **New method** — centralizes safe reading of `diagonal.random` |
| `_validate_diagonal_config` | Replaced inline reading with `_get_diagonal_random_cfg()` |
| `sample_diagonal_gains` | Replaced inline reading with `_get_diagonal_random_cfg()` |
| `forward` | Replaced inline reading with `_get_diagonal_random_cfg()` |

The changes are overall of **refactoring and compatibility** nature: they do not alter the functional behavior of the channel for already correct configurations, but improve robustness with respect to OmegaConf/Hydra configurations and remove duplication of diagonal config access logic.
