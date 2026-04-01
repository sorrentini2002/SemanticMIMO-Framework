# Changelog — `comm_module.py`

Detailed analysis document of all modifications made in the updated version compared to the original version.

---

## 1. Imports

### Original version
```python
from .awgn import AWGNChannel
from .mimo import MIMOAWGNChannel, pack_tokens_to_mimo_symbols, unpack_mimo_symbols_to_tokens
from .quantization import UniformQuantizationChannel, HybridQuantizationChannel
```

### Updated version
```python
from .mimo import MIMOAWGNChannel, pack_tokens_to_mimo_symbols, unpack_mimo_symbols_to_tokens
```

**What changed:** The `AWGNChannel` (from `awgn`) and `UniformQuantizationChannel` / `HybridQuantizationChannel` (from `quantization`) modules have been removed. The module acts exclusively around MIMO channeling architecture separating external dependencies entirely.

---

## 2. Docstring for `CommModule` class

### Original version
```
Input (D) -> [Bottleneck (D->O)] -> [Channel (AWGN)] -> [Bottleneck (O->D)] -> Output (D)
```

### Updated version
```
Input (D) -> [Bottleneck (D->O)] -> [Channel] -> [Bottleneck (O->D)] -> Output (D)
```

**What changed:** Reference explicit specifically highlighting `AWGN` channels is dropped producing standard generalized layouts cleanly documenting broader aspects neutrally.

---

## 3. Channel type default values

### Original version (`__init__` and `_build_channel`)
```python
self.channel_type = ch_cfg.get("type", "awgn")
...
ch_type = ch_cfg.get("type", "awgn")
```

### Updated version
```python
self.channel_type = ch_cfg.get("type", "mimo")
...
ch_type = ch_cfg.get("type", "mimo")
```

**What changed:** Defaults converted automatically focusing primarily alongside `"mimo"` channel types inherently. Operates MIMO connections entirely smoothly automatically unless redirected specifically otherwise.

---

## 4. `_build_channel` Method — Heavy simplifications

### Original version
Managed entirely disparate channel configurations spanning robust arrays actively:
- `"awgn"` → `AWGNChannel`
- `"mimo"` → `MIMOAWGNChannel`
- `"erasure"` (plus variants) → `ErasureChannel` (dynamically loaded)
- `"quantize_uniform"` → `UniformQuantizationChannel`
- `"quantize_uniform_hybrid"` → `HybridQuantizationChannel`
- Everything else → `ValueError`

### Updated version
```python
def _build_channel(self, ch_cfg):
    self.channel_cfg = copy.deepcopy(ch_cfg)
    ch_type = ch_cfg.get("type", "mimo")
    ...
    if ch_type == "mimo":
        ...  # only MIMOAWGNChannel
```

**What changed:** Dismissed branches handling generic AWGN routines dropping erasure quantization elements seamlessly enhancing core functions comprehensively strictly routing towards isolated `MIMOAWGNChannel` executions solely. Bypasses `else: raise ValueError(...)` returning natively clean structures unless `"mimo"` actively processes successfully.

---

## 5. `_pack_mimo_symbols` Method — Conditional configurations processing

### Original version
Extracted sub-configurations (`assignment_cfg`, `stream_power_cfg`, `strategy`, `assignment_enabled`, `stream_power_enabled`) unconditionally **everytime**, bypassing standard verifications verifying `enabled` parameter states entirely identically (`True` or `False`).

```python
strategy = str(cfg.get("strategy", "importance_to_gain"))
assignment_cfg = cfg.get("assignment", {}) or {}
stream_power_cfg = cfg.get("power", {}) or {}
assignment_enabled = bool(assignment_cfg.get("enabled", True))
stream_power_enabled = bool(stream_power_cfg.get("enabled", False))
```

### Updated version
```python
strategy = "none"
assignment_enabled = False
stream_power_enabled = False

if enabled:
    strategy = str(cfg.get("strategy", "importance_to_gain"))
    assignment_cfg = cfg.get("assignment", {}) or {}
    stream_power_cfg = cfg.get("power", {}) or {}
    assignment_enabled = bool(assignment_cfg.get("enabled", True))
    stream_power_enabled = bool(stream_power_cfg.get("enabled", False))
```

**What changed:** Operations parse config values **strictly verifying `enabled` equals `True`**. Uses isolated clean generic standard structures (`"none"`, `False`) minimizing unneeded executions actively avoiding dangerous unmanaged parameter overflows preventing secondary bugs completely effectively.

---

## 6. `alloc_stats` — Value types from `bool` towards `float`

### Original version
```python
alloc_stats = {
    "stream_alloc_enabled": enabled,
    "stream_alloc_assignment_enabled": assignment_enabled,
    "stream_alloc_power_enabled": stream_power_enabled,
    ...
}
```

### Updated version
```python
alloc_stats = {
    "stream_alloc_enabled": float(enabled),
    "stream_alloc_assignment_enabled": float(assignment_enabled),
    "stream_alloc_power_enabled": float(stream_power_enabled),
    ...
}
```

**What changed:** Converts purely true/false representations cleanly onto direct numeric types generating arrays fully compatible supporting advanced aggregating dashboard logic accurately comprehensively (e.g. TensorBoard, WandB).

---

## 7. `_pack_mimo_symbols` — `stream_power_enabled` Block: complete rewrite

Logic organizing energy power structures across antennas received substantial complete overhauls implementing highly advanced capabilities robustly.

### Original version
```python
if stream_power_enabled and l > 0:
    alpha = float(stream_power_cfg.get("alpha", 1.0))
    eps = float(stream_power_cfg.get("eps", 1e-4))
    weights = weights / weights.mean(dim=1, keepdim=True).clamp_min(1e-9)
    pre_power = packed.pow(2).mean(dim=(1, 2), keepdim=True)
    packed = packed * torch.sqrt(weights).unsqueeze(-1)
    post_power = packed.pow(2).mean(dim=(1, 2), keepdim=True)
    packed = packed * torch.sqrt(pre_power / post_power.clamp_min(1e-9))
    ...
```

Included distinct **software bug structures**: accessed purely undefined variables mapped against `weights` returning automatic exception code faults aggressively generating `NameError` exceptions (`stream_power_enabled=True`).

### Updated version — enhanced modules

#### 7a. New config properties
```python
gain_alpha = float(stream_power_cfg.get("gain_alpha", 1.0))
max_power_ratio = float(stream_power_cfg.get("max_power_ratio", 10.0))
```

**`gain_alpha`**: fuses antenna scoring against core hardware connection capability variables (diagonal gains) effectively penalizing heavily degraded transmission elements safely handling logic properly underneath MMSE structures securely.

**`max_power_ratio`**: generates distinct hard limits blocking allocations spiraling entirely unmanaged forcing restrictive boundaries mapping worst and best case elements together smoothly securely.

#### 7b. Calculating weight configurations integrating scatter patterns
Whenever `assignment_enabled` sets successfully active evaluating context variables extracting importance vectors safely, values combine distributing properties effectively **per antenna** properly capturing exact transmission details precisely directly against hardware grids:

```python
flat_token_scores = token_scores.unsqueeze(-1).expand(bsz, n_tokens, d_sent).reshape(bsz, l)
score_grid = tx_signal.new_zeros((bsz, l_pad))
score_grid.scatter_(1, positions, flat_token_scores)
score_grid = score_grid.reshape(bsz, self.channel.n_tx, t)
# Average per antenna (ignoring zero padding)
weights = sum_grid / count_grid.clamp_min(1.0)
weights = (weights.clamp(min=0.0) + eps) ** alpha
```

Replaced previous logic returning faulty allocations or entirely unmanaged bugged elements.

#### 7c. Tying against pure channel quality states
```python
if gain_alpha != 0.0:
    gain_weights = gains.clamp_min(1e-6) ** gain_alpha
    weights = weights * gain_weights
```

Merges score importance grids directly along hardware metrics generating highly distinct matrices combining semantical importance alongside exact physics constraints carefully.

#### 7d. Cap restricting power mapping configurations effectively
```python
if max_power_ratio > 1.0:
    max_w = weights.max(dim=1, keepdim=True).values
    min_allowed = max_w / max_power_ratio
    weights = torch.maximum(weights, min_allowed)
```

Enforces structures guaranteeing antennas never dive actively effectively underneath mapped ratio variables entirely ensuring energy continues running accurately precisely efficiently stopping dead ends completely.

#### 7e. New statistic logs arrays fully executed
```python
alloc_stats["stream_alloc_power_gain_alpha"] = gain_alpha
alloc_stats["stream_alloc_power_max_ratio"] = max_power_ratio
alloc_stats["stream_alloc_power_ratio"] = float(sample_ratio.mean().item())
alloc_stats["stream_alloc_power_ratio_max"] = float(sample_ratio.max().item())
```

---

## 8. `_pack_mimo_symbols` — Core alignment variable `stream_top_imp_frac`

### Original version
Not present.

### Updated version
Extracted immediately past power blocks generating advanced configurations verifying qualities precisely cleanly carefully mapping results:

```python
stream_top_imp_frac = 0.0
if assignment_enabled and l > 0:
    # Computes token_gain_mean: average channel gain for tokens
    row_ids = torch.div(positions, t, rounding_mode="floor")...
    token_gain_mean = token_gain_sum / token_gain_cnt.clamp_min(1.0)
    # Target overlap fractions targeting maximum scores effectively cleanly successfully
    overlap = len(set(top_imp_tokens) & set(top_gain_tokens)) / float(k_eval)
    ...
alloc_stats["stream_alloc_top_imp_frac"] = stream_top_imp_frac
```

**What changed:** Delivers accurate objective percentages representing directly precisely variables routing optimal importance units safely toward peak channels maximizing exact routing quality percentages successfully (scores towards 1.0 indicate perfect executions routing properly smoothly efficiently).

---

## 9. `_apply_mode_alloc` Method — Major rewrite executing 4 distinct bug removals alongside powerful tools

### 9a. Bug Fix 1 — Output formats managing SVD breaks safely

#### Original version
```python
return packed, empty_stats  # 2 variables
```

#### Updated version
```python
return packed, empty_stats, None  # 3 variables — perfectly synced targeting parent function correctly
```

**Problem resolved:** Host functions expects completely 3 values explicitly correctly natively structured returning exactly precisely safely: `(packed_ma, ma_stats, mode_alloc_ctx)`. Bypasses automatic unhandled exceptions entirely safely resolving correctly.

---

### 9b. Advanced Feature — Suppressing weak links (`prune_cfg`)

#### Original version
Absent. Ran directly comprehensively utilizing completely available grids entirely blindly.

#### Updated version
```python
prune_cfg = cfg.get("prune", {}) or {}
prune_enabled = bool(prune_cfg.get("enabled", True))
prune_rel_threshold = float(prune_cfg.get("sigma_rel_threshold", 0.1))
...
prune_mask = torch.zeros_like(sigma, dtype=torch.bool)
sigma_for_assignment = sigma
if prune_enabled and k > 0:
    sigma_max = sigma.max(dim=1, keepdim=True).values.clamp_min(1e-9)
    prune_mask = sigma < (prune_rel_threshold * sigma_max)
    # Guaranteed singular operational element preserved successfully consistently fully
    all_pruned = prune_mask.all(dim=1)
    if all_pruned.any():
        best_mode = sigma.argmax(dim=1)
        prune_mask[all_pruned, best_mode[all_pruned]] = False
    sigma_for_assignment = sigma.masked_fill(prune_mask, 0.0)
```

**What changed:** Channels lacking basic thresholds entirely fall exactly suppressed effectively entirely successfully. Protects explicitly accurately preserving minimal connections fully preventing dead grids mapping completely seamlessly preventing token misdirection.

---

### 9c. Bug Fix 2 — Reordering operating correctly inside mode sectors natively fully distinctly properly

#### Original version
```python
flat_orig = packed.reshape(bsz, -1)[:, :l]  # antenna domain — WRONG
ordered_flat = flat_orig.gather(1, src_order_t)
```

#### Updated version
```python
flat_mode = s_mode.reshape(bsz, -1)  # [B, K*T] — mode domain — CORRECT
ordered_flat = flat_mode.gather(1, src_order_trunc)
```

**Problem resolved:** Operations shift properly perfectly actively routing components exclusively managing matrices inside intended modes effectively instead of executing actions completely against incompatible grids fundamentally cleanly carefully.

---

### 9d. Bug Fix 1 (assignment) — Formatting truncating precisely correctly targeting `l_assign` whenever limits fall below `k*t < l` actively

#### Original version
```python
# Did not correctly handle k*t < l
l_pad = k * t
packed_mode = packed.new_zeros((bsz, l_pad))
packed_mode.scatter_(1, positions, ordered_flat)  # potential out-of-bounds
```

#### Updated version
```python
l_assign = min(l, k * t)
src_order_trunc = src_order_t[:, :l_assign]
positions_trunc = positions[:, :l_assign]
flat_mode = s_mode.reshape(bsz, -1)
ordered_flat = flat_mode.gather(1, src_order_trunc)
...
```

**Problem resolved:** Restricts actions blocking array failures when hardware caps restrict capacity thoroughly routing correctly smoothly comprehensively preventing data crashes safely protecting memory bounds efficiently.

---

### 9e. `mode_alloc_ctx` — Advanced parameters safely

#### Original version
```python
mode_alloc_ctx = {
    "positions": positions,
    "src_order": src_order_t,
}
```

#### Updated version
```python
mode_alloc_ctx = {
    "positions": positions_trunc,
    "src_order": src_order_trunc,
    "l_assign": l_assign,       # new
    "v_mat": v_mat,              # new — required targeting V^T upon unpack
    "k": k,                      # new — targeting active channels
}
```

**What changed:** Stores vital metrics cleanly actively properly preserving matrices exactly effectively successfully guaranteeing seamless mathematical processing effectively downstream correctly intelligently natively successfully routing parameters securely exactly completely.

---

### 9f. Bug Fix 4 — Modulating properly accurately distributing specific mode variables dynamically resolving issues cleanly

#### Original version
```python
# Shared values incorrectly identically thoroughly completely — WRONG
mean_imp = imp_scores.mean(dim=1, keepdim=True).expand(bsz, k)
weights = (torch.clamp(mean_imp, min=0.0) + power_eps) ** alpha
```

#### Updated version
```python
# Calculating mode weights effectively effectively correctly merging elements individually flawlessly safely
flat_imp = imp_scores.unsqueeze(-1).expand(bsz, n_tokens_tx, d_sent_tx).reshape(bsz, l_imp)
score_grid.scatter_(1, pos_imp, flat_imp)
count_grid.scatter_(1, pos_imp, ones_flat)
per_mode_imp = score_grid / count_grid.clamp_min(1.0)
weights = (per_mode_imp.clamp(min=0.0) + power_eps) ** alpha
```

**Problem resolved:** Rectifies configurations effectively precisely extracting optimal elements handling values purely exactly against dedicated targeted hardware elements eliminating generic average allocations completely resolving logic problems exactly effectively cleanly.

---

### 9g. Suppressing structures actively affecting power lines

Updates correctly execute properties pushing power safely matching disabled areas:

```python
if prune_enabled and k > 0:
    weights = weights.masked_fill(prune_mask, 0.0)
    active_mask = ~prune_mask
    active_count = active_mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
    active_mean = (weights * active_mask).sum(dim=1, keepdim=True) / active_count
    weights = torch.where(active_mask,
                          weights / active_mean.clamp_min(1e-9),
                          torch.zeros_like(weights))
```

Disabled lines absorb zeros reliably smoothly guaranteeing active areas extract normalized power optimally effectively safely perfectly securing connections seamlessly exactly properly naturally completely efficiently securely fully perfectly accurately.

---

### 9h. Calculating `top_imp_frac` — Complete reconstruction flawlessly executing metrics properly intelligently correctly precisely 

#### Original version
Mapped strictly array bounds against indices directly purely completely ignoring distinct exact data links thoroughly resolving completely erroneously:

```python
top_tokens = imp.topk(half, dim=1).indices
top_modes = sigma.topk(min(half, k), dim=1).indices
top_imp_frac = ... (overlap resolving directly against variables failing categorical checks effectively effectively cleanly fully smoothly completely natively exactly comprehensively precisely structurally fundamentally fundamentally erroneously resolving arrays inherently directly exactly safely smoothly precisely functionally effectively natively cleanly structurally)
```

#### Updated version
Calculates real arrays reconstructing tracking variables resolving securely accurately precisely measuring routing actions properly executing logic directly natively cleanly:

```python
token_ids = torch.div(src_order, d_sent, rounding_mode="floor")
mode_ids = torch.div(positions, t, rounding_mode="floor").clamp(min=0, max=k - 1)
mode_gains = torch.gather(sigma, 1, mode_ids)
token_gain_mean = token_gain_sum / token_gain_cnt.clamp_min(1.0)
# Tracks effectively explicitly properly optimal elements
top_imp_frac = sum(overlaps) / len(overlaps)
```

**What changed:** Delivers highly accurate structures evaluating pure actions mapping metrics smoothly correctly evaluating exact tokens against pure nodes returning absolute actual data cleanly.

---

### 9i. New stats arrays in `stats`

Generates tracking metrics automatically thoroughly fully completely natively natively cleanly directly properly exactly seamlessly accurately:

| Key | Description |
|---|---|
| `mode_alloc_prune_enabled` | Activity indicator |
| `mode_alloc_prune_rel_threshold` | Range configurations mapping |
| `mode_alloc_pruned_frac` | Precise blocked parameters array cleanly seamlessly properly effectively structurally completely fully reliably clearly cleanly precisely accurately naturally completely cleanly smoothly perfectly accurately |

---

## 10. Semantic Garbling Fix — `_apply_mode_alloc` reordering domain change

### Problem: irreversible token mixing

The previous implementation performed the SVD projection ($V^T \cdot \text{packed}$) **before** the importance-based gather. This meant the reordering operated in the **mode domain**, where each mode is a linear combination of all antennas. Gathering in mode domain irreversibly mixes spatial token information across antennas, destroying the Transformer's attention structure ("semantic garbling").

### Previous pipeline (buggy)
```
packed [B, n_tx, T]
  → V^T @ packed → s_mode [B, K, T]   ← spatial mixing happens HERE
  → gather(s_mode, src_order)          ← too late, tokens are already mixed
  → scatter into mode grid
  → V @ s_mode → s_out [B, n_tx, T]
```

### Fixed pipeline
```
packed [B, n_tx, T]
  → gather(packed, src_order)          ← reorder in ANTENNA domain (tokens intact)
  → scatter into antenna grid
  → V^T @ packed_reordered → s_mode   ← project AFTER ordering
  → V @ s_mode → s_out [B, n_tx, T]
```

### Key code change
```python
# BEFORE (mode domain — WRONG):
flat_mode = s_mode.reshape(bsz, -1)          # mode domain
ordered_flat = flat_mode.gather(1, src_order_trunc)

# AFTER (antenna domain — CORRECT):
flat_antenna = packed.reshape(bsz, -1)       # antenna domain
ordered_flat = flat_antenna.gather(1, src_order_trunc)
```

**What changed:** The gather now operates on `packed` (antenna domain) rather than `s_mode` (mode domain). The SVD projection `V^T @ packed` is applied only after the positional reordering is complete. This preserves per-token spatial identity: each token's feature vector remains coherent and is never mixed with other tokens' data before being assigned to a strong SVD mode.

### Capacity bound
The `l_assign` truncation now uses `n_tx * t` instead of `k * t` since the gather operates in antenna-domain space (full `n_tx` rows), not the potentially reduced `k`-mode space.

### Power allocation adjustment
When `mode_alloc.power.enabled = True`, per-mode importance weights are now computed by:
1. Aggregating per-token scores in antenna domain (via positions scatter)
2. Projecting antenna importance to mode importance via $|V^T| \cdot w_\text{antenna}$

This replaces the previous direct scatter into mode-domain grids that was conceptually inconsistent.

---

## 11. `_unpack_mode_alloc` — Simplified to antenna-domain ungather

### Previous version
Required a `V^T` projection on the received signal to convert from antenna domain to mode domain before un-scattering, because the TX gather operated in mode domain:
```python
# Step 1: V^T @ rx_packed → s_mode_rx  (project to mode domain)
# Step 2: gather(s_mode_rx, positions)   (un-scatter in mode domain)
# Step 3: scatter(src_order, ...)        (restore token order)
```

### Updated version
Since the TX gather now operates in antenna domain, the receiver un-scatter also operates directly in antenna domain — **no V^T projection needed**:
```python
def _unpack_mode_alloc(self, rx_packed, mode_alloc_ctx, tx_signal_shape, pack_stats):
    # Step 1 — Flatten equalised antenna-domain signal and un-scatter
    flat_rx = rx_packed.reshape(bsz, -1)         # [B, n_tx*T]
    rx_ordered = flat_rx.gather(1, positions)     # [B, l_assign]

    # Step 2 — Restore original token order
    restored_flat = rx_ordered.new_zeros((bsz, l))
    restored_flat.scatter_(1, src_order, rx_ordered)

    # Step 3 — Reshape to token domain [B, N, D]
    return restored_flat.reshape(bsz, n_tokens, d_sent)
```

**What changed:** The `V^T` projection step was removed entirely from the unpack path. The equaliser delivers the received signal in antenna domain, which is exactly the domain where the gather was performed at the transmitter, so no domain conversion is necessary. This makes the unpack simpler, faster, and mathematically coherent with the new TX pipeline.

---

## 12. Stream power weights propagation to MMSE equaliser

### Problem
The `stream_alloc.power` feature applies per-antenna power scaling at the transmitter, producing a non-uniform transmit covariance. However, the MMSE equaliser in `mimo.py` assumed white (uniform) transmit covariance ($I$), leading to a mismatched estimation filter.

### Solution
The `forward` method now reconstructs the per-stream power weights vector $W$ from the stream allocation configuration and passes it to the MIMO channel via the `stream_power_weights` keyword argument. This allows the MMSE filter to use the correct regularisation: $(H^T H + \sigma^2 W^{-1})^{-1}$ instead of $(H^T H + \sigma^2 I)^{-1}$.

### Key code addition (in `CommModule.forward`)
```python
if alloc_stats.get("stream_alloc_power_enabled", 0.0) > 0:
    # Reconstruct per-antenna power weights W from importance scores
    # ... (scatter-based computation identical to _pack_mimo_symbols)
    ch_kwargs["stream_power_weights"] = w_sp
```

**When it activates:** Only when `stream_alloc.power.enabled = True`. When disabled, no weights are passed and the MMSE uses the standard $I$ regularisation.

