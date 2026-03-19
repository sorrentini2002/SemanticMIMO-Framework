# Detailed Changelog — `proposal.py`

Document describing all modifications made in the updated file compared to the original version.

---

## 1. Imports and dependencies

### 1.1 New import: `CommModuleWrapper`
**Original:** the file only imported `VisionTransformer`, `torch.nn`, `torch`, `kmeans` and `torch.nn.functional`.

**Updated:** added the import of the wrapper for the advanced channel:
```python
from comm.comm_module_wrapper import CommModuleWrapper
```
This import enables support for the complete `CommModule` channel (bottleneck + MIMO + power allocation) in addition to pre-existing channels.

---

## 2. Class `Compress_Batches_and_Select_Tokens_Block_Wrapper`

### 2.1 New attribute `last_adc_scores`
**Original:** the `__init__` method did not provide any attribute for saving token selection scores.

**Updated:** added:
```python
self.last_adc_scores = None
```
This attribute will be populated on each forward pass with the importance scores of selected tokens, making them available for resource allocation in the communication channel.

### 2.2 Method `merge_batches_and_select_tokens` — ADC score saving
**Original:** the method accumulated only compressed activations in `clustered_activations` and returned the final stack. No saving of importance scores for selected tokens was provided.

**Updated:** a `clustered_scores` accumulator is added in parallel, which, for each cluster, saves the scores of selected tokens. To maintain shape consistency with the activation tensor (which includes the CLS token), a dummy score equal to `1.0` is prepended for the CLS:

```python
clustered_scores = []
...
sel_patch_scores = cluster_class_token_attention[top_k_tokens]
cls_score = torch.ones(1, dtype=sel_patch_scores.dtype, device=device)
clustered_scores.append(torch.cat([cls_score, sel_patch_scores]))
...
self.last_adc_scores = torch.stack(clustered_scores, dim=0)  # [n_new_batches, n_new_tokens]
```

The final result has shape `[n_new_batches, n_new_tokens]`, where the first position of each row corresponds to the CLS token (fixed score `1.0`) and subsequent positions to the real scores of selected patch tokens.

### 2.3 Method `forward` — removal of `if self.training` guard
**Original:** compression (batch merging + token selection) was applied **only during training**:
```python
if self.training:
    x = self.merge_batches_and_select_tokens(x)
```
During validation, the block behaved as a standard transformer block without compressing batches or tokens.

**Updated:** the guard is removed and compression is applied **always**, both in training and in evaluation:
```python
x = self.merge_batches_and_select_tokens(x)
```
This ensures that communication and compression statistics are consistent between the two phases, making validation metrics truly representative of deployment behavior.

---

## 3. Class `model` — method `__init__`

### 3.1 More flexible compression resolution logic
**Original:** when `desired_compression` was `None`, both `batch_compression` and `token_compression` had to be provided mandatorily, otherwise an `AssertionError` was raised:
```python
assert batch_compression is not None and token_compression is not None,     'Both batch_compression and token_compressions must be not None'
compression = (batch_compression, token_compression)
self.compression_ratio = batch_compression * token_compression
```

**Updated:** the logic is expanded with an `if/elif/else` structure that handles three distinct cases:

```python
if desired_compression is None:
    if batch_compression is not None and token_compression is not None:
        # Both provided: use the pair directly
        compression = (batch_compression, token_compression)
        self.compression_ratio = batch_compression * token_compression
    elif batch_compression is not None:
        # Only batch_compression: use as single ratio
        compression = batch_compression
        self.compression_ratio = batch_compression
    elif token_compression is not None:
        # Only token_compression: use as single ratio
        compression = token_compression
        self.compression_ratio = token_compression
    else:
        raise ValueError(
            'Set either desired_compression or at least one between '
            'batch_compression/token_compression'
        )
```

This way it is sufficient to specify only one of `batch_compression` or `token_compression`, and `build_model` will derive both through the square root (logic already present). The error changes from `AssertionError` to `ValueError` with an explicit message.

### 3.2 Message of the assert on `desired_compression`
**Original:**
```python
assert batch_compression is None and token_compression is None, 'desired_compression must be not None'
```
The message was misleading (it said `desired_compression` had to be non-None, while it was checking the opposite).

**Updated:**
```python
assert batch_compression is None and token_compression is None,     'When desired_compression is set, batch_compression and token_compression must be None'
```
The message now correctly describes the condition being checked.

---

## 4. Class `model` — method `build_model`

### 4.1 Updated descriptive comment
**Original:** the method was preceded by the simple comment `# Function to build model`.

**Updated:** replaced with a more descriptive comment that reflects the actual role:
```python
# --------------------------------------------------------
# build_model: assembles the complete split-learning model
# --------------------------------------------------------
```

### 4.2 More detailed inline comments
**Original:** comments described operations generically (e.g., `# Wrap last block with our compression method`, `# Split the original model`, `# Add comm pipeline and compression modules`).

**Updated:** each step is documented more accurately, with explanation of the reasons behind choices (e.g., why attention is wrapped, why `CommModuleWrapper` is already compatible with `nn.Sequential`, etc.). Comments contextualize operations within the split-learning architecture.

### 4.3 Wiring of attention scores to `CommModuleWrapper` (new block)
**Original:** after building the `nn.Sequential` with client blocks, channel, and server blocks, the function directly returned the model without any additional channel configuration.

**Updated:** a block is added that, if the channel is an instance of `CommModuleWrapper`, provides it with a reference to the compressor block:

```python
if isinstance(channel, CommModuleWrapper):
    channel.set_score_source(self.compressor_module)
```

This connection is the mechanism that allows `CommModuleWrapper` to read, on each forward pass, the `last_adc_scores` calculated by the compressor and pass them to `CommModule` as `selection_scores`. In this way power allocation, stream, and transmission modes are guided by token importance, without modifying the `forward` signature or breaking compatibility with `nn.Sequential`. If the channel is not a `CommModuleWrapper` (e.g., it is a legacy `Gaussian_Noise_Analogic_Channel`), the block is skipped without side effects.

---

## Summary of changes

| Area | Modification |
|---|---|
| Imports | Added `from comm.comm_module_wrapper import CommModuleWrapper` |
| `Compress_Batches_and_Select_Tokens_Block_Wrapper.__init__` | New attribute `self.last_adc_scores = None` |
| `merge_batches_and_select_tokens` | Saving ADC scores (`clustered_scores`, `last_adc_scores`) with dummy CLS score |
| `forward` of compressor | Removed guard `if self.training`: compression active in eval too |
| `model.__init__` | More flexible compression logic: accepts single `batch_compression`/`token_compression`; corrected assert message |
| `build_model` | Improved comments; added wiring `channel.set_score_source(self.compressor_module)` for `CommModuleWrapper` |
