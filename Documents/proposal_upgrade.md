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

### 2.1 New attributes: `last_adc_scores` and `_model_ref`
**Original:** the `__init__` method did not provide any attribute for saving scores or referencing the parent model.

**Updated:** added:
```python
self.last_adc_scores = None
self._model_ref = None
```
- `last_adc_scores`: stores token importance scores for the communication channel.
- `_model_ref`: a soft reference to the parent `model` instance, used to check global flags like `clean_validation` during the forward pass.

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

### 2.3 Method `forward` — conditional compression and `clean_validation`
**Original:** compression (batch merging + token selection) was applied **only during training**:
```python
if self.training:
    x = self.merge_batches_and_select_tokens(x)
```
During validation, the block behaved as a standard transformer block without compressing batches or tokens.

**Updated:** the guard `if self.training` is removed, but a new check is added for the evaluation phase. Compression is skipped only if the `clean_validation` flag is active on the parent model:
```python
# Check if we should bypass ADC (only in eval mode and if clean_validation is True)
clean_val = False
if not self.training and self._model_ref is not None:
    clean_val = getattr(self._model_ref, 'clean_validation', False)

if not clean_val:
    x = self.merge_batches_and_select_tokens(x)
```
This ensures that by default (noisy validation) compression is active in both phases, but allows for a "clean" bypass when requested for specific evaluations.

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

### 3.2 New attribute `clean_validation`
**Updated:** added a boolean flag to toggle between noisy (default) and clean evaluation:
```python
self.clean_validation = False
```
This flag is typically set from the Hydra configuration and dictates whether the model should bypass ADC and the communication channel during the validation phase.

### 3.3 Message of the assert on `desired_compression`
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

### 4.3 Wiring of attention scores and model reference
**Original:** the function directly returned the model without any additional channel or compressor configuration.

**Updated:** several wiring steps are added to establish communication between the model components:

1.  **Reference to parent model**: the compressor block receives a reference to the `model` instance to access its internal state (like `clean_validation`):
    ```python
    self.compressor_module._model_ref = self
    ```

2.  **Wiring scores to Channel**: if the channel is an instance of `CommModuleWrapper`, it is connected to the compressor:
    ```python
    if isinstance(channel, CommModuleWrapper):
        channel.set_score_source(self.compressor_module)
    ```

This infrastructure allows `CommModuleWrapper` to access `last_adc_scores` for weighted power allocation and enables the compressor to conditionally bypass ADC during evaluation.

---

## 5. Class `model` — method `forward`

### 5.1 Manual pipeline bypass for clean validation
**Original:** the `forward` method simply called `self.split_model(x)`.

**Updated:** a conditional logic is implemented to allow bypassing the communication channel during evaluation:

```python
if not self.training and self.clean_validation:
    # Manual forward pass to bypass the channel layer
    x = self.split_model[0](x) # Patch Embed
    x = self.split_model[1](x) # Pos Drop
    for i, blk in enumerate(self.split_model[2]):
        if i == split_index: 
            continue # SKIP CHANNEL
        x = blk(x)
    x = self.split_model[3](x) # Norm
    x = self.split_model[4](x) # Head
    return x
else:
    return self.split_model(x)
```
When `clean_validation` is active, the model manually iterates through the ViT blocks and skips the one at `split_index` (which represents the communication channel), effectively performing a standard inference on a perfect channel without ADC interference.

---

## Summary of changes

| Area | Modification |
|---|---|
| Imports | Added `from comm.comm_module_wrapper import CommModuleWrapper` |
| `Compress_Batches_and_Select_Tokens_Block_Wrapper.__init__` | New attributes `self.last_adc_scores` and `self._model_ref` |
| `forward` of compressor | ADC bypass logic checking `self._model_ref.clean_validation` |
| `model.__init__` | More flexible compression logic; new `clean_validation` flag |
| `model.forward` | Manual pipeline traversal to bypass channel during clean evaluation |
| `build_model` | Added `_model_ref` wiring and score source link for `CommModuleWrapper` |
