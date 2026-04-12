# Detailed Documentation -- `gumbel_method.py`

---

## General Overview

`gumbel_method.py` implements a split-learning wrapper for ViT/DeiT models based on
**Gumbel-Softmax token selection**. 

Unlike previous iterations, this module now follows a **"Native Wrapper"** design: it does not implement the compression algorithm itself but imports it directly from the core `methods/gumbel/` directory. This ensures 100% algorithmic fidelity with the pristine Gumbel implementation while providing the necessary integration hooks for the SemanticMIMO framework.

Main objectives of the module:

1. Select the most relevant patch tokens using the core Gumbel-Softmax algorithm.
2. Maintain a drop-in replacement interface for the SemanticMIMO pipeline.
3. Expose per-token semantic scores (`last_adc_scores`) to the communication channel for adaptive resource allocation.
4. Support dynamic "Clean Validation" and evaluation-only channel toggling.

---

## File Structure

```
gumbel_method.py
|
|-- IMPORTS
|    |-- sample_gumbel_topk()          # From .gumbel.gumbel
|    `-- compute_tau()                 # From .gumbel.schedules
|
|-- Gumbel_Token_Selection_Block_Wrapper
|    |-- register_step()
|    |-- current_tau (property)
|    |-- gumbel_compress()
|    |-- forward()                     # Forward + Clean Bypass check
|    `-- compress_labels()              # Dummy wrapper (K-means removed)
|
|-- Store_Class_Token_Attn_Wrapper     # Stores CLS attention row [B, N]
|
`-- model                              # Outer split-learning model wrapper
	  |-- __init__()
	  |-- build_model()
	  `-- forward()
```

---

## Imports and Dependencies

```python
from .gumbel.gumbel import sample_gumbel_topk
from .gumbel.schedules import compute_tau
from comm.comm_module_wrapper import CommModuleWrapper
```

### Key dependencies

- **`methods.gumbel`**: The source of truth for the Gumbel-Softmax algorithm.
- **`timm`**: Used for the Vision Transformer backbone.
- **`CommModuleWrapper`**: Advanced communication wrapper with semantic score support.

---

## Core Algorithm Integration

The module no longer defines the mathematical logic for selection. Instead, it delegates to:

1. **`compute_tau`**: Handles the annealing schedule (linear, cosine, exp) of the Gumbel temperature.
2. **`sample_gumbel_topk`**: Handles Gumbel noise sampling, Straight-Through Estimator (STE) masks, and token gathering.

This ensures that any improvement to the core Gumbel logic is automatically reflected in the MIMO pipeline.

---

## Class `Gumbel_Token_Selection_Block_Wrapper`

This class wraps one transformer block and applies token compression using the core Gumbel logic.

### Interface-critical attributes

- `last_adc_scores`: `[B, N_selected]` semantic scores consumed by `CommModuleWrapper` for waterfilling.
- `_model_ref`: weak back-reference to parent model, used to read the `clean_validation` flag.
- `n_new_tokens`: tracks the current sequence length after compression (CLS + K patches).

---

## Constructor (`__init__`)

### Parameters

- `token_compression`: patch retention ratio (e.g., 0.5 to keep half the patches).
- `tau_max`, `tau_min`, `anneal_steps`, `anneal_mode`: temperature annealing hyper-parameters.
- `hard`, `straight_through`: flags for the STE gradient estimator.
- `compression_enabled`: master toggle to bypass compression entirely.

---

## `gumbel_compress(x)`

The bridge between the ViT features and the Gumbel core.

### Steps

1. **Attention Extraction**: Retrieves the CLS attention scores stored by `Store_Class_Token_Attn_Wrapper`.
2. **Shape Alignment**: Unsqueezes the attention to `[B, 1, N]` to meet the internal expectations of the core algorithm.
3. **Core Call**: Executes `sample_gumbel_topk`.
4. **Score Export**: Gather the raw importance scores for the *selected* tokens and prepends a `1.0` dummy score for the CLS token, storing the result in `self.last_adc_scores`.

---

## `forward(x)`

Implements the phase-aware forward pass.

1. **Standard block pass**: Residual attention and MLP.
2. **Bypass Check**: If `self.training` is False and `clean_validation` is enabled, it returns the full token sequence.
3. **Compression**: If active, calls `gumbel_compress`.

---

## `compress_labels(labels, num_classes)`

Since batch compression (K-Means) has been removed to prioritize Gumbel algorithmic purity, this function now simply returns standard one-hot encoded labels:
`F.one_hot(labels, num_classes=num_classes).float()`.

---

## Class `model` (Outer Wrapper)

Top-level integration class that assembles the split-learning pipeline.

### Constructor & build_model()

- **Split Index**: Injects the compressor and channel at the specified block index.
- **Semantic Wiring**: Automatically links the compressor's `last_adc_scores` to the `CommModuleWrapper`.
- **Logic Toggle**: Forward the `semantic_waterfilling` and `channel_eval_only` flags to the communication module.

---

## End-to-End Execution Flow

```
Input image batch
	 |
	 v
ViT blocks before split
	 |
	 v
Wrapped split block
	 |-- attention wrapper stores CLS values
	 |-- Gumbel logic samples top patches based on CLS scores
	 |-- last_adc_scores populated for waterfilling
	 v
CommModuleWrapper channel
	 |-- reads scores from Gumbel output
	 |-- applies SNR/Mode allocation (Semantic Waterfilling)
	 v
ViT blocks after split
	 v
Classifier output
```

---

## Final Summary

The current architecture of `gumbel_method.py` prioritizes **separation of concerns**:
- **Math/Algorithm**: Isolated in `methods/gumbel/`.
- **System Integration**: Managed by `gumbel_method.py`.
- **MIMO Physics**: Managed by `comm_module.py`.

By removing the legacy K-means logic, we focus exclusively on the impact of Gumbel-Softmax token pruning in MIMO scenarios, providing a cleaner baseline for comparison against the original "proposal" method.
