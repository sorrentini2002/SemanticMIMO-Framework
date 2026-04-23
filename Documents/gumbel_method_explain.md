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


---

### Technical Implementation Details and Code Modifications

The following updates have been implemented to stabilize the training of the Gumbel-Softmax token selection policy and ensure optimal gradient flow through the end-to-end communication pipeline.

### 1. Dynamic SNR Training (main.py)
To prevent the model from collapsing under constant high-noise conditions during training, we implemented **Dynamic SNR Sampling**. 
- For every batch, the training SNR is sampled uniformly from the interval `[0.0, 20.0]` dB.
- High-SNR batches act as "beacons" for the gradient, allowing the ViT backbone to receive clear semantic feedback through the Straight-Through Estimator (STE) without the signal being entirely masked by channel noise.

### 2. "The Cage": Native Attention Scoring (gumbel_method.py)
The dedicated MLP-based scoring head (linear layers + GELU) has been removed. 
- Scores are now derived directly from the native `class_token_attention` of the wrapped ViT block.
- Because these scores originate from the transformer's own Softmax layer, they are mathematically bounded within the `[0, 1]` range. This eliminates the risk of logit "explosions" and stabilizes the Gumbel sampling process without requiring ad-hoc clipping hacks.

### 3. Gumbel-Softmax Overhaul (gumbel.py)
We have refined the stochastic selection logic to better align with the linear nature of attention weights:
- **Removal of Logarithm**: The formula `torch.log(scores)` was removed. Noisy logits are now computed as `scores + (gumbel_noise * tau)`.
- **Decaying Noise Amplitude**: The physical magnitude of the Gumbel noise now scales with the temperature `tau`. As training progresses and `tau` decreases, the perturbation fades, allowing the model to solidify its selection policy.
- **Double Softmax (Exponential Amplification)**: To exaggerate micro-differences in attention weights and force discrete selection, we use `F.softmax(noisy_scores / tau, dim=-1)`.

### 4. Batch Entropy Regularization (gumbel_method.py & main.py)
To ensure spatial diversity and prevent the "Index Collapse" (where the model always selects the same patch locations regardless of image content), we introduced an **Active Entropy Loss**:
- We calculate the **mean selection distribution** across the entire batch (`p_mean`).
- We maximize the entropy of this mean distribution by subtracting it from the total training loss (with a weighting factor of `0.05`). 
- This forces the model to explore different regions of the image across the dataset while still allowing it to be sharp and selective for any individual image.

### 5. Fixed Temperature Annealing (main.py)
A critical logic bug was resolved by hooking the `register_step()` method directly into the training loop. 
- The Gumbel temperature `tau` now correctly decays from the starting value (2.0) to the target value (0.1). 
- Without this fix, the selection remained stochastic/random forever, preventing the convergence to a hard, semantically-driven policy.

### 6. Gradient Flow Optimization (communication.py & mimo.py)
To maintain the integrity of the "gradient thread" from the final loss back to the Client-side ViT, several `.detach()` calls were removed:
- **Channel Power Tracking**: Removed `.detach()` from the signal power and noise variance calculations. The gradient now understands how the power allocation strategy (including Waterfilling) affects the effective SNR.
- **Attention Flow**: Removed `.detach()` from `class_token_attention`, enabling the ViT to learn which tokens are semantically relevant for the task through the Straight-Through Estimator.

### 7. Removal of Gradient Noise Hooks (communication.py)
Experimental hooks that injected AWGN noise into the **backward pass** (gradients) have been removed. We determined that noise should only affect the forward pass (the data), while the backward pass must stay as clean as possible to provide high-fidelity optimization signals to the scoring mechanism.

### 8. Removal of Spatial Reconstruction (comm_module_wrapper.py)
The zero-padding reconstruction previously performed at the Server-side has been removed. 
- The Server now processes the condensed feature set directly. 
- This prevents the model from learning "fake" geometric cues from the absolute zeros used in padding, which was previously causing instabilities in the Server's LayerNorm and Self-Attention layers.

---

### Advanced Architectural and Training Enhancements

The following features were subsequently added to refine the geometric understanding and training stability of the Gumbel-Softmax pipeline.

#### 9. Geometric Simplicial Scoring Branch
To refine token selection beyond raw class attention, we introduced a **Simplicial Interacting Graph** logic. This branch uses two dedicated linear projections (`w_u` and `w_tri`) and LayerNorm layers to compute a "contextual marginal vector" ($m_{cls}$) and a "simplicial interaction triangle" ($patch_{tri}$). The resulting `interaction_strength` augments the base attention scores, allowing the model to capture higher-order relationships between the CLS token and the image patches.

#### 10. Initial Score Polarization ($\gamma = 5.0$)
A learnable "Brute Force" parameter ($\gamma$) was initialized at **5.0** to immediately polarize the selection scores during the early stages of training. This high initial value ensures that the Straight-Through Estimator (STE) receives sharp signals, forcing the scoring head to converge quickly on a semantically coherent policy before the Gumbel temperature $\tau$ decays significantly.

#### 11. Multi-Stage Differential Learning Rates
The optimizer configuration in `main.py` uses a **Tri-Stage Learning Rate** strategy to stabilize the split-learning architecture:
- **Encoder/Backbone**: Limited to a lower LR ($3\times 10^{-5}$) to prevent catastrophic forgetting of pre-trained spatial features.
- **Server/Decoder**: Set to a standard LR ($3\times 10^{-4}$) to allow rapid adaptation to the sparse token payload.
- **Classification & Scoring Heads**: Boosted with a $10\times$ multiplier ($3\times 10^{-3}$) to act as a "hydraulic shield"—absorbing initial entropy shocks and providing stabilized gradient signals to the upstream modules.

#### 12. Stabilization: Clipping and Weight Decay
To further steady the training process, two critical mechanisms were implemented:
- **Global Gradient Clipping**: All gradients are capped at a `max_norm` of 1.0 using `torch.nn.utils.clip_grad_norm_`. This prevents "gradient spikes" from the decoder from destabilizing the core backbone.
- **Aggressive Weight Decay**: The `AdamW` weight decay was increased to **0.05**. This penalizes the model for arbitrarily inflating weights to bypass channel constraints, encouraging a more efficient and generalized representation.

#### 13. Stochastic Multi-Budget Training ("Vaccination")
To prevent the Server-side decoder from overfitting to a specific sequence length, we implemented **Multi-Budget Training**. During training, the number of selected tokens ($n_{\alpha}$) is sampled randomly for each batch within a range (e.g., between 8 and 64 tokens). This "vaccination" strategy forces the model to be robust to varying bandwidth conditions, ensuring superior performance when a fixed budget is enforced during evaluation.

#### 14. Z-Score Standardization for Scores
The legacy L1 normalization and log-transformations for scores were replaced with **Z-Score Standardization** across the spatial token dimension. By centering the scores (zero-mean) and scaling them (unit-variance) before the Gumbel addition, we ensure the raw logits operate consistently with the temperature parameter $\tau$, preventing signal disintegration.

#### 15. Power Normalization ("The Crystal Ceiling")
To prevent "reward hacking"—where the model might increase signal energy to artificially improve the effective SNR—we introduced a mandatory **RMS Power Normalization** block in `comm_module_wrapper.py`. This physically constrains the average energy of the compressed tensor to 1.0 before it enters the channel simulator.

#### 16. Joint Loss Optimization
The training objective was unified in `main.py` into a single, cohesive calculation: `loss = task_loss + (entropy_weight * entropy_loss)`. This ensures that entropy regularization (promoting spatial diversity) is balanced correctly against the classification objective in every optimization step.

#### 17. Index Synchronization
The CommModuleWrapper now extracts last_indices_sel from the Gumbel selector and passes it to the CommModule. This enables Unequal Error Protection (UEP): tokens deemed most important by the Gumbel branch are now physically routed through the most robust MIMO spatial streams (those associated with the highest singular values).

#### 18. Dynamic Logit Scaling (`_compute_logit_scale()`)
Implements a cosine-based scaling factor $\alpha$ (growing from 0.1 to 1.0) applied to raw logits. This prevents early entropy collapse by counteracting low initial weight magnitudes, ensuring a controlled transition from random exploration to sharp, semantically-driven selection.

#### 19. Per-Instance Entropy Bottleneck
Replaced batch-entropy maximization with a "moving ceiling" loss: $Loss_{ent} = \lambda \cdot \max(0, H_{actual} - H_{target})$. By only penalizing entropy when it exceeds a scheduled target (5.2 $\to$ 2.0), the model can focus on classification accuracy while maintaining the necessary exploration budget.

#### 20. Differentiated Weight Decay ($WD = 10^{-3}$)
Applied a specific weight decay to the score head parameters ($w_u, w_{tri}$). This acts as a restorative force that prevents unbounded weight growth, keeping the logit standard deviation within the optimal 1.5–2.5 range for stable Straight-Through estimation.

#### 21. Selection Stability Bonus (EMA)
Introduced an optional stability mechanism using an Exponential Moving Average (EMA) of selection frequencies. By rewarding systematically useful patches, this helps "lock in" emerging semantic choices and prevents jittery selection switching in the late training phases.
