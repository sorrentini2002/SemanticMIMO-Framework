# Detailed Documentation -- `gumbel_method.py`

---

## General Overview

`gumbel_method.py` implements a split-learning wrapper for ViT/DeiT models based on **Gumbel-Softmax token selection**. 

This module acts as a bridge between the Vision Transformer backbone and the SemanticMIMO communication channel, ensuring that token selection is not only efficient but also mathematically sound for gradient-based training and robust under physical channel noise.

Main objectives of the module:

1. **Top-K Selection**: Select the most relevant patch tokens using Gumbel-Softmax for differentiable pruning.
2. **Gradient Integrity**: Use the Straight-Through Estimator (STE) to maintain gradient flow from the classification loss to the selection module.
3. **Logits-Domain Sampling**: Sample directly from attention logits to prevent noise-dominance issues associated with probability-domain (post-softmax) sampling.
4. **MIMO Semantic Scaling**: Expose normalized semantic scores (`last_adc_scores`) to the channel module to guide adaptive power allocation (Waterfilling).
5. **Energy Preservation Under Compression**: Rescale selected tokens before channel transmission so the average transmitted energy remains consistent when only a small Top-K subset is sent.
6. **Warmup-Aware Training**: Keep full-token transmission during early optimization steps so the ViT backbone stabilizes before enabling token pruning.
7. **Diversity Enforcement**: Apply entropy and covariance regularization to prevent mode collapse and encourage diverse token selection.

---

## Technical Architecture

### 1. Logits Extraction and Stable Sampling
To ensure Gumbel noise is properly scaled, the module bypasses the standard attention softmax and captures raw Dot-Product logits. The `Store_Class_Token_Attn_Wrapper` intercedes during the attention forward pass:
```python
attn_logits = q @ k.transpose(-2, -1)
# Captured for Gumbel sampling:
self.class_token_logits = attn_logits[:, :, 0, :].mean(dim=1) 
# Standard ViT continues:
attn = attn_logits.softmax(dim=-1)
```
When scores are already in probability-like range `[0, 1]`, they are converted to logits with `log(clamp(scores, min=1e-10))` before adding Gumbel noise. This prevents random-selection behavior caused by adding high-variance Gumbel perturbations directly to bounded probabilities.

### 2. Standardized Semantic Scores (`last_adc_scores`)
For the physical MIMO channel (SVD/Waterfilling), it is critical that the **CLS token (index 0)** receives the highest possible power allocation. 
- **The Issue**: Raw Gumbel logits are unbounded. If used directly, a logit of `10.0` for a patch would make the dummy CLS score of `1.0` look "weak" to the power allocator.
- **The Solution**: Scores are passed through a `sigmoid` function before export. This ensures all semantic weights are in the range `(0, 1]`, making the CLS dummy value of `1.0` the absolute power priority.

### 3. Gradient Flow (Straight-Through Estimator)
The selection uses a "Hard" Top-K mask in the forward pass for maximum compression utility, but remains differentiable in the backward pass via the STE identity:
$$m_{final} = m_{hard} + (n_{\alpha} \cdot m_{soft} - (n_{\alpha} \cdot m_{soft}).detach())$$
This allows the classification loss to "vote" on which patches are semantically important for the task.

The scaling factor $n_{\alpha}$ matches the gradient magnitude to the Top-K hard selection budget, avoiding weak gradients when soft probabilities sum to 1 but hard selection activates multiple tokens.

### 4. Channel-Consistent Backpropagation and Energy Normalization
The channel path has been aligned with physically consistent training dynamics:

- In the AWGN analog channel, noise is applied in forward only; gradient-time noise injection has been removed to preserve STE learning signal.
- In `gumbel_compress`, selected tokens are scaled by $\sqrt{N / N_{sel}}$ before transmission so compression does not collapse effective per-token SNR when only a small subset is sent.

Together, these changes prevent gradient destruction and stabilize optimization under noisy channels.

---

## Regularization Mechanisms

To ensure stable training, the wrapper implements two key regularization losses:

| Loss Type | Logic | Purpose |
| :--- | :--- | :--- |
| **Global Entropy** | Minimizes $- \sum \bar{p} \log \bar{p}$ over the batch mean probability $\bar{p}$. | Prevents **Mode Collapse**. Ensures the model explores different patches across the dataset. |
| **Covariance** | Penalizes cosine similarity between full patch embeddings, weighted by selection probability. | Promotes **Feature Diversity**. Encourages the model to pick complementary tokens rather than redundant ones. |

In addition, warmup and annealing are synchronized so temperature decay starts only after warmup, and annealing length is aligned with total training steps ($epochs \times steps\_per\_epoch$).

---

## File Structure

```
gumbel_method.py
|
|-- Store_Class_Token_Attn_Wrapper     # Captures raw logits and attn scores.
|
|-- Gumbel_Token_Selection_Block_Wrapper
|    |-- gumbel_compress()             # Main selection logic + MIMO normalization.
|    |-- compute_reg_loss()            # Entropy & Covariance implementation.
|    `-- forward()                     # Residual pass + warmup/clean-bypass gating.
|
`-- model                              # Split-Learning assembler.
```

---

## End-to-End Execution Flow

1. **Client Blocks**: Initial transformer layers process the image.
2. **Split Point (Wrapper)**:
    - **Capture**: Raw attention logits are stored.
    - **Sample**: Gumbel-Softmax selects Top-K patches.
    - **Scale**: Selected scores are sigmoided for MIMO-ready `last_adc_scores`; selected token embeddings are energy-normalized before channel transmission.
    - **Warmup Control**: Before `warmup_steps`, compression is bypassed and the full token sequence is forwarded.
3. **Channel (`CommModuleWrapper`)**: 
    - Performs SVD-based communication using semantic scores to allocate power/modes, while AWGN noise is injected only in the forward signal path.
4. **Server Blocks**: The decoder layers process the received (and noisy) tokens to perform final classification.

---

## Evaluation Modes

- **MC-Evaluation**: When enabled, the model aggregates weights over multiple Gumbel noise samples to produce a robust, deterministic ranking for testing.
- **Clean Validation**: A diagnostic mode that bypasses all selection and channel noise to establish a noise-free performance ceiling.

Default configuration has been aligned with these dynamics (`tau_start: 1.0`, `warmup_steps: 1000`) in both Gumbel method configurations, including the eval-channel profile.
