# Report: Integration of ViT Intermediate Bottleneck with MIMO Channel

## 1. Overview
The merged codebase implements an advanced Split Learning architecture that bridges a Vision Transformer (ViT) with a simulated physical wireless layer. Specifically, intermediate representations (compressed image patch tokens) are transmitted from a client to a server over a simulated **Multiple-Input Multiple-Output (MIMO)** wireless channel. 

The most defining feature of this integration is its **Joint Source-Channel Coding (JSCC)** optimization: the system extracts and leverages the ViT's self-attention scores (the semantic "importance" of each image patch) to dynamically dictate the physical layer allocation of transmit power, antennas, and spatial streams. 

## 2. Architectural Highlights & New Features
*   **MIMO Channel Simulation**: Native PyTorch realization of a generic $N_{tx} \times N_{rx}$ MIMO channel with varying fading distributions (Rayleigh, Diagonal, Identity) and Additive White Gaussian Noise (AWGN).
*   **Signal Equalization**: Uses robust Zero-Forcing (ZF) or Minimum Mean Square Error (MMSE) linear equalizers at the receiver end to reconstruct the transmitted ViT tokens.
*   **Attention-Guided Physical Allocation**:
    *   **Power Allocation**: Adjusts transmit power per token globally. Important ViT patches are dynamically given higher transmit power amplitudes, improving their resilience against noise.
    *   **Stream/Antenna Assignment**: Ranks physical antennas by gain matrices and assigns the most important payload tokens to the antennas with the strongest channels.
    *   **SVD Spatial Multiplexing**: For complex interference channels (Rayleigh fading), the transmitter computes the Singular Value Decomposition (SVD) of the channel matrix to discover independent orthogonal paths (eigenmodes). The most attention-heavy tokens are projected precisely onto the strongest eigenmodes.

## 3. Detailed File Breakdown

### `comm/mimo.py`
This file introduces the core signal processing math needed to simulate the physical layer inside a deep learning pipeline.
*   **`MIMOAWGNChannel`**: An `nn.Module` managing physical channel states ($Y = HS + N$). It is strictly responsible for sampling channel fading gains, calculating accurate noise bounds based on target SNR, and executing matrix inversions for ZF/MMSE equalizers.
*   **Dimensionality Transformation**: Introduces `pack_tokens_to_mimo_symbols` to map a 3D token tensor $(Batch, Tokens, D_{sent})$ into a packed, zero-padded 2D antenna grid $(Batch, N_{tx}, T_{time})$ representing actual physical transmissions over time periods. 

### `comm/comm_module.py`
This acts as the pipeline controller handling the end-to-end `Compressor -> Channel -> Decompressor` bottleneck strategy.
*   **Token Reordering & Mode Allocation**: Implements the complex routing logic utilizing the attention scores (`_resolve_stream_alloc_scores`). Calculates SVD values (`_compute_svd_modes`) and physically repackages the tensor data to prioritize strong modes/streams. 

### `comm/comm_module_wrapper.py`
To seamlessly embed the complex `CommModule` into the existing generic sequential transformer (`methods/proposal.py`), this adapter manages data routing:
*   **Cross-Layer Score Wiring**: Exposes `_get_selection_scores` which automatically captures the Class Token (`[CLS]`) attention maps from the upstream ViT architecture and funnels them directly into the communications module.
*   **State Transparency**: Extracts and caches comprehensive physics and allocation statistics into a `last_info` dictionary so that per-batch transmission metadata is accessible to the training loop despite running inside an opaque `nn.Sequential` block.

### `main.py`
The orchestration/training entry point using the `Hydra` configuration framework.
*   **Expanded Experiment Epoch Tracking**: Iterates over standard DL steps (loss, backprop), but now actively polls `model.channel.get_last_info()` on every single batch.
*   **Comprehensive Logging**: Accurately aggregates spatial logic success rates (like `mode_alloc_top_imp_frac` and `symbol_rate`) alongside prediction accuracy, saving detailed joint records inside a newly structured `best_training_results.json` artifact for reproducibility. 

### `analyze_mimo_scenarios.py`
A newly created post-experiment diagnostic tool designed to automatically interpret large-scale Hydra hyperparameter grid sweeps.
*   **Adaptive Configuration Parsing**: Recursively searches output directories, reading `.hydra/overrides.yaml` to dynamically separate *Constant* scenario parameters (e.g., fading type, model size) from *Variable* parameters (e.g., varying SNR levels, antenna counts).
*   **Hierarchical Grouping**: Assembles disparate JSON iterations into structured folders mapping "Constants $\to$ Array of Variables". It yields clean `comparison_results.json` files specifically formatted for downstream metric plotting.

## 4. Summary
By bridging these disparate codebases, the user effectively upgraded a static semantic feature-transmission split-learning network into a highly intelligent, physical-layer-aware communication framework. The ViT is now no longer blind to transmission conditions; rather, it uses its learned self-attention understanding of image context to proactively defend critical information using modern MIMO spatial allocation tactics.
