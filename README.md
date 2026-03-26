# Integration of ViT Intermediate Bottleneck with MIMO Channel

## 1. Project Overview
This repository implements an advanced Split Learning architecture that bridges a **Vision Transformer (ViT)** with a simulated physical wireless layer. This framework enables the transmission of intermediate representations (compressed image patch tokens) from a client to a server over a simulated **Multiple-Input Multiple-Output (MIMO)** wireless channel.

The core of this integration relies on **Joint Source-Channel Coding (JSCC)** optimization: the system extracts the ViT's self-attention scores (the semantic importance of each image patch) and uses them to dynamically allocate physical layer resources such as transmit power, antennas, and spatial streams. Instead of treating all data packets equally, the model "defends" the most relevant information based on channel conditions and token importance.

---

## 2. Key Architectural Features

### 📡 Advanced MIMO Channel Simulation
* **MIMO Channel (MIMOAWGNChannel)**: Native PyTorch implementation of an {tx} \times N_{rx}$ MIMO channel with simulated fading distributions (Rayleigh, Diagonal, Identity) and AWGN noise.
* **Signal Equalization**: Uses robust **Zero-Forcing (ZF)** and **Minimum Mean Square Error (MMSE)** linear equalizers at the receiver end to accurately reconstruct the tokens transmitted by the ViT.
* **Dynamic Configuration Resolution**: Native support for complex structured inputs like ListConfig (e.g., dynamic SNR sweeps) managed resiliently during execution.

### 🧠 Attention-Guided Physical Allocation (JSCC)
* **Token Power Allocation (power_alloc)**: Global transmission power is distributed at the semantic token level. Crucial patches identified by the ViT receive higher power amplitudes to resist noisy channel conditions.
* **Stream/Antenna Assignment (stream_alloc_power)**: Power is allocated considering not only packet importance but also the physical gain (gain_alpha) of the channel, naturally penalizing weaker antennas to maximize efficiency (preventing improper energy exhaustion via max_power_ratio).
* **SVD Spatial Multiplexing & Mode Pruning**: For channels with strong interference (Rayleigh), Singular Value Decomposition (SVD) is applied to the channel matrix to discover independent paths (modes). Priority tokens are projected directly onto the strongest eigenvectors, while transmissions on weak modes are prudently interrupted (sigma_rel_threshold) to save resources and minimize inverse ^T$ reconstruction errors.

---

## 3. Detailed Component Breakdown & Documentation

To understand the specific upgrades and modifications made to the system, please refer to the detailed markdown documentation located in the Vit_with_MIMO/Documents/ directory:

### comm/mimo.py
Contains the core mathematical operations for the physical layer. The simulation accurately computes fading, injects noise scaled to the target SNR, and applies SVD precoding and MMSE/ZF equalization. 
* 📖 **Read more:** [MIMO Upgrade Overview](Documents/mimo_upgrade.md)

### comm/comm_module.py
The pivot module for the conversion between the ML model and telecom operations. It exclusively focuses on MIMO channels now, managing token reordering, spatial stream power optimization, and defining spatial efficiency metrics like stream_top_imp_frac.
* 📖 **Read more:** [Comm Module Upgrade](Documents/comm_module_upgrade.md)

### comm/comm_module_wrapper.py
Acts as a transparent bridge (Adapter) encapsulating the complex physical logic within the ViT's existing 
n.Sequential blocks. It elegantly routes attention maps (set_score_source) from the ViT encoder to the physical channel.
* 📖 **Read more:** [Comm Module Wrapper Details](Documents/comm_module_wrapper_explain.md)

### methods/proposal.py (The Split-Learning Compressor)
This module compresses and fragments neural weights prior to transmission. It has been heavily upgraded to hook into survived token attention scores (last_adc_scores) and ensures proper evaluation stability regardless of the model mode.
* 📖 **Read more:** [Proposal Upgrade](Documents/proposal_upgrade.md)

### main.py
The primary execution controller. It now records extensive telemetry regarding MIMO metrics by periodically fetching model.channel.get_last_info() and robustly handles dynamic multi-SNR sweeps explicitly during the validation_phase.
* 📖 **Read more:** [Main Optimization Guide](Documents/main_upgrade.md)

### Analyze_mimo_scenarios.py
A heavily redesigned script built to untangle complex organizational hurdles during intensive hyperparameter sweeps. It parses directory trees using pathlib and cleanly structures outputs into a unified JSON formatted layout for downstream visualization.
* 📖 **Read more:** [Scenario Analysis Explanation](Documents/analyze_mimo_scenarios_explain.md)

---

## 4. Configuration Guide
If you need to tweak hyperparameters, datasets (cifar_100, imagenette, etc.), or alter the simulated radio profiles (such as fading profiles or the equalizer types), please refer to the specific configuration modification guide:
* 📖 **Configuration Manual:** [How to Modify Configs](Documents/how_to_modify_configs.md)

---

## 5. Experimental Tests

### Documents/test_description.md
Provides a detailed description of the ongoing experimental test, including model architecture, split-computing configuration, and communication channel profiles.
* 📖 **Read more:** [Current Test Description](Documents/test_description.md)

### Documents/result_description.md
Provides a detailed description of the results of the ongoing experimental test, including a set of plots that summarize the results.
* 📖 **Read more:** [Current Result Description](Documents/result_description.md)

---

## 6. Conclusion
The convergence of these refactoring upgrades endows this research platform with state-of-the-art neural telecommunications characteristics. Originally isolated, the Split-Learning Vision Transformer now learns to strategically interface with its physical MIMO carrier, intentionally discarding irrelevant dependencies while pushing the most powerful visual representations through the cleanest spatial linkages available at execution time.
