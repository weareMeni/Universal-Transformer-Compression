# Dynamic $\mathcal{O}(1)$ Edge Compressors

This repository contains a PyTorch micro-framework for training and evaluating dynamic-compute local agents. It explores the memory bandwidth bottlenecks of local-to-remote agent collaboration (e.g., edge-to-cloud compression) by implementing a modified Universal Transformer (UT) designed to trade spatial VRAM footprint for dynamic ALU cycles.

## The Bottleneck Problem

In local/remote agentic systems, edge devices are heavily bottlenecked by memory capacity and bandwidth. A local "compressor" model must read long contexts and transmit a dense, continuous vector to a remote predictor model. 

Standard transformers require scaling layers (and thus parameter count / VRAM) to process complex hierarchical logic. This framework demonstrates that a recurrent Universal Transformer can act as a highly efficient edge compressor, utilizing an $\mathcal{O}(1)$ parameter core block that dynamically scales its ponder depth to maximize the information density of the transmitted payload.

## Architectural Innovations

This implementation seeks to solves two historical instabilities of Universal Transformers (ACT gradient conflicts and representation collapse):

### 1. Penalty-Free Routing via Differential Weight Decay
Standard Adaptive Computation Time (ACT) relies on an explicit ponder loss ($\mathcal{L}_{task} + \lambda \mathcal{P}$), which causes gradient conflicts and requires brittle hyperparameter tuning. 
* **The Fix:** Explicit ponder loss is entirely removed. The halting classifier's bias is initialized to naturally target ~4 ponder loops. A heavy, differential weight decay (`gate_wd=0.5` vs `base_wd=0.1`) is applied exclusively to the gating parameters. The model implicitly routes its own depth purely driven by task complexity overpowering the optimizer friction, stabilizing training.

### 2. Temporal RNN Fusion
Looping a single core layer repeatedly via Backpropagation Through Time (BPTT) often leads to token oversmoothing and representation collapse.
* **The Fix:** The traditional linear halting gate is replaced with an RNN controller (GRU). Rather than acting passively, the RNN projects its hidden state back into the residual stream at each cycle. This replaces static timestep embeddings and injects historical pondering context, mathematically breaking BPTT symmetry and preserving sharp token representations across deep unrolls.

## Benchmarks & Results

Both models were tasked with reading a 40-token sequence and compressing it through a strict **$1 \times 256$ informational bottleneck** before passing it to a frozen Cloud Predictor MLP. 

* **Standard Compressor:** 4 Static Layers (~4.3M parameters)
* **Universal Compressor:** 1 Recurrent Core Layer (~1.6M parameters)

The UT successfully matched or exceeded the static baseline's compression density using roughly **1/2.6x the parameters**.

| Task | Architecture | Train Acc | Test Acc | Avg Ponder Steps |
| :--- | :--- | :--- | :--- | :--- |
| **LISTOPS** (Hierarchical) | Standard (4 Layer) | 32.5% | 23.5% | N/A |
| **LISTOPS** (Hierarchical) | **Universal (1 Layer)** | 65.1% | **41.2%** | ~9.0 |
| **DYCK** (State Tracking) | Standard (4 Layer) | 73.3% | 74.9% | N/A |
| **DYCK** (State Tracking) | **Universal (1 Layer)** | 92.3% | **93.4%** | ~7.0 |

*Note: The standard model physically lacks the spatial depth to resolve nested ListOps trees or track long Dyck states before the bottleneck. The Universal Compressor dynamically scales its arithmetic intensity (up to 9 loops) to resolve the logic within a flat memory footprint.*
