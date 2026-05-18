# SPPO: Sequence-Level PPO for Long-Horizon Reasoning Tasks

[![arXiv](https://img.shields.io/badge/arXiv-2604.08865-b31b1b.svg)](https://arxiv.org/abs/2604.08865)

This repository serves as the official implementation of the paper **"SPPO: Sequence-Level PPO for Long-Horizon Reasoning Tasks"**.

## News

- **[2026.04]** 🎉 This paper has been accepted to the **ACL 2026 Main Conference**!
- **[2026.05]** 🎉 This paper has been selected as Oral in ACL 2026. See you in San Diego!

## 💡 Abstract

Proximal Policy Optimization (PPO) was central to aligning Large Language Models (LLMs) in reasoning tasks with verifiable rewards. However, standard token-level PPO struggles in this setting due to the instability of temporal credit assignment over long Chain-of-Thought (CoT) horizons and the prohibitive memory cost of the value model. While critic-free alternatives like GRPO mitigate these issues, they incur significant computational overhead by requiring multiple samples for baseline estimation, severely limiting training throughput. 

**Sequence-Level PPO (SPPO)** introduces a scalable algorithm that harmonizes the sample efficiency of PPO with the stability of outcome-based updates: shifting from a Multi-Step MDP to a **Sequence-Level Contextual Bandit**.
- **Sequence-Level Optimization**: Treats the entire reasoning chain as a single atomic action, utilizing a decoupled scalar value function to derive low-variance advantage signals without multi-sampling.
- **Decoupled Small Critic**: Because scalar solvability estimation is significantly simpler than generative reasoning, SPPO enables training with a lightweight critic (e.g., 1.5B Critic for a 7B Policy), radically reducing VRAM requirements without sacrificing performance.

<div align="center">
  <img src="image/main.png" alt="SPPO Architecture" width="800"/>
  <p><em>Figure 1: Overall Architecture of Sequence-Level PPO (SPPO).</em></p>
</div>

<br>

<div align="center" style="display: flex; justify-content: center; gap: 20px;">
  <div style="text-align: center;">
    <img src="image/memory_usage_percentage.png" alt="Memory Footprint Comparison" width="400"/>
    <p><em>Figure 2: Peak VRAM Allocation Analysis.</em></p>
  </div>
  <div style="text-align: center;">
    <img src="image/efficiency_plot.png" alt="Training Efficiency" width="400"/>
    <p><em>Figure 3: Training Efficiency and Performance.</em></p>
  </div>
</div>


## 🚀 Key Features

- **Exclusive SPPO Implementation**: Full support for the Sequence-Level Contextual Bandit formulation with Single-Sample Efficiency ($N=1$).
- **Efficient & Stable**: Resolves the temporal credit assignment problem in long-horizon CoT tasks while avoiding the computational bottleneck of multi-sampling.
- **Extreme Memory Efficiency**: Natively supports "Small Critic" architectures (e.g., training a 7B policy with a 1.5B critic), making efficient RL alignment accessible on consumer-grade hardware.
- **Scalable**: Built on top of `verl`, supporting FSDP and Megatron for training large-scale models.

## 🛠️ Quick Start

### Installation

**Option 1: Automated Setup (Recommended)**

```bash
bash uv_verl.sh
```

**Option 2: Manual Setup**

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip

# Install package in editable mode
pip install --no-deps -e .

# Add project root to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)
```


## ⚙️ Training with SPPO

We provide pre-configured scripts for various model sizes and settings. 

```bash
# 1. DeepSeek-R1-Distill-Qwen 1.5B (SPPO DeepscaleR)
bash run_scripts/run_ds1.5B_PPO_SEQUENCE_shuffle.sh

# 2. DeepSeek-R1-Distill-Qwen 7B (DAPO-17k)
bash run_scripts/run_R1-7B_DAPO_SEQUENCE.sh

# 3. DeepSeek-R1-Distill-Qwen 7B (DAPO-17k with Small Critic)
# Utilizes a 1.5B critic to align the 7B policy.
bash run_scripts/run_R1-7B_DAPO_SEQUENCE_small_critic.sh
```

## 📊 Evaluation

Models can be evaluated on the provided AIME24/25, AMC23, MATH, and Minerva benchmarks out of the box using the `verl` evaluation toolkit. Training logs and checkpoints will automatically be populated in the current working directory.

## 📜 Citation

If you find SPPO useful for your research, please cite our paper:

```bibtex
@misc{wang2026spposequencelevelppolonghorizon,
      title={SPPO: Sequence-Level PPO for Long-Horizon Reasoning Tasks}, 
      author={Tianyi Wang and Yixia Li and Long Li and Yibiao Chen and Shaohan Huang and Yun Chen and Peng Li and Yang Liu and Guanhua Chen},
      year={2026},
      eprint={2604.08865},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2604.08865}, 
}
```
