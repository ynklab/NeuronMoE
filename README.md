# NeuronMoE

> **NeuronMoE: Neuron-Guided Mixture-of-Experts for Efficient Multilingual LLM Extension**

[![arXiv](https://img.shields.io/badge/arXiv-2603.05046-b31b1b.svg)](https://arxiv.org/abs/2603.05046)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

## Overview

NeuronMoE is a method for efficiently extending large language models (LLMs) to new languages using a neuron-guided Mixture-of-Experts (MoE) architecture. By analyzing language-specific neuron distributions across model layers, NeuronMoE determines the optimal number of experts per layer, achieving performance comparable to uniform expert allocation (LayerMoE baseline) while reducing the number of parameters by approximately 40%.

![NeuronMoE Overview](figures/overview.png)

![Allocation Comparison](figures/allocation_comparison.png)

**Key Results:**
- Equivalent multilingual performance to LayerMoE with ~40% parameter reduction
- Neuron-guided expert allocation based on language-specific neuron analysis
- Two-stage training: expert training + router training

## Citation

If you use this repository, please cite NeuronMoE and the works this codebase
builds on:

```bibtex
@misc{li2026neuronmoe,
  title = {NeuronMoE: Neuron-Guided Mixture-of-Experts for Efficient Multilingual LLM Extension},
  author = {Li, Rongzhi and Yanaka, Hitomi},
  year = {2026},
  eprint = {2603.05046},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  doi = {10.48550/arXiv.2603.05046},
  url = {https://arxiv.org/abs/2603.05046}
}

@inproceedings{zhang-etal-2025-less,
  title = {Less, but Better: Efficient Multilingual Expansion for {LLM}s via Layer-wise Mixture-of-Experts},
  author = {Zhang, Xue and Liang, Yunlong and Meng, Fandong and Zhang, Songming and Chen, Yufeng and Xu, Jinan and Zhou, Jie},
  booktitle = {Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  year = {2025},
  address = {Vienna, Austria},
  publisher = {Association for Computational Linguistics},
  pages = {17948--17963},
  doi = {10.18653/v1/2025.acl-long.878},
  url = {https://aclanthology.org/2025.acl-long.878/}
}

@inproceedings{kojima-etal-2024-multilingual,
  title = {On the Multilingual Ability of Decoder-based Pre-trained Language Models: Finding and Controlling Language-Specific Neurons},
  author = {Kojima, Takeshi and Okimura, Itsuki and Iwasawa, Yusuke and Yanaka, Hitomi and Matsuo, Yutaka},
  booktitle = {Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)},
  year = {2024},
  address = {Mexico City, Mexico},
  publisher = {Association for Computational Linguistics},
  pages = {6919--6971},
  doi = {10.18653/v1/2024.naacl-long.384},
  url = {https://aclanthology.org/2024.naacl-long.384/}
}
```

## Repository Structure

```
NeuronMoE/
├── neuron_analysis/         # Language-specific neuron analysis (based on ml-selfcond)
│   ├── selfcond/            # Core analysis package
│   ├── scripts/             # Analysis scripts
│   ├── assets/              # Language data for neuron analysis
│   └── main_prod_env.sh     # Orchestration script
├── expert_allocation/       # Neuron-guided expert number determination
│   ├── analyze_neuron_distribution.py
│   ├── analyze_neuron_distribution_3lang.py
│   ├── visualize_neuron_distribution.py
│   ├── create_sense_data.py
│   └── configs/             # Expert configuration files
├── peft/                    # Custom PEFT with MoE tuner
│   └── src/peft/tuners/moe/
├── training/scripts/        # Training scripts (Stage 1 & 2)
├── evaluation/scripts/      # Evaluation scripts
├── patches/                 # Patches for dependencies
│   ├── llama_factory/       # MoE modifications for LLaMA-Factory v0.5.0
│   ├── transformers/        # MoE loss functions for transformers 4.45.0
│   └── lm_eval_tasks/       # Custom MMLU tasks (Greek, Turkish)
├── scripts/                 # Pipeline scripts
│   ├── install_patches.sh   # Apply patches to dependencies
│   ├── prepare_data.sh      # Step 1: Data download & preprocessing
│   ├── run_neuron_analysis.sh  # Step 2: Neuron analysis
│   └── run_expert_allocation.sh  # Step 3: Expert allocation
├── similarity/              # LayerMoE baseline similarity analysis
├── data/                    # Data download and preprocessing
└── figures/                 # Paper figures
```

## Installation


### Environment Setup

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install base dependencies
uv venv --python 3.10
source .venv/bin/activate
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
uv pip install transformers==4.45.0
```

### Install Dependencies

#### 1. NeuronMoE package and training extras

Install this repository as an editable package and pull in the optional
`training` extras (deepspeed, flash-attn, seaborn, wandb, …) needed for
the full pipeline:

```bash
uv pip install -e ".[training]"
```

(Optional) For faster attention, install `flash-attn` separately. It does not
declare `torch` as a build dependency, so it must be installed with
build-isolation disabled and only after `torch` is already in the venv:

```bash
uv pip install flash-attn==2.8.3 --no-build-isolation
```

#### 2. LLaMA-Factory (for training)

Install LLaMA-Factory v0.5.0 directly from GitHub:

```bash
uv pip install "git+https://github.com/hiyouga/LLaMA-Factory.git@v0.5.0"
```

#### 3. Custom PEFT (MoE support)

Install the custom PEFT library included in this repository, which adds MoE tuner support.
This must be installed **after** LLaMA-Factory to override the peft dependency it installs:

```bash
cd peft
uv pip install -e .
```

#### 4. lm-evaluation-harness (for evaluation)

Install lm-evaluation-harness v0.4.4 directly from GitHub:

```bash
uv pip install "git+https://github.com/EleutherAI/lm-evaluation-harness.git@v0.4.4"
```

#### 5. Apply patches

Apply MoE modifications to LLaMA-Factory, transformers, and lm-evaluation-harness:

```bash
bash scripts/install_patches.sh
```

This script applies:
- **LLaMA-Factory**: `moe` finetuning type, MoE-specific arguments (`ada_moe_num_experts_list`, `topk`, `aux_loss_coef`, `lpr_loss_coef`, etc.), MoE adapter initialization via `MoeConfig`, group-based routing and MoE loss computation
- **transformers**: LPR loss, load balancing loss, classification loss, and sequential adding loss in `LlamaForCausalLM`
- **lm-evaluation-harness**: Custom MMLU task definitions for Greek (`mmlu_el`) and Turkish (`mmlu_tr`)

#### 6. Neuron Analysis Dependencies

```bash
uv pip install -r neuron_analysis/frozen_requirements.txt
```

## Pipeline

### Configuration

Set all environment variables once before running the pipeline:

```bash
export OUTPUT_DIR=/path/to/data                    # Data download destination
export BASE_MODEL_PATH=meta-llama/Llama-3.2-3B     # Base model
export LLAMA_FACTORY_DIR=/path/to/LLaMA-Factory    # LLaMA-Factory installation
export LM_EVAL_DIR=/path/to/lm-evaluation-harness  # lm-evaluation-harness installation
export OUTPUT_BASE_DIR=/path/to/outputs             # Training output directory
```

### Step 1: Data Preparation

```bash
bash scripts/prepare_data.sh
```

See `scripts/prepare_data.sh` for additional variables (`NEW_LANGS`, `OLD_LANGS`).

### Step 2: Language-Specific Neuron Analysis

```bash
export NEURONMOE_OUTPUT_DIR=$OUTPUT_BASE_DIR/neuron_output
export SAMPLE_DATA_DIR=$OUTPUT_DIR/sample-data
bash scripts/run_neuron_analysis.sh
```

See `scripts/run_neuron_analysis.sh` for additional variables (`LANGUAGES`, `MODEL`).

### Step 3: Neuron-Guided Expert Allocation

```bash
export NEURON_RESULTS_DIR=$NEURONMOE_OUTPUT_DIR
bash scripts/run_expert_allocation.sh
```

Set `MODE=single` for single new language, `MODE=3lang` (default) for multiple.

### Step 4: MoE Training

#### Stage 1: Expert Training

```bash
export DATA_DIR=$OUTPUT_DIR/sample-data
export G1_DATASETS="el2b,hu2b,tr2b"
export G1_LANG_FILES="el-llama-2B.jsonl,hu-llama-2B.jsonl,tr-llama-2B.jsonl"
bash training/scripts/stage1_neuronmoe.sh
```

#### Stage 2: Router Training

```bash
export MOE_MODEL_PATH=$OUTPUT_BASE_DIR/stage1/checkpoint
bash training/scripts/stage2_neuronmoe.sh
```

### Step 5: Evaluation

```bash
export PEFT_MODEL_PATH=$OUTPUT_BASE_DIR/stage2/checkpoint
export OUTPUT_PATH=$OUTPUT_BASE_DIR/eval_results
export LM_EVAL=$LM_EVAL_DIR/lm_eval
bash evaluation/scripts/eval_g1.sh
```

## LayerMoE Baseline Reproduction

To reproduce the LayerMoE baseline (uniform expert allocation):

```bash
# Compute layer similarity
uv run python similarity/cal_similarity_dense.py \
    -m $BASE_MODEL_PATH \
    --data_dir $OUTPUT_DIR/sample-data

# Train with layer-similarity config
export EXPERT_CONFIG_PATH=expert_allocation/configs/expert_config_layer_similarity.txt
bash training/scripts/stage1_neuronmoe.sh
```

## Acknowledgements

This project builds upon the following works:

- [MoE-LPR](https://github.com/NJUNLP/MoE-LPR) - MoE framework for multilingual LLM extension
- [On the Multilingual Ability of Decoder-based Pre-trained Language Models](https://aclanthology.org/2024.naacl-long.384/) (Kojima et al., NAACL 2024) - Language-specific neuron analysis
- [ml-selfcond](https://github.com/apple/ml-selfcond) (Apple) - Self-conditioning framework used by the neuron analysis code
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) - Training framework
- [LayerMoE](https://github.com/NJUNLP/LayerMoE) / [Less, but Better](https://aclanthology.org/2025.acl-long.878/) (Zhang et al., ACL 2025) - Layer-wise MoE baseline

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
