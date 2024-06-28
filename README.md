# A Comprehensive Survey on Low-Rank Adaptation


This repository provides a comprehensive survey of Low-Rank Adaptation (LoRA) methods and their applications. We welcome contributions to keep this list up-to-date. If you find this repository useful, please consider starring it.

## Table of Contents

1. [LoRA Settings](#1-lora-settings)
   1.1 [Initialization](#11-initialization)
   1.2 [Hyperparameters](#12-hyperparameters)
   1.3 [Optimization](#13-optimization)
   1.4 [Regularization](#14-regularization)
2. [Dynamic Rank](#2-dynamic-rank)
3. [LoRA Variants](#3-lora-variants)
4. [Other Low-rank Decomposition](#4-other-low-rank-decomposition)
5. [LoRA with Model Compressions](#5-lora-with-model-compressions)
   5.1 [LoRA with Pruning](#51-lora-with-pruning)
   5.2 [LoRA with Quantization](#52-lora-with-quantization)
   5.3 [LoRA with NAS](#53-lora-with-nas)
   5.4 [Memory-efficient LoRA](#54-memory-efficient-lora)
6. [LoRA Extensions](#6-lora-extensions)
   6.1 [Multiple LoRA](#61-multiple-lora)
   6.2 [Mixture-of-Experts (MOE) LoRA](#62-mixture-of-experts-moe-lora)
   6.3 [LoRA Merge](#63-lora-merge)
7. [LoRA applications](#7-lora-applications)
   7.1 [Visual Understanding](#71-visual-understanding)
   7.2 [Visual Generation](#72-visual-generation)
   7.3 [Language Understanding](#73-language-understanding)
   7.4 [Multimodal learning](#74-multimodal-learning)
   7.5 [Other](#75-other)

## 1. LoRA Settings
| Year | Title | Venue | Paper | Code |
|------|-------|-------|-------|------|
| 2022     | **LoRA: Low-Rank Adaptation of Large Language Models**      |    ICLR   |     [Link](https://arxiv.org/abs/2106.09685)  |   [Link](https://github.com/microsoft/LoRA)   |

### 1.1 Initialization

| Year | Title | Venue | Paper | Code |
|------|-------|-------|-------|------|
|      |       |       |       |      |

### 1.2 Hyperparameters

| Year | Title | Venue | Paper | Code |
|------|-------|-------|-------|------|
| 2024 | **LoRA+: Efficient Low Rank Adaptation of Large Models** |    arXiv   | [Link](https://arxiv.org/abs/2402.12354) |  [Link](https://github.com/nikhil-ghosh-berkeley/loraplus) |
| 2023 | **The expressive power of low-rank adaptation** |    ICLR   | [Link](https://arxiv.org/abs/2310.17513) |  [Link](https://github.com/UW-Madison-Lee-Lab/Expressive_Power_of_LoRA) |
### 1.3 Optimization

| Year | Title | Venue | Paper | Code |
|------|-------|-------|-------|------|
| 2024 | **Derivative-Free Optimization for Low-Rank Adaptation in Large Language Models** |    arXiv   | [Link](https://arxiv.org/abs/2403.01754) |  [Link](https://github.com/stan-anony/derivative_free_lora_rank) |
| 2024 | **AFLoRA: Adaptive Freezing of Low Rank Adaptation in Parameter Efficient Fine-Tuning of Large Models** |    arXiv   | [Link](https://arxiv.org/abs/2403.13269) |  - |
| 2023 | **Bayesian Low-rank Adaptation for Large Language Models** |    ICLR   | [Link](https://arxiv.org/abs/2308.13111) |  [Link](https://github.com/MaximeRobeyns/bayesian_lora) |

=
### 1.4 Regularization

| Year | Title | Venue | Paper | Code |
|------|-------|-------|-------|------|
| 2024 | **LoRA Meets Dropout under a Unified Framework** |    arXiv   | [Link](https://arxiv.org/abs/2403.00812) |  - |
| 2024 | **AdvLoRA: Adversarial Low-Rank Adaptation of Vision-Language Models** |    arXiv   | [Link](https://arxiv.org/pdf/2404.13425) |  - |
| 2024 | **PeriodicLoRA: Breaking the Low-Rank Bottleneck in LoRA Optimization** |    arXiv   | [Link](https://arxiv.org/abs/2402.16141) |  - |
## 2. Dynamic Rank

| Year | Title | Venue | Paper | Code |
|------|-------|-------|-------|------|
| 2023 | **DyLoRA: Parameter-Efficient Tuning of Pre-trained Models using Dynamic Search-Free Low-Rank Adaptation** |    EACL   | [Link](https://arxiv.org/pdf/2404.13425) |  [Link](https://github.com/huawei-noah/KD-NLP/tree/main/DyLoRA) |
| 2024 | **MoRA: High-Rank Updating for Parameter-Efficient Fine-Tuning** |    arXiv   | [Link](https://arxiv.org/pdf/2405.12130) |  [Link](https://github.com/kongds/MoRA) |
| 2024 | **BiLoRA: A Bi-level Optimization Framework for Overfitting-Resilient Low-Rank Adaptation of Large Pre-trained Models** |    arXiv   | [Link](https://arxiv.org/abs/2403.13037) |  - |

## 3. LoRA Variants

| Year | Title | Venue | Paper | Code |
|------|-------|-------|-------|------|
| 2024 | **FLoRA: Low-Rank Core Space for N-dimension** |    arXiv   | [Link](https://arxiv.org/abs/2405.14739) |  [Link](https://github.com/SJTU-DeepVisionLab/FLoRA) |
| 2023 | **Lora-fa: Memory-efficient low-rank adaptation for large language models fine-tuning** |    arXiv   | [Link](https://arxiv.org/abs/2308.03303) |  - |
## 4. Other Low-rank Decomposition

| Year | Title | Venue | Paper | Code |
|------|-------|-------|-------|------|
| 2024 | **Parameter-Efficient Fine-Tuning with Discrete Fourier Transform** |    ICML   | [Link](https://arxiv.org/pdf/2405.03003) |  [Link](https://github.com/Chaos96/fourierft) |

## 5. LoRA with Model Compressions

### 5.1 LoRA with Pruning

| Year | Title | Venue | Paper | Code |
|------|-------|-------|-------|------|
|      |       |       |       |      |

### 5.2 LoRA with Quantization

| Year | Title | Venue | Paper | Code |
|------|-------|-------|-------|------|
|      |       |       |       |      |

### 5.3 LoRA with NAS

| Year | Title | Venue | Paper | Code |
|------|-------|-------|-------|------|
| 2024 | **LoNAS: Elastic Low-Rank Adapters for Efficient Large Language** |    COLING   | [Link](https://aclanthology.org/2024.lrec-main.940.pdf) |  [Link](https://github.com/IntelLabs/Hardware-Aware-Automated-Machine-Learning?tab=readme-ov-file) |

### 5.4 Memory-efficient LoRA

| Year | Title | Venue | Paper | Code |
|------|-------|-------|-------|------|
| 2024 | **Galore: Memory-efficient llm training by gradient low-rank projection** |    ICML   | [Link](https://arxiv.org/abs/2403.03507) |  [Link](https://github.com/jiaweizzhao/GaLore) |
## 6. LoRA Extensions

### 6.1 Multiple LoRA

| Year | Title | Venue | Paper | Code |
|------|-------|-------|-------|------|
|      |       |       |       |      |

### 6.2 Mixture-of-Experts (MOE) LoRA

| Year | Title | Venue | Paper | Code |
|------|-------|-------|-------|------|
| 2023 | **Loramoe: Revolutionizing mixture of experts for maintaining world knowledge in language model alignment** |    arXiv   | [Link](https://simg.baai.ac.cn/paperfile/96f0cfd7-79c7-4110-88e5-4ea80a7fbc8d.pdf) |  - |

### 6.3 LoRA Merge

| Year | Title | Venue | Paper | Code |
|------|-------|-------|-------|------|
|      |       |       |       |      |

## 7. LoRA applications

### 7.1 Visual Understanding

| Year | Title | Venue | Paper | Code |
|------|-------|-------|-------|------|
| 2024 | **Multi-LoRA Composition for Image Generation** |    arXiv   | [Link](https://arxiv.org/pdf/2402.16843) |  [Link](https://github.com/maszhongming/Multi-LoRA-Composition) |
| 2023 | **Motion Style Transfer: Modular Low-Rank Adaptation for Deep Motion Forecasting** |    PMLR   | [Link](https://proceedings.mlr.press/v205/kothari23a/kothari23a.pdf) |  [Link](https://github.com/vita-epfl/motion-style-transfer) |
### 7.2 Visual Generation

| Year | Title | Venue | Paper | Code |
|------|-------|-------|-------|------|
|      |       |       |       |      |

### 7.3 Language Understanding

| Year | Title | Venue | Paper | Code |
|------|-------|-------|-------|------|
| 2023 | **Exploring the impact of low-rank adaptation on the performance, efficiency, and regularization of RLHF** |    arXiv   | [Link](https://arxiv.org/abs/2309.09055) |  [Link](https://github.com/SimengSun/alpaca_farm_lora) |

### 7.4 Multimodal learning

| Year | Title | Venue | Paper | Code |
|------|-------|-------|-------|------|
|      |       |       |       |      |

### 7.5 Other

| Year | Title | Venue | Paper | Code |
|------|-------|-------|-------|------|
| 2023 | **Low-Rank Adaptation of Large Language Model Rescoring for Parameter-Efficient Speech Recognition** |    ASRU   | [Link](https://ieeexplore.ieee.org/abstract/document/10389632) |  - |



## Contributing

We welcome contributions to this survey. Please feel free to submit a pull request to add new papers or update existing information.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
