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
| 2024 | The Impact of Initialization on LoRA Finetuning Dynamics     | -                | [Link](https://arxiv.org/abs/2406.08447) | -                                                            |
| 2024 | ShareLoRA: Parameter Efficient and Robust Large Language Model Fine-tuning via Shared Low-Rank Adaptation | -     | [Link](https://arxiv.org/abs/2406.10785) | -                                                            |
| 2024 | MiLoRA: Harnessing Minor Singular Components for Parameter-Efficient LLM Finetuning | -     | [Link](https://arxiv.org/abs/2406.09044) | -                                                            |
| 2024    | **PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models**      |    ICLR   |     [Link](https://arxiv.org/abs/2404.02948)  |   [Link](https://github.com/GraphPKU/PiSSA)   |
| 2024 | **CorDA: Context-Oriented Decomposition Adaptation of Large Language Models** |    arXiv   | [Link](https://arxiv.org/abs/2402.16141) |  [link](https://github.com/iboing/CorDA) |
| 2024 | **SVFT: Parameter-Efficient Fine-Tuning with Singular Vectors** |     arXiv   | [Link](https://arxiv.org/abs/2405.19597v1) |  [link](https://github.com/VijayLingam95/SVFT/) |






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
| 2024 | A Study of Optimizations for Fine-tuning Large Language Models | -                | [Link](https://arxiv.org/abs/2406.02290) | -                                                            |\
| 2024 | Bayesian-LoRA: LoRA based Parameter Efficient Fine-Tuning using Optimal Quantization levels and Rank Values trough Differentiable Bayesian Gates | -     | [Link](https://arxiv.org/abs/2406.13046) | -                                                            |
| 2024 | Understanding Linear Probing then Fine-tuning Language Models from NTK Perspective | -     | [Link](https://arxiv.org/abs/2405.16747) | -                                                            |
| 2024 | BLoB: Bayesian Low-Rank Adaptation by Backpropagation for Large Language Models | -     | [Link](https://arxiv.org/abs/2406.11675) | -                                                            |





### 1.4 Regularization

| Year | Title | Venue | Paper | Code |
|------|-------|-------|-------|------|
| 2024 | **LoRA Meets Dropout under a Unified Framework** |    arXiv   | [Link](https://arxiv.org/abs/2403.00812) |  - |
| 2024 | **AdvLoRA: Adversarial Low-Rank Adaptation of Vision-Language Models** |    arXiv   | [Link](https://arxiv.org/pdf/2404.13425) |  - |
| 2024 | **PeriodicLoRA: Breaking the Low-Rank Bottleneck in LoRA Optimization** |    arXiv   | [Link](https://arxiv.org/abs/2402.16141) |  - |



### 1.5 Sparse

| Year | Title | Venue | Paper | Code |
|------|-------|-------|-------|------|
| 2024 | Lottery Ticket Adaptation: Mitigating Destructive Interference in LLMs | -                | [Link](https://arxiv.org/abs/2406.16797) | [Link](https://github.com/kiddyboots216/lottery-ticket-adaptation) |
| 2024 | Sparse High Rank Adapters                                    | -                | [Link](https://arxiv.org/abs/2406.13175) | -                                                            |
| 2024 | SLTrain: a sparse plus low-rank approach for parameter and memory efficient pretraining | -                | [Link](https://arxiv.org/abs/2406.02214) | -                                                            |
| 2023 | Sparse Low-rank Adaptation of Pre-trained Language Models    | EMNLP    | [Link](https://arxiv.org/abs/2311.11696)     | [Link](https://github.com/TsinghuaC3I/SoRA)        |
| 2024 | SLoPe: Double-Pruned Sparse Plus Lazy Low-Rank Adapter Pretraining of LLMs | -     | [Link](https://arxiv.org/abs/2405.16325) | -                                                            |
| 2024 | MLAE: Masked LoRA Experts for Parameter-Efficient Fine-Tuning | -     | [Link](https://arxiv.org/abs/2405.18897) | -                                                            |

### 1.6  Bayesian
| 2023 | **Bayesian Low-rank Adaptation for Large Language Models** |    ICLR   | [Link](https://arxiv.org/abs/2308.13111) |  [Link](https://github.com/MaximeRobeyns/bayesian_lora) |
| 2024 | Bayesian-LoRA: LoRA based Parameter Efficient Fine-Tuning using Optimal Quantization levels and Rank Values trough Differentiable Bayesian Gates | -     | [Link](https://arxiv.org/abs/2406.13046) | -                                                            |
| 2024 | BLoB: Bayesian Low-Rank Adaptation by Backpropagation for Large Language Models | -     | [Link](https://arxiv.org/abs/2406.11675) | -      

### 1.7 Robust
| 2024 | ShareLoRA: Parameter Efficient and Robust Large Language Model Fine-tuning via Shared Low-Rank Adaptation | -                | [Link](https://arxiv.org/abs/2406.10785) | -                                                            |
| 2024 | RoSA: Accurate Parameter-Efficient Fine-Tuning via Robust Adaptation | -                | [Link](https://arxiv.org/abs/2401.04679) | [Link](https://github.com/IST-DASLab/RoS)                    |


## 2. Dynamic Rank

| Year | Title | Venue | Paper | Code |
|------|-------|-------|-------|------|
| 2023 | Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning | ICLR     | [Link](https://arxiv.org/pdf/2303.10512.pdf) | [Link](https://github.com/QingruZhang/AdaLoRA)     |
| 2023 | **DyLoRA: Parameter-Efficient Tuning of Pre-trained Models using Dynamic Search-Free Low-Rank Adaptation** |    EACL   | [Link](https://arxiv.org/pdf/2404.13425) |  [Link](https://github.com/huawei-noah/KD-NLP/tree/main/DyLoRA) |
| 2024 | **MoRA: High-Rank Updating for Parameter-Efficient Fine-Tuning** |    arXiv   | [Link](https://arxiv.org/pdf/2405.12130) |  [Link](https://github.com/kongds/MoRA) |
| 2024 | **BiLoRA: A Bi-level Optimization Framework for Overfitting-Resilient Low-Rank Adaptation of Large Pre-trained Models** |    arXiv   | [Link](https://arxiv.org/abs/2403.13037) |  - |
|

## 3. LoRA Variants

| Year | Title | Venue | Paper | Code |
|------|-------|-------|-------|------|
| 2023 | **Lora-fa: Memory-efficient low-rank adaptation for large language models fine-tuning** |    arXiv   | [Link](https://arxiv.org/abs/2308.03303) |  - |
| 2023 | **VERA: VECTOR-BASED RANDOM MATRIX ADAPTATION** |    arXiv   | [Link](https://arxiv.org/pdf/2310.11454) |  - |
| 2024 | **DoRA: Weight-Decomposed Low-Rank Adaptation** |   ICML   | [Link](https://arxiv.org/abs/2402.09353) |  [Link](https://github.com/NVlabs/DoRA) |
| 2024 | **FLoRA: Low-Rank Core Space for N-dimension** |    arXiv   | [Link](https://arxiv.org/abs/2405.14739) |  [Link](https://github.com/SJTU-DeepVisionLab/FLoRA) |
| 2024 | Mixture-of-Subspaces in Low-Rank Adaptation                  | -                | [Link](https://arxiv.org/abs/2406.11909) | [Link](https://github.com/wutaiqiang/MoSLoRA)                |
| 2024 | LoRA-XS: Low-Rank Adaptation with Extremely Small Number of Parameters | -     | [Link](https://arxiv.org/abs/2405.17604) | -                                                            |
| 2024 | ReFT: Representation Finetuning for Language Models          | Preprint | [Link](https://arxiv.org/abs/2404.03592)     | [Link](https://github.com/stanfordnlp/pyreft)      |
| 2024 | LaMDA: Large Model Fine-Tuning via Spectrally Decomposed Low-Dimensional Adaptation | -     | [Link](https://arxiv.org/abs/2406.12832) | -                                                            |
| 2024 | Structured Unrestricted-Rank Matrices for Parameter Efficient Fine-tuning | Preprint | [Link](https://arxiv.org/abs/2406.17740)     |                                                    |
| 2024 | LoRETTA: Low-Rank Economic Tensor-Train Adaptation for Ultra-Low-Parameter Fine-Tuning of Large Language Models | NAACL    | [Link](https://arxiv.org/abs/2311.11696)     | [Link](https://github.com/yifanycc/loretta)        |
| 2024 | Riemannian Preconditioned LoRA for Fine-Tuning Foundation Models | -                | [Link](https://arxiv.org/abs/2402.02347) | [Link](https://github.com/pilancilab/Riemannian_Preconditioned_LoRA) |
| 2024 | Trans-LoRA: towards data-free Transferable Parameter Efficient Finetuning | -     | [Link](https://arxiv.org/abs/2405.17258) | -                                                            |
| 2024 | VB-LoRA: Extreme Parameter Efficient Fine-Tuning with Vector Banks | -     | [Link](https://arxiv.org/abs/2405.15179) | -                                                            |





## 4. Other Low-rank Decomposition


| Year | Title | Venue | Paper | Code |
|------|-------|-------|-------|------|
| 2024 | **Parameter-Efficient Fine-Tuning with Discrete Fourier Transform** |    ICML   | [Link](https://arxiv.org/pdf/2405.03003) |  [Link](https://github.com/Chaos96/fourierft) |
| 2024 | OLoRA: Orthonormal Low-Rank Adaptation of Large Language Models | -                | [Link](https://arxiv.org/abs/2406.01775) | -                                                            |
| 2024 | **Bridging The Gap between Low-rank and Orthogonal Adaptation via Householder Reflection Adaptation** |     arXiv   | [Link](https://arxiv.org/abs/2405.17484) |  [link](https://github.com/DaShenZi721/HRA) |






## 5. LoRA with Model Compressions

### 5.1 LoRA with Pruning

| Year | Title | Venue | Paper | Code |
|------|-------|-------|-------|------|
| 2024 | RankAdaptor: Hierarchical Dynamic Low-Rank Adaptation for Structural Pruned LLMs | -                | [Link](https://arxiv.org/abs/2406.15734) | -                                                            |
| 2024 | PRILoRA: Pruned and Rank-Increasing Low-Rank Adaptation      | EACL             | [Link](https://arxiv.org/abs/2401.11316) | -                                                            |


### 5.2 LoRA with Quantization

| Year | Title | Venue | Paper | Code |
|------|-------|-------|-------|------|
|  2023    |    QA-LoRA: Quantization-Aware Low-Rank Adaptation of Large Language Models   |  ICLR      | [Link](https://arxiv.org/pdf/2309.14717) |  [Link](https://github.com/yuhuixu1993/qa-lora) |
| 2024 | Low-Rank Quantization-Aware Training for LLMs                | -                | [Link](https://arxiv.org/abs/2406.06385) | -                                                            |
| 2024 | QDyLoRA: Quantized Dynamic Low-Rank Adaptation for Efficient Large Language Model Tuning | AAAI Workshop    | [Link](https://arxiv.org/abs/2402.10462) | -                                                            |
| 2024 | LoQT: Low Rank Adapters for Quantized Training               | -     | [Link](https://arxiv.org/abs/2405.16528) | -                                                            |
| 2024 | One QuantLLM for ALL: Fine-tuning Quantized LLMs Once for Efficient Deployments | -     | [Link](https://arxiv.org/abs/2405.20202) | -                                                            |
| 2023 | QLORA: Efficient Finetuning of Quantized LLMs                | NeurIPS  | [Link](https://arxiv.org/pdf/2305.14314.pdf) | [Link](https://github.com/artidoro/qlora)          |
| 2024 |  Accurate LoRA-Finetuning Quantization of LLMs via Information Retention|ICML| [Link](https://arxiv.org/abs/2402.05445) | [Link](https://github.com/htqin/IR-QLoRA)          |

### 5.3 LoRA with NAS

| Year | Title | Venue | Paper | Code |
|------|-------|-------|-------|------|
| 2024 | **LoNAS: Elastic Low-Rank Adapters for Efficient Large Language** |    COLING   | [Link](https://aclanthology.org/2024.lrec-main.940.pdf) |  [Link](https://github.com/IntelLabs/Hardware-Aware-Automated-Machine-Learning?tab=readme-ov-file) |

### 5.4 Memory-efficient LoRA

| Year | Title | Venue | Paper | Code |
|------|-------|-------|-------|------|
| 2024 | **Galore: Memory-efficient llm training by gradient low-rank projection** |    ICML   | [Link](https://arxiv.org/abs/2403.03507) |  [Link](https://github.com/jiaweizzhao/GaLore) |
| 2024 | Flora: Low-Rank Adapters Are Secretly Gradient Compressors   | ICML             | [Link](https://arxiv.org/abs/2402.03293) | -                                                            |
| 2024 | BlockLLM: Memory-Efficient Adaptation of LLMs by Selecting and Optimizing the Right Coordinate Blocks | Preprint | [Link](https://arxiv.org/abs/2406.17296)     |                                                    |


### 5.4  Knowledge Distillation LoRA

| Year | Title | Venue | Paper | Code |
| 2024 | PC-LoRA: Low-Rank Adaptation for Progressive Model Compression with Knowledge Distillation | -                | [Link](https://arxiv.org/abs/2406.09117) | -                                                            |




## 6. LoRA Extensions

### 6.1 Multiple LoRA

| Year | Title | Venue | Paper | Code |
|------|-------|-------|-------|------|
| 2024 | LoRA-Ensemble: Efficient Uncertainty Modelling for Self-attention Networks | -     | [Link](https://arxiv.org/abs/2405.14438) | -                                                            |
| 2024 | MeteoRA: Multiple-tasks Embedded LoRA for Large Language Models | -     | [Link](https://arxiv.org/abs/2405.13053) | -                                                            |
| 2024 | MELoRA: Mini-Ensemble Low-Rank Adapters for Parameter-Efficient Fine-Tuning | ACL              | [Link](https://arxiv.org/abs/2402.17263) | -                                                            |


### 6.2 Mixture-of-Experts (MOE) LoRA

| Year | Title | Venue | Paper | Code |
|------|-------|-------|-------|------|
| 2023 | **Loramoe: Revolutionizing mixture of experts for maintaining world knowledge in language model alignment** |    arXiv   | [Link](https://simg.baai.ac.cn/paperfile/96f0cfd7-79c7-4110-88e5-4ea80a7fbc8d.pdf) |  - |
| 2024 | MoLE: Mixture of LoRA Experts                                | ICLR     | [Link](https://arxiv.org/abs/2404.13628)     | [Link](https://github.com/yushuiwx/MoLE)           |
| 2024 | Uni-MoE: Scaling Unified Multimodal LLMs with Mixture of Experts | -     | [Link](https://arxiv.org/abs/2405.11273) | -                                                            |
| 2024 | AdaMoLE: Fine-Tuning Large Language Models with Adaptive Mixture of Low-Rank Adaptation Experts | -     | [Link](https://arxiv.org/abs/2405.00361) | -                                                            |
| 2024 | Mixture of Experts Using Tensor Products                     | -     | [Link](https://arxiv.org/abs/2405.16671) | -                                                            |


### 6.3 LoRA Merge

| Year | Title | Venue | Paper | Code |
|------|-------|-------|-------|------|
|      |       |       |       |      |

## 7. LoRA applications

### 7.1 Visual Understanding

| Year | Title | Venue | Paper | Code |
|------|-------|-------|-------|------|
| 2024 | Convolution Meets LoRA: Parameter Efficient Finetuning for Segment Anything Model | ICLR             | [Link](https://arxiv.org/abs/2401.17868) | -                                                            |
| 2024 | Low-Rank Rescaled Vision Transformer Fine-Tuning: A Residual Design Approach | -                | [Link](https://arxiv.org/abs/2403.19067) | [Link](https://github.com/zstarN70/RLRR.git)                 |
| 2024 | ExPLoRA: Parameter-Efficient Extended Pre-Training to Adapt Vision Transformers under Domain Shifts | -                | [Link](https://arxiv.org/abs/2406.10973) | -                                                            |
|    2023  |  **MeLo: Low-rank Adaptation is Better than Finetuning for Medical Image**     |       |       |  [Link](https://github.com/JamesQFreeman/LoRA-ViT)     |



### 7.2 Visual Generation
| Year | Title | Venue | Paper | Code |
|------|-------|-------|-------|------|
| 2024 | ExPLoRA: Parameter-Efficient Extended Pre-Training to Adapt Vision Transformers under Domain Shifts | -     | [Link](https://arxiv.org/abs/2406.10973) | -                                                            |
| 2024 | MoE-FFD: Mixture of Experts for Generalized and Parameter-Efficient Face Forgery Detection | -                | [Link](https://arxiv.org/abs/2404.08452) | [Link](https://github.com/LoveSiameseCat/MoE-FFD)            |
| 2024 | Mixture of Low-rank Experts for Transferable AI-Generated Image Detection | -                | [Link](https://arxiv.org/abs/2404.04883) | [Link](https://github.com/zhliuworks/CLIPMoLE)               |
| 2024 | LoRA-Composer: Leveraging Low-Rank Adaptation for Multi-Concept Customization in Training-Free Diffusion Models | -                | [Link](https://arxiv.org/abs/2403.11627) | [Link](https://github.com/Young98CN/LoRA_Composer)           |
| 2024 | Low-Rank Few-Shot Adaptation of Vision-Language Models       | -     | [Link](https://arxiv.org/abs/2405.18541) | -                                                            |
| 2024 | FouRA: Fourier Low Rank Adaptation                           | -     | [Link](https://arxiv.org/abs/2406.08798) | -                                                            |
| 2023 | Intrinsic LoRA: A Generalist Approach for Discovering Knowledge in Generative Models | -                | [Link](https://arxiv.org/abs/2311.17137) | [Link](https://intrinsic-lora.github.io/)                    |
| 2023 | Orthogonal Adaptation for Modular Customization of Diffusion Models | Preprint | [Link](https://arxiv.org/abs/2312.02432)     |                                                    |
| 2023 | ZipLoRA: Any Subject in Any Style by Effectively Merging LoRAs | Preprint | [Link](https://arxiv.org/abs/2311.13600)     | [Link](https://github.com/mkshing/ziplora-pytorch) |
| 2023 | Cones: Concept Neurons in Diffusion Models for Customized Generation | ICML     | [Link](https://arxiv.org/abs/2303.05125)     |                                                    
| 2023 | Multi-Concept Customization of Text-to-Image Diffusion       | CVPR     | [Link](https://arxiv.org/abs/2212.04488)     |                                                    
| 2023 | Cones 2: Customizable Image Synthesis with Multiple Subjects | Preprint | [Link](https://arxiv.org/abs/2305.19327)     | [Link](https://github.com/ali-vilab/Cones-V2)      |
| 2024 | Block-wise LoRA: Revisiting Fine-grained LoRA for Effective Personalization and Stylization in Text-to-Image Generation | AAAI     | [Link](https://arxiv.org/abs/2403.07500)     |                                                    |
| 2023 | Mix-of-Show: Decentralized Low-Rank Adaptation for Multi-Concept Customization of Diffusion Models | NeurIPS  | [Link](https://arxiv.org/abs/2305.18292)     | [Link](https://github.com/TencentARC/Mix-of-Show)  |
| 2024 | SELMA: Learning and Merging Skill-Specific Text-to-Image Experts with Auto-Generated Data | Preprint | [Link](https://arxiv.org/abs/2403.06952)     | [Link](https://github.com/jialuli-luka/SELMA)      |
| 2024 | MACE: Mass Concept Erasure in Diffusion Models               | CVPR     |                                              | [Link](https://github.com/Shilin-LU/MACE)          |
| 2024 | DiffuseKronA: A Parameter Efficient Fine-tuning Method for Personalized Diffusion Model | Preprint |                                              | [Link](https://diffusekrona.github.io/)            |
| 2024 | **Multi-LoRA Composition for Image Generation** |    arXiv   | [Link](https://arxiv.org/pdf/2402.16843) |  [Link](https://github.com/maszhongming/Multi-LoRA-Composition) |
| 2023 | **Motion Style Transfer: Modular Low-Rank Adaptation for Deep Motion Forecasting** |    PMLR   | [Link](https://proceedings.mlr.press/v205/kothari23a/kothari23a.pdf) |  [Link](https://github.com/vita-epfl/motion-style-transfer) |


### 7.3 Language Understanding

| Year | Title | Venue | Paper | Code |
|------|-------|-------|-------|------|
| 2023 | **Exploring the impact of low-rank adaptation on the performance, efficiency, and regularization of RLHF** |    arXiv   | [Link](https://arxiv.org/abs/2309.09055) |  [Link](https://github.com/SimengSun/alpaca_farm_lora) |

### 7.4 Multimodal learning

| Year | Title | Venue | Paper | Code |
|------|-------|-------|-------|------|
| 2024 | LaMDA: Large Model Fine-Tuning via Spectrally Decomposed Low-Dimensional Adaptation | -                | [Link](https://arxiv.org/abs/2406.12832) | -                                                            |
| 2024 | MoVA: Adapting Mixture of Vision Experts to Multimodal Context | -                | [Link](https://arxiv.org/abs/2404.13046) | [Link](https://github.com/TempleX98/MoVA)                    |
| 2024 | AdvLoRA: Adversarial Low-Rank Adaptation of Vision-Language Models | -     | [Link](https://arxiv.org/abs/2404.13425)                                    |                                                              |


### 7.5 Federated Learning

| Year | Title | Venue | Paper | Code |
|------|-------|-------|-------|------|
| 2024 | Improving LoRA in Privacy-preserving Federated Learning      | ICLR             | [Link](https://arxiv.org/abs/2403.12313) | -                                                            |
| 2024 | FeDeRA:Efficient Fine-tuning of Language Models in Federated Learning Leveraging Weight Decomposition | -     | [Link](https://arxiv.org/abs/2404.18848) | -                                                            |
| 2024 | FLoRA: Enhancing Vision-Language Models with Parameter-Efficient Federated Learning | -     | [Link](https://arxiv.org/abs/2404.15182) | -                                                            |
| 2024 | FL-TAC: Enhanced Fine-Tuning in Federated Learning via Low-Rank, Task-Specific Adapter Clustering | -     | [Link](https://arxiv.org/abs/2404.15384) | -                                                            |
| 2024 | DP-DyLoRA: Fine-Tuning Transformer-Based Models On-Device under Differentially Private Federated Learning using Dynamic Low-Rank Adaptation | -     | [Link](https://arxiv.org/abs/2405.06368) | -                                                            |


### 7.6 Other

| Year | Title | Venue | Paper | Code |
|------|-------|-------|-------|------|
| 2023 | **Low-Rank Adaptation of Large Language Model Rescoring for Parameter-Efficient Speech Recognition** |    ASRU   | [Link](https://ieeexplore.ieee.org/abstract/document/10389632) |  - |
| 2024 | Low-Rank Adaptation of Time Series Foundational Models for Out-of-Domain Modality Forecasting | -     | [Link](https://arxiv.org/abs/2405.10216) | -                                                            |
| 2023 | Continual Learning with Low Rank Adaptation                  | NeurIPS Workshop | [Link](https://arxiv.org/abs/2311.17601) | -                                                            |



## Contributing

We welcome contributions to this survey. Please feel free to submit a pull request to add new papers or update existing information.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
