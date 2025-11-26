<div align="center">

<h1>MolSight: Optical Chemical Structure Recognition with SMILES Pretraining, Multi-Granularity Learning and Reinforcement Learning</h1>

AAAI 2026 Accepted Paper

Wenrui Zhang<sup>1</sup> · Xinggang Wang<sup>1</sup> · Bin Feng<sup>1</sup> · Wenyu Liu<sup>1</sup>

<sup>1</sup>School of Electronic Information and Communications, Huazhong University of Science and Technology

<a href="https://arxiv.org/pdf/2511.17300"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b.svg" alt="Paper"></a> <a href="https://github.com/hustvl/MolSight"><img src="https://img.shields.io/badge/Code-GitHub-black.svg" alt="Code"></a> <a href="https://github.com/hustvl/MolSight/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-blue.svg" alt="License"></a>

</div>

## 📖 Introduction

**MolSight** is a comprehensive learning framework for Optical Chemical Structure Recognition (OCSR), designed to bridge the gap between computer vision and chemical informatics (AI4S). [cite\_start] [cite: 1]

Accurately translating molecular images into machine-readable formats (like SMILES) is critical for drug discovery and digital chemistry. MolSight addresses the limitations of previous methods—particularly in handling complex **stereoisomers**—through a novel three-stage training paradigm:

1.  **SMILES Pretraining:** Aligns visual representations with chemical strings.
2.  **Multi-Granularity Fine-Tuning:** Captures both global structure and local functional group details.
3.  **RL Post-Training:** Utilizes Reinforcement Learning to optimize for chemical semantic correctness rather than simple token matching.

### ✨ Key Features

  * **First RL-based OCSR:** MolSight is the first OCSR system to integrate **Reinforcement Learning**. We utilize Group Relative Policy Optimization (GRPO) to directly optimize chemical validity[c.
  * **Stereo-200k Dataset:** We introduce a new annotated dataset consisting of **200,000 challenging stereoisomeric molecules** specifically curated to address confusion in 3D chiral structures.

<img width="966" height="564" alt="image" src="https://github.com/user-attachments/assets/d6bbe0b9-d890-4f5b-99f7-f11601a4be65" />

  * **SOTA Performance:** Extensive experiments demonstrate that MolSight achieves state-of-the-art results in accuracy, similarity, and robustness, outperforming classical and learning-based baselines.

## 🔥 News

  * **[2025-11-26]** 🎉 MolSight has been accepted to **AAAI 2026**\!
  * **[2025-11-26]** 🚀 Code released.

## 🛠️ Getting Started

### Installation

Clone the repository and install the required dependencies:

```bash
# Clone the repository
git clone https://github.com/hustvl/MolSight
cd MolSight

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

*Note: The **Stereo-200k** dataset will be released shortly. Please check the [Updates](https://www.google.com/search?q=%23-news) section.*

## 🚀 Training

MolSight employs a multi-stage training pipeline. You can reproduce the training process using the provided scripts:

**1. Supervised Fine-Tuning (SFT)**
Train the backbone model using multi-granularity learning:

```bash
bash train.sh
```

**2. Coordinate Predictor Training**
Train the auxiliary coordinate predictor module:

```bash
bash train_loc_predictor.sh
```

**3. Reinforcement Learning (RL) Post-Training**
Optimize the model using GRPO for chemical semantic correctness:

```bash
bash post_train.sh
```

## 🧪 Evaluation

To evaluate the model on the benchmark datasets:

```bash
# Run evaluation script (ensure model weights are loaded)
bash eval.sh
```

## 📝 Citation

If you find MolSight or the Stereo-200k dataset useful for your research in AI4Science or Chemistry, please cite our paper:

```bibtex
@article{zhang2025molsight,
  title={MolSight: Optical Chemical Structure Recognition with SMILES Pretraining, Multi-Granularity Learning and Reinforcement Learning},
  author={Zhang, Wenrui and Wang, Xinggang and Feng, Bin and Liu, Wenyu},
  journal={arXiv preprint arXiv:2511.17300},
  year={2025}
}
```

## 📄 License

This project is licensed under the [Apache 2.0 License](https://www.google.com/search?q=LICENSE).

