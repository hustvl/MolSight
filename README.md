<div align="center">


<h1>MolSight: Optical Chemical Structure Recognition with SMILES Pretraining, Multi-Granularity Learning and Reinforcement Learning</h1>

<b>Wenrui Zhang</b>, <b>Xinggang Wang</b>, <b>Bin Feng</b>, <b>Wenyu Liu</b>

School of Electronic Information and Communications, Huazhong University of Science and Technology

</div>

## Introduction

+ We present MolSight, a comprehensive learning framework for Optical Chemical Structure Recognition (OCSR) that enhances model performance across diverse molecular types, particularly stereoisomers, through a three-stage training approach consisting of pre-training, multi-granularity fine-tuning, and RL post-training.
+ MolSight represents the first OCSR system to incorporate reinforcement learning methods. By integrating the Group Relative Policy Optimization (GRPO) algorithm, the model optimization process overcomes the limitations of token-level accuracy and directly optimizes for chemical semantic correctness, effectively improving recognition accuracy for stereoisomeric molecules.
+ We construct a new annotated molecular image dataset, Stereo-200k, consisting entirely of challenging stereoisomeric molecules that are prone to confusion. This dataset supports MolSight's RL training process and will be made publicly available to the research community.
+ Extensive experiments demonstrate that MolSight achieves state-of-the-art performance in terms of accuracy, similarity, and robustness, outperforming both classical and learning-based methods across most scenarios, while showing broad potential for downstream applications.

## Updates

- [x] Release code
- [ ] Release Stereo-200k dataset
- [ ] Release model weights

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/hustvl/MolSight
cd MolSight

# Install dependencies
pip install -r requirements.txt
```

## Training

Start MolSight training with:

```bash
# SFT
bash train.sh
# train the additional coord predictor
bash train_loc_predictor.sh
# post training with RL
bash post_train.sh
```