# Fire Risk Assessment
Final Project for Stanford's CS 230: Deep Learning, Fall 2025
Elisabeth Holm, Evan Robert, Kimberly Cheung

This repository contains code and experiments for assessing fire risk using geospatial and machine learning techniques

Required: Python 3.10.0

---

## Installation

Clone the repository and set up a Python virtual environment:

```bash
git clone https://github.com/evanr1232/fire-risk-assessment.git
cd fire-risk-assessment

# Create and activate a virtual environment
python3.10 -m venv fire
source fire/bin/activate

# Install dependencies
pip install -r requirements.txt
```

# Running the code
## Activate your virtual environment (if not done already)
```
source fire/bin/activate
```
## Download and visualize the FireRisk data
```
python visualize_firerisk.py
```
To only download and not visualize, change the num_vis flag to 0
```
python visualize_firerisk.py --num_vis 0
```
Note: it will take a while (~30 mins) to download the 15GB of data
# ImageNet1k MAE + ViT Finetuned Model

## Download the MAE ImageNet1k pretrained base model weights
```
wget https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth -O models/mae_vit_encoder_imagenet1k_base.pth
```

## Go into the finetuning folder
```
cd finetuning
```

## Train the Baseline Model (Image Only)
```
python train_baseline_mae.py --encoder_path "../models/mae_vit_encoder_imagenet1k_base.pth"
```

## Train the Multimodal Model (Image + Metadata)
```
python train_multimodal_mae.py --encoder_path "../models/mae_vit_encoder_imagenet1k_base.pth"
```
