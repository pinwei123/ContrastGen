# ContrastGen

![Framework](https://img.shields.io/badge/Framework-PyTorch-red.svg)
![Task](https://img.shields.io/badge/Task-3D_Medical_Image_Synthesis-blue.svg)

**ContrastGen** is a **3D Conditional Diffusion Model** designed to synthesize Contrast-Enhanced CT (CECT) volumes from Non-Contrast CT (NCCT) inputs. The project aims to reduce the need for contrast agents in medical imaging by learning the non-linear mapping between NCCT and CECT scans.

## 🚀 Project Introduction

This project implements a **Conditional Gaussian Diffusion Model** tailored for 3D volumetric data. By conditioning the diffusion process on an input NCCT scan, the model generates a corresponding high-fidelity CECT scan.

The codebase is built on **PyTorch** and optimized for **NIfTI (.nii/.nii.gz)** medical image formats. It includes a complete pipeline for data loading, preprocessing, conditional training, and multi-GPU distribution.

## 📂 Data Structure

The training script expects paired data in separate folders. Files must be named such that sorting them results in correct input-target pairs.

```text
/dataset/
├── data/
│   ├── cacs/          # Input: Non-Contrast CT (NCCT)
│   ├── cas/           # Target: Contrast-Enhanced CT (CECT)
│   ├── cacs_test/     # Validation Input
│   └── cas_test/      # Validation Target
└── ...
```
## 🛠️ Getting Started
### Prerequisites

* Python **3.x**
* **PyTorch** and **torchvision**
* Necessary medical imaging libraries (`nibabel`, `torchio`)
* Scientific computing libraries (`numpy`, `scikit-learn`)

### Installation

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/pinwei123/ContrastGen.git](https://github.com/pinwei123/ContrastGen.git)
    ```

2.  **Install Dependencies:**
    ```bash
    cd ContrastGen
    pip install -r requirements.txt
    ```
## ⚙️ Training Guide

The training script `train.py` is configured to run the diffusion model trainer using a comprehensive set of command-line arguments.

### Execution

To train the conditional diffusion model, you **must** include the `--with_condition` flag and ensure the input/target paths are correctly set.

```bash
# Example command for conditional training
python train.py \
    --inputfolder /path/to/cacs \
    --targetfolder /path/to/cas \
    --val_input /path/to/cacs_test \
    --val_target /path/to/cas_test \
    --input_size 128 \
    --depth_size 128 \
    --batchsize 2 \
    --epochs 50000 \
    --with_condition
```
### 🤝 Customization (Dataset)

**The `dataset.py` is designed for easy modification** to accommodate different data preparation or augmentation strategies without touching the core diffusion model logic.
