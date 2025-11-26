# ContrastGen: Conditional 3D Contrast-Enhanced CT Synthesis

![Framework](https://img.shields.io/badge/Framework-PyTorch-red.svg)
![Task](https://img.shields.io/badge/Task-3D_Medical_Image_Synthesis-blue.svg)

**ContrastGen** is a state-of-the-art **3D Conditional Diffusion Model** designed to synthesize Contrast-Enhanced CT (CECT) volumes from Non-Contrast CT (NCCT) inputs. The project aims to reduce the need for contrast agents in medical imaging by learning the non-linear mapping between NCCT and CECT scans.

## 🚀 Project Introduction

This project implements a **Conditional Gaussian Diffusion Model** tailored for 3D volumetric data. By conditioning the diffusion process on an input NCCT scan, the model generates a corresponding high-fidelity CECT scan.

The codebase is built on **PyTorch** and optimized for **NIfTI (.nii/.nii.gz)** medical image formats. It includes a complete pipeline for data loading, preprocessing, conditional training, and multi-GPU distribution.

## ✨ Key Features

| Feature | Description |
| :--- | :--- |
| **3D Conditional Diffusion** | Uses `GaussianDiffusion` conditioned on input volumes to guide generation. |
| **3D U-Net Architecture** | Employes a specialized 3D U-Net backbone for volumetric noise prediction. | 
| **NIfTI Data Pipeline** | Custom `Dataset` handling `.nii` files, resizing, and normalization. | 
| **3D Augmentation** | Supports on-the-fly 3D augmentation (Flip, Gamma, Noise) using `torchio`. | 
| **Distributed Training** | Built-in support for `nn.DataParallel` for multi-GPU setups. | 

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
    cd ContrastGen
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
## ⚙️ Training Guide

The training script `train.py` is configured to run the diffusion model trainer using a comprehensive set of command-line arguments.

### Key Configuration Arguments

| Argument | Type | Default Value | Description |
| :--- | :--- | :--- | :--- |
| `--inputfolder` | `str` | `/dataset/data/cacs` | Path to the training NCCT images. |
| `--targetfolder` | `str` | `/dataset/data/cas` | Path to the training CECT images. |
| `--input_size` | `int` | `128` | Image height/width after resizing (X and Y dimensions). |
| `--depth_size` | `int` | `128` | Image depth after resizing (Z dimension). |
| `--timesteps` | `int` | `250` | Number of diffusion timesteps. |
| `--epochs` | `int` | `50000` | Total number of training steps (iterations). |
| `--batchsize` | `int` | `2` | Batch size for training. |
| `--with_condition` | `action`| `False` | **Must be set** to enable conditional generation (using NCCT as input). |
| `-r` / `--resume_weight` | `str` | `""` | Path to a checkpoint for resuming training. |

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

> **The `dataset.py` is designed for easy modification** to accommodate different data preparation or augmentation strategies without touching the core diffusion model logic.

* **`NiftiPairImageGenerator`**: This class handles the paired Non-Contrast CT (NCCT) and Contrast-Enhanced CT (CECT) data loading.
* **Custom Augmentation**: Users can easily modify or extend the `self.augment_transform` within the `NiftiPairImageGenerator` class to incorporate more sophisticated **3D augmentation techniques** (e.g., `tio.RandomAffine`, `tio.RandomElasticDeformation`).
