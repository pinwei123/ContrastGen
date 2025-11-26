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
    A `requirements.txt` file is required for this step, but assuming common dependencies:
    ```bash
    # It is highly recommended to use a virtual environment
    pip install torch torchvision numpy scikit-learn nibabel torchio
    # You might also need the custom diffusion model packages:
    # pip install -e .  (If package is installable)
    ```
