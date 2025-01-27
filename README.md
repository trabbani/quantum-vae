

# Quantum Variational Autoencoder (QVAE)

A **Quantum-Inspired Variational Autoencoder** for reconstructing low-dimensional manifolds embedded in higher-dimensional spaces. This repository demonstrates the approach using a **1D ring** (circle) with added noise, embedded in **3D**.

---

## 1. Description

This repository provides a **Quantum VAE** implementation that encodes data into a complex latent space (interpreted as quantum states) and decodes them via Hermitian transformations. The model aims to learn a smooth latent representation and accurately reconstruct the original data distribution.

---

## 2. Animated/Static Preview

<p align="center">
  <img src="examples/ring_reconstruction.gif" alt="Ring Reconstruction GIF" width="400"/>
</p>

_**Fig:** Visualization of noisy ring data (red) and reconstructed points (blue)._

---

## 3. Table of Contents

1. [Description](#1-description)
2. [Animated/Static Preview](#2-animatedstatic-preview)
3. [Table of Contents](#3-table-of-contents)
4. [Project Overview](#4-project-overview)
5. [Repository Structure](#5-repository-structure)
6. [Installation](#6-installation)
7. [Google Colab Support](#7-google-colab-support)
8. [Results and Visualization](#8-results-and-visualization)

---

## 4. Project Overview

### Original Iterative Method

Previous quantum-inspired approaches often **iterated** between:

1. Computing local ground states $|\psi_0(x)\rangle$ of an error Hamiltonian $H(x)$.
2. Updating the Hermitian matrices $A_1, \ldots, A_D$ (which define $$H(x)$$) based on the reconstruction error.

While this **iterative** process was effective at capturing underlying geometry, it could be **computationally heavy** and challenging to scale.

---

### What We Propose — Quantum VAE with Von Mises-Fisher Latent Space

In this repository, we propose a **Quantum Variational Autoencoder (QVAE)** that embeds quantum-inspired concepts within a VAE framework. Unlike iterative methods, the QVAE learns **end-to-end** through backpropagation, making it computationally efficient and scalable.

---

### Architecture Overview

#### **Description**:

**Input Vector ($x$)**:  
The input data (e.g., 3D points) is fed into the encoder.

**Encoder**:  
Extracts features through fully connected layers with ReLU activations.  
Outputs two key latent space parameters:

- $\mu$: A normalized vector representing the mean direction in the latent space.
- $\kappa$: A scalar value representing the concentration (spread) of the vMF distribution.

**Latent Sampling**:  
Samples from the **Von Mises-Fisher (vMF) distribution**, parameterized by $\mu$ and $\kappa$.  
Converts the sampled vectors into a **complex latent representation**, respecting quantum-inspired constraints.

**Decoder**:  
Uses a Hermitian layer composed of **learnable Hermitian matrices**.  
Transforms the latent vector back into the original data space.

**Reconstructed Data ($x'$)**:  
The output of the decoder is the model's reconstruction of the input data.

---

### Architecture Flow

```plaintext
    Input Vector (x) ─────────────► Encoder
                                     │
                                     ├──► μ (Mean Direction - Complex Latent Vector)
                                     │
                                     └──► κ (Concentration - Spread)
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      ▼                                               ▼
       Normalize μ (Unit Norm)                           Softplus Activation
                      │                                                │
                      └────────────────────┬───────────────────────────┘
                                           ▼        
                                 Sample from vMF Distribution
                                           │
                                           ▼
                           Latent Vector (Complex Representation)
                                           │
                                           ▼
                                  Decoder (Hermitian Layer)
                                           │
                                           ▼
                               Reconstructed Data (x')
```

---

### Dataset: Rotated 3D Ring with Optional Noise

The dataset for training consists of a **1D ring** (circle), uniformly sampled in angle $\theta \in [0, 2\pi]$. This circle is embedded in 3D space by applying a fixed rotation matrix. Optionally, **Gaussian noise** can be added to simulate real-world imperfections.


---

### Why Quantum VAE? Key Benefits

- **Probabilistic and Geometric Representation**: The vMF distribution ensures latent space points naturally adhere to quantum constraints (unit norm). This makes the latent space interpretable and geometrically meaningful.
- **Single-pass Optimization**: Unlike iterative quantum-state optimization, the QVAE trains in a single end-to-end pipeline, leveraging deep learning tools like PyTorch.
- **Generative and Denoising Power**: The VAE framework allows the model to reconstruct data, denoise noisy inputs, and even generate new samples by sampling from the learned vMF latent space.
- **Hermitian Decoding**: The quantum-inspired Hermitian decoder learns to represent the structural relationships of the data, making it robust and flexible.

---

## 5. Repository Structure


```

quantum-vae/
├── quantum_vae/
│   ├── data/
│   │   ├── __init__.py
│   │   └── datamodule.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── quantum_vae.py
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── logger.py
│   └── __init__.py
├── examples/
│   └── ring_reconstruction.gif
├── README.md
├── requirements.txt
└── .gitignore


```

- **`quantum_vae/data`**: Data loading (e.g., ring generation, DataLoader wrappers).  
- **`quantum_vae/models`**: QVAE model definitions (encoder, decoder, Hermitian layers).  
- **`quantum_vae/training`**: Training logic and main training loop.  
- **`quantum_vae/utils`**: Shared utilities (logging, plotting).  
- **`examples/`**: Contains GIFs or images demonstrating model outcomes.  


## 6. Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/trabbani/quantum-vae.git
   cd quantum-vae
   ```

2. **Set up a virtual environment** (recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/macOS
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Install PyTorch**:

   Visit the [PyTorch installation page](https://pytorch.org/get-started/locally/) to find the appropriate command for your environment.

   Example:
   ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

---

## 7. Google Colab Support

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/trabbani/quantum-vae/blob/main/notebooks/qvae_demo_1.ipynb)

1. Click the **Open in Colab** badge above.
2. Run the first cell to install dependencies.
3. Execute subsequent cells to train and visualize the QVAE.


### Notes for Colab Users
- Ensure you’re using a **GPU runtime** for faster training:
  - Go to `Runtime` → `Change runtime type` → Select `GPU` under Hardware Accelerator.
- If you encounter issues, restart the runtime (`Runtime` → `Restart runtime`) and rerun the cells.



---

## 8. Results and Visualization

After training, the model outputs reconstruction/kl/variance losses and visualizes the original vs. reconstructed data:

- **Red**: Original noisy data.
- **Blue**: Reconstructed outputs.


