# Quantum Variational Autoencoder (QVAE)

A **Quantum-Inspired Variational Autoencoder** for reconstructing low-dimensional manifolds in higher-dimensional spaces. This project showcases the method using a **1D ring** with added noise embedded in **3D**.

---

## Description

The **Quantum VAE** encodes data into a quantum-inspired latent space using complex representations and reconstructs it via Hermitian transformations. The model learns a smooth latent representation while accurately reconstructing the original data distribution.

---

## Visualization

<p align="center">
  <img src="examples/ring_reconstruction.gif" alt="Ring Reconstruction GIF" width="400"/>
</p>

_**Fig:** Visualization of noisy ring data (red) and reconstructed points (blue)._

---

## Architecture Overview

### Highlights
- **Encoder**: Extracts features and outputs latent parameters:
  - $\mu$: Mean direction (unit vector).
  - $\kappa$: Concentration (spread of distribution).
- **Latent Space**: Samples from the **Von Mises-Fisher (vMF) distribution** for smooth, quantum-inspired representations.
- **Decoder**: Uses **Hermitian transformations** to reconstruct data.

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

## Key Benefits

- **Quantum-Inspired Representation**: vMF distribution ensures latent points follow quantum constraints (unit norm), enabling meaningful geometrical interpretations.
- **Efficiency**: Single-pass, end-to-end training using modern deep learning tools like PyTorch.
- **Denoising & Generation**: Handles noisy data and generates new samples from the learned latent space.
- **Scalability**: Overcomes the limitations of traditional iterative quantum optimization.

---

## Dataset

The model trains on a **1D ring** (circle) sampled uniformly in angle $\theta \in [0, 2\pi]$, embedded in 3D using a rotation matrix. Optional **Gaussian noise** adds realism to the dataset.

---

## Repository Structure

```
quantum-vae/
├── quantum_vae/
│   ├── data/            # Data loading utilities
│   ├── models/          # Encoder, decoder, Hermitian layers
│   ├── training/        # Training logic
│   ├── utils/           # Logging and visualization tools
├── examples/            # GIFs and example outputs
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/trabbani/quantum-vae.git
   cd quantum-vae
   ```
2. Create a virtual environment (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Install PyTorch (follow the [PyTorch guide](https://pytorch.org/get-started/locally/)):
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

---

## Google Colab Support

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/trabbani/quantum-vae/blob/main/notebooks/qvae_demo_1.ipynb)

- Use **GPU runtime** for faster training: `Runtime` → `Change runtime type` → Select `GPU`.
- If issues arise, restart the runtime (`Runtime` → `Restart runtime`) and rerun the cells.

---

## Results

The model reconstructs noisy data (red) into clean representations (blue), demonstrating its ability to denoise and reconstruct underlying structures effectively.