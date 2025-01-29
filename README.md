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

---

## Appendix

### **Understanding the Problem: Why Quantum-Inspired Methods?**

### **The Challenge of High-Dimensional Data**
Real-world data—like medical images, financial records, or sensor readings—often appears high-dimensional but actually lies on simpler, **low-dimensional manifolds**. For example:
- A **3D scan** of a rotating object can be described by just **3 parameters** (angle, lighting, distance), despite having thousands of pixels.
- **Patient health records** with 30+ features might depend on just **2–3 latent factors** (e.g., metabolic health, genetic risk).

Identifying this **"hidden" intrinsic dimension** is crucial for:
- **Compression** (reducing data redundancy).  
- **Denoising** (removing irrelevant noise).  
- **Better Learning** (extracting meaningful features).  

However, traditional methods (e.g., **PCA, t-SNE, k-NN-based estimators**) struggle with **noise**, which introduces **"shadow dimensions"**, artificially inflating estimates of data complexity.

---

### **A Quantum-Inspired Solution**
Recent advances in **quantum geometry** and **quantum cognition** provide a **noise-resistant** approach by representing data as **quantum states**. This allows us to:
1. **Filter out artificial noise dimensions** using spectral properties of a quantum metric.
2. **Capture global manifold structure** while maintaining robustness to perturbations.
3. **Learn smooth latent representations** that preserve geometric consistency.

Instead of modeling data as a simple vector in $\mathbb{R}^D$, we embed it into a **quantum state** $|\psi(x)\rangle$, which encodes both:
- **Local properties** (feature values).
- **Global geometric relations** (structure of the data manifold).

---

### **Key Equations and Their Role**
Below are the **core equations** that define this quantum-inspired learning approach. These are **simplified for intuition**—for full details, see **[the research paper](https://arxiv.org/abs/2409.12805)**.

#### **1. Error Hamiltonian** (Encodes the "cost" of deviations from the ideal quantum representation)

$$H(x) = \frac{1}{2} \sum_{k=1}^D (A_k - a_k \cdot I_N)^2$$

where:
- $A_k$  are **Hermitian operators** representing learned feature transformations.
- $a_k$ are **observed feature values** in the original dataset.
- **Goal**: Minimizing $H(x)$ aligns **quantum states** with the underlying **data manifold**.

---

#### **2. Energy Decomposition** (Separates "bias" and "variance" effects in manifold estimation)

$$
E_0(x) = \frac{1}{2} \|A(\psi_0(x)) - x\|^2 + \frac{1}{2} \sigma^2(\psi_0(x))
$$

where:
- **Bias**: Measures the deviation of quantum representation from the actual data point.
- **Variance**: Represents **quantum fluctuations** (uncertainty in the representation).
- **Impact**: Balancing bias and variance allows the model to **filter noise** while preserving essential **data structure**.

---

#### **3. Quantum Metric** (Extracts Intrinsic Dimension from Spectral Gaps)

$$
g_{\mu\nu}(x) = 2 \sum_{n=1}^{N-1} \text{Re}\left( 
\frac{\langle \psi_0(x)|A_\mu|\psi_n(x) \rangle \langle \psi_n(x)|A_\nu|\psi_0(x) \rangle}{E_n(x) - E_0(x)} 
\right)
$$


where:
- $g(x)$ is a **local Riemannian metric** induced by quantum geometry.
- **Spectral Gaps** in the eigenvalues of $g(x)$ reveal **true intrinsic dimension** $d$, filtering out noise-induced dimensions.

**Why is this better than PCA or k-NN-based methods?**  
- **Quantum metrics are robust** to noise, while classical local estimators tend to overestimate $d$.
- **No assumption of linearity**—this method adapts to **highly curved** manifolds.

---

### **Why This Matters**

- **Robust Intrinsic Dimension Estimation**: Suppresses **"shadow dimensions"** from noise.
- **Efficient Manifold Learning**: Uses **Von Mises-Fisher priors** to enforce **structured latent spaces**.
- **Scalable Training**: Implemented in **PyTorch**, compatible with **real-world datasets**.

For a **full theoretical background** and **benchmark comparisons**, refer to:

📄 **[Robust Estimation of Intrinsic Dimension with Quantum Cognition Machine Learning](https://arxiv.org/abs/2409.12805)**


---
