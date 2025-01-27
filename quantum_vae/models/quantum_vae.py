import torch
import torch.nn as nn
import torch.nn.functional as F
from hyperspherical_vae.distributions import VonMisesFisher, HypersphericalUniform


class HermitianLayer(nn.Module):
    def __init__(self, quantum_dim: int, input_dim: int) -> None:
        super().__init__()
        self.quantum_dim = quantum_dim
        self.input_dim = input_dim
        self.hermitian_base = nn.Parameter(
            torch.randn(input_dim, quantum_dim, quantum_dim)
        )

        # Fixed complex structure matrix
        constant_matrix = torch.zeros(quantum_dim, quantum_dim, dtype=torch.complex64)
        constant_matrix.fill_diagonal_(0.5)
        constant_matrix += torch.triu(torch.ones(quantum_dim, quantum_dim), diagonal=1)
        constant_matrix += torch.tril(
            1j * torch.ones(quantum_dim, quantum_dim), diagonal=-1
        )
        self.register_buffer("constant_matrix", constant_matrix)

    def forward(self) -> torch.Tensor:
        structured = self.constant_matrix * self.hermitian_base
        return structured + structured.conj().mT


class QuantumDecoder(nn.Module):
    def __init__(self, quantum_dim: int, input_dim: int):
        super().__init__()
        self.hermitian_layer = HermitianLayer(quantum_dim, input_dim)
        self.quantum_dim = quantum_dim
        self.input_dim = input_dim

    def forward(self, z_complex: torch.Tensor) -> torch.Tensor:
        H = self.hermitian_layer()
        z = z_complex.unsqueeze(1)
        zH = z.conj()
        return torch.einsum("bki,dij,bkj->bd", zH, H, z).real


class QuantumEncoder(nn.Module):
    def __init__(self, input_dim: int, quantum_dim: int):
        super().__init__()
        self.quantum_dim = quantum_dim
        self.input_dim = input_dim
        hidden_dim = 8 * quantum_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * quantum_dim + 1),
        )

    def forward(self, x: torch.Tensor):
        out = self.net(x)
        real_mu, imag_mu, raw_kappa = torch.split(
            out, [self.quantum_dim] * 2 + [1], dim=-1
        )
        mu = torch.cat([real_mu, imag_mu], dim=-1)
        mu = mu / (mu.norm(dim=-1, keepdim=True) + 1e-8)
        kappa = F.softplus(raw_kappa) + 1e-6
        return mu, kappa


class QVAE(nn.Module):
    def __init__(self, input_dim: int, quantum_dim: int, w: float, beta: float):
        super().__init__()
        self.input_dim = input_dim
        self.quantum_dim = quantum_dim
        self.encoder = QuantumEncoder(input_dim, quantum_dim)
        self.decoder = QuantumDecoder(quantum_dim, input_dim)

        self.register_buffer("_dummy", torch.empty(0))
        self.prior = HypersphericalUniform(2 * quantum_dim - 1, device=self._dummy.device)
        self.w = w
        self.beta = beta

    def forward(self, x: torch.Tensor):
        mu, kappa = self.encoder(x)
        q_z = VonMisesFisher(mu, kappa)
        z_sample = q_z.rsample()
        z_complex = torch.view_as_complex(z_sample.view(-1, self.quantum_dim, 2))
        x_recon = self.decoder(z_complex)
        return x_recon, q_z, z_complex

    def compute_variance_fluctuation(self, z_complex: torch.Tensor) -> torch.Tensor:
        H = self.decoder.hermitian_layer()
        A2 = H @ H
        z_conj = z_complex.conj()
        exp_A = torch.einsum("bi,dij,bj->bd", z_conj, H, z_complex)
        exp_A2 = torch.einsum("bi,dij,bj->bd", z_conj, A2, z_complex)
        return (exp_A2.real - exp_A.real**2).mean()

    def loss_function(self, x, x_recon, q_z, z_complex):
        recon_loss = F.mse_loss(x_recon, x)
        kl_loss = torch.distributions.kl_divergence(q_z, self.prior).mean()
        var_loss = self.compute_variance_fluctuation(z_complex)
        return {
            "loss": recon_loss + self.beta * kl_loss + self.w * var_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "var_loss": var_loss,
        }
    
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        # Force the distribution’s device to match the updated _dummy buffer
        self.prior.device = self._dummy.device
        return self