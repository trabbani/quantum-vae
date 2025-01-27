import torch
import torch.optim as optim
import logging
import numpy as np
from ..models.quantum_vae import QVAE
import plotly.graph_objs as go

def train_qvae(config):
    logging.basicConfig(
        level=logging.INFO, 
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # DataLoaders
    from ..data.datamodule import create_dataloaders
    train_loader, val_loader = create_dataloaders(
        num_points=config['num_points'],
        noise_level=config['noise_level'],
        batch_size=config['batch_size'],
        seed=config['seed']
    )
    
    # Model
    model = QVAE(
        input_dim=config['input_dim'],
        quantum_dim=config['quantum_dim'],
        w=config['w'],
        beta=config['beta']
    )
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(1, config['num_epochs'] + 1):
        model.train()
        train_loss, train_recon, train_kl, train_var = 0., 0., 0., 0.
        
        for x_batch, in train_loader:
            x_batch = x_batch.to(device)
            optimizer.zero_grad()
            
            x_recon, q_z, z_complex = model(x_batch)
            losses = model.loss_function(x_batch, x_recon, q_z, z_complex)
            
            losses["loss"].backward()
            optimizer.step()
            
            train_loss += losses["loss"].item() * x_batch.size(0)
            train_recon += losses["recon_loss"].item() * x_batch.size(0)
            train_kl += losses["kl_loss"].item() * x_batch.size(0)
            train_var += losses["var_loss"].item() * x_batch.size(0)
        
        # Validation
        model.eval()
        val_loss, val_recon, val_kl, val_var = 0., 0., 0., 0.
        with torch.no_grad():
            for x_val, in val_loader:
                x_val = x_val.to(device)
                x_recon_val, q_z_val, z_complex_val = model(x_val)
                val_losses = model.loss_function(x_val, x_recon_val, q_z_val, z_complex_val)
                
                val_loss += val_losses["loss"].item() * x_val.size(0)
                val_recon += val_losses["recon_loss"].item() * x_val.size(0)
                val_kl += val_losses["kl_loss"].item() * x_val.size(0)
                val_var += val_losses["var_loss"].item() * x_val.size(0)
        
        # Calculate epoch metrics
        train_samples = len(train_loader.dataset)
        train_loss /= train_samples
        train_recon /= train_samples
        train_kl /= train_samples
        train_var /= train_samples
        
        val_samples = len(val_loader.dataset)
        val_loss /= val_samples
        val_recon /= val_samples
        val_kl /= val_samples
        val_var /= val_samples
        
        # Logging
        logging.info(
            f"Epoch {epoch:03d}/{config['num_epochs']} | "
            f"Train Loss: {train_loss:.4f} (Recon: {train_recon:.4f}, KL: {train_kl:.4f}, Var: {train_var:.4f}) | "
            f"Val Loss: {val_loss:.4f} (Recon: {val_recon:.4f}, KL: {val_kl:.4f}, Var: {val_var:.4f})"
        )
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config['save_path'])
    
    # Final logging
    logging.info(f"Training complete. Best validation loss: {best_val_loss:.4f}")
    
    # Visualization
    model.load_state_dict(torch.load(config['save_path']))
    model.eval()
    
    # Generate sample visualization
    with torch.no_grad():
        # Collect all validation data
        all_original = []
        all_reconstructed = []
        
        for x_val, in val_loader:
            x_val = x_val.to(device)
            x_recon_val, _, _ = model(x_val)
            
            all_original.append(x_val.cpu().numpy())
            all_reconstructed.append(x_recon_val.cpu().numpy())
        
        # Concatenate all batches
        original_full = np.concatenate(all_original, axis=0)
        reconstructed_full = np.concatenate(all_reconstructed, axis=0)
        
        # Create 3D visualization
        plotter = PlotlyPointCloudPlotter3D()
        plotter.add_point_cloud(original_full, "Original", color="red", size=1, opacity=0.5)
        plotter.add_point_cloud(reconstructed_full, "Reconstructed", color="blue", size=2, opacity=0.8)
        plotter.configure_layout(
            title=f"Full Validation Set: {len(original_full)} Points",
        )
        plotter.show()

class PlotlyPointCloudPlotter3D:
    def __init__(self):
        self.fig = go.Figure()

    def add_point_cloud(self, points: np.ndarray, name: str, color: str, size: int = 4, opacity: float = 0.7):
        radius = np.sqrt(points[:, 0]**2 + points[:, 1]**2 + points[:, 2]**2)
        hover_text = [f"Radius: {r:.3f}" for r in radius]

        scatter3d = go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            name=name,
            marker=dict(color=color, size=size, opacity=opacity),
            hovertext=hover_text,
            hoverinfo="text"
        )
        self.fig.add_trace(scatter3d)

    def configure_layout(self, title: str = "3D Point Cloud Plot"):
        self.fig.update_layout(
            title=title,
            scene=dict(
                xaxis=dict(title="X", range=[-1.5, 1.5]),
                yaxis=dict(title="Y", range=[-1.5, 1.5]),
                zaxis=dict(title="Z", range=[-1.5, 1.5]),
                aspectmode="cube"
            ),
            width=800,
            height=800,
            showlegend=True
        )

    def show(self):
        self.fig.show()