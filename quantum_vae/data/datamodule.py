import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

def generate_rotated_3d_ring(num_points: int, noise_level: float) -> np.ndarray:
    theta = np.random.uniform(0, 2*np.pi, num_points)
    x = np.cos(theta)
    y = np.sin(theta)
    circle_2d = np.stack([x, y], axis=1)
    angle = np.radians(45)
    rot_matrix = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ])
    circle_3d = np.hstack([circle_2d, np.zeros((num_points, 1))]) @ rot_matrix.T
    noise = np.random.normal(scale=noise_level, size=circle_3d.shape)
    return circle_3d + noise

class RotatedRingDataset(TensorDataset):
    def __init__(self, num_points: int, noise_level: float, seed: int = None):
        if seed is not None:
            np.random.seed(seed)
        data = generate_rotated_3d_ring(num_points, noise_level)
        super().__init__(torch.tensor(data, dtype=torch.float32))

def create_dataloaders(num_points: int, noise_level: float, 
                      batch_size: int = 64, val_ratio: float = 0.2,
                      seed: int = None):
    full_dataset = RotatedRingDataset(num_points, noise_level, seed)
    val_size = int(val_ratio * num_points)
    train_size = num_points - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed) if seed else None
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, 
        shuffle=True, generator=torch.Generator().manual_seed(seed) if seed else None
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader