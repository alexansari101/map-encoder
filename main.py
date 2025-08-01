import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Tuple, List
from dataclasses import dataclass

# --- Configuration ---
@dataclass
class TrainConfig:
    """Configuration parameters for training."""
    # Model Hyperparameters
    in_channels: int = 2
    hidden_dim: int = 128
    mask_ratio: float = 0.45
    
    # Data & Training Hyperparameters
    num_samples: int = 256
    batch_size: int = 16
    learning_rate: float = 1e-4
    num_epochs: int = 50
    
    # DataLoader
    num_workers: int = 8

# A siplified version of the PointNetPolylineEncoder class
# https://github.com/sshaoshuai/MTR
class PolylineEncoder(nn.Module):
    """
    Encodes a batch of polylines into a fixed-size global feature vector for each polyline.
    It uses a PointNet-like architecture where each point is processed by an MLP,
    and a symmetric function (max-pooling) aggregates point features.

    A siplified version of the PointNetPolylineEncoder class
    https://github.com/sshaoshuai/MTR
    """
    def __init__(self, in_channels: int, hidden_dim: int):
        """
        Args:
            in_channels (int): The number of input channels for each point (e.g., 2 for x, y).
            hidden_dim (int): The dimensionality of the hidden and output features.
        """
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, polylines: torch.Tensor, polylines_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the PolylineEncoder.

        Args:
            polylines (torch.Tensor): The input polylines tensor.
                                      Shape: (Batch, NumPolylines, NumPoints, Channels).
            polylines_mask (torch.Tensor): A boolean mask indicating valid points.
                                           Shape: (Batch, NumPolylines, NumPoints), True for valid.

        Returns:
            torch.Tensor: The global feature vector for each polyline.
                          Shape: (Batch, NumPolylines, HiddenDim).
        """
        batch_size, num_polylines, num_points, _ = polylines.shape
        hidden_dim = self.point_mlp[-1].out_features

        # Process only the valid points for efficiency
        valid_features = self.point_mlp(polylines[polylines_mask])

        # Scatter the computed features back into a zero-padded tensor
        point_features = polylines.new_zeros(batch_size, num_polylines, num_points, hidden_dim)
        point_features[polylines_mask] = valid_features

        # Set ignored points to a large negative value for robust max-pooling
        # The mask needs to be inverted (~polylines_mask) to select the points to ignore.
        point_features[~polylines_mask] = -1e9

        # Max-pool across the points of each polyline to get a global feature.
        polyline_features, _ = torch.max(point_features, dim=2)
        # Handle cases where a polyline has no points after masking
        polyline_features[polyline_features == -1e9] = 0
        return polyline_features


class PointwisePredictionDecoder(nn.Module):
    """
    Decodes a combined feature vector (global context + local point info)
    to predict the coordinates of each point in a polyline.
    """
    def __init__(self, hidden_dim: int, point_channels: int):
        """
        Args:
            hidden_dim (int): The dimensionality of the global polyline feature.
            point_channels (int): The number of output channels for each point (e.g., 2 for x, y).
        """
        super().__init__()
        # The decoder's input is the concatenation of the global feature and the point's own data.
        decoder_in_dim = hidden_dim + point_channels
        self.decoder_mlp = nn.Sequential(
            nn.Linear(decoder_in_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, point_channels)
        )

    def forward(self, combined_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the PointwisePredictionDecoder.

        Args:
            combined_features (torch.Tensor): Concatenated global and local features.
                                              Shape: (B, N, P, HiddenDim + PointChannels).

        Returns:
            torch.Tensor: The predicted coordinates for each point.
                          Shape: (B, N, P, PointChannels).
        """
        return self.decoder_mlp(combined_features)


# Main model for the self-supervised training task
class SelfSupervisedModel(nn.Module):
    """
    Main model that orchestrates the self-supervised masked point prediction task.
    It masks a portion of the input polylines, encodes them, and then decodes
    to reconstruct the original, unmasked points.
    """
    def __init__(self, encoder: PolylineEncoder, decoder: PointwisePredictionDecoder, mask_ratio: float):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mask_ratio = mask_ratio
        self.loss_fn = nn.L1Loss()

    def forward(self, polylines: torch.Tensor, padding_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs one forward pass of self-supervised masked point prediction.
        """
        # 1. Create a random mask to hide a portion of the points for the self-supervised task
        # True for points we want to HIDE.
        task_mask = torch.rand(polylines.shape[:-1], device=polylines.device) < self.mask_ratio

        # 2. Combine the padding mask and the task mask. A point is visible to the encoder
        #    only if it's NOT padding AND it's NOT masked for the task.
        visible_points_mask = padding_mask & ~task_mask
        
        # 3. For the decoder's input, zero out points that are either padding or task-masked.
        masked_polylines_for_decoder = polylines.clone()
        # Create a combined mask for zeroing out points.
        combined_mask_for_zeroing = ~visible_points_mask
        masked_polylines_for_decoder[combined_mask_for_zeroing.unsqueeze(-1).expand_as(polylines)] = 0.0

        # 4. Encode the polylines using the visbility mask.
        global_feature = self.encoder(polylines, visible_points_mask)

        # 5. Prepare decoder input.
        _, _, num_points, _ = polylines.shape
        global_feature_expanded = global_feature.unsqueeze(2).expand(-1, -1, num_points, -1)
        decoder_input = torch.cat([global_feature_expanded, masked_polylines_for_decoder], dim=-1)

        # 6. Decode to predict all points.
        predicted_polylines = self.decoder(decoder_input)

        # 7. Calculate loss ONLY on points that were part of the task mask (and not padding).
        target_points_mask = padding_mask & task_mask
        target_points = polylines[target_points_mask.unsqueeze(-1).expand_as(polylines)]
        predicted_points = predicted_polylines[target_points_mask.unsqueeze(-1).expand_as(polylines)]

        
        # Avoid calculating loss if no points were masked (edge case)
        if target_points.numel() == 0:
            return torch.tensor(0.0, device=polylines.device, requires_grad=True), predicted_polylines

        loss = self.loss_fn(predicted_points, target_points)
        
        return loss, predicted_polylines

# --- Data Handling for Variable-Length Inputs ---
def generate_variable_length_polylines(channels: int) -> torch.Tensor:
    """Generates a single sample with a variable number of polylines and points."""
    num_polylines = torch.randint(low=512, high=1024, size=(1,)).item()
    num_points = torch.randint(low=10, high=20, size=(1,)).item()
    
    steps = torch.randn(1, num_polylines, num_points, channels) * 0.1
    steps[:, :, 0, :] = torch.randn(1, num_polylines, channels)
    return torch.cumsum(steps, dim=2).squeeze(0)

class VariablePolylineDataset(Dataset):
    """Dataset that generates variable-length polylines, simulating real-world data."""
    def __init__(self, num_samples: int, channels: int):
        self.num_samples = num_samples
        self.channels = channels

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        return generate_variable_length_polylines(self.channels)

def variable_length_collate_fn(batch: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Custom collate function to handle variable-length polylines by padding.
    
    Args:
        batch (List[torch.Tensor]): A list of tensors from the dataset, where each
                                     tensor has shape (NumPolylines, NumPoints, Channels).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - padded_polylines (torch.Tensor): Padded tensor of shape (B, MaxN, MaxP, C).
            - padding_mask (torch.Tensor): Boolean mask of shape (B, MaxN, MaxP).
    """
    # Find the maximum number of polylines and points in the batch.
    max_num_polylines = max(p.shape[0] for p in batch)
    max_num_points = max(p.shape[1] for p in batch)
    channels = batch[0].shape[2]
    
    # Create padded tensors and mask.
    padded_polylines = torch.zeros(len(batch), max_num_polylines, max_num_points, channels)
    padding_mask = torch.zeros(len(batch), max_num_polylines, max_num_points, dtype=torch.bool)
    
    for i, polylines in enumerate(batch):
        num_polylines, num_points, _ = polylines.shape
        padded_polylines[i, :num_polylines, :num_points] = polylines
        padding_mask[i, :num_polylines, :num_points] = True
        
    return padded_polylines, padding_mask

class Trainer:
    """Encapsulates the training loop and related logic."""
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, dataloader: DataLoader, config: TrainConfig, device: torch.device):
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.config = config
        self.device = device
        self.loss_history = []

    def train(self):
        print("\n🔥 Starting training...")
        for epoch in range(self.config.num_epochs):
            self.model.train()
            epoch_loss = 0.0
            
            progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}", leave=False)
            for polylines_batch, padding_mask_batch in progress_bar:
                polylines_batch = polylines_batch.to(self.device, non_blocking=True)
                padding_mask_batch = padding_mask_batch.to(self.device, non_blocking=True)
                
                self.optimizer.zero_grad()
                loss, _ = self.model(polylines_batch, padding_mask_batch)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                progress_bar.set_postfix(loss=f'{loss.item():.6f}')
            
            avg_epoch_loss = epoch_loss / len(self.dataloader)
            self.loss_history.append(avg_epoch_loss)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{self.config.num_epochs}], Average Loss: {avg_epoch_loss:.6f}")

        print("✅ Training complete.")

def main() -> None:
    config = TrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Instantiate Model Components ---
    encoder = PolylineEncoder(config.in_channels, config.hidden_dim)
    decoder = PointwisePredictionDecoder(config.hidden_dim, config.in_channels)
    model = SelfSupervisedModel(encoder, decoder, config.mask_ratio).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # --- Create Dataset and DataLoader ---
    dataset = VariablePolylineDataset(config.num_samples, config.in_channels)
    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=variable_length_collate_fn
    )
    
    trainer = Trainer(model, optimizer, dataloader, config, device)
    trainer.train()

    # --- Visualize Results ---
    plt.figure(figsize=(10, 5))
    plt.plot(trainer.loss_history)
    plt.title("Training Loss Over Time (Corrected Model with Structured Data)")
    plt.xlabel("Epoch")
    plt.ylabel("L1 Reconstruction Loss")
    plt.grid(True)
    plt.savefig('training_loss.png')
    print("📈 Plot saved to training_loss.png")


if __name__ == "__main__":
    main()
