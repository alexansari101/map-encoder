import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Tuple, List
from dataclasses import dataclass
import numpy as np
import random

import osmnx as ox
from pyproj import Transformer
from shapely.geometry import LineString, MultiLineString

# --- Configuration ---
@dataclass
class TrainConfig:
    """Configuration parameters for training."""
    # Model Hyperparameters
    in_channels: int = 2
    hidden_dim: int = 128
    mask_ratio: float = 0.45
    
    # Data & Training Hyperparameters
    data_dir: str = './data/sf_scenes'
    val_split_ratio: float = 0.2
    scene_size_meters: int = 500 # The width and height of a scene
    batch_size: int = 4
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

# --- Data Generation and Loading ---
def build_osm_dataset(config: TrainConfig):
    """
    Fetches data from OpenStreetMap, processes it, splits it into train/val sets,
    and saves it to disk with descriptive filenames.
    """
    print(f"Building OSM dataset at: {config.data_dir}")
    
    sf_locations = [
        "Mission Dolores Park, San Francisco", "Coit Tower, San Francisco",
        "Conservatory of Flowers, San Francisco", "Portsmouth Square, San Francisco",
        "Oracle Park, San Francisco", "Salesforce Park, San Francisco",
        "Lafayette Park, San Francisco", "Alamo Square, San Francisco",
        "Palace of Fine Arts, San Francisco", "Cole Valley, San Francisco", 
        "Western Addition, San Francisco", "Outer Sunset, Sunset District, San Francisco"
    ]   

    tags = {"highway": True}
    # Create a pyproj transformer to convert from WGS84 (lat/lon) to a local projected CRS (in meters).
    # EPSG:32610 is UTM Zone 10N, suitable for San Francisco.
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32610", always_xy=True)
    
    all_scenes_with_names = []
    for place_query in tqdm(sf_locations, desc="Generating scenes from SF locations"):
        try:
            center_lat, center_lng = ox.geocode(place_query)
            bbox = ox.utils_geo.bbox_from_point((center_lat, center_lng), dist=config.scene_size_meters)
            features_gdf = ox.features_from_bbox(bbox, tags)
            if features_gdf.empty: continue
            
            features_gdf_proj = ox.projection.project_gdf(features_gdf, to_crs="EPSG:32610")

            # Convert the scene's center point to projected coordinates for normalization.
            center_x, center_y = transformer.transform(center_lng, center_lat)
            center_coords = np.array([center_x, center_y])
            
            polylines = []
            for _, row in features_gdf_proj.iterrows():
                geom = row.geometry
                if isinstance(geom, (LineString, MultiLineString)):
                    geoms = [geom] if isinstance(geom, LineString) else list(geom.geoms)
                    for line in geoms:
                        # Normalize coordinates by subtracting the scene center. This makes the model
                        # translation-invariant, as all scenes are centered around (0,0).
                        coords = np.array(line.coords) - center_coords
                        polylines.append(torch.tensor(coords, dtype=torch.float32))
            
            if polylines:
                # Each scene is stored as a tuple: (string_name, List[polyline_tensors])
                short_name = place_query.split(',')[0].lower().replace(' ', '_')
                all_scenes_with_names.append((short_name, polylines))

        except Exception as e:
            print(f"Could not process scene for '{place_query}': {e}")
            continue

    # Split scenes into train and validation sets
    random.shuffle(all_scenes_with_names)
    split_idx = int(len(all_scenes_with_names) * (1 - config.val_split_ratio))
    train_scenes = all_scenes_with_names[:split_idx]
    val_scenes = all_scenes_with_names[split_idx:]

    # Save scenes to their respective directories
    for split, scenes_with_names in [('train', train_scenes), ('val', val_scenes)]:
        split_dir = os.path.join(config.data_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        for name, scene_data in scenes_with_names:
            torch.save(scene_data, os.path.join(split_dir, f"{name}.pt"))

    print(f"✅ Dataset built. Train scenes: {len(train_scenes)}, Val scenes: {len(val_scenes)}")

class OSMDataset(Dataset):
    """Loads pre-processed polyline data scenes from a specific split directory."""
    def __init__(self, data_dir: str, split: str):
        self.split_dir = os.path.join(data_dir, split)
        if not os.path.exists(self.split_dir):
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}. Please build the dataset first.")
        self.file_paths = [os.path.join(self.split_dir, f) for f in os.listdir(self.split_dir) if f.endswith('.pt')]
        if not self.file_paths:
            raise FileNotFoundError(f"No data files found in {self.split_dir}.")

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> List[torch.Tensor]:
        """
        Loads and validates a single data sample (a "scene").

        Returns:
            List[torch.Tensor]: A list of polyline tensors. Each tensor in the list
                                has a shape of (NumPoints_i, Channels), where NumPoints_i
                                can vary for each polyline.
        """
        filepath = self.file_paths[idx]
        data = torch.load(filepath)

        # Add a validation check to ensure data integrity.
        # This will catch corrupted files immediately and provide a clear error.
        if not isinstance(data, list) or not all(isinstance(p, torch.Tensor) for p in data):
            raise TypeError(f"Data in file {filepath} is not a list of Tensors as expected.")

        return data

def variable_length_collate_fn(batch: List[List[torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Custom collate function to handle variable-length scenes by padding.

    Args:
        batch (List[List[torch.Tensor]]): A list of scenes from the dataset.
            - Each `scene` is a `List[torch.Tensor]`.
            - Each `torch.Tensor` is a polyline of shape (NumPoints_i, Channels).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - padded_polylines (torch.Tensor): Padded tensor of shape (B, N, P, C).
                B: Batch Size
                N: Max number of polylines in a scene within the batch.
                P: Max number of points in a polyline within the batch.
                C: Number of channels (e.g., 2 for x, y).
            - padding_mask (torch.Tensor): Boolean mask of shape (B, N, P).
                `True` indicates a real data point, `False` indicates padding.
    """
    # Filter out empty scenes that might have been saved
    batch = [scene for scene in batch if scene]
    if not batch:
        return torch.tensor([]), torch.tensor([])

    max_num_polylines = max(len(scene) for scene in batch)
    max_num_points = max(max(p.shape[0] for p in scene) if scene else 0 for scene in batch)
    channels = batch[0][0].shape[1] if batch[0] else 2 # Default to 2 channels if a scene is empty
    
    padded_polylines = torch.zeros(len(batch), max_num_polylines, max_num_points, channels)
    padding_mask = torch.zeros(len(batch), max_num_polylines, max_num_points, dtype=torch.bool)
    
    for i, scene in enumerate(batch):
        for j, polyline in enumerate(scene):
            num_points = polyline.shape[0]
            padded_polylines[i, j, :num_points] = polyline
            padding_mask[i, j, :num_points] = True
            
    return padded_polylines, padding_mask

# --- Trainer Class ---
class Trainer:
    """Encapsulates the training and validation loops."""
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, train_loader: DataLoader, val_loader: DataLoader, config: TrainConfig, device: torch.device):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.train_loss_history = []
        self.val_loss_history = []

    def _run_epoch(self, epoch_num: int, is_training: bool) -> float:
        """Runs a single epoch of either training or validation."""
        loader = self.train_loader if is_training else self.val_loader
        self.model.train(is_training)
        
        epoch_loss = 0.0
        desc = "Training" if is_training else "Validating"
        progress_bar = tqdm(loader, desc=f"Epoch {epoch_num+1}/{self.config.num_epochs} - {desc}", leave=False)
        
        for polylines_batch, padding_mask_batch in progress_bar:
            if polylines_batch.numel() == 0: continue
            
            polylines_batch = polylines_batch.to(self.device, non_blocking=True)
            padding_mask_batch = padding_mask_batch.to(self.device, non_blocking=True)
            
            if is_training:
                self.optimizer.zero_grad()
            
            loss, _ = self.model(polylines_batch, padding_mask_batch)
            
            if is_training:
                loss.backward()
                self.optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=f'{loss.item():.6f}')
        
        return epoch_loss / len(loader) if len(loader) > 0 else 0.0

    def train(self):
        print("\n🔥 Starting training...")
        for epoch in range(self.config.num_epochs):
            train_loss = self._run_epoch(epoch, is_training=True)
            self.train_loss_history.append(train_loss)
            
            with torch.no_grad():
                val_loss = self._run_epoch(epoch, is_training=False)
            self.val_loss_history.append(val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{self.config.num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        print("✅ Training complete.")

def main() -> None:
    config = TrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Build Dataset if it doesn't exist ---
    if not os.path.exists(config.data_dir) or not os.listdir(config.data_dir):
        build_osm_dataset(config)

    # --- 2. Setup Model, Optimizer, and DataLoaders ---
    encoder = PolylineEncoder(config.in_channels, config.hidden_dim)
    decoder = PointwisePredictionDecoder(config.hidden_dim, config.in_channels)
    model = SelfSupervisedModel(encoder, decoder, config.mask_ratio).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    train_dataset = OSMDataset(data_dir=config.data_dir, split='train')
    val_dataset = OSMDataset(data_dir=config.data_dir, split='val')
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=(device.type == 'cuda'), collate_fn=variable_length_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=(device.type == 'cuda'), collate_fn=variable_length_collate_fn
    )
    # --- 3. Instantiate and run the Trainer ---
    trainer = Trainer(model, optimizer, train_loader, val_loader, config, device)
    trainer.train()

    # --- 4. Visualize and Save Results ---
    plt.figure(figsize=(10, 5))
    plt.plot(trainer.train_loss_history, label='Training Loss')
    plt.plot(trainer.val_loss_history, label='Validation Loss')
    plt.title("Training and Validation Loss Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Average L1 Reconstruction Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss.png')
    print("📈 Plot saved to training_loss.png")    # --- Visualize Results ---


if __name__ == "__main__":
    main()
