import argparse
import os
import random
import shutil
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models import (
    PointwisePredictionDecoder,
    PolylineEncoder,
    SelfSupervisedModel,
)

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
    checkpoint_dir: str = './checkpoints'
    val_split_ratio: float = 0.2
    scene_size_meters: int = 500 # The width and height of a scene
    batch_size: int = 4
    learning_rate: float = 1e-4
    num_epochs: int = 50
    
    # DataLoader
    num_workers: int = 8

    # Visualization
    visualization_dir: str = './visualizations'
    max_num_visualizations: int = 10 # Max number of example plots to generate


# --- Data Generation and Loading ---
def build_osm_dataset(config: TrainConfig):
    """
    Fetches data from OpenStreetMap, processes it, splits it into train/val sets,
    and saves it to disk with descriptive filenames.
    """
    # Localize heavy/optional imports to avoid import-time failures when only using the models
    import osmnx as ox
    from pyproj import Transformer
    from shapely.geometry import LineString, MultiLineString

    print(f"Building OSM dataset at: {config.data_dir}")
    
    sf_locations = [
        "Mission Dolores Park, San Francisco", "Coit Tower, San Francisco",
        "Conservatory of Flowers, San Francisco", "Portsmouth Square, San Francisco",
        "Oracle Park, San Francisco", "Salesforce Park, San Francisco",
        "Lafayette Park, San Francisco", "Alamo Square, San Francisco",
        "Palace of Fine Arts, San Francisco", "Cole Valley, San Francisco", 
        "Western Addition, San Francisco", "Outer Sunset, Sunset District, San Francisco"
    ]   

    tags: Dict[str, Union[bool, str, List[str]]] = {"highway": True}
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

class Trainer:
    """Encapsulates the training loop, checkpointing, and validation."""
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, train_loader: DataLoader, val_loader: DataLoader, config: TrainConfig, device: torch.device):
        self.model, self.optimizer, self.train_loader, self.val_loader, self.config, self.device = model, optimizer, train_loader, val_loader, config, device
        self.train_loss_history, self.val_loss_history = [], []
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

    def save_checkpoint(self, epoch: int, is_best: bool):
        """Saves the current training state to a checkpoint file."""
        state = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
        }
        latest_path = os.path.join(self.config.checkpoint_dir, 'latest_checkpoint.pt')
        torch.save(state, latest_path)
        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, 'best_model.pt')
            shutil.copyfile(latest_path, best_path)
            print(f"   -> New best model saved with val_loss: {self.best_val_loss:.6f}")

    def load_checkpoint(self):
        """Loads training state from the latest checkpoint."""
        latest_path = os.path.join(self.config.checkpoint_dir, 'latest_checkpoint.pt')
        if os.path.exists(latest_path):
            print(f"Resuming training from checkpoint: {latest_path}")
            checkpoint = torch.load(latest_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch']
            self.best_val_loss = checkpoint['best_val_loss']
        else:
            print("No checkpoint found. Starting training from scratch.")

    def _run_epoch(self, epoch_num: int, is_training: bool) -> float:
        loader = self.train_loader if is_training else self.val_loader
        self.model.train(is_training)
        epoch_loss = 0.0
        desc = "Training" if is_training else "Validating"
        progress_bar = tqdm(loader, desc=f"Epoch {epoch_num+1}/{self.config.num_epochs} - {desc}", leave=False)
        for polylines_batch, padding_mask_batch in progress_bar:
            if polylines_batch.numel() == 0: continue
            polylines_batch, padding_mask_batch = polylines_batch.to(self.device, non_blocking=True), padding_mask_batch.to(self.device, non_blocking=True)
            if is_training: self.optimizer.zero_grad()
            loss, _ = self.model(polylines_batch, padding_mask_batch)
            if is_training:
                loss.backward()
                self.optimizer.step()
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=f'{loss.item():.6f}')
        return epoch_loss / len(loader) if len(loader) > 0 else 0.0

    def train(self):
        print("\n🔥 Starting training...")
        for epoch in range(self.start_epoch, self.config.num_epochs):
            train_loss = self._run_epoch(epoch, is_training=True)
            self.train_loss_history.append(train_loss)
            with torch.no_grad(): val_loss = self._run_epoch(epoch, is_training=False)
            self.val_loss_history.append(val_loss)
            
            is_best = val_loss < self.best_val_loss
            if is_best: self.best_val_loss = val_loss
            
            self.save_checkpoint(epoch, is_best)
            print(f"Epoch [{epoch+1}/{self.config.num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        print("✅ Training complete.")

def generate_visualizations(model: SelfSupervisedModel, dataloader: DataLoader, device: torch.device, config: TrainConfig):
    """
    Generates and saves a gallery of model prediction visualizations from a single batch.
    """
    print(f"Generating up to {config.max_num_visualizations} visualizations...")
    os.makedirs(config.visualization_dir, exist_ok=True)
    
    model.eval()
    with torch.no_grad():
        # Get one batch from the validation set
        polylines_batch, padding_mask_batch = next(iter(dataloader))
        if polylines_batch.numel() == 0:
            print("Skipping visualization: first validation batch was empty.")
            return

        polylines_batch = polylines_batch.to(device)
        padding_mask_batch = padding_mask_batch.to(device)

        # Run the forward pass, getting the masks back for visualization
        _, predictions, task_mask = model(polylines_batch, padding_mask_batch, return_masks=True)

        # Move results to CPU for plotting
        polylines_batch, padding_mask_batch, predictions, task_mask = \
            polylines_batch.cpu(), padding_mask_batch.cpu(), predictions.cpu(), task_mask.cpu()

        # Base the number of visualizations on the ACTUAL batch size, not the configured one.
        actual_batch_size = polylines_batch.shape[0]
        num_to_generate = min(config.max_num_visualizations, actual_batch_size)
        
        for sample_idx in range(num_to_generate):
            scene_polylines = polylines_batch[sample_idx]
            scene_padding_mask = padding_mask_batch[sample_idx]
            scene_predictions = predictions[sample_idx]
            scene_task_mask = task_mask[sample_idx]

            # Find all valid polylines in the current scene
            valid_polyline_indices = torch.where(scene_padding_mask.sum(dim=-1) > 0)[0]
            if len(valid_polyline_indices) == 0: continue

            # Select a random valid polyline to visualize
            target_polyline_idx = random.choice(valid_polyline_indices).item()
            
            plt.figure(figsize=(12, 12))
            
            # 1. Plot all polylines in the scene as faint context
            for i, polyline in enumerate(scene_polylines):
                valid_points = polyline[scene_padding_mask[i]]
                plt.plot(valid_points[:, 0], valid_points[:, 1], color='gray', alpha=0.3, zorder=1)
                
            # 2. Extract data for the target polyline
            target_polyline_gt = scene_polylines[target_polyline_idx][scene_padding_mask[target_polyline_idx]]
            target_polyline_pred = scene_predictions[target_polyline_idx][scene_padding_mask[target_polyline_idx]]
            is_masked = scene_task_mask[target_polyline_idx][scene_padding_mask[target_polyline_idx]]

            # 3. Plot the target polyline segment by segment for clarity
            for i in range(1, len(target_polyline_gt)):
                p1, p2 = target_polyline_gt[i-1], target_polyline_gt[i]
                if is_masked[i-1] or is_masked[i]: plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r--', zorder=2)
                else: plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', zorder=3)

            # Create dummy plots just for the legend handles
            plt.plot([], [], 'b-', label='Visible Segments')
            plt.plot([], [], 'r--', label='Masked Segments (Ground Truth)')
            
            # 4. Plot the masked and predicted points on top
            plt.scatter(target_polyline_gt[is_masked, 0], target_polyline_gt[is_masked, 1], 
                        edgecolors='red', facecolors='none', s=150, 
                        label='Masked Points (Ground Truth)', zorder=4)
            plt.scatter(target_polyline_pred[is_masked, 0], target_polyline_pred[is_masked, 1], 
                        c='green', marker='x', s=100, label='Predicted Points', zorder=5)

            plt.title(f"Prediction for Sample {sample_idx}, Polyline {target_polyline_idx}")
            plt.xlabel("X (meters)"); plt.ylabel("Y (meters)")  # noqa: E702
            plt.legend(); plt.grid(True); plt.axis('equal')  # noqa: E702
            
            save_path = os.path.join(config.visualization_dir, f"sample_{sample_idx}_polyline_{target_polyline_idx}.png")
            plt.savefig(save_path)
            plt.close() # Close the figure to free memory

    print(f"📈 {num_to_generate} visualizations saved to {config.visualization_dir}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Train and visualize a self-supervised polyline model.")
    parser.add_argument('--force-rebuild', action='store_true', help="Force deletion and rebuilding of the dataset.")
    parser.add_argument('--train', action='store_true', help="Run the training loop.")
    parser.add_argument('--visualize', action='store_true', help="Generate visualizations using the best model.")
    args = parser.parse_args()

    config = TrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Handle Data ---
    if args.force_rebuild and os.path.exists(config.data_dir):
        print(f"Deleting existing dataset at {config.data_dir}...")
        shutil.rmtree(config.data_dir)
    if not os.path.exists(config.data_dir) or not os.listdir(os.path.join(config.data_dir, 'train')):
        build_osm_dataset(config)

    # --- 2. Setup Model, Optimizer, and DataLoaders ---
    encoder = PolylineEncoder(config.in_channels, config.hidden_dim)
    decoder = PointwisePredictionDecoder(config.hidden_dim, config.in_channels)
    model = SelfSupervisedModel(encoder, decoder, config.mask_ratio).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # --- 3. Conditional Execution Logic ---
    best_model_path = os.path.join(config.checkpoint_dir, 'best_model.pt')
    # If no flags are specified, default to training if no model exists, else visualize.
    if not args.train and not args.visualize:
        if os.path.exists(best_model_path):
            print("No action specified, but a trained model was found. Running visualization.")
            args.visualize = True
        else:
            print("No action specified and no trained model found. Running training.")
            args.train = True

    # --- 4. Run Training ---
    if args.train:
        train_dataset = OSMDataset(data_dir=config.data_dir, split='train')
        val_dataset = OSMDataset(data_dir=config.data_dir, split='val')
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=(device.type == 'cuda'), collate_fn=variable_length_collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=(device.type == 'cuda'), collate_fn=variable_length_collate_fn)
        trainer = Trainer(model, optimizer, train_loader, val_loader, config, device)
        trainer.load_checkpoint()
        trainer.train()

        # Plot training history after training is complete
        plt.figure(figsize=(10, 5))
        plt.plot(trainer.train_loss_history, label='Training Loss')
        plt.plot(trainer.val_loss_history, label='Validation Loss')
        plt.title("Training and Validation Loss Over Time")
        plt.xlabel("Epoch"); plt.ylabel("Average L1 Reconstruction Loss")  # noqa: E702
        plt.legend(); plt.grid(True)  # noqa: E702
        plt.savefig('training_loss.png')
        print("📈 Plot saved to training_loss.png")

    # --- 5. Run Visualization ---
    if args.visualize:
        if not args.train: # If we only visualize, we need to load the best model
            if os.path.exists(best_model_path):
                print(f"Loading best model from {best_model_path} for visualization.")
                checkpoint = torch.load(best_model_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                print("Cannot run visualization: No trained model found. Please run with --train first.")
                return
        
        # We need a dataloader for visualization regardless
        val_dataset = OSMDataset(data_dir=config.data_dir, split='val')
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=(device.type == 'cuda'), collate_fn=variable_length_collate_fn)
        generate_visualizations(model, val_loader, device, config)


if __name__ == "__main__":
    main()
