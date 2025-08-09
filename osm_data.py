import os
import random
from typing import Dict, List, Tuple, Union

import torch
from torch.utils.data import Dataset


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
    channels = batch[0][0].shape[1] if batch[0] else 2  # Default to 2 channels if a scene is empty

    padded_polylines = torch.zeros(len(batch), max_num_polylines, max_num_points, channels)
    padding_mask = torch.zeros(len(batch), max_num_polylines, max_num_points, dtype=torch.bool)

    for i, scene in enumerate(batch):
        for j, polyline in enumerate(scene):
            num_points = polyline.shape[0]
            padded_polylines[i, j, :num_points] = polyline
            padding_mask[i, j, :num_points] = True

    return padded_polylines, padding_mask


def build_osm_dataset(config) -> None:
    """
    Fetches data from OpenStreetMap, processes it, splits it into train/val sets,
    and saves it to disk with descriptive filenames.

    Heavy/optional dependencies are imported locally to keep base imports light.
    """
    # Local imports to avoid hard dependency for consumers who only use the models
    import numpy as np
    import osmnx as ox
    from pyproj import Transformer
    from shapely.geometry import LineString, MultiLineString
    from tqdm import tqdm

    print(f"Building OSM dataset at: {config.data_dir}")

    sf_locations = [
        "Mission Dolores Park, San Francisco",
        "Coit Tower, San Francisco",
        "Conservatory of Flowers, San Francisco",
        "Portsmouth Square, San Francisco",
        "Oracle Park, San Francisco",
        "Salesforce Park, San Francisco",
        "Lafayette Park, San Francisco",
        "Alamo Square, San Francisco",
        "Palace of Fine Arts, San Francisco",
        "Cole Valley, San Francisco",
        "Western Addition, San Francisco",
        "Outer Sunset, Sunset District, San Francisco",
    ]

    tags: Dict[str, Union[bool, str, List[str]]] = {"highway": True}
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32610", always_xy=True)

    all_scenes_with_names: List[Tuple[str, List[torch.Tensor]]] = []
    for place_query in tqdm(sf_locations, desc="Generating scenes from SF locations"):
        try:
            center_lat, center_lng = ox.geocode(place_query)
            bbox = ox.utils_geo.bbox_from_point((center_lat, center_lng), dist=config.scene_size_meters)
            features_gdf = ox.features_from_bbox(bbox, tags)
            if features_gdf.empty:
                continue

            features_gdf_proj = ox.projection.project_gdf(features_gdf, to_crs="EPSG:32610")

            center_x, center_y = transformer.transform(center_lng, center_lat)
            center_coords = np.array([center_x, center_y])

            polylines: List[torch.Tensor] = []
            for _, row in features_gdf_proj.iterrows():
                geom = row.geometry
                if isinstance(geom, (LineString, MultiLineString)):
                    geoms = [geom] if isinstance(geom, LineString) else list(geom.geoms)
                    for line in geoms:
                        coords = np.array(line.coords) - center_coords
                        polylines.append(torch.tensor(coords, dtype=torch.float32))

            if polylines:
                short_name = place_query.split(',')[0].lower().replace(' ', '_')
                all_scenes_with_names.append((short_name, polylines))

        except Exception as e:  # noqa: BLE001
            print(f"Could not process scene for '{place_query}': {e}")
            continue

    random.shuffle(all_scenes_with_names)
    split_idx = int(len(all_scenes_with_names) * (1 - config.val_split_ratio))
    train_scenes = all_scenes_with_names[:split_idx]
    val_scenes = all_scenes_with_names[split_idx:]

    for split, scenes_with_names in [("train", train_scenes), ("val", val_scenes)]:
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
            raise FileNotFoundError(
                f"Split directory not found: {self.split_dir}. Please build the dataset first."
            )
        self.file_paths = [
            os.path.join(self.split_dir, f) for f in os.listdir(self.split_dir) if f.endswith(".pt")
        ]
        if not self.file_paths:
            raise FileNotFoundError(f"No data files found in {self.split_dir}.")

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> List[torch.Tensor]:
        filepath = self.file_paths[idx]
        data = torch.load(filepath)
        if not isinstance(data, list) or not all(isinstance(p, torch.Tensor) for p in data):
            raise TypeError(f"Data in file {filepath} is not a list of Tensors as expected.")
        return data


__all__ = ["build_osm_dataset", "OSMDataset", "variable_length_collate_fn"]


