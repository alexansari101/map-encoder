from typing import Tuple, Union

import torch
import torch.nn as nn


class PolylineEncoder(nn.Module):
    """
    Encodes a batch of polylines into a fixed-size global feature vector for each polyline.
    It uses a PointNet-like architecture where each point is processed by an MLP,
    and a symmetric function (max-pooling) aggregates point features.

    A simplified version of the PointNetPolylineEncoder class
    https://github.com/sshaoshuai/MTR
    """

    def __init__(self, in_channels: int, hidden_dim: int):
        """
        Args:
            in_channels (int): The number of input channels for each point (e.g., 2 for x, y).
            hidden_dim (int): The dimensionality of the hidden and output features.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.point_mlp = nn.Sequential(
            nn.Linear(in_channels, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
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

        valid_features = self.point_mlp(polylines[polylines_mask])

        point_features = polylines.new_zeros(batch_size, num_polylines, num_points, self.hidden_dim)
        point_features[polylines_mask] = valid_features

        point_features[~polylines_mask] = -1e9

        polyline_features, _ = torch.max(point_features, dim=2)
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
        decoder_in_dim = hidden_dim + point_channels
        self.decoder_mlp = nn.Sequential(
            nn.Linear(decoder_in_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, point_channels),
        )

    def forward(self, combined_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            combined_features (torch.Tensor): Concatenated global and local features.
                                              Shape: (B, N, P, HiddenDim + PointChannels).

        Returns:
            torch.Tensor: The predicted coordinates for each point.
                          Shape: (B, N, P, PointChannels).
        """
        return self.decoder_mlp(combined_features)


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

    def forward(
        self,
        polylines: torch.Tensor,
        padding_mask: torch.Tensor,
        return_masks: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Performs one forward pass of self-supervised masked point prediction.

        Args:
            polylines (torch.Tensor): Ground truth polylines.
            padding_mask (torch.Tensor): Mask for padded elements.
            return_masks (bool): If True, returns the internal task mask for visualization.

        Returns:
            If return_masks is False: (loss, predictions)
            If return_masks is True: (loss, predictions, task_mask)
        """
        task_mask = torch.rand(polylines.shape[:-1], device=polylines.device) < self.mask_ratio

        visible_points_mask = padding_mask & ~task_mask

        masked_polylines_for_decoder = polylines.clone()
        combined_mask_for_zeroing = ~visible_points_mask
        masked_polylines_for_decoder[combined_mask_for_zeroing.unsqueeze(-1).expand_as(polylines)] = 0.0

        global_feature = self.encoder(polylines, visible_points_mask)

        _, _, num_points, _ = polylines.shape
        global_feature_expanded = global_feature.unsqueeze(2).expand(-1, -1, num_points, -1)
        decoder_input = torch.cat([global_feature_expanded, masked_polylines_for_decoder], dim=-1)

        predicted_polylines = self.decoder(decoder_input)

        target_points_mask = padding_mask & task_mask
        target_points = polylines[target_points_mask.unsqueeze(-1).expand_as(polylines)]
        predicted_points = predicted_polylines[target_points_mask.unsqueeze(-1).expand_as(polylines)]

        if target_points.numel() == 0:
            loss = torch.tensor(0.0, device=polylines.device, requires_grad=True)
        else:
            loss = self.loss_fn(predicted_points, target_points)

        if return_masks:
            return loss, predicted_polylines, task_mask
        return loss, predicted_polylines


__all__ = [
    "PolylineEncoder",
    "PointwisePredictionDecoder",
    "SelfSupervisedModel",
]


