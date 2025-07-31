import torch
import torch.nn as nn

# A minimal PointNet-style encoder for polylines
class PolylineEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, polylines):
        """
        Args:
            polylines (Tensor): Shape (B, N, P, C) -> Batch, NumPolylines, NumPoints, Channels
        Returns:
            Tensor: Shape (B, N, H) -> Batch, NumPolylines, HiddenDim
        """
        # Apply MLP to each point independently
        point_features = self.point_mlp(polylines) # (B, N, P, H)
        
        # Max-pool across the points of each polyline to get a global feature
        polyline_features, _ = torch.max(point_features, dim=2) # (B, N, H)
        return polyline_features


# A simple decoder to predict point coordinates from a global feature vector
class MaskedPredictionDecoder(nn.Module):
    def __init__(self, hidden_dim, num_points, point_channels):
        super().__init__()
        self.num_points = num_points
        self.point_channels = point_channels
        self.decoder_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, num_points * point_channels)
        )

    def forward(self, polyline_features):
        """
        Args:
            polyline_features (Tensor): Shape (B, N, H)
        Returns:
            Tensor: Shape (B, N, P, C) -> Predicted coordinates for all points
        """
        # Predict all points for each polyline
        predicted = self.decoder_mlp(polyline_features) # (B, N, P*C)
        
        # Reshape to match polyline format
        batch_size, num_polylines, _ = polyline_features.shape
        predicted = predicted.view(batch_size, num_polylines, self.num_points, self.point_channels)
        return predicted


# Main model for the self-supervised task
class SelfSupervisedModel(nn.Module):
    def __init__(self, encoder, decoder, mask_ratio=0.3):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mask_ratio = mask_ratio
        self.loss_fn = nn.L1Loss()

    def forward(self, polylines):
        """
        Performs the self-supervised masked point prediction task.
        
        Args:
            polylines (Tensor): The ground truth polylines of shape (B, N, P, C)
        Returns:
            loss (Tensor): The reconstruction loss on the masked points.
            predictions (Tensor): The reconstructed polylines.
        """
        # 1. Create mask and apply it to the input
        # We create a random mask for the points (dimension 2)
        point_mask = torch.rand(polylines.shape[:-1], device=polylines.device) < self.mask_ratio
        # Add a dimension to match the channels
        point_mask_expanded = point_mask.unsqueeze(-1)

        masked_polylines = polylines.clone()
        masked_polylines[point_mask_expanded.expand_as(polylines)] = 0.0 # Zero out masked points

        # 2. Encode the masked polylines
        encoded_features = self.encoder(masked_polylines)

        # 3. Decode to predict all points
        predicted_polylines = self.decoder(encoded_features)

        # 4. Calculate loss ONLY on the points that were masked
        target_points = polylines[point_mask_expanded.expand_as(polylines)]
        predicted_points = predicted_polylines[point_mask_expanded.expand_as(polylines)]
        
        loss = self.loss_fn(predicted_points, target_points)
        
        return loss, predicted_polylines


def main():
    # --- Model Hyperparameters ---
    IN_CHANNELS = 2  # (x, y) coordinates
    HIDDEN_DIM = 128
    
    # --- Data Shape ---
    BATCH_SIZE = 4
    NUM_POLYLINES = 50
    NUM_POINTS = 20
    
    # --- Instantiate Model Components ---
    encoder = PolylineEncoder(in_channels=IN_CHANNELS, hidden_dim=HIDDEN_DIM)
    decoder = MaskedPredictionDecoder(hidden_dim=HIDDEN_DIM, num_points=NUM_POINTS, point_channels=IN_CHANNELS)
    ssl_model = SelfSupervisedModel(encoder, decoder, mask_ratio=0.3)
    
    # --- Create Dummy Data ---
    # Represents a batch of scenes, each with multiple polylines
    dummy_osm_data = torch.randn(BATCH_SIZE, NUM_POLYLINES, NUM_POINTS, IN_CHANNELS)
    
    # --- Run a Forward Pass ---
    print(f"Input shape: {dummy_osm_data.shape}")
    
    reconstruction_loss, predictions = ssl_model(dummy_osm_data)
    
    print(f"Prediction shape: {predictions.shape}")
    print(f"Calculated reconstruction loss: {reconstruction_loss.item():.4f}")
    
    # In a real training loop, you would call loss.backward() and optimizer.step() here.
    print("\n✅ Minimal example ran successfully.")


if __name__ == "__main__":
    main()
