import torch
import torch.nn as nn
import matplotlib.pyplot as plt

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
            polylines (Tensor): Shape (B, N, P, C). Masked points are expected to be zero.
        Returns:
            Tensor: Shape (B, N, H) -> Batch, NumPolylines, HiddenDim
        """
        # Apply MLP to each point independently
        point_features = self.point_mlp(polylines) # (B, N, P, H)

        # Max-pool across the points of each polyline to get a global feature
        polyline_features, _ = torch.max(point_features, dim=2) # (B, N, H)
        return polyline_features


# This decoder remains the same.
class PointwisePredictionDecoder(nn.Module):
    def __init__(self, hidden_dim, point_channels):
        super().__init__()
        decoder_in_dim = hidden_dim + point_channels
        self.decoder_mlp = nn.Sequential(
            nn.Linear(decoder_in_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, point_channels)
        )

    def forward(self, combined_features):
        return self.decoder_mlp(combined_features)


# Main model for the self-supervised task
class SelfSupervisedModel(nn.Module):
    def __init__(self, encoder, decoder, mask_ratio=0.25):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mask_ratio = mask_ratio
        self.loss_fn = nn.L1Loss()

    def forward(self, polylines):
        """
        Performs the self-supervised masked point prediction task.
        """
        # --- 1. Create mask and apply it to the input ---
        point_mask = torch.rand(polylines.shape[:-1], device=polylines.device) < self.mask_ratio
        
        # Create a single version of the polylines where masked points are zeroed out.
        masked_polylines = polylines.clone()
        masked_polylines[point_mask.unsqueeze(-1).expand_as(polylines)] = 0.0

        # --- 2. Encode the masked polylines to get a GLOBAL feature ---
        # The modified encoder will now correctly handle the zeroed-out points.
        global_feature = self.encoder(masked_polylines)

        # --- 3. Prepare input for the Pointwise Decoder ---
        batch_size, num_polylines, num_points, _ = polylines.shape
        
        global_feature_expanded = global_feature.unsqueeze(2).expand(
            batch_size, num_polylines, num_points, -1
        )
        
        # Decoder input uses the same masked polylines.
        decoder_input = torch.cat([global_feature_expanded, masked_polylines], dim=-1)

        # --- 4. Decode to predict all points ---
        predicted_polylines = self.decoder(decoder_input)

        # --- 5. Calculate loss ONLY on the points that were masked ---
        target_points = polylines[point_mask.unsqueeze(-1).expand_as(polylines)]
        predicted_points = predicted_polylines[point_mask.unsqueeze(-1).expand_as(polylines)]
        
        # Avoid calculating loss if no points were masked (edge case)
        if target_points.numel() == 0:
            return torch.tensor(0.0, device=polylines.device, requires_grad=True), predicted_polylines

        loss = self.loss_fn(predicted_points, target_points)
        
        return loss, predicted_polylines

# --- NEW FUNCTION TO GENERATE STRUCTURED DATA ---
def generate_structured_polylines(batch_size, num_polylines, num_points, channels):
    """
    Generates polylines that have actual structure by making them random walks.
    This provides a meaningful correlation for the model to learn.
    """
    # Start with a random initial point for each polyline
    polylines = torch.randn(batch_size, num_polylines, 1, channels)
    
    # Sequentially generate the rest of the points
    all_points = [polylines]
    for _ in range(1, num_points):
        # Add a small random step to the last point to get the next point
        next_step = polylines[:, :, -1:, :] + torch.randn(batch_size, num_polylines, 1, channels) * 0.1
        all_points.append(next_step)
        
    return torch.cat(all_points, dim=2)


def main():
    # --- 1. Setup Device (GPU or CPU) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Model Hyperparameters ---
    IN_CHANNELS = 2
    HIDDEN_DIM = 128
    
    # --- Data Shape ---
    BATCH_SIZE = 4
    NUM_POLYLINES = 1024
    NUM_POINTS = 20
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 500
    
    # --- Instantiate Model Components ---
    encoder = PolylineEncoder(in_channels=IN_CHANNELS, hidden_dim=HIDDEN_DIM)
    decoder = PointwisePredictionDecoder(hidden_dim=HIDDEN_DIM, point_channels=IN_CHANNELS)
    ssl_model = SelfSupervisedModel(encoder, decoder, mask_ratio=0.45).to(device)
    optimizer = torch.optim.Adam(ssl_model.parameters(), lr=LEARNING_RATE)
    
    # --- Create Structured Dummy Data ---
    # Instead of pure random noise, we generate data with inherent structure.
    dummy_osm_data = generate_structured_polylines(BATCH_SIZE, NUM_POLYLINES, NUM_POINTS, IN_CHANNELS).to(device)

    # --- Training Loop ---
    print("\n🔥 Starting training with structured data...")
    loss_history = []
    for epoch in range(NUM_EPOCHS):
        ssl_model.train()
        optimizer.zero_grad()
        loss, _ = ssl_model(dummy_osm_data)
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.6f}")

    print("✅ Training complete.")

    # --- 4. Visualize Results ---
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.title("Training Loss Over Time (Corrected Model with Structured Data)")
    plt.xlabel("Epoch")
    plt.ylabel("L1 Reconstruction Loss")
    plt.grid(True)
    plt.savefig('training_loss.png')
    print("📈 Plot saved to training_loss.png")


if __name__ == "__main__":
    main()
