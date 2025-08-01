import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

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
    def __init__(self, in_channels, hidden_dim):
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

    def forward(self, polylines, polylines_mask):
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
        return polyline_features


class PointwisePredictionDecoder(nn.Module):
    """
    Decodes a combined feature vector (global context + local point info)
    to predict the coordinates of each point in a polyline.
    """
    def __init__(self, hidden_dim, point_channels):
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

    def forward(self, combined_features):
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
    def __init__(self, encoder, decoder, mask_ratio=0.25):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mask_ratio = mask_ratio
        self.loss_fn = nn.L1Loss()

    def forward(self, polylines):
        """
        Performs one forward pass of self-supervised masked point prediction.

        Args:
            polylines (torch.Tensor): The ground truth polylines.
                                      Shape: (B, N, P, C).

        Returns:
            - loss (torch.Tensor): The reconstruction loss on the masked points.
            - predictions (torch.Tensor): The reconstructed polylines.
        """
        # 1. Create a random mask to hide a portion of the points.
        # `point_mask` is True for points we want to HIDE.
        point_mask = torch.rand(polylines.shape[:-1], device=polylines.device) < self.mask_ratio

        # The encoder expects a mask where True means the point is VISIBLE
        visible_points_mask = ~point_mask
        
        # The decoder needs a version of the polylines where masked points are zeroed out.
        masked_polylines_for_decoder = polylines.clone()
        masked_polylines_for_decoder[point_mask.unsqueeze(-1).expand_as(polylines)] = 0.0

        # 2. Encode the polylines using the visbility mask.
        global_feature = self.encoder(polylines, visible_points_mask)

        # 3. Prepare the decoder input by combining global and local information.
        batch_size, num_polylines, num_points, _ = polylines.shape
        global_feature_expanded = global_feature.unsqueeze(2).expand(-1, -1, num_points, -1)
        decoder_input = torch.cat([global_feature_expanded, masked_polylines_for_decoder], dim=-1)
        
        # 4. Decode to predict all points.
        predicted_polylines = self.decoder(decoder_input)

        # 5. Calculate loss ONLY on the points that were masked.
        target_points = polylines[point_mask.unsqueeze(-1).expand_as(polylines)]
        predicted_points = predicted_polylines[point_mask.unsqueeze(-1).expand_as(polylines)]
        
        # Avoid calculating loss if no points were masked (edge case)
        if target_points.numel() == 0:
            return torch.tensor(0.0, device=polylines.device, requires_grad=True), predicted_polylines

        loss = self.loss_fn(predicted_points, target_points)
        
        return loss, predicted_polylines

# --- FUNCTION TO GENERATE STRUCTURED POLYLINE DATA ---
def generate_structured_polylines(num_samples, num_polylines, num_points, channels):
    """
    Generates polylines that have actual structure by making them random walks.
    This provides a meaningful correlation for the model to learn.
    """
    # Generate all the small random steps at once
    steps = torch.randn(num_samples, num_polylines, num_points, channels) * 0.1
    # Set the first "step" to be the initial random position
    steps[:, :, 0, :] = torch.randn(num_samples, num_polylines, channels)
    # Use cumulative sum to create the random walk paths
    return torch.cumsum(steps, dim=2)

class PolylineDataset(Dataset):
    """
    A memory-efficient dataset that generates structured polyline data on-the-fly.
    This pattern is scalable for large datasets that cannot fit into memory.
    """
    def __init__(self, num_samples, num_polylines, num_points, channels):
        self.num_samples = num_samples
        self.num_polylines = num_polylines
        self.num_points = num_points
        self.channels = channels

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate one sample (e.g., one "scene" of polylines) on the fly.
        return generate_structured_polylines(1, self.num_polylines, self.num_points, self.channels).squeeze(0)

def main():
    # --- Setup Device (GPU or CPU) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Model Hyperparameters ---
    IN_CHANNELS = 2
    HIDDEN_DIM = 128
    MASK_RATIO = 0.45
    
    # --- Data Shape ---
    NUM_SAMPLES = 128 # Total number of "scenes" in our dataset
    BATCH_SIZE = 16 
    NUM_POLYLINES = 1024
    NUM_POINTS = 20
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 50
    
    # --- Instantiate Model Components ---
    encoder = PolylineEncoder(in_channels=IN_CHANNELS, hidden_dim=HIDDEN_DIM)
    decoder = PointwisePredictionDecoder(hidden_dim=HIDDEN_DIM, point_channels=IN_CHANNELS)
    ssl_model = SelfSupervisedModel(encoder, decoder, mask_ratio=MASK_RATIO).to(device)

    optimizer = torch.optim.Adam(ssl_model.parameters(), lr=LEARNING_RATE)

    # --- Create Dataset and DataLoader ---
    use_gpu = device.type == 'cuda'
    dataset = PolylineDataset(NUM_SAMPLES, NUM_POLYLINES, NUM_POINTS, IN_CHANNELS)
    train_dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # --- Training Loop ---
    print("\n🔥 Starting training with structured data...")
    print(f"Dataset size: {len(dataset)} samples, Batch size: {BATCH_SIZE}, Steps per epoch: {len(train_dataloader)}")
    
    loss_history = []
    for epoch in range(NUM_EPOCHS):
        ssl_model.train()
        epoch_loss = 0.0

        # Use tqdm for a clean progress bar
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)
        for batch_data in progress_bar:
            # Move the batch to the selected device
            polylines_batch = batch_data.to(device, non_blocking=use_gpu)
            
            optimizer.zero_grad()
            loss, _ = ssl_model(polylines_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=f'{loss.item():.6f}')
        
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        loss_history.append(avg_epoch_loss)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_epoch_loss:.6f}")

    print("✅ Training complete.")

    # --- Visualize Results ---
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
