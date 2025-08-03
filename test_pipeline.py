import unittest
import torch
from torch.utils.data import DataLoader, Dataset
from typing import List
import os
import shutil

# We need to import the core classes from your main script.
# Assuming your main script is saved as 'main.py'
from main import (
    TrainConfig,
    PolylineEncoder,
    PointwisePredictionDecoder,
    SelfSupervisedModel,
    variable_length_collate_fn, # Import the production collate_fn
    Trainer
)

# --- Dummy Data Generation (Self-contained for testing) ---

def generate_variable_length_polylines(channels: int) -> List[torch.Tensor]:
    """
    Generates a single "scene" as a list of variable-length polyline tensors,
    mimicking the structure of the real OSMDataset.
    """
    num_polylines = torch.randint(low=64, high=128, size=(1,)).item()
    scene = []
    for _ in range(num_polylines):
        num_points = torch.randint(low=10, high=20, size=(1,)).item()
        # Create a single polyline tensor
        steps = torch.randn(1, 1, num_points, channels) * 0.1
        steps[:, :, 0, :] = torch.randn(1, 1, channels)
        polyline = torch.cumsum(steps, dim=2).squeeze(0).squeeze(0)
        scene.append(polyline)
    return scene

class VariablePolylineDataset(Dataset):
    """Dataset that generates variable-length polylines for testing purposes."""
    def __init__(self, num_samples: int, channels: int):
        self.num_samples = num_samples
        self.channels = channels

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> List[torch.Tensor]:
        return generate_variable_length_polylines(self.channels)


class TestTrainingPipeline(unittest.TestCase):
    """
    Unit and integration tests for the self-supervised training pipeline.
    """

    def setUp(self):
        """Set up common components and a temporary directory for artifacts."""
        self.config = TrainConfig()
        # THE FIX: Define test-specific parameters locally instead of relying on the config object.
        self.test_num_samples = 8
        # Override config for fast testing
        self.config.batch_size = 4
        self.config.num_epochs = 1 # Run only for one epoch in tests
        self.config.checkpoint_dir = './test_checkpoints' # Use a temporary dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

    def tearDown(self):
        """Clean up the temporary checkpoint directory after tests."""
        if os.path.exists(self.config.checkpoint_dir):
            shutil.rmtree(self.config.checkpoint_dir)

    def test_model_forward_pass(self):
        """
        Tests that a single forward pass through the SelfSupervisedModel
        executes without errors and returns valid loss and prediction tensors.
        """
        dataset = VariablePolylineDataset(self.test_num_samples, self.config.in_channels)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, collate_fn=variable_length_collate_fn)
        polylines_batch, padding_mask_batch = next(iter(dataloader))
        
        encoder = PolylineEncoder(self.config.in_channels, self.config.hidden_dim)
        decoder = PointwisePredictionDecoder(self.config.hidden_dim, self.config.in_channels)
        model = SelfSupervisedModel(encoder, decoder, self.config.mask_ratio).to(self.device)
        
        polylines_batch = polylines_batch.to(self.device)
        padding_mask_batch = padding_mask_batch.to(self.device)
        
        loss, predictions = model(polylines_batch, padding_mask_batch)
        
        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.is_tensor(predictions))
        self.assertEqual(predictions.shape, polylines_batch.shape)
        print("✅ test_model_forward_pass: PASSED")

    def test_trainer_integration(self):
        """
        An integration smoke test for the Trainer class, including the validation loop.
        """
        encoder = PolylineEncoder(self.config.in_channels, self.config.hidden_dim)
        decoder = PointwisePredictionDecoder(self.config.hidden_dim, self.config.in_channels)
        model = SelfSupervisedModel(encoder, decoder, self.config.mask_ratio).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        
        # Create separate datasets and dataloaders for train and val
        train_dataset = VariablePolylineDataset(self.test_num_samples, self.config.in_channels)
        val_dataset = VariablePolylineDataset(self.test_num_samples // 2, self.config.in_channels)
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, collate_fn=variable_length_collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, collate_fn=variable_length_collate_fn)

        trainer = Trainer(model, optimizer, train_loader, val_loader, self.config, self.device)

        try:
            trainer.train()
            # Check that both loss histories were recorded correctly
            self.assertEqual(len(trainer.train_loss_history), self.config.num_epochs)
            self.assertEqual(len(trainer.val_loss_history), self.config.num_epochs)
            self.assertIsInstance(trainer.train_loss_history[0], float)
            self.assertIsInstance(trainer.val_loss_history[0], float)
            print("✅ test_trainer_integration: PASSED")
        except Exception as e:
            self.fail(f"Trainer integration test failed with exception: {e}")

    def test_checkpointing(self):
        """
        Tests that the Trainer can save and load checkpoints correctly.
        """
        # 1. Setup and run a trainer for one epoch to create a checkpoint
        encoder = PolylineEncoder(self.config.in_channels, self.config.hidden_dim)
        decoder = PointwisePredictionDecoder(self.config.hidden_dim, self.config.in_channels)
        model = SelfSupervisedModel(encoder, decoder, self.config.mask_ratio).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        train_dataset = VariablePolylineDataset(self.test_num_samples, self.config.in_channels)
        val_dataset = VariablePolylineDataset(self.test_num_samples, self.config.in_channels)
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, collate_fn=variable_length_collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, collate_fn=variable_length_collate_fn)
        
        trainer1 = Trainer(model, optimizer, train_loader, val_loader, self.config, self.device)
        trainer1.train()

        # 2. Verify that checkpoint files were created
        latest_path = os.path.join(self.config.checkpoint_dir, 'latest_checkpoint.pt')
        best_path = os.path.join(self.config.checkpoint_dir, 'best_model.pt')
        self.assertTrue(os.path.exists(latest_path))
        self.assertTrue(os.path.exists(best_path))

        # 3. Instantiate a NEW trainer and load the checkpoint
        model2 = SelfSupervisedModel(encoder, decoder, self.config.mask_ratio).to(self.device)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=self.config.learning_rate)
        trainer2 = Trainer(model2, optimizer2, train_loader, val_loader, self.config, self.device)
        trainer2.load_checkpoint()

        # 4. Assert that the state was restored correctly
        self.assertEqual(trainer2.start_epoch, self.config.num_epochs) # Should be 1 after one epoch
        self.assertNotEqual(trainer2.best_val_loss, float('inf'))
        print("✅ test_checkpointing: PASSED")


if __name__ == '__main__':
    # To run the tests, save the main script as 'main.py', this test script
    # as 'test_pipeline.py', and run 'python -m unittest test_pipeline.py'
    unittest.main()
