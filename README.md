# Self-Supervised Polyline Encoder for Map Data

This project provides a pipeline for training a map encoder on real-world OpenStreetMap (OSM) data. The primary purpose is to produce a PoinNet style Polyline Encoder model that can convert complex, variable-length road networks into fixed-size feature vectors.

To ensure the encoder can learn meaningful representations, it is trained on a self-supervised task: masked point prediction. The model learns to reconstruct missing points in a polyline based on the context of the surrounding map scene. The resulting pre-trained encoder can be used as a component for downstream tasks in robotics and autonomous systems, such as motion forecasting or behavior prediction.

## Installation

This project uses `uv` for fast and reliable dependency management. The required packages are listed in the `pyproject.toml` file.

1. Clone and cd into the repository:

2. Create and activate the virtual environment:
    `uv` will create a new virtual environment in a `.venv` directory in your project root.

    ```bash
    # Create the virtual environment
    uv venv

    # Activate the environment (on Linux/macOS)
    source .venv/bin/activate
    ```

3. Install dependencies:
    `uv sync` will read the dependencies from the `pyproject.toml` file and install them into your active environment.

    ```bash
    # Sync the dependencies
    uv sync
    ```

## Usage

The main script, `main.py`, is controlled via command-line arguments. It handles data acquisition, training, checkpointing, and visualization.

1. ***Build the Dataset***
    The first time you run the script, it will automatically download map data from OpenStreetMap for several locations in San Francisco, process it into scenes, and save it to the `./data/` directory.

    To force a rebuild of the dataset (e.g., if you change the locations or processing logic), use the `--force-rebuild` flag:

    ```bash
    python main.py --force-rebuild

    ```

2. ***Train the Model***
    To start or resume training, use the `--train` flag. The script will automatically load the latest checkpoint from `./checkpoints/` if one exists.

    ```bash
    # Start training from scratch or resume from the last epoch
    python main.py --train
    ```

    The trainer will save two checkpoints:

    - checkpoints/latest_checkpoint.pt: The state of the model and optimizer at the end of the most recent epoch.
    - checkpoints/best_model.pt: The model state that achieved the lowest validation loss.

3. ***Generate Visualizations***
    To generate visualizations using the best-trained model, use the `--visualize` flag.

    ```bash
    # Load the best model and create visualizations
    python main.py --visualize
    ```

    This will create a gallery of up to 10 example predictions in the `./visualizations/` directory.

### Default Behavior

If you run the script with no arguments, it will behave intelligently:

- **If no trained model is found:** It will automatically start the training process.
- **If a trained model exists:** It will automatically run the visualization process.

```bash
# Will train if no model exists, otherwise will visualize
python main.py
```

### Cleanup

Use the `--clean` flag to remove all generated artifacts and exit. This deletes the dataset, checkpoints, visualizations, and the training loss plot. This replaces manual commands like `rm -rf data visualizations checkpoints`.

```bash
python main.py --clean
```

## Running Tests

The project includes a suite of unit and integration tests to verify the functionality of the data pipeline and model components.

To run the tests, execute the following command from the root directory of the project:

```bash
python -m unittest test/test_pipeline.py
```

The tests will use a self-contained dummy data generator and will create and clean up their own temporary checkpoint directory, so they will not interfere with your main training artifacts.
