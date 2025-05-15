# Configuration and hyperparameters

import os
from pathlib import Path

class Config:
    # Paths
    data_dir = Path("./data/3D")
    train_dir = data_dir / "Train"
    val_dir = data_dir / "Val"
    test_dir = data_dir / "Test"
    output_dir = Path("./outputs")
    checkpoints_dir = output_dir / "checkpoints"
    logs_dir = output_dir / "logs"
    metrics_dir = output_dir / "metrics"

    # Training Parameters
    num_epochs = 50
    batch_size = 8
    learning_rate = 1e-4
    weight_decay = 1e-5
    scheduler_step = 10
    scheduler_gamma = 0.1
    save_every_n_epochs = 10

    # Model
    project_name = "ViT-Nicara-3D"   # Name of model class in models.py
    model_name = "ViT3D"
    num_classes = 3              # AD, CN, MCI
    input_channels = 1          # Adjust based on your 3D input (e.g., grayscale)
    input_channels = 1
    img_size = 64
    patch_size = 16
    emb_dim = 128
    num_heads = 4
    num_layers = 6
    dropout = 0.1
    num_classes = 3

    # System
    device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") else "cpu"
    num_workers = 4
    seed = 42
