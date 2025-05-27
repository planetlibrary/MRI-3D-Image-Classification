# Helper functions (metrics, visualizations, etc.)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import datetime
import pyfiglet
from prettytable import PrettyTable
import time
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from torchinfo import summary
from typing import Union
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def project_intro(name):
    ascii_art = pyfiglet.figlet_format(name,font="big")
    print(ascii_art)

def model_summary(model, input_size, **kwargs):
    print("Model Details as follows: 👇🏽")
    # Get summary data
    info = summary(model, input_size=input_size, verbose=0, **kwargs)

    # Left: Model layers table
    model_table = PrettyTable()
    model_table.field_names = ["Layer (type)", "Output Shape", "Param #"]
    for layer in info.summary_list:
        model_table.add_row([
            layer.class_name,
            str(layer.output_size),
            f"{layer.num_params:,}"
        ])
    model_lines = model_table.get_string().splitlines()

    # Right: Stats table
    stats_table = PrettyTable()
    stats_table.field_names = ["Stats", "Count"]
    stats_table.add_row(["Total", f"{info.total_params:,}"])
    stats_table.add_row(["Trainable", f"{info.trainable_params:,}"])
    stats_table.add_row(["Non-trainable", f"{info.total_params - info.trainable_params:,}"])
    stats_table.add_row(["Total mult-adds", f"{info.total_mult_adds:,}"])
    stats_lines = stats_table.get_string().splitlines()

    # Pad the shorter table
    max_len = max(len(model_lines), len(stats_lines))
    model_lines += [" " * len(model_lines[0])] * (max_len - len(model_lines))
    stats_lines += [" " * len(stats_lines[0])] * (max_len - len(stats_lines))

    # Combine line by line
    for m_line, s_line in zip(model_lines, stats_lines):
        print(f"{m_line}   {s_line}")


def training_knobs(cfg_dict):
    print("Training Knobs:  👇🏽")
    items = list(cfg_dict.items())
    mid = len(items) // 2 + len(items) % 2  # Left half gets extra if odd

    left_items = items[:mid]
    right_items = items[mid:]

    # Prepare row strings
    left_table = PrettyTable()
    left_table.field_names = ["Parameter", "Value"]
    for k, v in left_items:
        left_table.add_row([k, v])
    left_lines = left_table.get_string().splitlines()

    right_table = PrettyTable()
    right_table.field_names = ["Parameter", "Value"]
    for k, v in right_items:
        right_table.add_row([k, v])
    right_lines = right_table.get_string().splitlines()

    # Pad shorter table if needed
    max_lines = max(len(left_lines), len(right_lines))
    left_lines += [" " * len(left_lines[0])] * (max_lines - len(left_lines))
    right_lines += [" " * len(right_lines[0])] * (max_lines - len(right_lines))

    # Print side by side
    for l, r in zip(left_lines, right_lines):
        print(f"{l}   {r}")


def compute_metrics(y_true, y_pred):
    """Compute common classification metrics and return them as a dictionary.
    
    Calculates accuracy, precision, recall, and F1 score using scikit-learn metrics.
    For precision, recall, and F1, macro averaging is used to handle multi-class classification.
    
    Args:
        y_true (array-like): Ground truth (correct) target values.
        y_pred (array-like): Estimated targets as returned by a classifier.
        
    Returns:
        dict: Dictionary containing the following metrics:
            - accuracy (float): Accuracy score
            - precision (float): Macro-averaged precision score
            - recall (float): Macro-averaged recall score
            - f1_score (float): Macro-averaged F1 score
            
    Example:
        >>> y_true = [0, 1, 0, 1]
        >>> y_pred = [0, 0, 0, 1]
        >>> compute_metrics(y_true, y_pred)
        {
            'accuracy': 0.75,
            'precision': 0.75,
            'recall': 0.75,
            'f1_score': 0.6666666666666666
        }
        
    Note:
        - Uses 'macro' averaging for multi-class metrics (treats all classes equally)
        - Sets zero_division=0 to handle cases with no predicted samples
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='macro', zero_division=0),
        "recall": recall_score(y_true, y_pred, average='macro', zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average='macro', zero_division=0),
    }


def save_checkpoint(model: torch.nn.Module, path: str | Path) -> None:
    """Save a model's state dictionary to the specified file path.
    
    Preserves the model's learned parameters for later reuse or continued training.
    Uses PyTorch's native serialization method for efficient storage.
    
    Args:
        model (torch.nn.Module): PyTorch model instance to be saved
        path (str | Path): File path where the model weights will be stored.
            Should have .pt or .pth file extension
            
    Returns:
        None: The model weights are written to disk but nothing is returned
        
    Example:
        >>> model = torch.nn.Linear(10, 2)
        >>> save_checkpoint(model, "trained_model.pth")
        >>> loaded_model = torch.nn.Linear(10, 2)
        >>> loaded_model.load_state_dict(torch.load("trained_model.pth"))
        
    Notes:
        - Creates a full snapshot of model parameters (weights and biases)
        - Does not save optimizer state or training progress - only model weights
        - Ensure directory structure exists before saving
        - File can be loaded with torch.load() and applied to a compatible architecture
        - For multi-GPU models, considers model wrapped in DataParallel/MultiGPU wrappers
    """
    torch.save(model.state_dict(), path)



def save_at_n_epoch(
    model: torch.nn.Module,
    epoch: int,
    path: Union[str, Path],
    config: object
) -> str:
    """Conditionally save model checkpoint at specified epoch intervals and final epoch.
    
    Checks if either:
    1. Current epoch is a multiple of `save_every_n_epochs` (when > 0), or
    2. Current epoch is the final epoch specified in config
    Saves model weights if either condition is met.

    Args:
        model: PyTorch model instance to save
        epoch: Current epoch number (1-based or 0-based depending on training loop)
        path: Destination path for model checkpoint. Recommended extensions: .pt, .pth
        config: Configuration object with attributes:
            - save_every_n_epochs (int): Save interval (0 disables intermediate saves)
            - num_epochs (int): Total number of training epochs

    Returns:
        str: Informational message if saved, empty string otherwise. Message contains
            a leading newline for cleaner output formatting in training logs.

    Example:
        >>> class Config:
        ...     save_every_n_epochs = 2
        ...     num_epochs = 5
        >>> config = Config()
        >>> save_at_n_epoch(model, 2, "model.pth", config)
        '\nThe model is saved at 2 epoch'
        >>> save_at_n_epoch(model, 5, "model.pth", config)
        '\nThe model is saved at 5 epoch'
        >>> save_at_n_epoch(model, 3, "model.pth", config)
        ''

    Notes:
        - Uses save_checkpoint() for actual saving (only saves model state_dict)
        - Always saves at final epoch regardless of save_every_n_epochs value
        - Returns message with leading newline to separate from progress bars
        - Ensure parent directories exist before calling
        - Set config.save_every_n_epochs=0 to only save final model
        - Epoch numbering should match training loop's actual epoch count
    """
    if (config.save_every_n_epochs > 0 and (epoch) % config.save_every_n_epochs == 0) or epoch == config.num_epochs:
        save_checkpoint(model, path)
        return f'\nThe model is saved at {epoch} epoch'
    return ''


def get_date() -> str:
    """Get current date and time in standardized format.
    
    Returns:
        str: Current local date/time formatted as ISO 8601 string (YYYY-MM-DD HH:MM:SS)
        
    Example:
        >>> get_date()
        '2023-08-05 15:30:00'
        
    Notes:
        - Uses local system time and datetime
        - Consistent 24-hour time format
        - Fixed length format (19 characters) for reliable parsing
        - Suitable for timestamps in logging or file naming
    """
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def get_time_diff(start: float, end: float) -> str:
    """Calculate and format time difference between two timestamps.
    
    Converts the absolute difference between two timestamps into a human-readable
    string using appropriate units (seconds, minutes, hours). Automatically selects
    the largest suitable unit for representation.

    Args:
        start: Start timestamp in seconds (e.g., from time.time())
        end: End timestamp in seconds (e.g., from time.time())

    Returns:
        str: Formatted time difference with unit, rounded to 3 decimal places.
            Possible formats:
            - "X.XXX seconds" (for differences < 60 seconds)
            - "X.XXX minutes" (for differences < 1 hour)
            - "X.XXX hours" (for differences >= 1 hour)

    Example:
        >>> get_time_diff(1622505600, 1622505600.123)
        '0.123 seconds'
        >>> get_time_diff(1622505600, 1622505660)  # 60 seconds difference
        '1.0 minutes'
        >>> get_time_diff(1622505600, 1622509200)  # 3600 seconds (1 hour)
        '1.0 hours'
        >>> get_time_diff(1622505600, 1622505600)  # Zero difference
        '0.0 seconds'

    Notes:
        - Always returns absolute difference (order of timestamps doesn't matter)
        - Uses 3 decimal places for all measurements
        - Units are always pluralized ("1.0 minutes" instead of "1.0 minute")
        - Thresholds: 
            60 seconds = 1 minute boundary
            3600 seconds = 1 hour boundary
    """
    interval = abs(end - start)  # ensure positive interval
        
    if interval < 60:
        return f"{round(interval, 3)} seconds"
    elif interval < 3600:
        minutes = round(interval / 60, 3)
        return f"{minutes} minutes"
    else:
        hours = round(interval / 3600, 3)
        return f"{hours} hours"



def print_dist(*data):
    print('Class Distribution as follows👇🏽: ')
    table = PrettyTable()
    table.field_names = ["Class", "Train", "Test", "Validation"]
    for cls in ['CN', 'MCI', 'AD']:
        table.add_row([cls, data[0][cls], data[1][cls], data[2][cls]])

    print(table)

def plot_metrics(config):
    # Load metrics from JSON
    metric_path = os.path.join(config.metrics_dir, 'metrics.json')
    with open(metric_path, 'r') as f:
        metrics = json.load(f)

    # Create DataFrame
    epochs = list(range(1, len(metrics['train_loss']) + 1))
    train_df = pd.DataFrame({
        'epoch': epochs,
        'loss': metrics['train_loss'],
        'accuracy': metrics['train_accuracy'],
        'precision': metrics['train_precision'],
        'recall': metrics['train_recall'],
        'f1_score': metrics['train_f1_score']
    })

    test_df = pd.DataFrame({
        'epoch': epochs,
        'loss': metrics['test_loss'],
        'accuracy': metrics['test_accuracy'],
        'precision': metrics['test_precision'],
        'recall': metrics['test_recall'],
        'f1_score': metrics['test_f1_score']
    })

    # Melt DataFrames for easier Seaborn plotting
    train_melted = train_df.melt(id_vars='epoch', var_name='metric', value_name='scores')
    test_melted = test_df.melt(id_vars='epoch', var_name='metric', value_name='scores')

    figure_dir = config.figures_dir
    os.makedirs(figure_dir, exist_ok=True)

    # Plot training metrics
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=train_melted, x='epoch', y='scores', hue='metric')
    plt.title('(a). Training Metrics vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend(title='Metrics')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, 'train_metrics.png'))
    plt.close()

    # Plot testing metrics
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=test_melted, x='epoch', y='scores', hue='metric')
    plt.title('(b). Testing Metrics vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend(title='Metrics')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, 'test_metrics.png'))
    plt.close()


def plot_cm(y_true, y_pred, config, epoch, phase = 'Train'):
    # confusion matrix plot

    cm = confusion_matrix(y_true, y_pred)
    labels = ['CN','MCI','AD']

    figure_dir = config.figures_dir
    os.makedirs(figure_dir, exist_ok=True)
    
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels = labels)
    plt.xlabel("Prediction")
    plt.ylabel("Actual")
    plt.title(f"{phase}-Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir,f'{phase.lower()}_confusion_matrix_{epoch}.png'))
    plt.close()

def plot_save_config(config):
    def convert(v):
        if isinstance(v, Path):
            return str(v)
        elif isinstance(v, torch.device):
            return str(v)
        else:
            return v

    # Convert all values in the config dictionary
    converted_dict = {k: convert(v) for k, v in config.__dict__.items()}

    obj_dict = {
        "configs": converted_dict
    }

    # Optional: Check types
    # check = {k: [v, type(v)] for k, v in converted_dict.items()}
    # print("Type Check:", check)

    # Save to JSON
    config_path = os.path.join(config.metrics_dir, 'configs.json')
    os.makedirs(config.metrics_dir, exist_ok=True)  # Ensure directory exists

    with open(config_path, 'w') as f:
        json.dump(obj_dict, f, indent=4)
