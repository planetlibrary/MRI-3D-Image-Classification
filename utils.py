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


def compute_metrics(y_true, y_pred):
    return {
    "accuracy": accuracy_score(y_true, y_pred),
    "precision": precision_score(y_true, y_pred, average='macro', zero_division=0),
    "recall": recall_score(y_true, y_pred, average='macro', zero_division=0),
    "f1_score": f1_score(y_true, y_pred, average='macro', zero_division=0),
    }

def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)

def save_at_n_epoch(model, epoch, path, config):

    if (config.save_every_n_epochs > 0 and (epoch) % config.save_every_n_epochs == 0) or epoch == config.num_epochs:
        save_checkpoint(model, path)
        return f'\nThe model is saved at {epoch} epoch'
    return ''

def get_date():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def get_time_diff(start, end):

    interval = abs(end - start)  # ensure positive interval
        
    if interval < 60:
        return f"{round(interval, 3)} seconds"
    elif interval < 3600:
        minutes = round(interval / 60, 3)
        return f"{minutes} minutes"
    else:
        hours = round(interval / 3600, 3)
        return f"{hours} hours"


def project_intro(name):
    ascii_art = pyfiglet.figlet_format(name,font="big")
    print(ascii_art)

def print_dist(*data):
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
