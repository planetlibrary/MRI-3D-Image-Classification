# CLI-enabled training pipeline
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
import textwrap
import os
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from config import Config
from data_loader import get_dataloaders
from model import get_model

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0
    correct, total = 0, 0

    all_preds, all_labels = [], []

    for inputs, labels in tqdm(loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    metrics = compute_metrics(all_labels, all_preds)
    avg_loss = running_loss / total
    # accuracy = correct / total
    return avg_loss, metrics


def evaluate(model, loader, criterion, device, phase="Validation"):
    model.eval()
    running_loss = 0
    correct, total = 0, 0

    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc=phase):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    metrics = compute_metrics(all_labels, all_preds)

    avg_loss = running_loss / total
    # accuracy = correct / total
    return avg_loss, metrics

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



def folder_structure():
    # return "├── Train/\n"\
    # "│ ├── AD/\n"\
    # "│ ├── CN/\n"\
    # "│ └── MCI/\n"\
    # "├── Val/\n"\
    # "│ ├── AD/\n"\
    # "│ ├── CN/\n"\
    # "│ └── MCI/\n"\
    # "└── Test/\n"\
    # "│ ├── AD/\n"\
    # "│ ├── CN/\n"\
    # "│ └── MCI/\n"

    return textwrap.dedent("""
    Expected directory structure:\n
    ├── Train/\n
    │ ├── AD/\n
    │ ├── CN/\n
    │ └── MCI/\n
    ├── Val/\n
    │ ├── AD/\n
    │ ├── CN/\n
    │ └── MCI/\n
    └── Test/\n
     ├── AD/\n
     ├── CN/\n
     └── MCI/
    """)

    

if __name__ == '__main__':
    # fs = folder_structure()
    parser = argparse.ArgumentParser(description="Train a 3D image classification model.\n")
    parser.add_argument('--data_dir', type=str, required=True, help=f'Path to root dataset.')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save logs and checkpoints')
    parser.add_argument('--epochs', type=int, default=Config.num_epochs)
    parser.add_argument('--batch_size', type=int, default=Config.batch_size)
    args = parser.parse_args()

    os.makedirs(os.path.join(args.output_dir, 'tensorboard'), exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard'))

    device = torch.device(Config.device)

    # Update config dynamically if needed
    Config.batch_size = args.batch_size
    Config.num_epochs = args.epochs
    Config.train_dir = os.path.join(args.data_dir, 'Train')
    # Config.val_dir = os.path.join(args.data_dir, 'Val')
    Config.test_dir = os.path.join(args.data_dir, 'Test')

    print(Config.num_epochs)

    train_loader, val_loader, test_loader = get_dataloaders(
        Config.train_dir, Config.val_dir, Config.test_dir,
        batch_size=args.batch_size,
        num_workers=Config.num_workers
    )

    model = get_model(Config.model_name, Config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)

    best_test_loss = np.inf
    history = {
    'train_loss': [],
    # 'val_loss': [],
    'test_loss': [],
    'train_accuracy': [],
    'train_precision': [],
    'train_recall': [],
    'train_f1_score': [],
    # 'val_accuracy': [],
    # 'val_precision': [],
    # 'val_recall': [],
    # 'val_f1_score': [],
    'test_accuracy': [],
    'test_precision': [],
    'test_recall': [],
    'test_f1_score': [],
    }

    for epoch in range(1, args.epochs + 1):
        print(f'\nEpoch {epoch}/{args.epochs}')
        train_loss, train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        # val_loss, val_metrics = evaluate(model, val_loader, criterion, device, phase="Validation")
        test_loss, test_metrics = evaluate(model, test_loader, criterion, device, phase="Test")

        print(f"\nTrain Loss: {train_loss:.4f} | Acc: {train_metrics['accuracy']:.4f}")
        # print(f'Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f}')
        print(f"Test  Loss: {test_loss:.4f} | Acc: {test_metrics['accuracy']:.4f}")


        # TensorBoard logging
        for metric_name in train_metrics:
            writer.add_scalar(f'{metric_name}/train', train_metrics[metric_name], epoch)
            # writer.add_scalar(f'{metric_name}/val', val_metrics[metric_name], epoch)
            writer.add_scalar(f'{metric_name}/test', test_metrics[metric_name], epoch)
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        # # writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)
        # writer.add_scalar('Accuracy/train', train_acc, epoch)
        # # writer.add_scalar('Accuracy/val', val_acc, epoch)
        # writer.add_scalar('Accuracy/test', test_acc, epoch)

        history['train_loss'].append(train_loss)
        # history['val_loss'].append(val_loss)
        history['test_loss'].append(test_loss)
        
        # Save metrics       
        for metric_name in train_metrics:
            history[f'train_{metric_name}'].append(train_metrics[metric_name])
            # history[f'val_{metric_name}'].append(val_metrics[metric_name])
            history[f'test_{metric_name}'].append(test_metrics[metric_name])

        # save checkpoint
        save_path = os.path.join(args.output_dir, 'model_checkpt.pth')
        save_at_n_epoch(model, epoch, save_path, Config)

        # Save best checkpoint
        if test_loss < best_test_loss and test_metrics['accuracy'] > best_val_acc:
            best_test_loss = test_loss
            best_val_acc = test_metrics['accuracy']
            save_path = os.path.join(args.output_dir, 'best_model.pth')
            save_checkpoint(model, save_path)

    # Save final metrics
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(history, f, indent=4)

    print(f"\nTraining completed. Best Test Loss: {best_test_loss:.4f} | Best Test Accuracy: {best_val_acc:.4f} | ")
    writer.close()
