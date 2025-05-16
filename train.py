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
from models import get_model
from utils import *

def train_one_epoch(model, loader, criterion, optimizer, scheduler, device):
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

    scheduler.step()

    metrics = compute_metrics(all_labels, all_preds)
    avg_loss = running_loss / total
    # accuracy = correct / total
    return avg_loss, metrics, all_labels, all_preds


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
    return avg_loss, metrics, all_labels, all_preds





    

if __name__ == '__main__':
    # fs = folder_structure()
    cfg = Config()
    parser = argparse.ArgumentParser(description="Train a 3D image classification model.")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to root dataset.')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save logs and checkpoints')
    parser.add_argument('--model_name', type=str, default='ViTForClassification')
    parser.add_argument('--epochs', type=int, default=cfg.num_epochs)
    parser.add_argument('--batch_size', type=int, default=cfg.batch_size)
    args = parser.parse_args()

    project_intro(cfg.project_name)


    os.makedirs(os.path.join(args.output_dir,f'{cfg.model_name}', 'tensorboard'), exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir,f'{cfg.model_name}', 'tensorboard'))

    device = torch.device(cfg.device)

    # Update config dynamically if needed
    cfg.batch_size = args.batch_size
    cfg.model_name = args.model_name
    cfg.num_epochs = args.epochs
    cfg.train_dir = os.path.join(args.data_dir, 'Train')
    cfg.test_dir = os.path.join(args.data_dir, 'Test')

    # print(cfg.num_epochs)

    train_loader, val_loader, test_loader = get_dataloaders(
        cfg.train_dir, cfg.val_dir, cfg.test_dir,
        batch_size=args.batch_size,
        num_workers=cfg.num_workers
    )

    # print(device)

    model = get_model(cfg.model_name, cfg).to(device)
    print(f"Loaded {cfg.model_name} successfully on {next(model.parameters()).device}")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40, 50], gamma=0.7)

    best_test_loss = np.inf
    best_test_acc = 0
    
    history = {
    'train_loss': [],
    'test_loss': [],
    'train_accuracy': [],
    'train_precision': [],
    'train_recall': [],
    'train_f1_score': [],
    'test_accuracy': [],
    'test_precision': [],
    'test_recall': [],
    'test_f1_score': [],
    }
    
    os.makedirs(cfg.logs_dir, exist_ok=True)
    with open(os.path.join(cfg.logs_dir, 'log.txt'), 'w') as f:
        f.write(f"The training started at: {get_date()}\n")

    f = open(os.path.join(cfg.logs_dir, 'terminal.log'), 'w')
    f.close()

    for epoch in range(1, args.epochs + 1):

        start = time.time()
        
        print(f'\nEpoch {epoch}/{args.epochs}')
        train_loss, train_metrics, train_labels, train_preds = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        test_loss, test_metrics, test_labels, test_preds = evaluate(model, test_loader, criterion, device, phase="Test")
        
        # Optionally log the learning rate
        current_lr = optimizer.param_groups[0]['lr']

        plot_cm(train_labels, train_preds, cfg, epoch,  phase='Train')
        plot_cm(test_labels, test_preds, cfg, epoch, phase='Test')

        

        # TensorBoard logging
        for metric_name in train_metrics:
            writer.add_scalar(f'{metric_name}/train', train_metrics[metric_name], epoch)
            writer.add_scalar(f'{metric_name}/test', test_metrics[metric_name], epoch)
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar("LR", current_lr, epoch)


        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        
        # Save metrics       
        for metric_name in train_metrics:
            history[f'train_{metric_name}'].append(train_metrics[metric_name])
            history[f'test_{metric_name}'].append(test_metrics[metric_name])

        # save checkpoint
        os.makedirs(cfg.checkpoints_dir, exist_ok=True)
        save_path = os.path.join(cfg.checkpoints_dir, f'model_checkpt_{epoch}.pth')
        save_status = save_at_n_epoch(model, epoch, save_path, cfg)

        log_txt = f"""Train Loss: {train_loss:.4f} | Accuracy: {train_metrics['accuracy']:.4f} | Precision: {train_metrics['precision']:.4f} | Recall: {train_metrics['recall']:.4f} | F1-score: {train_metrics['f1_score']:.4f}\nTest Loss: {test_loss:.4f} | Accuracy: {test_metrics['accuracy']:.4f} | Precision: {test_metrics['precision']:.4f} | Recall: {test_metrics['recall']:.4f} | F1-score: {test_metrics['f1_score']:.4f}| Learning Rate: {current_lr:.6f}"""

        with open(os.path.join(cfg.logs_dir, 'log.txt'), 'a') as f:
            f.write(f"\n[{get_date()}]: "+log_txt+'\n')



        end = time.time()
        print(log_txt+f"| Time Taken: {get_time_diff(start, end)}"+f"{save_status}\n")

        with open(os.path.join(cfg.logs_dir, 'terminal.log'), 'a') as f:
            f.write(f"\n[{get_date()}]: "+log_txt+f"| Time Taken: {get_time_diff(start, end)}"+f"{save_status}\n")


        # Save best checkpoint
        if test_loss < best_test_loss and test_metrics['accuracy'] > best_test_acc:
            best_test_loss = test_loss
            best_test_acc = test_metrics['accuracy']
            save_path = os.path.join(cfg.checkpoints_dir, 'best_model.pth')
            save_checkpoint(model, save_path)


    # Save final metrics
    os.makedirs(cfg.metrics_dir, exist_ok=True)
    with open(os.path.join(cfg.metrics_dir, 'metrics.json'), 'w') as f:
        json.dump(history, f, indent=4)
    
    plot_metrics(cfg)
    plot_save_config(cfg)

    print(f"\nTraining completed. Best Test Loss: {best_test_loss:.4f} | Best Test Accuracy: {best_test_acc:.4f}")
    writer.close()
