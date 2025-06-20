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
from utils.utils import *
from torchinfo import summary
from utils.model_tracker import ModelChangeTracker
import torchio as tio
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_





def train_one_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    running_loss = 0
    correct, total = 0, 0

    all_preds, all_labels = [], []

    pbar = tqdm(
            loader,
            desc="Training",
            unit="batch",
        )
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device,non_blocking=True)

        # if torch.isnan(inputs).any():
        #     print(inputs.shape)

        optimizer.zero_grad()
        outputs = model(inputs)

        # if torch.isnan(outputs).any():
        #    print(outputs.shape, torch.isnan(inputs).any())

        # print(outputs.shape, labels.shape)
        # print(outputs, labels)


        loss = criterion(outputs, labels)
        loss.backward()
        # efficient gradient norm via clip_grad_norm_ (no clipping if max_norm is huge)
        total_norm = clip_grad_norm_(model.parameters(), max_norm=1e9, norm_type=2)
        # # compute total gradient norm
        # total_norm = 0.0
        # for p in model.parameters():
        #     if p.grad is not None:
        #         total_norm += p.grad.data.norm(2).item() ** 2
        # total_norm = total_norm ** 0.5
        # print(f"Gradient L2-norm before step: {total_norm:.6g}")

        

        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(
                loss=f"{running_loss/total:.4f}",
                acc=f"{correct/total:.4f}",
                grad_norm=f"{total_norm:.4g}",
            )
        # print(preds, labels)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    scheduler.step()
    torch.cuda.empty_cache()

    metrics = compute_metrics(all_labels, all_preds)
    avg_loss = running_loss / total
    # accuracy = correct / total
    return avg_loss, metrics, all_labels, all_preds


def evaluate(model, loader, criterion, device, phase="Validation"):
    model.eval()
    running_loss = 0
    correct, total = 0, 0

    all_preds, all_labels = [], []

    pbar = tqdm(
            loader,
            desc=f"{phase}",
            unit="batch",
        )

    with torch.no_grad():
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix(
                loss=f"{running_loss/total:.4f}",
                acc=f"{correct/total:.4f}",
            )

    torch.cuda.empty_cache()
    metrics = compute_metrics(all_labels, all_preds)

    avg_loss = running_loss / total
    # accuracy = correct / total
    return avg_loss, metrics, all_labels, all_preds


    

if __name__ == '__main__':
    # fs = folder_structure()
    parser = argparse.ArgumentParser(description="Train a 3D image classification model.")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to root dataset.')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save logs and checkpoints')
    parser.add_argument('--model_name', type=str, default='ViTForClassification')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=2)

    prj_start = time.time() 
    args = parser.parse_args()

    cfg = Config(args.model_name)
    project_intro(cfg.project_name)
    

    torch.cuda.empty_cache()


    os.makedirs(os.path.join(args.output_dir,f'{cfg.model_name}', 'tensorboard'), exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir,f'{cfg.model_name}', 'tensorboard'))

    device = torch.device(cfg.device)

    # Update config dynamically if needed
    cfg.batch_size = args.batch_size
    cfg.num_epochs = args.epochs
    cfg.train_dir = os.path.join(args.data_dir, 'Train')
    cfg.test_dir = os.path.join(args.data_dir, 'Test')

    
    tracker = ModelChangeTracker(
        kl_threshold=1e-4,
        track_kl=False,
        track_norms=True,
        track_grads=True,
        config = cfg
    )

    training_knobs(cfg.__dict__)

    # print(cfg.num_epochs)

    transform = tio.Compose([
        tio.ZNormalization(),
        tio.RandomFlip(axes=('LR',), flip_probability=0.5),
        tio.RandomAffine(scales=(0.9, 1.1), degrees=10),
    ])

    train_loader, val_loader, test_loader, label_dist = get_dataloaders(
        cfg.train_dir, cfg.val_dir, cfg.test_dir,
        batch_size=args.batch_size,
        num_workers=cfg.num_workers,
        transforms=None
    )

    train_dist, _, _ = label_dist
    # print(train_dist.values())
    total_labels = sum(i for i in train_dist.values())
    weight_train = torch.tensor([val for key, val in train_dist.items()], dtype=torch.float32).to(device)
    # print(weight_train)

    # Memory optimization configurations
    # torch.backends.cudnn.benchmark = True  # Enable CuDNN auto-tuner
    # torch.set_float32_matmul_precision('high')  # For Ampere+ GPUs
    
    # # Add memory fragmentation prevention
    # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    # print(device)

    model = get_model(cfg.model_name, cfg).to(device)
    model_summary(model, [cfg.batch_size, cfg.input_channels, cfg.image_size, cfg.image_size, cfg.image_size])
    # res = summary(model, [cfg.batch_size, cfg.input_channels, cfg.image_size, cfg.image_size, cfg.image_size])

    try:
        prev_model_path = os.path.join(cfg.checkpoints_dir, 'best_model.pth')
        model.load_state_dict(torch.load(prev_model_path))
    except FileNotFoundError as e:
        print('Pre-trained model path is not there, training started from scratch ...')

    print(f"Loaded {cfg.model_name} successfully on {next(model.parameters()).device}")
    criterion = nn.CrossEntropyLoss(weight_train)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 50, 100, 120, 150, 180, 200, 230, 250, 270, 280], gamma=0.72)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=2e-7)

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

        tracker.update(model, epoch)

        
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
        save_path = os.path.join(cfg.checkpoints_dir, f'model_checkpt.pth')
        save_status = save_at_n_epoch(model, epoch, save_path, cfg)

        log_txt = f"""Train Loss: {train_loss:.4f} | Accuracy: {train_metrics['accuracy']:.4f} | Precision: {train_metrics['precision']:.4f} | Recall: {train_metrics['recall']:.4f} | F1-score: {train_metrics['f1_score']:.4f}\nTest Loss: {test_loss:.4f} | Accuracy: {test_metrics['accuracy']:.4f} | Precision: {test_metrics['precision']:.4f} | Recall: {test_metrics['recall']:.4f} | F1-score: {test_metrics['f1_score']:.4f}| Learning Rate: {current_lr:.6f}"""

        with open(os.path.join(cfg.logs_dir, 'log.txt'), 'a') as f:
            f.write(f"\n[{get_date()}]: "+log_txt+'\n')



        end = time.time()
        print(log_txt+f"| Time Taken: {get_time_diff(start, end)}"+f"{save_status}\n")

        with open(os.path.join(cfg.logs_dir, 'terminal.log'), 'a') as f:
            f.write(f"\n[{get_date()}]: "+log_txt+f"| Time Taken: {get_time_diff(start, end)}"+f"{save_status}\n")


        # Save best checkpoint
        if test_loss < best_test_loss or test_metrics['accuracy'] > best_test_acc:
            best_test_loss = test_loss
            best_test_acc = test_metrics['accuracy']
            save_path = os.path.join(cfg.checkpoints_dir, 'best_model.pth')
            save_checkpoint(model, save_path)
        
        # except torch.cuda.OutOfMemoryError:
        #     print(f"\nOOM at epoch {epoch}. Attempting recovery...")
        #     torch.cuda.empty_cache()
        #     args.batch_size = max(1, args.batch_size // 2)
        #     print(f"Reducing batch size to {args.batch_size} and retrying...")
            
        #     # # Reinitialize data loaders with new batch size
        #     # train_loader, val_loader, test_loader = get_dataloaders(
        #     #     cfg.train_dir, cfg.val_dir, cfg.test_dir,
        #     #     batch_size=args.batch_size,
        #     # )
            # continue


    # Save final metrics
    os.makedirs(cfg.metrics_dir, exist_ok=True)
    with open(os.path.join(cfg.metrics_dir, 'metrics.json'), 'w') as f:
        json.dump(history, f, indent=4)
    
    plot_metrics(cfg)
    plot_single_metrics(cfg)
    plot_save_config(cfg)

    prj_end = time.time()
    print(f"\nTraining completed. Best Test Loss: {best_test_loss:.4f} | Best Test Accuracy: {best_test_acc:.4f} | Total time taken: {get_time_diff(prj_start, prj_end)}")
    
    # print(tracker.get_param_norm_log())
    # print(tracker.get_grad_norm_log())
    writer.close()
