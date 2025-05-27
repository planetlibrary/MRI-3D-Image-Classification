# dataloader.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import glob
import os
from utils.utils import print_dist

# Optional: Add transformations later like torchio, torchvision, etc.

class FolderDataset(Dataset):
    def __init__(self, folder: str, transform=None):
        self.folder = Path(folder)
        self.image_paths = sorted(glob.glob(f'{self.folder}/**/*.pt', recursive=True))
        self.class_to_idx = {'CN': 0, 'MCI': 1, 'AD': 2}
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def _extract_label(self, path: str) -> int:
        for key in self.class_to_idx:
            if f'/{key}/' in path.replace('\\', '/'):
                return self.class_to_idx[key]
        raise ValueError(f"Unknown label in path: {path}")

    def __getitem__(self, idx: int):
        file_path = self.image_paths[idx]
        # print(f'File Path: {file_path}')
        tensor = torch.load(file_path)  # Should be a preprocessed 3D tensor
        label = self._extract_label(file_path)
        # print(label)

        if self.transform:
            tensor = self.transform(tensor)

        return tensor, label

    def label_distribution(self):
        dist = {key: 0 for key in self.class_to_idx}
        for path in self.image_paths:
            label = self._extract_label(path)
            class_name = list(self.class_to_idx.keys())[list(self.class_to_idx.values()).index(label)]
            dist[class_name] += 1
        return dist


def get_dataloaders(train_path, val_path, test_path, batch_size=8, num_workers=4, transforms=None):
    train_dataset = FolderDataset(train_path, transform=transforms)
    val_dataset = FolderDataset(val_path, transform=transforms)
    test_dataset = FolderDataset(test_path, transform=transforms)

    # print(train_dataset[0])
    train_dist = train_dataset.label_distribution()
    val_dist = val_dataset.label_distribution()
    test_dist = test_dataset.label_distribution()

    
    print_dist(train_dist, test_dist, val_dist)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


if __name__=='__main__':
    # from dataloader import get_dataloaders
    from config import Config

    train_loader, val_loader, test_loader = get_dataloaders(
        Config.train_dir, Config.val_dir, Config.test_dir,
        batch_size=Config.batch_size,
        num_workers=Config.num_workers
    )

    # print(train_loader)
