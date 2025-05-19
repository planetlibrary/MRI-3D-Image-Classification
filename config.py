# Configuration and hyperparameters

import os
from pathlib import Path
import torch

class Config:
    def __init__(self, model_name= 'ViTForClassification'):
        self.model_name = model_name

        # Paths
        self.data_dir = Path("./data/3D")
        self.train_dir =self.data_dir / "Train"
        self.val_dir = self.data_dir/ "Val"
        self.test_dir = self.data_dir/ "Test"
        self.output_dir = Path("./outputs")
        self.checkpoints_dir =self.output_dir /self.model_name/ "checkpoints"
        self.logs_dir = self.output_dir / self.model_name/"logs"
        self.metrics_dir = self.output_dir/ self.model_name/"metrics"
        self.figures_dir = self.output_dir/ self.model_name/"figures"

        # Training Parameters
        self.num_epochs = 50
        self.batch_size = 25
        self.learning_rate = 1e-3
        self.weight_decay = 2e-3
        self.scheduler_step = 10
        self.scheduler_gamma = 0.1

        
        self.save_every_n_epochs = 1

        # Model
        self.project_name = "ViT-Nicara-3D"   # Name of model class in models.py

        if self.model_name == "ViTForClassfication_V2":
            self.num_classes = 3              # AD, CN, MCI
            self.input_channels = 1          # Adjust based on your 3D input (e.g., grayscale)
            self.image_size = 192
            self.patch_size = 16
            self.num_heads = 4
            self.num_layers = 4
            self.hidden_size = 1728
            self.dropout = 0.1

        elif self.model_name == "ViT3D_V1":
            self.num_classes = 3              # AD, CN, MCI
            self.input_channels = 1          # Adjust based on your 3D input (e.g., grayscale)
            self.image_size = 192
            self.patch_size = 16
            self.emb_dim = 128
            self.num_heads = 4
            self.num_layers = 6
            self.hidden_size = 27
            self.dropout = 0.1


        elif self.model_name == "ViTForClassification":
            self.batch_size = 5 #3
            self.image_size = 192
            self.patch_size = 6
            self.hidden_size= 216
            self.num_hidden_layers= 2 #3
            self.num_attention_heads= 4 #8
            self.intermediate_size= 256 # 3 * 216 # 3 * hidden_size
            self.hidden_dropout_prob= 0.85 # 0.25
            self.attention_probs_dropout_prob= 0.8 #0.25
            self.initializer_range= 0.02
            self.num_classes= 3 # num_classes
            self.num_channels= 1
            self.qkv_bias= True
            self.use_faster_attention= True
            self.lr = 5e-5
            # save_model_every = 10
            # exp_name = '3D ViT Final'
            # model_name = 'Hybrid'
            # epochs = 150

        else:
            raise KeyError("Model name do not exist in the existing list: ['ViTForClassfication_V2', 'ViT3D_V1', 'ViTForClassification']")



        # System
        self.device = "cuda:1" if torch.cuda.is_available() else "cpu"
        self.num_workers = 4
        self.seed = 42
