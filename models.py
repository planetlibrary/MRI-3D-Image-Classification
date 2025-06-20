# Model architecture(s)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torchinfo import summary
from config import Config
import matplotlib.pyplot as plt

from einops import rearrange
import math

class ViT3D_V1(nn.Module):
    def __init__(self, 
                 in_channels=1,
                 image_size=64,
                 patch_size=16,
                 emb_dim=128,
                 num_heads=4,
                 num_layers=6,
                 num_classes=3,
                 dropout=0.1):
        super(ViT3D, self).__init__()

        self.patch_size = patch_size
        assert image_size % patch_size == 0, "Image dimensions must be divisible by patch size."
        self.image_size = image_size  # store for consistency, if needed

        self.num_patches = (image_size // patch_size) ** 3
        
        self.patch_dim = in_channels * patch_size**3

        self.patch_embed = nn.Linear(self.patch_dim, emb_dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, emb_dim))
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, num_classes)
        )

    def forward(self, x):
        B, C, D, H, W = x.shape
        p = self.patch_size

        # Create patches using unfold
        x = x.unfold(2, p, p).unfold(3, p, p).unfold(4, p, p)
        num_patches = x.shape[1]  # after patching and reshaping

        x = x.contiguous().view(B, C, -1, p**3)
        x = x.permute(0, 2, 1, 3).reshape(B, -1, self.patch_dim)

        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # x = x + self.pos_embed
        # Expand or interpolate position embedding to match
        if self.pos_embed.shape[1] != x.shape[1]:
            pos_embed = nn.functional.interpolate(
                self.pos_embed[:, 1:].transpose(1, 2),
                size=x.shape[1] - 1,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
            pos_embed = torch.cat([self.pos_embed[:, :1], pos_embed], dim=1)
        else:
            pos_embed = self.pos_embed

        x = x + pos_embed

        x = self.dropout(x)
        x = self.transformer(x)

        x = x[:, 0]
        x = self.mlp_head(x)
        return x

import torch.nn as nn
import torch.nn.functional as F
import math

# class TransformerEncoderLayerWithAttn(nn.TransformerEncoderLayer):
#     def __init__(self, *args, **kwargs):
#         super(TransformerEncoderLayerWithAttn, self).__init__(*args, **kwargs)

#     def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
#         # Standard forward with attention capture
#         src2, attn_weights = self.self_attn(
#             src, src, src,
#             attn_mask=src_mask,
#             key_padding_mask=src_key_padding_mask,
#             need_weights=True,
#             average_attn_weights=False  # to get per-head weights
#         )
#         src = src + self.dropout1(src2)
#         src = self.norm1(src)
#         src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
#         src = src + self.dropout2(src2)
#         src = self.norm2(src)
#         return src, attn_weights

# class TransformerEncoderWithAttn(nn.Module):
#     def __init__(self, layer, num_layers):
#         super().__init__()
#         self.layers = nn.ModuleList([layer for _ in range(num_layers)])

#     def forward(self, src):
#         attn_maps = []
#         for mod in self.layers:
#             src, attn = mod(src)
#             attn_maps.append(attn)  # attn shape: [B, num_heads, seq_len, seq_len]
#         return src, attn_maps



class ViTForClassification_V2(nn.Module):
    def __init__(self, input_channels, hidden_size, num_classes, num_heads,attn_dim_feedforward, num_layers, initializer_range, attn_dropout, embed_dropout):
        super().__init__()
        num_patches = 512
        self.initializer_range = initializer_range
        
        # Using built-in PyTorch modules for 3D patch extraction
        self.embedding = nn.Sequential(
            nn.Conv3d(input_channels, num_patches, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.MaxPool3d(kernel_size=2),

            nn.Conv3d(num_patches,num_patches, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.MaxPool3d(kernel_size=2),

            nn.Conv3d(num_patches,num_patches, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.MaxPool3d(kernel_size=2),

            nn.Conv3d(num_patches,num_patches, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.MaxPool3d(kernel_size=2),
            
            # nn.Conv3d(num_patches,num_patches, kernel_size=3, stride=1, padding=1),
            # nn.Tanh(),
            # nn.MaxPool3d(kernel_size=2),
            )
        self.conv1 = nn.Conv3d(num_patches,num_patches, kernel_size=3, stride=1, padding=1)
        self.max_pool1 = nn.MaxPool3d(kernel_size=2)
        self.relu1 = nn.ReLU()
        self.pre_trans_proj = nn.Linear(4096, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.embed_dropout = nn.Dropout(embed_dropout)
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        # self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches+1, hidden_size))
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,  # Should divide hidden_size
                dim_feedforward=attn_dim_feedforward,
                activation='gelu',
                batch_first=True,
                dropout=attn_dropout
            ),
            num_layers=num_layers
        )
        # Attention pooling
        # self.attention_pool = nn.Linear(hidden_size, 1)
        # Classification head
        self.classifier = nn.Sequential(nn.Linear(hidden_size, num_classes))
        # nn.Linear(hidden_size * 2, num_classes)
        self.apply(self._init_weights)
        
    def forward(self, x, output_attns = False):
        x = self.embedding(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.max_pool1(x)
        # x = self.bn1(x)
        x = rearrange(x, 'b c d w h -> b (c d w h)')
        # print(x.shape)
        x=self.pre_trans_proj(x)
        x = self.bn2(x)
        x = self.embed_dropout(x)
        # cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        # x = torch.cat((cls_tokens, x), dim=1) 
        # x += self.position_embeddings
        # print(x.shape)
        x= self.transformer(x)
        # cls_logits, activation_logits = x[:,0,:], x[:,1:,:]
        # x = torch.matmul(nn.functional.softmax(self.attention_pool(activation_logits), dim=1).transpose(-1, -2), activation_logits).squeeze(-2)
        # x = torch.cat((cls_logits, x), dim=1)
        x = self.classifier(x)
        return x

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv3d)):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=self.initializer_range, a=-2, b=2)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm) or isinstance(module, nn.BatchNorm3d) or isinstance(module, nn.BatchNorm1d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=self.initializer_range, a=-2, b=2)

        

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv =  nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.act = nn.ReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=2)
        
        
    def forward(self, x):
        return self.maxpool(self.act((self.bn(self.conv(x)))))

class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415

    Taken from https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py
    """

    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


class PatchEmbeddings(nn.Module):
    """
    Convert the image into patches and then project them into a vector space.
    """

    def __init__(self, cfg):
        super().__init__()
        self.image_size = cfg.image_size
        self.patch_size = cfg.patch_size
        self.num_channels = cfg.num_channels
        self.hidden_size = cfg.hidden_size
        # Calculate the number of patches from the image size and patch size
        # self.num_patches = (self.image_size // self.patch_size) ** 3
        # Create a projection layer to convert the image into patches
        # The layer projects each patch into a vector of size hidden_size
        self.conv_1 = ConvBlock(self.num_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv_2 = ConvBlock(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv_3 = ConvBlock(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv_4 = ConvBlock(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv_5 = ConvBlock(256, 512, kernel_size=3, stride=1, padding=1)
        self.num_patches = 512
        #self.projection = nn.Conv3d(self.num_channels, self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)
        # Consider modifying the PatchEmbeddings class to add initial downsampling:
        self.downsample = nn.MaxPool3d(kernel_size=2, stride=2)  # Add this to reduce spatial dimensions

    def forward(self, x):
        # (batch_size, num_channels, image_depth, image_size, image_size) -> (batch_size, num_patches, hidden_size)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        # x = self.downsample(x)
        #x = self.projection(x)
        x = rearrange(x, 'b c d w h -> b c (d w h)')
        
        return x


class Embeddings(nn.Module):
    """
    Combine the patch embeddings with the class token and position embeddings.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.patch_embeddings = PatchEmbeddings(cfg)
        # Create a learnable [CLS] token
        # Similar to BERT, the [CLS] token is added to the beginning of the input sequence
        # and is used to classify the entire sequence
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.cfg.hidden_size))
        self.cls_pos_embed = nn.Parameter(torch.zeros(1, self.cfg.hidden_size))  # or make this learnable
        # Create position embeddings for the [CLS] token and the patch embeddings
        # Add 1 to the sequence length for the [CLS] token
        # self.position_embeddings = nn.Parameter(torch.randn(1, self.patch_embeddings.num_patches + 1, self.cfg.hidden_size))
        position_embeddings = self.get_3d_sinusoidal_embedding(self.cfg.hidden_size)  # (512, 216)
        self.register_buffer("position_embeddings", position_embeddings)
        self.dropout = nn.Dropout(cfg.hidden_dropout_prob)

    def get_3d_sinusoidal_embedding(self, dim):
        """
        grid_positions: (N, 3) where each row is (x, y, z)
        dim: total dimension of embedding (should be divisible by 3 and even)
        """

        grid_size = 8
        grid_positions = torch.stack(torch.meshgrid(
            torch.arange(grid_size),
            torch.arange(grid_size),
            torch.arange(grid_size),
            indexing='ij'

        ), dim=-1).reshape(-1, 3)  # shape: (512, 3)
        dim_each = dim // 3  # for x, y, z
        assert dim % 6 == 0, "Embedding dimension must be divisible by 6 for 3D sinusoidal encoding."

        def pe(pos, d):
            pe = torch.zeros((pos.shape[0], d))
            div_term = torch.exp(torch.arange(0, d, 2) * -(torch.log(torch.tensor(10000.0)) / d))
            # print('pe, pos shape: ', pe.shape, pos.shape)
            pe[:, 0::2] = torch.sin(pos.unsqueeze(1) * div_term)
            pe[:, 1::2] = torch.cos(pos.unsqueeze(1) * div_term)
            return pe

        pe_x = pe(grid_positions[:, 0], dim_each)
        pe_y = pe(grid_positions[:, 1], dim_each)
        pe_z = pe(grid_positions[:, 2], dim_each)

        return torch.cat([pe_x, pe_y, pe_z], dim=1)  # shape: (512, dim)

    def forward(self, x):
        x = self.patch_embeddings(x)
        batch_size, _, _ = x.size()
        # Expand the [CLS] token to the batch size
        # (1, 1, hidden_size) -> (batch_size, 1, hidden_size)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # Concatenate the [CLS] token to the beginning of the input sequence
        # This results in a sequence length of (num_patches + 1)
        x = torch.cat((cls_tokens, x), dim=1)
        # Generate (512, 216) 3D sinusoidal PE
        pos_embed = torch.cat([self.cls_pos_embed, self.position_embeddings], dim=0)  # (513, 216)
        pos_embed = pos_embed.unsqueeze(0).expand(batch_size, -1, -1)  # (3, 513, 216)

        x = x + pos_embed
        # x = x + self.position_embeddings
        x = self.dropout(x)
        return x


class AttentionHead(nn.Module):
    """
    A single attention head.
    This module is used in the MultiHeadAttention module.

    """
    def __init__(self, hidden_size, attention_head_size, dropout, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        # Create the query, key, and value projection layers
        self.query = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.key = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.value = nn.Linear(hidden_size, attention_head_size, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Project the input into query, key, and value
        # The same input is used to generate the query, key, and value,
        # so it's usually called self-attention.
        # (batch_size, sequence_length, hidden_size) -> (batch_size, sequence_length, attention_head_size)
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        # Calculate the attention scores
        # softmax(Q*K.T/sqrt(head_size))*V
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        # Calculate the attention output
        attention_output = torch.matmul(attention_probs, value)
        return (attention_output, attention_probs)


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module.
    This module is used in the TransformerEncoder module.
    """

    def __init__(self, cfg):
        super().__init__()
        self.hidden_size = cfg.hidden_size
        self.num_attention_heads = cfg.num_attention_heads
        # The attention head size is the hidden size divided by the number of attention heads
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # Whether or not to use bias in the query, key, and value projection layers
        self.qkv_bias = cfg.qkv_bias
        # Create a list of attention heads
        self.heads = nn.ModuleList([])
        for _ in range(self.num_attention_heads):
            head = AttentionHead(
                self.hidden_size,
                self.attention_head_size,
                cfg.attention_probs_dropout_prob,
                self.qkv_bias
            )
            self.heads.append(head)
        # Create a linear layer to project the attention output back to the hidden size
        # In most cases, all_head_size and hidden_size are the same
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(cfg.hidden_dropout_prob)

    def forward(self, x, output_attentions=False):
        # Calculate the attention output for each attention head
        attention_outputs = [head(x) for head in self.heads]
        # Concatenate the attention outputs from each attention head
        attention_output = torch.cat([attention_output for attention_output, _ in attention_outputs], dim=-1)
        # Project the concatenated attention output back to the hidden size
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)
        # Return the attention output and the attention probabilities (optional)
        if not output_attentions:
            return (attention_output, None)
        else:
            attention_probs = torch.stack([attention_probs for _, attention_probs in attention_outputs], dim=1)
            return (attention_output, attention_probs)


class FasterMultiHeadAttention(nn.Module):
    """
    Multi-head attention module with some optimizations.
    All the heads are processed simultaneously with merged query, key, and value projections.
    """

    def __init__(self, cfg):
        super().__init__()
        self.hidden_size = cfg.hidden_size
        self.num_attention_heads = cfg.num_attention_heads
        # The attention head size is the hidden size divided by the number of attention heads
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # Whether or not to use bias in the query, key, and value projection layers
        self.qkv_bias = cfg.qkv_bias
        # Create a linear layer to project the query, key, and value
        self.qkv_projection = nn.Linear(self.hidden_size, self.all_head_size * 3, bias=self.qkv_bias)
        self.attn_dropout = nn.Dropout(cfg.attention_probs_dropout_prob)
        # Create a linear layer to project the attention output back to the hidden size
        # In most cases, all_head_size and hidden_size are the same
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(cfg.hidden_dropout_prob)

    def forward(self, x, output_attentions=False):
        # Project the query, key, and value
        # (batch_size, sequence_length, hidden_size) -> (batch_size, sequence_length, all_head_size * 3)
        qkv = self.qkv_projection(x)
        # Split the projected query, key, and value into query, key, and value
        # (batch_size, sequence_length, all_head_size * 3) -> (batch_size, sequence_length, all_head_size)
        query, key, value = torch.chunk(qkv, 3, dim=-1)
        # Resize the query, key, and value to (batch_size, num_attention_heads, sequence_length, attention_head_size)
        batch_size, sequence_length, _ = query.size()
        query = query.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        key = key.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        value = value.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        # Calculate the attention scores
        # softmax(Q*K.T/sqrt(head_size))*V
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.attn_dropout(attention_probs)
        # Calculate the attention output
        attention_output = torch.matmul(attention_probs, value)
        # Resize the attention output
        # from (batch_size, num_attention_heads, sequence_length, attention_head_size)
        # To (batch_size, sequence_length, all_head_size)
        attention_output = attention_output.transpose(1, 2) \
                                           .contiguous() \
                                           .view(batch_size, sequence_length, self.all_head_size)
        # Project the attention output back to the hidden size
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)
        # Return the attention output and the attention probabilities (optional)
        if not output_attentions:
            return (attention_output, None)
        else:
            return (attention_output, attention_probs)


class MLP(nn.Module):
    """
    A multi-layer perceptron module.
    """

    def __init__(self, cfg):
        super().__init__()
        self.dense_1 = nn.Linear(cfg.hidden_size, cfg.intermediate_size)
        self.activation = NewGELUActivation()
        self.dense_2 = nn.Linear(cfg.intermediate_size, cfg.hidden_size)
        self.dropout = nn.Dropout(cfg.hidden_dropout_prob)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """
    A single transformer block.
    """

    def __init__(self, cfg):
        super().__init__()
        self.use_faster_attention = cfg.use_faster_attention
        if self.use_faster_attention:
            self.attention = FasterMultiHeadAttention(cfg)
        else:
            self.attention = MultiHeadAttention(cfg)
        self.layernorm_1 = nn.LayerNorm(cfg.hidden_size)
        self.mlp = MLP(cfg)
        self.layernorm_2 = nn.LayerNorm(cfg.hidden_size)

    def forward(self, x, output_attentions=False):
        # Self-attention
        attention_output, attention_probs = self.attention(self.layernorm_1(x), output_attentions=output_attentions)
        # Skip connection
        x = x + attention_output
        # Feed-forward network
        mlp_output = self.mlp(self.layernorm_2(x))
        # Skip connection
        x = x + mlp_output
        # Return the transformer block's output and the attention probabilities (optional)
        if not output_attentions:
            return (x, None)
        else:
            return (x, attention_probs)


class Encoder(nn.Module):
    """
    The transformer encoder module.
    """

    def __init__(self, cfg):
        super().__init__()
        # Create a list of transformer blocks
        self.blocks = nn.ModuleList([])
        for _ in range(cfg.num_hidden_layers):
            block = Block(cfg)
            self.blocks.append(block)

    def forward(self, x, output_attentions=False):
        # Calculate the transformer block's output for each block
        all_attentions = []
        for block in self.blocks:
            x, attention_probs = block(x, output_attentions=output_attentions)
            if output_attentions:
                all_attentions.append(attention_probs)
        # Return the encoder's output and the attention probabilities (optional)
        if not output_attentions:
            return (x, None)
        else:
            return (x, all_attentions)


class ViTForClassification(nn.Module):
    """
    The ViT model for classification.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.image_size = cfg.image_size
        self.hidden_size = cfg.hidden_size
        self.num_classes = cfg.num_classes
        # Create the embedding module
        self.embedding = Embeddings(cfg)
        # Create the transformer encoder module
        self.encoder = Encoder(cfg)
        # Create a linear layer to project the encoder's output to the number of classes
        self.attention_pool = nn.Linear(self.hidden_size, 1)
        self.classifier = nn.Linear(2*self.hidden_size, self.num_classes)
        # Initialize the weights
        self.apply(self._init_weights)

    def forward(self, x, output_attentions=False):
        # Calculate the embedding output
        embedding_output = self.embedding(x)
        # Calculate the encoder's output
        encoder_output, all_attentions = self.encoder(embedding_output, output_attentions=output_attentions)
        # Calculate the logits, take the [CLS] token's output as features for classification
        cls_logits, activation_logits = encoder_output[:, 0, :], encoder_output[:, 1:, :]
        activation_logits = torch.matmul(nn.functional.softmax(self.attention_pool(activation_logits), dim=1).transpose(-1, -2), activation_logits).squeeze(-2)
        logits = torch.cat((cls_logits, activation_logits), dim=1)
        logits = self.classifier(logits)
        # Return the logits and the attention probabilities (optional)

        return logits
        if not output_attentions:
            return (logits, None)
        else:
            return (logits, all_attentions)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.cfg.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, Embeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.cfg.initializer_range,
            ).to(module.position_embeddings.dtype)

            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=self.cfg.initializer_range,
            ).to(module.cls_token.dtype)


def get_model(model_name, cfg):
    # cfg = config(model_name)
    if model_name == "ViT3D_V1":
        return ViT3D_V1(
            in_channels=cfg.input_channels,
            image_size=cfg.image_size,
            patch_size=cfg.patch_size,
            emb_dim=cfg.emb_dim,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
            num_classes=cfg.num_classes,
            dropout=cfg.dropout
        )
    elif model_name == "ViTForClassification_V2":
        # print(cfg.__dict__)
        return ViTForClassification_V2(
            input_channels=cfg.input_channels, 
            hidden_size=cfg.hidden_size, 
            num_classes=cfg.num_classes, 
            num_heads= cfg.num_heads,
            attn_dim_feedforward=cfg.attn_dim_feedforward,
            num_layers=cfg.num_layers,
            attn_dropout = cfg.attn_dropout,
            embed_dropout = cfg.embed_dropout,
            initializer_range = cfg.initializer_range
        )
    elif model_name == "ViTForClassification":
        return ViTForClassification(cfg)
    else:
        raise ValueError(f"Model {model_name} do not exist in the existing list: ['ViTForClassification_V2', 'ViT3D_V1', 'ViTForClassification']")


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

if __name__ == '__main__':
    from torchinfo import summary
    # cfg = Config('ViTForClassification_V2')
    model = ViTForClassification_V2(input_channels=1, hidden_size=300, num_classes=3, num_heads = 30,attn_dim_feedforward=600, num_layers=3, initializer_range=0.02,attn_dropout=0.1, embed_dropout=0.1)
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # x = torch.randn(1,1,64,64,64)
    # model(x)
    # # print(next(model.parameters()).device)
    # # print(model.max_pool1)
    # h = model.max_pool1.register_forward_hook(get_activation('last_layer'))
    # output, atnma= model(x, output_attentions=True)
    # h.remove()
    # forward_features = activation['last_layer']
    # print(activation['last_layer'].shape, )

    # attention_weights = model.transformer.state_dict()
    # wq, wk, wv = attention_weights['layers.0.self_attn.in_proj_weight'].chunk(3, dim = 0)
    # bq, bk, bv = attention_weights['layers.0.self_attn.in_proj_bias'].chunk(3, dim = 0)

    # print(wq.shape, wk.shape, wv.shape)
    # print(bq.shape, bk.shape, bv.shape)
    # for i,j in model.named_parameters():
    #     print(i)

    res = summary(model, [16,1,64,64,64])

    # Suppose model is an instance of ViTForClassification_V2
    for i, layer in enumerate(model.transformer.layers):
        print(f"Layer {i}: {layer.self_attn.num_heads} heads")

    # print(res.total_mult_adds/(1000**2))
    # print(res.total_input/(1000**2))
    # print(res.input_size)
    # print(res.total_output_bytes/(1000**2))
    # print((res.total_input+res.total_output_bytes)/(1000**2))
    # res = summary(model, [5,1,192,192,192],col_width=16,
    # col_names=["input_size", "output_size", "num_params", "mult_adds"],
    # row_settings=["var_names"],verbose=2)
