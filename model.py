# Model architecture(s)

import torch
import torch.nn as nn

class ViT3D(nn.Module):
    def __init__(self, 
                 in_channels=1,
                 img_size=64,
                 patch_size=16,
                 emb_dim=128,
                 num_heads=4,
                 num_layers=6,
                 num_classes=3,
                 dropout=0.1):
        super(ViT3D, self).__init__()

        self.patch_size = patch_size
        assert img_size % patch_size == 0, "Image dimensions must be divisible by patch size."
        self.img_size = img_size  # store for consistency, if needed

        self.num_patches = (img_size // patch_size) ** 3
        
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


def get_model(model_name, config):
    if model_name == "ViT3D":
        return ViT3D(
            in_channels=config.input_channels,
            img_size=config.img_size,
            patch_size=config.patch_size,
            emb_dim=config.emb_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            num_classes=config.num_classes,
            dropout=config.dropout
        )
    else:
        raise ValueError(f"Model {model_name} is not implemented.")

        
if __name__ == '__main__':
    from torchinfo import summary
    model = ViT3D(img_size=192)
    res = summary(model, [1,1,192,192,192],col_width=16,
    col_names=["input_size", "output_size", "num_params", "mult_adds"],
    row_settings=["var_names"],verbose=2)
