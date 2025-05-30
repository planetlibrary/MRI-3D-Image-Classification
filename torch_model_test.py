# # import torch
# # import torch.nn as nn
# # layer = nn.TransformerEncoderLayer(
# #         d_model=64,
# #         nhead=32,  # Should divide hidden_size
# #         dim_feedforward=64,
# #         activation='gelu',
# #         batch_first=True,
# #         dropout=0.75) 
# # # params  = list(dict(layer.self_attn.__dict__['_parameters']).values())[0]
# # # print(params.shape)

# # # for name, param in layer.self_attn.named_parameters():
# # #     print(name, param.shape)

# # a = nn.TransformerEncoder(
# #         encoder_layer=layer,num_layers=4
# #     )

# # # for name, param in a.state_dict().items():
# # #     print(name, param.shape)

# # # d = torch.randn(1,512,64)
# # # print(d)
# # # o = a(d)
# # # print(o)
# # # print(o.shape)

# # attention_maps = []

# # def hook_fn(module, input, output):
# #     # output[1] contains the attention weights
# #     print(output)
# #     attention_maps.append(output[1].detach())

# # for layer in a.layers:
# #     layer.self_attn.forward = lambda *args, **kwargs: nn.MultiheadAttention.forward(
# #         layer.self_attn, *args, **kwargs, need_weights=True, average_attn_weights=False
# #     )
# #     layer.self_attn.register_forward_hook(hook_fn)

# # print(attention_maps)

# import torch
# import torch.nn as nn


# # layer = nn.TransformerEncoderLayer(
# #     d_model=64,
# #     nhead=32,
# #     dim_feedforward=64,
# #     activation='gelu',
# #     batch_first=True,
# #     dropout=0.1
# # )

# x = torch.rand(20, 64)  # batch, seq_len, dim
# class custNet(nn.Module):
#     def __init__(self, dim = 3):
#         super().__init__()
#         self.ll = nn.Linear(64, 1024)
#         self.layer = nn.TransformerEncoderLayer(
#             d_model=2,
#             nhead=2,
#             dim_feedforward=64,
#             activation='gelu',
#             batch_first=True,
#             dropout=0.1
#         )
#         self.a = nn.TransformerEncoder(self.layer, num_layers=4)

#         self.ll1 = nn.Linear(1024,3)
#     def forward(self, x):
#         x= self.ll(x)
#         x=x.view(-1,512,2)    
#         print(x.shape)
#         x = self.a(x)
#         x = x.view(-1,512*2)
#         x = self.ll1(x)
#         return x

# model= custNet()
# out = model(x)
# print(out.shape)

# attention_maps = []
# def hook_fn(module, input, output):
#     attention_maps.append(output[1].detach())

# for i, layer in enumerate(model.a.layers):
#     layer.self_attn.register_forward_hook(hook_fn)

# # Safe monkey-patching
# original_forward = nn.MultiheadAttention.forward

# def custom_forward(self, query, key, value, **kwargs):
#     kwargs.pop("need_weights", None)
#     kwargs.pop("average_attn_weights", None)
#     return original_forward(self, query, key, value,
#                             need_weights=True,
#                             average_attn_weights=True,
#                             **kwargs)

# # Apply patch
# nn.MultiheadAttention.forward = custom_forward

# # Run
# model.eval()
# out = model(x)
# print(f'out shape: {out.shape}')

# # Restore original
# nn.MultiheadAttention.forward = original_forward


# print(f"Captured {len(attention_maps)} attention maps.")
# print("Shape of one attention map:", attention_maps[0].shape)


import torch 
import torch.nn as nn

class custNet(nn.Module):
    def __init__(self):
        super().__init__()
        num_patches = 10
        self.embedding = nn.Sequential(
            nn.Conv3d(1, num_patches, kernel_size=3, stride=2, padding=1),
            nn.Conv3d(num_patches,num_patches, kernel_size=3, stride=2, padding=1),
            nn.Conv3d(num_patches,num_patches, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(num_patches,num_patches, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(num_patches,num_patches, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(num_patches,num_patches, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_patches),
            nn.PReLU(),
            )
    def forward(self, x):
        x = self.embedding(x)
        return x
x = torch.randn(1,1,64,64,64)
model = custNet()
out = model(x)
print(out.shape)

print(model.embedding[5])
# print(list(model.children()))