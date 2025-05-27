# vizualisation utils

import torch
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from scipy.ndimage import zoom

import os
import sys
import json
import time
import datetime
from models import get_model
from types import SimpleNamespace

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


def load_model(config):
    config_obj = SimpleNamespace(**config["configs"])
    model_name = config["configs"]['model_name']
    device = config["configs"]['device']

    # print(config_obj)
    model = get_model(model_name, config_obj).to(device)
    model_path = os.path.join(config['configs']['checkpoints_dir'], 'best_model.pth')
    model.load_state_dict(torch.load(model_path))
    return model


def get_last_layer_activations_grds(model, x, config):
    device = config["configs"]['device']
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    if x.dim()==4:
        x = torch.unsqueeze(x, dim=0)

    model_name = config["configs"]['model_name']
    x = x.to(device)
    if model_name == 'ViTForClassification':
        h = model.embedding.patch_embeddings.conv_5.maxpool.register_forward_hook(get_activation('last_layer'))
        output, attention_maps = model(x, output_attentions=True)
        h.remove()
        forward_features = activation['last_layer']
        #forward_features[forward_features < 0] = 0
        
        return output, forward_features, attention_maps

    elif model_name == 'ViTForClassification_V2':
        h = model.max_pool1.register_forward_hook(get_activation('last_layer'))
        output, attention_maps = model(x, output_attns=True)
        h.remove()
        forward_features = activation['last_layer']
        return output, forward_features, attention_maps

def get_outputs(model, x, config):
    device = config["configs"]['device']
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if x.dim() == 4:
        x = x.unsqueeze(0)

    x = x.to(device)
    print(f'Shape of x: {x.shape}')
    outs = model(x)
    return outs


def get_class_attention_coefficients(attention_maps: list):
    attention_maps = torch.cat(attention_maps, dim=1)
    # select only the attention maps of the CLS token
    attention_coeff = attention_maps[:, :, 0, 1:] 
    attention_coeff = torch.sum(attention_coeff, dim=1)
    attention_coeff = torch.squeeze(attention_coeff)
    attention_coeff = attention_coeff/torch.max(attention_coeff)
    return attention_coeff


def get_heatmap(model, forward_attentions, attention_coeff, return_map=True):
    forward_attentions = torch.squeeze(forward_attentions)
    for i in range(512):
        #forward_attentions[i, :, :, :] *= pooled_gradients[i]
        forward_attentions[i, :, :, :] *= attention_coeff[i]
    heatmap = torch.mean(forward_attentions, dim=0)
    heatmap = heatmap.detach().cpu().numpy()
    heatmap = np.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0)

    upscaled_heatmap = zoom(heatmap, (24, 24, 24), mode='nearest')

    upscaled_heatmap = np.uint8(upscaled_heatmap*255)
    print(f'Shape of up_heatmap: {upscaled_heatmap.shape}')

    if return_map:
        return heatmap, upscaled_heatmap
    else:
        return upscaled_heatmap

# def plot_class_attention_map(attention_coeff, tag='label', cmap='hot'):
#     h, w = 32, 16
#     assert attention_coeff.size(dim=0) == h*w

#     class_attention_map = attention_coeff.view(h, w)
#     class_attention_map = class_attention_map/torch.max(class_attention_map)
#     class_attention_map = class_attention_map.detach().cpu().numpy()
#     class_attention_map = np.uint8(class_attention_map*255)
    
#     return class_attention_map


from pathlib import Path 
def read_image_folder(img_folder, config):
    """
    test
    """
    # check if the path is folder or not
    img_folder = Path(img_folder)
    img_paths = []

    if img_folder.is_file():
        img_paths  = [img_folder]

    elif img_folder.is_dir():
        img_paths = [p for p in img_folder.iterdir() if p.is_file()]
        # img_paths = os.listdir(img_folder)
        # print(img_folder)

    else:
        raise FileNotFoundError('Given image path is invalid')

    print(len(img_paths))
    image_affines = []
    
    for image_path in img_paths:

        image_path = os.path.join(img_folder, image_path)
        file_name = image_path.split('/')[-1].split('.')[0]
        # image_path = os.path.join(image_list, image_path)
        print(image_path)
        original_image = nib.as_closest_canonical(nib.load(image_path))
        original_affine = original_image.affine
        image = original_image.get_fdata()
        xdim, ydim, zdim = image.shape
        image = np.pad(image, [((512-xdim)//2, (512-xdim)//2), ((512-ydim)//2, (512-ydim)//2), ((512-zdim)//2, (512-zdim)//2)], 'constant', constant_values=0)
        #image = image.reshape(image.shape[2], image.shape[1], image.shape[0])

        # print(config)

        zoom_factors = (
        config["configs"]['processed_image_size'] / image.shape[0],
        config["configs"]['processed_image_size'] / image.shape[1],
        config["configs"]['processed_image_size'] / image.shape[2]
        )

        # width_factor= config["configs"]['processed_image_size'] / image.shape[0]
        # height_factor=config[ "configs"]['processed_image_size'] / image.shape[1]
        # depth_factor= config["configs"]['processed_image_size'] / image.shape[-1]

        image = zoom(image, zoom_factors, order=1)
        original_image = image.copy()
        original_image = original_image[None, ...]
        # image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
        idxs = np.linspace(0, 191, 64).astype(int)
        image = image[np.ix_(idxs, idxs, idxs)]
        image = image[None, ...]

        # Compute new affine
        scale_matrix = np.diag(list(1/np.array(zoom_factors)) + [1])
        new_affine = original_affine @ scale_matrix

        # nan_mask = np.isnan(image)
        # inf_mask = np.isinf(image)

        # num_nan = np.sum(nan_mask)
        # num_inf = np.sum(inf_mask)
        # num_bads = num_inf + num_nan

        # if num_bads == 0:
        #     is_bad = False
        # else:
        #     is_bad = True

        # Check for NaNs/Infs
        # nan_mask = np.isnan(image)
        # inf_mask = np.isinf(image)
        # # Replace NaNs/Infs with 0
        # image[nan_mask] = 0.0
        # image[inf_mask] = 0.0
        image = np.nan_to_num(image, nan=np.nanmedian(image))

        image = image.astype('float32')

        image_affines.append((original_image, image, new_affine,file_name))

    return image_affines


     
