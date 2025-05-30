# For model inference and prediction

import torch
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from scipy.ndimage import zoom
import torch.nn as nn
import os
import sys
import json
import time
import datetime
import argparse
from utils.xai_util import *

torch.cuda.empty_cache()
torch.manual_seed(0)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Run inference on a directory of MRI images stored as .nii files.")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to .nii folder.')
    parser.add_argument('--output_dir', type=str, required=False, default='/DATA1/sayantan/nicara_prepro_vit/inference_outputs', help='Path to save the inference results')
    parser.add_argument('--config', type=str, required=True, help="Path to the JSON file containing the model pipeline configuration (config.json).")
    # parse.add_argument('--labels', type=str, default='', help="Optional path to the ground truth labels file, if available.")


    prj_start = time.time() 
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    model = load_model(config)
    print(args.data_dir)
    image_pkgs = read_image_folder(args.data_dir, config)
    # print(image_pkgs)
    # print(len(image_pkgs))
    
    for img_pkg in image_pkgs:
        print(f'file name: {img_pkg[-1]}')
        heats = []
        # epoch_start_time = time.time()
    
        # Create figure and axes with proper size
        fig, axes = plt.subplots(1, 3, figsize=(12, 8), squeeze=True)
    
        # Get image and model outputs
        x1 = img_pkg[1]
        model.eval()

        logits = get_outputs(model, x1, config)
        softmax = nn.Softmax(dim=1)
        probs = softmax(logits)
        probs = probs.detach().cpu().numpy().flatten()

        # Format all probabilities as strings with the desired format
        prob_strings = []
        for class_idx, prob in enumerate(probs):
            prob_strings.append(f"{prob*100:.2f}%")
        # Join all probability strings
        all_probs_str = ", ".join(prob_strings)


        os.makedirs(os.path.join(args.output_dir, img_pkg[-1]), exist_ok = True)
        pred_path = os.path.join(args.output_dir, img_pkg[-1], 'prediction.txt')
        with open(pred_path, 'w') as f:
            f.write(f"Prediction Probs -> ['CN', 'MCI', 'AD']: [{all_probs_str}]")


        fig.suptitle(f"Heatmap for {img_pkg[-1]}", fontsize=5,  y=0.75)
        yhat, forward_features, attention_maps = get_last_layer_activations_grds(model, x1, config)
        attention_coeff = get_class_attention_coefficients(attention_maps)
        heatmap, upscaled_heatmap = get_heatmap(model, forward_features, attention_coeff)
    
        x = img_pkg[1]
        print('shape of x: ', x.shape)
        # Process heatmap
        upscaled_heatmap = upscaled_heatmap * x[0]
        upscaled_heatmap = upscaled_heatmap / np.max(upscaled_heatmap)
        heats.append(heatmap)
    
        # Process image slices
        x = np.squeeze(x)
        mri_gif_maker(x, upscaled_heatmap, os.path.join(args.output_dir, img_pkg[-1]))
        slice_idx = x.shape[0] // 2
        
        # Prepare sagital view
        sagital = np.rot90(x[slice_idx, :, :])
        sagital_heatmap = np.rot90(upscaled_heatmap[slice_idx, :, :])

        
        # Prepare coronal view
        coronal = np.rot90(x[:, slice_idx, :])
        coronal_heatmap = np.rot90(upscaled_heatmap[:, slice_idx, :])
        
        # Prepare axial view
        axial = np.rot90(x[:, :, slice_idx])
        axial_heatmap = np.rot90(upscaled_heatmap[:, :, slice_idx])
        
        # Create mask for side-by-side visualization
        mask = np.concatenate((np.ones((64,64)), np.zeros((64,64))), axis=1)
        
        # Sagital view processing
        sagital = np.concatenate((sagital, sagital), axis=1)
        sagital_heatmap = np.concatenate((sagital_heatmap, sagital_heatmap), axis=1)
        sagital_heatmap = np.ma.masked_where(mask == 1, sagital_heatmap)
        print(sagital.shape)
        print(sagital_heatmap.shape)
        
        # Coronal view processing
        coronal = np.concatenate((coronal, coronal), axis=1)
        coronal_heatmap = np.concatenate((coronal_heatmap, coronal_heatmap), axis=1)
        coronal_heatmap = np.ma.masked_where(mask == 1, coronal_heatmap)
        
        # Axial view processing
        axial = np.concatenate((axial, axial), axis=1)
        axial_heatmap = np.concatenate((axial_heatmap, axial_heatmap), axis=1)
        axial_heatmap = np.ma.masked_where(mask == 1, axial_heatmap)
        
        # Plot sagital view
        axes[0].imshow(sagital, cmap='gray')
        axes[0].imshow(sagital_heatmap, alpha=0.7, cmap='jet')  # Adjusted alpha for better overlay visibility
        axes[0].set_title("Sagital", fontsize=5)
        axes[0].axis('off')
        
        # Plot coronal view
        axes[1].imshow(coronal, cmap='gray')
        axes[1].imshow(coronal_heatmap, alpha=0.7, cmap='jet')
        axes[1].set_title("Coronal", fontsize=5)
        axes[1].axis('off')
        
        # Plot axial view
        axes[2].imshow(axial, cmap='gray')
        axes[2].imshow(axial_heatmap, alpha=0.7, cmap='jet')
        axes[2].set_title("Axial", fontsize=5)
        axes[2].axis('off')
        
        # Adjust layout and add spacing between subplots
        plt.subplots_adjust(wspace=0.01)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

    
        # Save visualization with high resolution
        plt.savefig(
            os.path.join(os.path.join(args.output_dir, img_pkg[-1], 'heatmap_vis.png')),
            dpi=300,  # Reduced from 900 for faster processing - increase if needed
            bbox_inches='tight'
            )
    
        # Save NIFTI file
        new_image = nib.Nifti1Image(upscaled_heatmap, affine=img_pkg[2])
        nib.save(new_image, os.path.join(args.output_dir, img_pkg[-1],"heatmap.nii.gz"))
        
        
        # Close the figure to free memory
        plt.close(fig)


    