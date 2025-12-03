import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import argparse
from datetime import datetime
from sklearn.decomposition import PCA

from model import DiT_Llama, modulate
from train import MNISTModularArithmeticDataset


def collect_residual_streams(model, dataloader, max_samples=None):
    """
    Collect residual stream representations after attn and ffn for each layer.
    Also collects attn_out and ffn_out (pure outputs before residual connection).
    Uses Case 2 approach: keeps patch dimension.
    
    Returns:
        residual_streams_attn: dict mapping layer_idx -> [num_samples, num_patches, dim]
        residual_streams_ffn: dict mapping layer_idx -> [num_samples, num_patches, dim]
        attn_outputs: dict mapping layer_idx -> [num_samples, num_patches, dim]
        ffn_outputs: dict mapping layer_idx -> [num_samples, num_patches, dim]
        labels: array of target class labels [num_samples]
    """
    # We'll manually track residual streams by modifying the forward pass
    # Store original forward method
    original_forwards = {}
    for layer_idx, layer in enumerate(model.layers):
        original_forwards[layer_idx] = layer.forward
    
    residual_streams_attn = {}  # layer_idx -> list of [batch_size, num_patches, dim]
    residual_streams_ffn = {}   # layer_idx -> list of [batch_size, num_patches, dim]
    attn_outputs = {}  # layer_idx -> list of [batch_size, num_patches, dim]
    ffn_outputs = {}   # layer_idx -> list of [batch_size, num_patches, dim]
    labels_list = []
    
    # Create wrapper functions to capture residual streams
    def create_wrapper(layer_idx, original_forward):
        def wrapper(x, freqs_cis, adaln_input=None):
            # Before attention
            x_before_attn = x
            
            # Run attention
            if adaln_input is not None:
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                    layer.adaLN_modulation(adaln_input).chunk(6, dim=1)
                )
                attn_out = layer.attention(
                    modulate(layer.attention_norm(x), shift_msa, scale_msa), freqs_cis
                )
                x_after_attn = x + gate_msa.unsqueeze(1) * attn_out
            else:
                attn_out = layer.attention(layer.attention_norm(x), freqs_cis)
                x_after_attn = x + attn_out
            
            # Store attention output (pure output before residual connection)
            if layer_idx not in attn_outputs:
                attn_outputs[layer_idx] = []
            attn_outputs[layer_idx].append(attn_out.detach().cpu())
            
            # Store residual stream after attention
            if layer_idx not in residual_streams_attn:
                residual_streams_attn[layer_idx] = []
            residual_streams_attn[layer_idx].append(x_after_attn.detach().cpu())
            
            # Run FFN
            if adaln_input is not None:
                ffn_out = layer.feed_forward(
                    modulate(layer.ffn_norm(x_after_attn), shift_mlp, scale_mlp)
                )
                x_after_ffn = x_after_attn + gate_mlp.unsqueeze(1) * ffn_out
            else:
                ffn_out = layer.feed_forward(layer.ffn_norm(x_after_attn))
                x_after_ffn = x_after_attn + ffn_out
            
            # Store FFN output (pure output before residual connection)
            if layer_idx not in ffn_outputs:
                ffn_outputs[layer_idx] = []
            ffn_outputs[layer_idx].append(ffn_out.detach().cpu())
            
            # Store residual stream after FFN
            if layer_idx not in residual_streams_ffn:
                residual_streams_ffn[layer_idx] = []
            residual_streams_ffn[layer_idx].append(x_after_ffn.detach().cpu())
            
            return x_after_ffn
        
        return wrapper
    
    # Replace forward methods
    for layer_idx, layer in enumerate(model.layers):
        layer.forward = create_wrapper(layer_idx, original_forwards[layer_idx])
    
    model.eval()
    total_samples = 0
    
    with torch.no_grad():
        for batch_idx, (_, x_src, label_tgt) in enumerate(tqdm(dataloader, desc="Collecting residual streams")):
            if max_samples is not None and total_samples >= max_samples:
                break
            
            x_src = x_src.cuda()
            label_tgt = label_tgt.cuda()
            batch_size = x_src.size(0)
            
            # Run forward pass
            x_init = torch.randn(batch_size, 1, 32, 32).cuda()
            t = torch.ones(batch_size).cuda() * 0.5
            _ = model(x_init, t, x_src)
            
            # Store labels
            labels_list.append(label_tgt.cpu().numpy())
            total_samples += batch_size
    
    # Restore original forward methods
    for layer_idx, layer in enumerate(model.layers):
        layer.forward = original_forwards[layer_idx]
    
    # Concatenate all batches
    for layer_idx in residual_streams_attn:
        residual_streams_attn[layer_idx] = torch.cat(residual_streams_attn[layer_idx], dim=0)
    for layer_idx in residual_streams_ffn:
        residual_streams_ffn[layer_idx] = torch.cat(residual_streams_ffn[layer_idx], dim=0)
    for layer_idx in attn_outputs:
        attn_outputs[layer_idx] = torch.cat(attn_outputs[layer_idx], dim=0)
    for layer_idx in ffn_outputs:
        ffn_outputs[layer_idx] = torch.cat(ffn_outputs[layer_idx], dim=0)
    
    labels = np.concatenate(labels_list)
    
    return residual_streams_attn, residual_streams_ffn, attn_outputs, ffn_outputs, labels


def visualize_residual_stream_pca_case1(residual_streams, labels, output_path, stream_type="attn"):
    """
    Case 1: Apply the same PCA to all patches (global PCA).
    Visualize each patch in a 16x16 subplot grid.
    
    Args:
        residual_streams: dict mapping layer_idx -> [num_samples, num_patches, dim]
        labels: array of target class labels [num_samples]
        output_path: path to save the visualization
        stream_type: "attn" or "ffn" for labeling
    """
    num_layers = len(residual_streams)
    layer_indices = sorted(residual_streams.keys())
    
    # Get unique classes for coloring
    unique_classes = np.unique(labels)
    num_classes = len(unique_classes)
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    class_to_color = {cls: colors[i] for i, cls in enumerate(unique_classes)}
    
    # Create a figure for each layer
    for layer_idx in layer_indices:
        residual_stream = residual_streams[layer_idx]  # [num_samples, num_patches, dim]
        num_samples, num_patches, dim = residual_stream.shape
        
        # Reshape patches to spatial grid
        patch_size = int(np.sqrt(num_patches))
        assert patch_size * patch_size == num_patches, f"num_patches ({num_patches}) must be a perfect square"
        
        # Case 1: Apply PCA to all patches together (global PCA)
        residual_flat = residual_stream.reshape(-1, dim).numpy()  # [num_samples * num_patches, dim]
        pca = PCA(n_components=4)
        pca_result = pca.fit_transform(residual_flat)  # [num_samples * num_patches, 4]
        pca_result = pca_result.reshape(num_samples, num_patches, 4)  # [num_samples, num_patches, 4]
        
        explained_var = pca.explained_variance_ratio_
        
        # Create 3 figures for 3 PCA pairs
        pca_pairs = [(0, 1, "pc0_pc1"), (1, 2, "pc1_pc2"), (2, 3, "pc2_pc3")]
        
        for comp_x, comp_y, pair_name in pca_pairs:
            fig, axes = plt.subplots(patch_size, patch_size, figsize=(20, 20))
            
            for i in range(patch_size):
                for j in range(patch_size):
                    patch_idx = i * patch_size + j
                    ax = axes[i, j]
                    
                    # Get PCA results for this patch: [num_samples, 4]
                    patch_pca = pca_result[:, patch_idx, :]
                    
                    # Plot each class with different color
                    for cls in unique_classes:
                        mask = labels == cls
                        ax.scatter(patch_pca[mask, comp_x], patch_pca[mask, comp_y],
                                  c=[class_to_color[cls]], s=5, alpha=0.6)
                    
                    ax.set_xticks([])
                    ax.set_yticks([])
            
            # Add overall title
            title_type = "Output" if stream_type.endswith("_output") else "Residual Stream"
            stream_name = stream_type.replace("_output", "").upper() if stream_type.endswith("_output") else stream_type.upper()
            fig.suptitle(f'Layer {layer_idx} - {stream_name} {title_type} (Case 1: Global PCA)\n'
                        f'PCA{comp_x} vs PCA{comp_y} - Explained var: PC{comp_x}={explained_var[comp_x]:.2%}, '
                        f'PC{comp_y}={explained_var[comp_y]:.2%}',
                        fontsize=14, y=0.98)
            
            # Add legend
            handles = [plt.Line2D([], [], marker='o', linestyle='', color=class_to_color[cls], 
                                 label=f'Class {cls}') for cls in unique_classes]
            fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.99, 0.995), fontsize=10)
            
            plt.tight_layout(rect=[0, 0, 0.95, 0.90])
            
            layer_output_path = output_path.replace('.png', f'_layer_{layer_idx}_{pair_name}.png')
            plt.savefig(layer_output_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Case 1 PCA visualization (Layer {layer_idx}, {stream_type}, PCA{comp_x} vs PCA{comp_y}) saved to: {layer_output_path}")
    
    print(f"Case 1 PCA visualization complete for all layers ({stream_type})")


def visualize_residual_stream_pca_case2(residual_streams, labels, output_path, stream_type="attn"):
    """
    Case 2: Apply independent PCA to each patch.
    Visualize each patch in a 16x16 subplot grid.
    
    Args:
        residual_streams: dict mapping layer_idx -> [num_samples, num_patches, dim]
        labels: array of target class labels [num_samples]
        output_path: path to save the visualization
        stream_type: "attn" or "ffn" for labeling
    """
    num_layers = len(residual_streams)
    layer_indices = sorted(residual_streams.keys())
    
    # Get unique classes for coloring
    unique_classes = np.unique(labels)
    num_classes = len(unique_classes)
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    class_to_color = {cls: colors[i] for i, cls in enumerate(unique_classes)}
    
    # Create a figure for each layer
    for layer_idx in layer_indices:
        residual_stream = residual_streams[layer_idx]  # [num_samples, num_patches, dim]
        num_samples, num_patches, dim = residual_stream.shape
        
        # Reshape patches to spatial grid
        patch_size = int(np.sqrt(num_patches))
        assert patch_size * patch_size == num_patches, f"num_patches ({num_patches}) must be a perfect square"
        
        # Case 2: Apply PCA independently to each patch
        pca_results = {}  # patch_idx -> [num_samples, 4]
        explained_vars = {}  # patch_idx -> [4] explained variance ratios
        
        for patch_idx in range(num_patches):
            patch_data = residual_stream[:, patch_idx, :].numpy()  # [num_samples, dim]
            pca = PCA(n_components=4)
            pca_result = pca.fit_transform(patch_data)  # [num_samples, 4]
            pca_results[patch_idx] = pca_result
            explained_vars[patch_idx] = pca.explained_variance_ratio_
        
        # Create 3 figures for 3 PCA pairs
        pca_pairs = [(0, 1, "pc0_pc1"), (1, 2, "pc1_pc2"), (2, 3, "pc2_pc3")]
        
        for comp_x, comp_y, pair_name in pca_pairs:
            fig, axes = plt.subplots(patch_size, patch_size, figsize=(20, 20))
            
            for i in range(patch_size):
                for j in range(patch_size):
                    patch_idx = i * patch_size + j
                    ax = axes[i, j]
                    
                    # Get PCA results for this patch: [num_samples, 4]
                    patch_pca = pca_results[patch_idx]
                    
                    # Plot each class with different color
                    for cls in unique_classes:
                        mask = labels == cls
                        ax.scatter(patch_pca[mask, comp_x], patch_pca[mask, comp_y],
                                  c=[class_to_color[cls]], s=5, alpha=0.6)
                    
                    ax.set_xticks([])
                    ax.set_yticks([])
            
            # Add overall title
            # Average explained variance across patches
            avg_explained_x = np.mean([explained_vars[p][comp_x] for p in range(num_patches)])
            avg_explained_y = np.mean([explained_vars[p][comp_y] for p in range(num_patches)])
            
            title_type = "Output" if stream_type.endswith("_output") else "Residual Stream"
            stream_name = stream_type.replace("_output", "").upper() if stream_type.endswith("_output") else stream_type.upper()
            fig.suptitle(f'Layer {layer_idx} - {stream_name} {title_type} (Case 2: Patch-wise PCA)\n'
                        f'PCA{comp_x} vs PCA{comp_y} - Avg explained var: PC{comp_x}={avg_explained_x:.2%}, '
                        f'PC{comp_y}={avg_explained_y:.2%}',
                        fontsize=14, y=0.98)
            
            # Add legend
            handles = [plt.Line2D([], [], marker='o', linestyle='', color=class_to_color[cls], 
                                 label=f'Class {cls}') for cls in unique_classes]
            fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.99, 0.995), fontsize=10)
            
            plt.tight_layout(rect=[0, 0, 0.95, 0.90])
            
            layer_output_path = output_path.replace('.png', f'_layer_{layer_idx}_{pair_name}.png')
            plt.savefig(layer_output_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Case 2 PCA visualization (Layer {layer_idx}, {stream_type}, PCA{comp_x} vs PCA{comp_y}) saved to: {layer_output_path}")
    
    print(f"Case 2 PCA visualization complete for all layers ({stream_type})")


def visualize_class_mean_pca_case1(residual_streams, labels, output_path, stream_type="attn", use_normalized=False):
    """
    Case 1: Compute class means across all patches, then apply global PCA.
    Fit PCA on class means, but transform data (normalized if use_normalized=True) for scatter plot.
    Visualize data points in PCA space (PC0 vs PC1).
    
    Args:
        residual_streams: dict mapping layer_idx -> [num_samples, num_patches, dim]
        labels: array of target class labels [num_samples]
        output_path: path to save the visualization
        stream_type: "attn" or "ffn" for labeling
        use_normalized: if True, normalize each sample before computing mean and for visualization
    """
    layer_indices = sorted(residual_streams.keys())
    unique_classes = np.unique(labels)
    num_classes = len(unique_classes)
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    class_to_color = {cls: colors[i] for i, cls in enumerate(unique_classes)}
    
    # Create a figure for each layer
    for layer_idx in layer_indices:
        residual_stream = residual_streams[layer_idx]  # [num_samples, num_patches, dim]
        num_samples, num_patches, dim = residual_stream.shape
        
        # Prepare data (normalize if needed)
        if use_normalized:
            # Normalize each sample (across patches and dims)
            # Flatten to [num_samples, num_patches * dim] for norm calculation
            residual_flat = residual_stream.view(num_samples, -1)  # [num_samples, num_patches * dim]
            norms = torch.norm(residual_flat, dim=1, keepdim=True)  # [num_samples, 1]
            # Reshape norms to [num_samples, 1, 1] for broadcasting
            norms = norms.view(num_samples, 1, 1)  # [num_samples, 1, 1]
            residual_stream_normalized = residual_stream / (norms + 1e-8)
        else:
            residual_stream_normalized = residual_stream
        
        # Compute class means for each patch (for PCA fitting)
        # class_means: [num_classes, num_patches, dim]
        class_means = []
        for cls in unique_classes:
            mask = labels == cls
            class_data = residual_stream_normalized[mask]  # [num_class_samples, num_patches, dim]
            
            # Compute mean across samples for each patch
            class_mean = class_data.mean(dim=0)  # [num_patches, dim]
            # Ensure it's 2D
            if class_mean.ndim == 1:
                class_mean = class_mean.unsqueeze(0)
            class_means.append(class_mean)
        
        class_means = torch.stack(class_means)  # [num_classes, num_patches, dim]
        # Ensure class_means is 3D: [num_classes, num_patches, dim]
        if class_means.ndim != 3:
            raise ValueError(f"class_means should be 3D but got shape {class_means.shape}")
        
        # Case 1: Flatten patches and apply global PCA on class means
        # Reshape to [num_classes * num_patches, dim]
        class_means_flat = class_means.view(-1, dim)
        if isinstance(class_means_flat, torch.Tensor):
            class_means_flat = class_means_flat.detach().cpu().numpy()
        else:
            class_means_flat = np.array(class_means_flat)
        # Ensure 2D array
        if class_means_flat.ndim > 2:
            class_means_flat = class_means_flat.reshape(-1, class_means_flat.shape[-1])
        
        # Fit PCA on class means
        pca = PCA(n_components=2)
        pca.fit(class_means_flat)
        
        explained_var = pca.explained_variance_ratio_
        
        # Create visualization: one subplot per patch
        patch_size = int(np.sqrt(num_patches))
        assert patch_size * patch_size == num_patches, f"num_patches ({num_patches}) must be a perfect square"
        
        fig, axes = plt.subplots(patch_size, patch_size, figsize=(20, 20))
        
        for i in range(patch_size):
            for j in range(patch_size):
                patch_idx = i * patch_size + j
                ax = axes[i, j]
                
                # Transform data using PCA fitted on class means
                # If use_normalized=True, this uses normalized data; otherwise uses original data
                patch_data = residual_stream_normalized[:, patch_idx, :]  # [num_samples, dim]
                if isinstance(patch_data, torch.Tensor):
                    patch_data = patch_data.detach().cpu().numpy()
                else:
                    patch_data = np.array(patch_data)
                # Ensure 2D array
                if patch_data.ndim > 2:
                    patch_data = patch_data.reshape(-1, patch_data.shape[-1])
                patch_pca = pca.transform(patch_data)  # [num_samples, 2]
                
                # Plot data points (normalized if use_normalized=True)
                for cls in unique_classes:
                    mask = labels == cls
                    ax.scatter(patch_pca[mask, 0], patch_pca[mask, 1],
                              c=[class_to_color[cls]], s=5, alpha=0.6)
                
                # Remove all axis information
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.set_title('')
                ax.grid(False)
        
        # Add overall title
        norm_suffix = " (Normalized)" if use_normalized else ""
        title_type = "Output" if stream_type.endswith("_output") else "Residual Stream"
        stream_name = stream_type.replace("_output", "").upper() if stream_type.endswith("_output") else stream_type.upper()
        fig.suptitle(f'Layer {layer_idx} - {stream_name} {title_type} Class Mean PCA (Case 1: Global PCA){norm_suffix}\n'
                    f'PC0 vs PC1 - Explained var: PC0={explained_var[0]:.2%}, PC1={explained_var[1]:.2%}',
                    fontsize=14, y=0.98)
        
        # Add legend (only once)
        handles = [plt.Line2D([], [], marker='o', linestyle='', color=class_to_color[cls], 
                             markersize=8, label=f'Class {cls}') for cls in unique_classes]
        fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.99, 0.995), fontsize=10)
        
        plt.tight_layout(rect=[0, 0, 0.95, 0.90])
        
        norm_suffix_file = "_normalized" if use_normalized else ""
        layer_output_path = output_path.replace('.png', f'_layer_{layer_idx}_pc0_pc1{norm_suffix_file}.png')
        plt.savefig(layer_output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Class mean PCA Case 1 (Layer {layer_idx}, {stream_type}, normalized={use_normalized}) saved to: {layer_output_path}")
    
    print(f"Class mean PCA Case 1 visualization complete for all layers ({stream_type}, normalized={use_normalized})")


def visualize_class_mean_pca_case2(residual_streams, labels, output_path, stream_type="attn", use_normalized=False):
    """
    Case 2: Compute class means for each patch independently, then apply patch-wise PCA.
    Fit PCA on class means for each patch, but transform data (normalized if use_normalized=True) for scatter plot.
    Visualize data points in PCA space (PC0 vs PC1) for each patch.
    
    Args:
        residual_streams: dict mapping layer_idx -> [num_samples, num_patches, dim]
        labels: array of target class labels [num_samples]
        output_path: path to save the visualization
        stream_type: "attn" or "ffn" for labeling
        use_normalized: if True, normalize each sample before computing mean and for visualization
    """
    layer_indices = sorted(residual_streams.keys())
    unique_classes = np.unique(labels)
    num_classes = len(unique_classes)
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    class_to_color = {cls: colors[i] for i, cls in enumerate(unique_classes)}
    
    # Create a figure for each layer
    for layer_idx in layer_indices:
        residual_stream = residual_streams[layer_idx]  # [num_samples, num_patches, dim]
        num_samples, num_patches, dim = residual_stream.shape
        
        # Prepare data (normalize if needed)
        if use_normalized:
            # Normalize each sample (across patches and dims)
            # Flatten to [num_samples, num_patches * dim] for norm calculation
            residual_flat = residual_stream.view(num_samples, -1)  # [num_samples, num_patches * dim]
            norms = torch.norm(residual_flat, dim=1, keepdim=True)  # [num_samples, 1]
            # Reshape norms to [num_samples, 1, 1] for broadcasting
            norms = norms.view(num_samples, 1, 1)  # [num_samples, 1, 1]
            residual_stream_normalized = residual_stream / (norms + 1e-8)
        else:
            residual_stream_normalized = residual_stream
        
        # Compute class means for each patch independently (for PCA fitting)
        # class_means: [num_classes, num_patches, dim]
        class_means = []
        for cls in unique_classes:
            mask = labels == cls
            class_data = residual_stream_normalized[mask]  # [num_class_samples, num_patches, dim]
            
            # Compute mean across samples for each patch
            class_mean = class_data.mean(dim=0)  # [num_patches, dim]
            # Ensure it's 2D
            if class_mean.ndim == 1:
                class_mean = class_mean.unsqueeze(0)
            class_means.append(class_mean)
        
        class_means = torch.stack(class_means)  # [num_classes, num_patches, dim]
        # Ensure class_means is 3D: [num_classes, num_patches, dim]
        if class_means.ndim != 3:
            raise ValueError(f"class_means should be 3D but got shape {class_means.shape}")
        
        # Case 2: Apply PCA independently to each patch (fit on class means)
        pca_models = {}  # patch_idx -> PCA model
        explained_vars = {}  # patch_idx -> [2] explained variance ratios
        
        for patch_idx in range(num_patches):
            patch_class_means = class_means[:, patch_idx, :]  # [num_classes, dim]
            if isinstance(patch_class_means, torch.Tensor):
                patch_class_means = patch_class_means.detach().cpu().numpy()
            else:
                patch_class_means = np.array(patch_class_means)
            # Ensure 2D array
            if patch_class_means.ndim > 2:
                patch_class_means = patch_class_means.reshape(-1, patch_class_means.shape[-1])
            elif patch_class_means.ndim == 1:
                patch_class_means = patch_class_means.reshape(1, -1)
            pca = PCA(n_components=2)
            pca.fit(patch_class_means)  # Fit on class means
            pca_models[patch_idx] = pca
            explained_vars[patch_idx] = pca.explained_variance_ratio_
        
        # Create visualization: one subplot per patch
        patch_size = int(np.sqrt(num_patches))
        assert patch_size * patch_size == num_patches, f"num_patches ({num_patches}) must be a perfect square"
        
        fig, axes = plt.subplots(patch_size, patch_size, figsize=(20, 20))
        
        for i in range(patch_size):
            for j in range(patch_size):
                patch_idx = i * patch_size + j
                ax = axes[i, j]
                
                # Transform data using PCA fitted on class means
                # If use_normalized=True, this uses normalized data; otherwise uses original data
                patch_data = residual_stream_normalized[:, patch_idx, :]  # [num_samples, dim]
                if isinstance(patch_data, torch.Tensor):
                    patch_data = patch_data.detach().cpu().numpy()
                else:
                    patch_data = np.array(patch_data)
                # Ensure 2D array
                if patch_data.ndim > 2:
                    patch_data = patch_data.reshape(-1, patch_data.shape[-1])
                patch_pca = pca_models[patch_idx].transform(patch_data)  # [num_samples, 2]
                
                # Plot data points (normalized if use_normalized=True)
                for cls in unique_classes:
                    mask = labels == cls
                    ax.scatter(patch_pca[mask, 0], patch_pca[mask, 1],
                              c=[class_to_color[cls]], s=5, alpha=0.6)
                
                # Remove all axis information
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.set_title('')
                ax.grid(False)
        
        # Add overall title
        # Average explained variance across patches
        avg_explained_0 = np.mean([explained_vars[p][0] for p in range(num_patches)])
        avg_explained_1 = np.mean([explained_vars[p][1] for p in range(num_patches)])
        
        norm_suffix = " (Normalized)" if use_normalized else ""
        title_type = "Output" if stream_type.endswith("_output") else "Residual Stream"
        stream_name = stream_type.replace("_output", "").upper() if stream_type.endswith("_output") else stream_type.upper()
        fig.suptitle(f'Layer {layer_idx} - {stream_name} {title_type} Class Mean PCA (Case 2: Patch-wise PCA){norm_suffix}\n'
                    f'PC0 vs PC1 - Avg explained var: PC0={avg_explained_0:.2%}, PC1={avg_explained_1:.2%}',
                    fontsize=14, y=0.98)
        
        # Add legend (only once)
        handles = [plt.Line2D([], [], marker='o', linestyle='', color=class_to_color[cls], 
                             markersize=8, label=f'Class {cls}') for cls in unique_classes]
        fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.99, 0.995), fontsize=10)
        
        plt.tight_layout(rect=[0, 0, 0.95, 0.90])
        
        norm_suffix_file = "_normalized" if use_normalized else ""
        layer_output_path = output_path.replace('.png', f'_layer_{layer_idx}_pc0_pc1{norm_suffix_file}.png')
        plt.savefig(layer_output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Class mean PCA Case 2 (Layer {layer_idx}, {stream_type}, normalized={use_normalized}) saved to: {layer_output_path}")
    
    print(f"Class mean PCA Case 2 visualization complete for all layers ({stream_type}, normalized={use_normalized})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--train_fraction", type=float, default=0.9)
    parser.add_argument("--num_images", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to analyze")
    parser.add_argument("--output_dir", type=str, default="ffn_analysis/")
    args = parser.parse_args()
    
    # Create output directory with datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f'residual_pca_train_fraction_{args.train_fraction}-num_images_{args.num_images}-{timestamp}'
    results_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Loading model from: {args.model_path}")
    print(f"Results will be saved to: {results_dir}")
    
    # Initialize model
    model = DiT_Llama(
        3, 32, dim=256, n_layers=10, n_heads=8,
    ).cuda()
    
    # Load trained weights
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    
    model_size = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {model_size}, {model_size / 1e6:.2f}M")
    
    # Load dataset (train split)
    dataset = MNISTModularArithmeticDataset(
        p=10, 
        split='train', 
        train_fraction=args.train_fraction, 
        num_images=args.num_images
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Starting residual stream collection...")
    
    # Residual stream PCA visualization (Case 2)
    print("\n" + "="*50)
    print("Collecting residual stream representations...")
    residual_streams_attn, residual_streams_ffn, attn_outputs, ffn_outputs, labels = collect_residual_streams(
        model, dataloader, max_samples=args.max_samples
    )
    
    print(f"\nResidual stream shapes:")
    for layer_idx in sorted(residual_streams_attn.keys()):
        print(f"Layer {layer_idx} - Attn Residual: {residual_streams_attn[layer_idx].shape}, "
              f"FFN Residual: {residual_streams_ffn[layer_idx].shape}")
    
    print(f"\nOutput shapes:")
    for layer_idx in sorted(attn_outputs.keys()):
        print(f"Layer {layer_idx} - Attn Output: {attn_outputs[layer_idx].shape}, "
              f"FFN Output: {ffn_outputs[layer_idx].shape}")
    
    # # Visualize residual stream PCA - Case 1 (Global PCA) for attn
    # print("\n" + "="*50)
    # print("Visualizing residual stream PCA Case 1 (Global PCA) - after attn...")
    # residual_pca_attn_case1_output = os.path.join(results_dir, f"residual_pca_attn_case1.png")
    # visualize_residual_stream_pca_case1(residual_streams_attn, labels, residual_pca_attn_case1_output, stream_type="attn")
    
    # # Visualize residual stream PCA - Case 2 (Patch-wise PCA) for attn
    # print("\n" + "="*50)
    # print("Visualizing residual stream PCA Case 2 (Patch-wise PCA) - after attn...")
    # residual_pca_attn_case2_output = os.path.join(results_dir, f"residual_pca_attn_case2.png")
    # visualize_residual_stream_pca_case2(residual_streams_attn, labels, residual_pca_attn_case2_output, stream_type="attn")
    
    # # Visualize residual stream PCA - Case 1 (Global PCA) for ffn
    # print("\n" + "="*50)
    # print("Visualizing residual stream PCA Case 1 (Global PCA) - after ffn...")
    # residual_pca_ffn_case1_output = os.path.join(results_dir, f"residual_pca_ffn_case1.png")
    # visualize_residual_stream_pca_case1(residual_streams_ffn, labels, residual_pca_ffn_case1_output, stream_type="ffn")
    
    # # Visualize residual stream PCA - Case 2 (Patch-wise PCA) for ffn
    # print("\n" + "="*50)
    # print("Visualizing residual stream PCA Case 2 (Patch-wise PCA) - after ffn...")
    # residual_pca_ffn_case2_output = os.path.join(results_dir, f"residual_pca_ffn_case2.png")
    # visualize_residual_stream_pca_case2(residual_streams_ffn, labels, residual_pca_ffn_case2_output, stream_type="ffn")
    
    # # Visualize class mean PCA - Case 1 (Global PCA) for attn (mean)
    # print("\n" + "="*50)
    # print("Visualizing class mean PCA Case 1 (Global PCA) - after attn (mean)...")
    # class_mean_pca_attn_case1_output = os.path.join(results_dir, f"class_mean_pca_attn_case1.png")
    # visualize_class_mean_pca_case1(residual_streams_attn, labels, class_mean_pca_attn_case1_output, stream_type="attn", use_normalized=False)
    
    # # Visualize class mean PCA - Case 1 (Global PCA) for attn (normalized)
    # print("\n" + "="*50)
    # print("Visualizing class mean PCA Case 1 (Global PCA) - after attn (normalized)...")
    # class_mean_pca_attn_case1_norm_output = os.path.join(results_dir, f"class_mean_pca_attn_case1_normalized.png")
    # visualize_class_mean_pca_case1(residual_streams_attn, labels, class_mean_pca_attn_case1_norm_output, stream_type="attn", use_normalized=True)
    
    # # Visualize class mean PCA - Case 2 (Patch-wise PCA) for attn (mean)
    # print("\n" + "="*50)
    # print("Visualizing class mean PCA Case 2 (Patch-wise PCA) - after attn (mean)...")
    # class_mean_pca_attn_case2_output = os.path.join(results_dir, f"class_mean_pca_attn_case2.png")
    # visualize_class_mean_pca_case2(residual_streams_attn, labels, class_mean_pca_attn_case2_output, stream_type="attn", use_normalized=False)
    
    # # Visualize class mean PCA - Case 2 (Patch-wise PCA) for attn (normalized)
    # print("\n" + "="*50)
    # print("Visualizing class mean PCA Case 2 (Patch-wise PCA) - after attn (normalized)...")
    # class_mean_pca_attn_case2_norm_output = os.path.join(results_dir, f"class_mean_pca_attn_case2_normalized.png")
    # visualize_class_mean_pca_case2(residual_streams_attn, labels, class_mean_pca_attn_case2_norm_output, stream_type="attn", use_normalized=True)
    
    # # Visualize class mean PCA - Case 1 (Global PCA) for ffn (mean)
    # print("\n" + "="*50)
    # print("Visualizing class mean PCA Case 1 (Global PCA) - after ffn (mean)...")
    # class_mean_pca_ffn_case1_output = os.path.join(results_dir, f"class_mean_pca_ffn_case1.png")
    # visualize_class_mean_pca_case1(residual_streams_ffn, labels, class_mean_pca_ffn_case1_output, stream_type="ffn", use_normalized=False)
    
    # # Visualize class mean PCA - Case 1 (Global PCA) for ffn (normalized)
    # print("\n" + "="*50)
    # print("Visualizing class mean PCA Case 1 (Global PCA) - after ffn (normalized)...")
    # class_mean_pca_ffn_case1_norm_output = os.path.join(results_dir, f"class_mean_pca_ffn_case1_normalized.png")
    # visualize_class_mean_pca_case1(residual_streams_ffn, labels, class_mean_pca_ffn_case1_norm_output, stream_type="ffn", use_normalized=True)
    
    # # Visualize class mean PCA - Case 2 (Patch-wise PCA) for ffn (mean)
    # print("\n" + "="*50)
    # print("Visualizing class mean PCA Case 2 (Patch-wise PCA) - after ffn (mean)...")
    # class_mean_pca_ffn_case2_output = os.path.join(results_dir, f"class_mean_pca_ffn_case2.png")
    # visualize_class_mean_pca_case2(residual_streams_ffn, labels, class_mean_pca_ffn_case2_output, stream_type="ffn", use_normalized=False)
    
    # # Visualize class mean PCA - Case 2 (Patch-wise PCA) for ffn (normalized)
    # print("\n" + "="*50)
    # print("Visualizing class mean PCA Case 2 (Patch-wise PCA) - after ffn (normalized)...")
    # class_mean_pca_ffn_case2_norm_output = os.path.join(results_dir, f"class_mean_pca_ffn_case2_normalized.png")
    # visualize_class_mean_pca_case2(residual_streams_ffn, labels, class_mean_pca_ffn_case2_norm_output, stream_type="ffn", use_normalized=True)
    
    # Visualize class mean PCA - Case 2 (Patch-wise PCA) for attn_output (normalized)
    print("\n" + "="*50)
    print("Visualizing class mean PCA Case 2 (Patch-wise PCA) - attn_output (normalized)...")
    class_mean_pca_attn_output_case2_norm_output = os.path.join(results_dir, f"class_mean_pca_attn_output_case2_normalized.png")
    visualize_class_mean_pca_case2(attn_outputs, labels, class_mean_pca_attn_output_case2_norm_output, stream_type="attn_output", use_normalized=True)
    
    # Visualize class mean PCA - Case 2 (Patch-wise PCA) for ffn_output (normalized)
    print("\n" + "="*50)
    print("Visualizing class mean PCA Case 2 (Patch-wise PCA) - ffn_output (normalized)...")
    class_mean_pca_ffn_output_case2_norm_output = os.path.join(results_dir, f"class_mean_pca_ffn_output_case2_normalized.png")
    visualize_class_mean_pca_case2(ffn_outputs, labels, class_mean_pca_ffn_output_case2_norm_output, stream_type="ffn_output", use_normalized=True)
    
    print(f"\n{'='*50}")
    print(f"Analysis complete! Results saved to: {results_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

