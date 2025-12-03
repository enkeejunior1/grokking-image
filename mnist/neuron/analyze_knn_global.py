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
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist

from model import DiT_Llama, modulate
from train import MNISTModularArithmeticDataset


def collect_residual_streams(model, dataloader, max_samples=None):
    """
    Collect residual stream representations after attn and ffn for each layer.
    Also collects attention output, FFN intermediate (4d), and FFN output.
    Uses Case 2 approach: keeps patch dimension.
    
    Returns:
        residual_streams_attn: dict mapping layer_idx -> [num_samples, num_patches, dim]
        residual_streams_ffn: dict mapping layer_idx -> [num_samples, num_patches, dim]
        attn_outputs: dict mapping layer_idx -> [num_samples, num_patches, dim]
        ffn_intermediates: dict mapping layer_idx -> [num_samples, num_patches, 4*dim]
        ffn_outputs: dict mapping layer_idx -> [num_samples, num_patches, dim]
        labels: array of target class labels [num_samples]
    """
    import torch.nn.functional as F
    
    # We'll manually track residual streams by modifying the forward pass
    # Store original forward method
    original_forwards = {}
    for layer_idx, layer in enumerate(model.layers):
        original_forwards[layer_idx] = layer.forward
    
    residual_streams_attn = {}  # layer_idx -> list of [batch_size, num_patches, dim]
    residual_streams_ffn = {}   # layer_idx -> list of [batch_size, num_patches, dim]
    attn_outputs = {}            # layer_idx -> list of [batch_size, num_patches, dim]
    ffn_intermediates = {}       # layer_idx -> list of [batch_size, num_patches, 4*dim]
    ffn_outputs = {}             # layer_idx -> list of [batch_size, num_patches, dim]
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
            
            # Store attention output (before residual connection)
            if layer_idx not in attn_outputs:
                attn_outputs[layer_idx] = []
            attn_outputs[layer_idx].append(attn_out.detach().cpu())
            
            # Store residual stream after attention
            if layer_idx not in residual_streams_attn:
                residual_streams_attn[layer_idx] = []
            residual_streams_attn[layer_idx].append(x_after_attn.detach().cpu())
            
            # Run FFN - need to capture intermediate 4d activation
            if adaln_input is not None:
                ffn_input = modulate(layer.ffn_norm(x_after_attn), shift_mlp, scale_mlp)
            else:
                ffn_input = layer.ffn_norm(x_after_attn)
            
            # Get FFN intermediate (4d): silu(w1(x)) * w3(x)
            w1_out = layer.feed_forward.w1(ffn_input)
            w3_out = layer.feed_forward.w3(ffn_input)
            ffn_intermediate = F.silu(w1_out) * w3_out  # [batch_size, num_patches, 4*dim]
            
            # Get FFN output
            ffn_out = layer.feed_forward.w2(ffn_intermediate)
            
            if adaln_input is not None:
                x_after_ffn = x_after_attn + gate_mlp.unsqueeze(1) * ffn_out
            else:
                x_after_ffn = x_after_attn + ffn_out
            
            # Store FFN intermediate (4d)
            if layer_idx not in ffn_intermediates:
                ffn_intermediates[layer_idx] = []
            ffn_intermediates[layer_idx].append(ffn_intermediate.detach().cpu())
            
            # Store FFN output (before residual connection)
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
    for layer_idx in ffn_intermediates:
        ffn_intermediates[layer_idx] = torch.cat(ffn_intermediates[layer_idx], dim=0)
    for layer_idx in ffn_outputs:
        ffn_outputs[layer_idx] = torch.cat(ffn_outputs[layer_idx], dim=0)
    
    labels = np.concatenate(labels_list)
    
    return residual_streams_attn, residual_streams_ffn, attn_outputs, ffn_intermediates, ffn_outputs, labels


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
            fig.suptitle(f'Layer {layer_idx} - {stream_type.upper()} Residual Stream (Case 1: Global PCA)\n'
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
            
            fig.suptitle(f'Layer {layer_idx} - {stream_type.upper()} Residual Stream (Case 2: Patch-wise PCA)\n'
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


def compute_distance_matrix(features, metric="cosine"):
    """
    Compute pairwise distance/similarity matrix.
    
    Args:
        features: [num_samples, dim] numpy array
        metric: "cosine", "dot", or "l2"
    
    Returns:
        distance_matrix: [num_samples, num_samples]
        For cosine and dot: higher value = more similar (use argmax for nearest neighbor)
        For l2: lower value = more similar (use argmin for nearest neighbor)
    """
    if metric == "cosine":
        # Cosine similarity: range [-1, 1], higher = more similar
        dist_matrix = cosine_similarity(features)
        np.fill_diagonal(dist_matrix, -np.inf)  # Exclude self
        return dist_matrix
    elif metric == "dot":
        # Dot product: higher = more similar (if features are normalized, similar to cosine)
        dist_matrix = np.dot(features, features.T)
        np.fill_diagonal(dist_matrix, -np.inf)  # Exclude self
        return dist_matrix
    elif metric == "l2":
        # L2 distance: lower = more similar
        dist_matrix = cdist(features, features, metric='euclidean')
        np.fill_diagonal(dist_matrix, np.inf)  # Exclude self
        return dist_matrix
    else:
        raise ValueError(f"Unknown metric: {metric}")


def analyze_knn_same_class(residual_streams, labels, output_path, stream_type="attn", metric="cosine"):
    """
    Analyze whether the nearest neighbor of each sample belongs to the same class.
    
    Two cases:
    - Case 1: Average all patches to get a single feature vector per sample
    - Case 2: Analyze each patch independently
    
    Args:
        residual_streams: dict mapping layer_idx -> [num_samples, num_patches, dim]
        labels: array of target class labels [num_samples]
        output_path: path to save the results
        stream_type: "attn" or "ffn" for labeling
        metric: "cosine", "dot", or "l2" for distance/similarity metric
    """
    layer_indices = sorted(residual_streams.keys())
    num_layers = len(layer_indices)
    
    # Results storage
    results_case1 = {}  # layer_idx -> accuracy
    results_case2 = {}  # layer_idx -> dict: patch_idx -> accuracy
    
    metric_name = {"cosine": "cosine similarity", "dot": "dot product", "l2": "L2 distance"}[metric]
    print(f"\nAnalyzing KNN same-class accuracy using {metric_name} ({stream_type})...")
    
    for layer_idx in tqdm(layer_indices, desc=f"Processing layers ({stream_type}, {metric})"):
        residual_stream = residual_streams[layer_idx]  # [num_samples, num_patches, dim]
        num_samples, num_patches, dim = residual_stream.shape
        
        # Case 1: Average all patches to get a single feature vector per sample
        # [num_samples, num_patches, dim] -> [num_samples, dim]
        features_avg = residual_stream.mean(dim=1).numpy()  # [num_samples, dim]
        
        # Compute distance/similarity matrix
        dist_matrix = compute_distance_matrix(features_avg, metric=metric)
        
        # Find nearest neighbor for each sample
        if metric == "l2":
            # For L2, lower is better (closer)
            nearest_neighbor_indices = np.argmin(dist_matrix, axis=1)  # [num_samples]
        else:
            # For cosine and dot, higher is better (more similar)
            nearest_neighbor_indices = np.argmax(dist_matrix, axis=1)  # [num_samples]
        
        # Check if nearest neighbor has same class
        same_class_mask = labels == labels[nearest_neighbor_indices]
        accuracy_case1 = same_class_mask.mean()
        results_case1[layer_idx] = accuracy_case1
        
        # Case 2: Analyze each patch independently
        patch_accuracies = {}
        for patch_idx in range(num_patches):
            patch_features = residual_stream[:, patch_idx, :].numpy()  # [num_samples, dim]
            
            # Compute distance/similarity matrix for this patch
            dist_matrix_patch = compute_distance_matrix(patch_features, metric=metric)
            
            # Find nearest neighbor for each sample
            if metric == "l2":
                nearest_neighbor_indices_patch = np.argmin(dist_matrix_patch, axis=1)
            else:
                nearest_neighbor_indices_patch = np.argmax(dist_matrix_patch, axis=1)
            
            # Check if nearest neighbor has same class
            same_class_mask_patch = labels == labels[nearest_neighbor_indices_patch]
            accuracy_patch = same_class_mask_patch.mean()
            patch_accuracies[patch_idx] = accuracy_patch
        
        results_case2[layer_idx] = patch_accuracies
    
    # Print results
    metric_name = {"cosine": "cosine similarity", "dot": "dot product", "l2": "L2 distance"}[metric]
    print(f"\n{'='*60}")
    print(f"KNN Same-Class Analysis Results ({stream_type.upper()}) - {metric_name.upper()}")
    print(f"{'='*60}")
    print(f"\nCase 1: Averaged patches (single feature vector per sample)")
    print(f"{'Layer':<10} {'Accuracy':<15}")
    print(f"{'-'*25}")
    for layer_idx in layer_indices:
        print(f"{layer_idx:<10} {results_case1[layer_idx]:.4f} ({results_case1[layer_idx]*100:.2f}%)")
    
    print(f"\nCase 2: Patch-wise analysis")
    print(f"{'Layer':<10} {'Mean Acc':<15} {'Std Acc':<15} {'Min Acc':<15} {'Max Acc':<15}")
    print(f"{'-'*70}")
    for layer_idx in layer_indices:
        patch_accs = list(results_case2[layer_idx].values())
        mean_acc = np.mean(patch_accs)
        std_acc = np.std(patch_accs)
        min_acc = np.min(patch_accs)
        max_acc = np.max(patch_accs)
        print(f"{layer_idx:<10} {mean_acc:.4f} ({mean_acc*100:.2f}%)  {std_acc:.4f}  {min_acc:.4f} ({min_acc*100:.2f}%)  {max_acc:.4f} ({max_acc*100:.2f}%)")
    
    # Visualize results
    visualize_knn_results(results_case1, results_case2, output_path, stream_type, metric)
    
    # Save results to file
    save_knn_results(results_case1, results_case2, output_path, stream_type, metric)
    
    return results_case1, results_case2


def visualize_knn_results(results_case1, results_case2, output_path, stream_type="attn", metric="cosine"):
    """Visualize KNN same-class accuracy results."""
    layer_indices = sorted(results_case1.keys())
    metric_name = {"cosine": "cosine similarity", "dot": "dot product", "l2": "L2 distance"}[metric]
    
    # Create figure with three subplots
    fig = plt.figure(figsize=(18, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.2], hspace=0.3)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    
    # Case 1: Averaged patches
    layers = list(layer_indices)
    accuracies_case1 = [results_case1[layer_idx] for layer_idx in layers]
    
    ax1.plot(layers, accuracies_case1, marker='o', linewidth=2, markersize=8)
    ax1.set_xlabel('Layer', fontsize=12)
    ax1.set_ylabel('KNN Same-Class Accuracy', fontsize=12)
    ax1.set_title(f'Case 1: Averaged Patches ({stream_type.upper()})\n{metric_name}', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])
    ax1.set_xticks(layers)
    
    # Case 2: Patch-wise (show mean and std)
    mean_accs = [np.mean(list(results_case2[layer_idx].values())) for layer_idx in layers]
    std_accs = [np.std(list(results_case2[layer_idx].values())) for layer_idx in layers]
    
    ax2.errorbar(layers, mean_accs, yerr=std_accs, marker='o', linewidth=2, 
                 markersize=8, capsize=5, capthick=2)
    ax2.set_xlabel('Layer', fontsize=12)
    ax2.set_ylabel('KNN Same-Class Accuracy (Mean ± Std)', fontsize=12)
    ax2.set_title(f'Case 2: Patch-wise Mean ({stream_type.upper()})\n{metric_name}', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    ax2.set_xticks(layers)
    
    # Case 2: Show summary statistics in third subplot
    # Get number of patches from first layer
    first_layer_idx = layer_indices[0]
    num_patches = len(results_case2[first_layer_idx])
    patch_size = int(np.sqrt(num_patches))
    
    # Calculate min, max, and std for each layer
    min_accs = [np.min(list(results_case2[layer_idx].values())) for layer_idx in layers]
    max_accs = [np.max(list(results_case2[layer_idx].values())) for layer_idx in layers]
    
    ax3.fill_between(layers, min_accs, max_accs, alpha=0.3, label='Range (Min-Max)')
    ax3.plot(layers, mean_accs, marker='o', linewidth=2, markersize=8, label='Mean')
    ax3.plot(layers, max_accs, marker='s', linewidth=2, markersize=8, label='Max (patch-wise)', linestyle='--')
    ax3.set_xlabel('Layer', fontsize=12)
    ax3.set_ylabel('KNN Same-Class Accuracy', fontsize=12)
    ax3.set_title(f'Case 2: Patch-wise Statistics ({stream_type.upper()})\n{metric_name}', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1.05])
    ax3.set_xticks(layers)
    ax3.legend()
    
    plt.tight_layout()
    
    # Save figure
    output_path_viz = output_path.replace('.txt', f'_knn_accuracy_{metric}.png')
    plt.savefig(output_path_viz, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nKNN accuracy visualization saved to: {output_path_viz}")


def save_knn_results(results_case1, results_case2, output_path, stream_type="attn", metric="cosine"):
    """Save KNN results to a text file."""
    layer_indices = sorted(results_case1.keys())
    metric_name = {"cosine": "cosine similarity", "dot": "dot product", "l2": "L2 distance"}[metric]
    
    output_path_txt = output_path.replace('.png', f'_knn_results_{metric}.txt')
    with open(output_path_txt, 'w') as f:
        f.write(f"KNN Same-Class Analysis Results ({stream_type.upper()}) - {metric_name.upper()}\n")
        f.write("="*60 + "\n\n")
        
        f.write("Case 1: Averaged patches (single feature vector per sample)\n")
        f.write(f"{'Layer':<10} {'Accuracy':<15}\n")
        f.write("-"*25 + "\n")
        for layer_idx in layer_indices:
            f.write(f"{layer_idx:<10} {results_case1[layer_idx]:.6f} ({results_case1[layer_idx]*100:.2f}%)\n")
        
        f.write("\nCase 2: Patch-wise analysis\n")
        f.write(f"{'Layer':<10} {'Mean Acc':<15} {'Std Acc':<15} {'Min Acc':<15} {'Max Acc':<15}\n")
        f.write("-"*70 + "\n")
        for layer_idx in layer_indices:
            patch_accs = list(results_case2[layer_idx].values())
            mean_acc = np.mean(patch_accs)
            std_acc = np.std(patch_accs)
            min_acc = np.min(patch_accs)
            max_acc = np.max(patch_accs)
            f.write(f"{layer_idx:<10} {mean_acc:.6f} ({mean_acc*100:.2f}%)  {std_acc:.6f}  "
                   f"{min_acc:.6f} ({min_acc*100:.2f}%)  {max_acc:.6f} ({max_acc*100:.2f}%)\n")
        
        # Also save per-patch results for Case 2
        f.write("\n\nCase 2: Detailed patch-wise results\n")
        f.write("="*60 + "\n")
        for layer_idx in layer_indices:
            f.write(f"\nLayer {layer_idx}:\n")
            patch_accs = results_case2[layer_idx]
            patch_size = int(np.sqrt(len(patch_accs)))
            for i in range(patch_size):
                for j in range(patch_size):
                    patch_idx = i * patch_size + j
                    if patch_idx in patch_accs:
                        f.write(f"  Patch [{i:2d},{j:2d}] (idx {patch_idx:3d}): {patch_accs[patch_idx]:.6f} ({patch_accs[patch_idx]*100:.2f}%)\n")
    
    print(f"KNN results saved to: {output_path_txt}")


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
    print("Collecting residual stream representations (Case 2)...")
    residual_streams_attn, residual_streams_ffn, attn_outputs, ffn_intermediates, ffn_outputs, labels = collect_residual_streams(
        model, dataloader, max_samples=args.max_samples
    )
    
    print(f"\nResidual stream shapes:")
    for layer_idx in sorted(residual_streams_attn.keys()):
        print(f"Layer {layer_idx} - Attn residual: {residual_streams_attn[layer_idx].shape}, "
              f"FFN residual: {residual_streams_ffn[layer_idx].shape}")
        print(f"  Attn output: {attn_outputs[layer_idx].shape}, "
              f"FFN intermediate: {ffn_intermediates[layer_idx].shape}, "
              f"FFN output: {ffn_outputs[layer_idx].shape}")
    
    # KNN same-class analysis with different metrics
    metrics = ["cosine", "dot", "l2"]
    
    for metric in metrics:
        metric_name = {"cosine": "cosine similarity", "dot": "dot product", "l2": "L2 distance"}[metric]
        
        # KNN same-class analysis - after attn residual
        print("\n" + "="*50)
        print(f"Analyzing KNN same-class accuracy ({metric_name}) - after attn residual...")
        knn_attn_output = os.path.join(results_dir, f"knn_attn_residual_results_{metric}.png")
        analyze_knn_same_class(residual_streams_attn, labels, knn_attn_output, stream_type="attn_residual", metric=metric)
        
        # KNN same-class analysis - after ffn residual
        print("\n" + "="*50)
        print(f"Analyzing KNN same-class accuracy ({metric_name}) - after ffn residual...")
        knn_ffn_output = os.path.join(results_dir, f"knn_ffn_residual_results_{metric}.png")
        analyze_knn_same_class(residual_streams_ffn, labels, knn_ffn_output, stream_type="ffn_residual", metric=metric)
        
        # KNN same-class analysis - attention output
        print("\n" + "="*50)
        print(f"Analyzing KNN same-class accuracy ({metric_name}) - attention output...")
        knn_attn_out_output = os.path.join(results_dir, f"knn_attn_output_results_{metric}.png")
        analyze_knn_same_class(attn_outputs, labels, knn_attn_out_output, stream_type="attn_output", metric=metric)
        
        # KNN same-class analysis - FFN intermediate (4d)
        print("\n" + "="*50)
        print(f"Analyzing KNN same-class accuracy ({metric_name}) - FFN intermediate (4d)...")
        knn_ffn_inter_output = os.path.join(results_dir, f"knn_ffn_intermediate_results_{metric}.png")
        analyze_knn_same_class(ffn_intermediates, labels, knn_ffn_inter_output, stream_type="ffn_intermediate", metric=metric)
        
        # KNN same-class analysis - FFN output
        print("\n" + "="*50)
        print(f"Analyzing KNN same-class accuracy ({metric_name}) - FFN output...")
        knn_ffn_out_output = os.path.join(results_dir, f"knn_ffn_output_results_{metric}.png")
        analyze_knn_same_class(ffn_outputs, labels, knn_ffn_out_output, stream_type="ffn_output", metric=metric)
    
    print(f"\n{'='*50}")
    print(f"Analysis complete! Results saved to: {results_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

