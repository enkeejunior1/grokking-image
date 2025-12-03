import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from train import RF, MNISTModularArithmeticDataset
from model import DiT_Llama
from classifier import MNISTClassifier
import argparse


class MaskedDiT_Llama(nn.Module):
    """Wrapper for DiT_Llama that supports masking patch-layer-module outputs"""
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.mask_dict = {}  # {(layer_idx, patch_idx, module_type): True/False}
        self.hooks = []
        
    def set_mask(self, layer_idx, patch_idx, module_type, mask_value=True):
        """Set mask for a specific (layer, patch, module_type) combination"""
        key = (layer_idx, patch_idx, module_type)
        self.mask_dict[key] = mask_value
        
    def clear_masks(self):
        """Clear all masks"""
        self.mask_dict = {}
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def register_hooks(self):
        """Register forward hooks to apply masks"""
        self.clear_masks()
        
        def make_hook(layer_idx, module_type):
            def hook(module, input, output):
                # output shape: (batch, seq_len, dim)
                batch_size, seq_len, dim = output.shape
                output_clone = output.clone()
                
                # Check which patches need to be masked
                for patch_idx in range(seq_len):
                    if (layer_idx, patch_idx, module_type) in self.mask_dict:
                        if self.mask_dict[(layer_idx, patch_idx, module_type)]:
                            # Mask this patch: set output to zero
                            output_clone[:, patch_idx, :] = 0.0
                
                return output_clone
            return hook
        
        # Register hooks for attention and FFN outputs
        for layer_idx, layer in enumerate(self.base_model.layers):
            # Hook for attention output
            hook_attn = make_hook(layer_idx, 'attn')
            self.hooks.append(
                layer.attention.register_forward_hook(hook_attn)
            )
            # Hook for FFN output
            hook_ffn = make_hook(layer_idx, 'ffn')
            self.hooks.append(
                layer.feed_forward.register_forward_hook(hook_ffn)
            )
    
    def forward(self, x, t, y):
        return self.base_model(x, t, y)


def compute_l2_change(model, rf, dataloader, layer_idx, patch_idx, module_type, num_samples=100):
    """
    Compute L2 norm change when masking a specific (layer, patch, module_type)
    Returns average L2 change across samples
    """
    model.eval()
    l2_changes = []
    
    with torch.no_grad():
        sample_count = 0
        for _, x_src, _ in dataloader:
            if sample_count >= num_samples:
                break
                
            x_src = x_src.cuda()
            batch_size = x_src.size(0)
            x_init = torch.randn(batch_size, 1, 32, 32).cuda()
            
            # Forward pass without masking
            model.clear_masks()
            model.register_hooks()
            output_original = rf.sample(x_init, x_src, sample_steps=1)
            output_original = output_original[-1]
            
            # Forward pass with masking
            model.clear_masks()
            model.set_mask(layer_idx, patch_idx, module_type, mask_value=True)
            model.register_hooks()
            output_masked = rf.sample(x_init, x_src, sample_steps=1)
            output_masked = output_masked[-1]
            
            # Compute L2 change
            l2_change = torch.norm(output_original - output_masked, p=2, dim=(1, 2, 3)).mean().item()
            l2_changes.append(l2_change)
            
            sample_count += batch_size
    
    return np.mean(l2_changes) if l2_changes else 0.0


def evaluate_accuracy(model, rf, classifier, dataloader, sample_steps=1):
    """Evaluate accuracy with current masking"""
    model.eval()
    classifier.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for _, x_src, label_tgt in dataloader:
            x_src = x_src.cuda()
            label_tgt = label_tgt.cuda()
            batch_size = x_src.size(0)
            
            x_init = torch.randn(batch_size, 1, 32, 32).cuda()
            images = rf.sample(x_init, x_src, sample_steps=sample_steps)
            generated_imgs = images[-1]
            
            preds = classifier(generated_imgs)
            preds = preds.argmax(dim=1)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(label_tgt.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    accuracy = (all_preds == all_labels).mean()
    
    return accuracy


def compute_accuracy_change(model, rf, classifier, dataloader, layer_idx, patch_idx, module_type, 
                            baseline_acc, sample_steps=1):
    """
    Compute accuracy change when masking a specific (layer, patch, module_type)
    Returns the accuracy after masking
    """
    model.eval()
    classifier.eval()
    
    # Set mask for this specific module
    model.clear_masks()
    model.set_mask(layer_idx, patch_idx, module_type, mask_value=True)
    model.register_hooks()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for _, x_src, label_tgt in dataloader:
            x_src = x_src.cuda()
            label_tgt = label_tgt.cuda()
            batch_size = x_src.size(0)
            
            x_init = torch.randn(batch_size, 1, 32, 32).cuda()
            images = rf.sample(x_init, x_src, sample_steps=sample_steps)
            generated_imgs = images[-1]
            
            preds = classifier(generated_imgs)
            preds = preds.argmax(dim=1)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(label_tgt.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    accuracy = (all_preds == all_labels).mean()
    
    return accuracy


def oneshot_ablation_visualization(
    model_path,
    classifier_weights_path,
    output_dir,
    train_fraction=1.0,
    num_images=1,
    batch_size=256,
    sample_steps=1,
    split="valid",
    num_samples_for_l2=50,
):
    """
    Perform oneshot ablation: remove each module one at a time and visualize
    accuracy and L2 difference heatmaps.
    
    Visualization structure:
    - Patch: 16x16 grid (reshaped from 256 patches)
    - Layer: horizontal axis (x-axis)
    - Module: vertical axis (y-axis) - attn and ffn as two rows
    - Two heatmaps: one for accuracy, one for L2 difference
    """
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    base_model = DiT_Llama(3, 32, dim=256, n_layers=10, n_heads=8).to(device)
    base_model.load_state_dict(torch.load(model_path))
    base_model.eval()
    
    model = MaskedDiT_Llama(base_model)
    rf = RF(base_model, ln=True)
    
    # Load classifier
    classifier = MNISTClassifier().to(device)
    classifier.load_state_dict(torch.load(classifier_weights_path))
    classifier.eval()
    
    # Load dataset
    dataset = MNISTModularArithmeticDataset(
        p=10, split=split, train_fraction=train_fraction, num_images=num_images
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    # Get model dimensions
    patch_size = model.base_model.patch_size
    input_size = model.base_model.input_size
    num_patches = (input_size // patch_size) ** 2
    num_layers = len(model.base_model.layers)
    patch_grid_size = int(np.sqrt(num_patches))  # Should be 16 for 16x16
    
    print(f"Model configuration:")
    print(f"  Patch size: {patch_size}")
    print(f"  Input size: {input_size}")
    print(f"  Number of patches: {num_patches}")
    print(f"  Patch grid size: {patch_grid_size}x{patch_grid_size}")
    print(f"  Number of layers: {num_layers}")
    print(f"  Module types: ['attn', 'ffn']")
    
    # Compute baseline (no masking)
    print("\nComputing baseline accuracy and L2...")
    model.clear_masks()
    model.register_hooks()
    baseline_acc = evaluate_accuracy(model, rf, classifier, dataloader, sample_steps)
    print(f"Baseline accuracy: {baseline_acc:.4f}")
    
    # Initialize arrays to store results
    # Shape: (num_layers, num_patches, 2) where last dim is [attn, ffn]
    accuracy_matrix = np.zeros((num_layers, num_patches, 2))
    l2_matrix = np.zeros((num_layers, num_patches, 2))
    
    # Compute accuracy and L2 for each module
    total_modules = num_layers * num_patches * 2
    print(f"\nComputing accuracy and L2 for {total_modules} modules...")
    
    for layer_idx in tqdm(range(num_layers), desc="Layers"):
        for patch_idx in tqdm(range(num_patches), desc="Patches", leave=False):
            for module_idx, module_type in enumerate(['attn', 'ffn']):
                # Compute accuracy change
                acc = compute_accuracy_change(
                    model, rf, classifier, dataloader, 
                    layer_idx, patch_idx, module_type,
                    baseline_acc, sample_steps
                )
                accuracy_matrix[layer_idx, patch_idx, module_idx] = acc
                
                # Compute L2 change
                l2_change = compute_l2_change(
                    model, rf, dataloader, layer_idx, patch_idx, module_type,
                    num_samples=num_samples_for_l2
                )
                l2_matrix[layer_idx, patch_idx, module_idx] = l2_change
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Reshape for visualization
    # We want: (patch_row, patch_col, layer, module)
    # Then for each patch position, we have (layer, module) heatmap
    # But we want to show: for each patch position, layer on x-axis, module on y-axis
    # So we need: (patch_row, patch_col, module, layer) -> (patch_row * module, patch_col * layer)
    
    # Alternative approach: Create a grid where each cell is a patch position
    # and within each cell, we show a small heatmap of (layer, module)
    # But that might be too small. Let's try a different approach:
    
    # Create heatmaps where:
    # - Each row corresponds to a module type (attn, ffn)
    # - Each column corresponds to a layer
    # - For each (module, layer) combination, we show a 16x16 grid of patches
    
    # Reshape: (num_layers, num_patches, 2) -> (2, num_layers, patch_grid_size, patch_grid_size)
    accuracy_reshaped = accuracy_matrix.transpose(2, 0, 1).reshape(2, num_layers, patch_grid_size, patch_grid_size)
    l2_reshaped = l2_matrix.transpose(2, 0, 1).reshape(2, num_layers, patch_grid_size, patch_grid_size)
    
    # Create visualization: For each module type, create a heatmap
    # Shape: (module, patch_row, patch_col, layer) -> we want to show this as
    # (module * patch_row, layer * patch_col) or similar
    
    # Better approach: Create a figure where:
    # - Rows: module types (attn, ffn) - 2 rows
    # - Columns: layers - num_layers columns
    # - Each subplot shows a 16x16 patch grid heatmap
    
    # Figure 1: Accuracy heatmap
    fig, axes = plt.subplots(2, num_layers, figsize=(2 * num_layers, 4))
    if num_layers == 1:
        axes = axes.reshape(2, 1)
    
    for module_idx, module_type in enumerate(['attn', 'ffn']):
        for layer_idx in range(num_layers):
            ax = axes[module_idx, layer_idx]
            patch_heatmap = accuracy_reshaped[module_idx, layer_idx, :, :]  # (16, 16)
            im = ax.imshow(patch_heatmap, cmap='viridis', aspect='auto')
            ax.set_title(f'L{layer_idx}', fontsize=8)
            if layer_idx == 0:
                ax.set_ylabel(module_type.upper(), fontsize=10)
            if module_idx == 1:
                ax.set_xlabel('Patch Col', fontsize=7)
            if module_idx == 0:
                ax.set_xticks([])
            else:
                ax.set_xticks([0, patch_grid_size-1])
            ax.set_yticks([0, patch_grid_size-1])
            plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.suptitle('Accuracy Heatmap: Module (rows) x Layer (cols) x Patch (16x16)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'oneshot_accuracy_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved accuracy heatmap: {os.path.join(output_dir, 'oneshot_accuracy_heatmap.png')}")
    
    # Print L2 statistics for debugging
    print(f"\nL2 Statistics:")
    print(f"  Min: {l2_matrix.min():.8f}")
    print(f"  Max: {l2_matrix.max():.8f}")
    print(f"  Mean: {l2_matrix.mean():.8f}")
    print(f"  Median: {np.median(l2_matrix):.8f}")
    print(f"  Std: {l2_matrix.std():.8f}")
    print(f"  25th percentile: {np.percentile(l2_matrix, 25):.8f}")
    print(f"  75th percentile: {np.percentile(l2_matrix, 75):.8f}")
    print(f"  95th percentile: {np.percentile(l2_matrix, 95):.8f}")
    print(f"  99th percentile: {np.percentile(l2_matrix, 99):.8f}")
    
    # Figure 2: L2 difference heatmap (raw values)
    fig, axes = plt.subplots(2, num_layers, figsize=(2 * num_layers, 4))
    if num_layers == 1:
        axes = axes.reshape(2, 1)
    
    for module_idx, module_type in enumerate(['attn', 'ffn']):
        for layer_idx in range(num_layers):
            ax = axes[module_idx, layer_idx]
            patch_heatmap = l2_reshaped[module_idx, layer_idx, :, :]  # (16, 16)
            im = ax.imshow(patch_heatmap, cmap='plasma', aspect='auto')
            ax.set_title(f'L{layer_idx}', fontsize=8)
            if layer_idx == 0:
                ax.set_ylabel(module_type.upper(), fontsize=10)
            if module_idx == 1:
                ax.set_xlabel('Patch Col', fontsize=7)
            if module_idx == 0:
                ax.set_xticks([])
            else:
                ax.set_xticks([0, patch_grid_size-1])
            ax.set_yticks([0, patch_grid_size-1])
            plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.suptitle('L2 Difference Heatmap (Raw): Module (rows) x Layer (cols) x Patch (16x16)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'oneshot_l2_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved L2 difference heatmap (raw): {os.path.join(output_dir, 'oneshot_l2_heatmap.png')}")
    
    # Figure 2b: L2 difference heatmap (normalized with percentiles)
    # Use percentile-based normalization to handle outliers
    l2_percentile_min = np.percentile(l2_matrix, 5)  # Use 5th percentile as min
    l2_percentile_max = np.percentile(l2_matrix, 95)  # Use 95th percentile as max
    
    fig, axes = plt.subplots(2, num_layers, figsize=(2 * num_layers, 4))
    if num_layers == 1:
        axes = axes.reshape(2, 1)
    
    for module_idx, module_type in enumerate(['attn', 'ffn']):
        for layer_idx in range(num_layers):
            ax = axes[module_idx, layer_idx]
            patch_heatmap = l2_reshaped[module_idx, layer_idx, :, :]  # (16, 16)
            im = ax.imshow(patch_heatmap, cmap='plasma', aspect='auto', 
                          vmin=l2_percentile_min, vmax=l2_percentile_max)
            ax.set_title(f'L{layer_idx}', fontsize=8)
            if layer_idx == 0:
                ax.set_ylabel(module_type.upper(), fontsize=10)
            if module_idx == 1:
                ax.set_xlabel('Patch Col', fontsize=7)
            if module_idx == 0:
                ax.set_xticks([])
            else:
                ax.set_xticks([0, patch_grid_size-1])
            ax.set_yticks([0, patch_grid_size-1])
            plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.suptitle(f'L2 Difference Heatmap (Normalized 5-95%): Module (rows) x Layer (cols) x Patch (16x16)\nRange: [{l2_percentile_min:.6f}, {l2_percentile_max:.6f}]', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'oneshot_l2_heatmap_normalized.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved L2 difference heatmap (normalized): {os.path.join(output_dir, 'oneshot_l2_heatmap_normalized.png')}")
    
    # Figure 2c: L2 difference heatmap (log scale)
    # Add small epsilon to avoid log(0)
    l2_log = np.log10(l2_matrix + 1e-10)
    l2_log_reshaped = l2_log.transpose(2, 0, 1).reshape(2, num_layers, patch_grid_size, patch_grid_size)
    
    fig, axes = plt.subplots(2, num_layers, figsize=(2 * num_layers, 4))
    if num_layers == 1:
        axes = axes.reshape(2, 1)
    
    for module_idx, module_type in enumerate(['attn', 'ffn']):
        for layer_idx in range(num_layers):
            ax = axes[module_idx, layer_idx]
            patch_heatmap = l2_log_reshaped[module_idx, layer_idx, :, :]  # (16, 16)
            im = ax.imshow(patch_heatmap, cmap='plasma', aspect='auto')
            ax.set_title(f'L{layer_idx}', fontsize=8)
            if layer_idx == 0:
                ax.set_ylabel(module_type.upper(), fontsize=10)
            if module_idx == 1:
                ax.set_xlabel('Patch Col', fontsize=7)
            if module_idx == 0:
                ax.set_xticks([])
            else:
                ax.set_xticks([0, patch_grid_size-1])
            ax.set_yticks([0, patch_grid_size-1])
            plt.colorbar(im, ax=ax, fraction=0.046, label='log10(L2)')
    
    plt.suptitle('L2 Difference Heatmap (Log Scale): Module (rows) x Layer (cols) x Patch (16x16)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'oneshot_l2_heatmap_log.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved L2 difference heatmap (log scale): {os.path.join(output_dir, 'oneshot_l2_heatmap_log.png')}")
    
    # Alternative visualization: Aggregate across patches to show layer x module
    # Average across all patches for each (layer, module) combination
    accuracy_agg = accuracy_matrix.mean(axis=1)  # (num_layers, 2)
    l2_agg = l2_matrix.mean(axis=1)  # (num_layers, 2)
    
    # Aggregated heatmap - raw values
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy aggregated
    im1 = axes[0].imshow(accuracy_agg.T, cmap='viridis', aspect='auto')
    axes[0].set_xlabel('Layer', fontsize=12)
    axes[0].set_ylabel('Module', fontsize=12)
    axes[0].set_yticks([0, 1])
    axes[0].set_yticklabels(['attn', 'ffn'])
    axes[0].set_title('Accuracy (averaged across patches)', fontsize=12)
    plt.colorbar(im1, ax=axes[0])
    
    # L2 aggregated (raw)
    im2 = axes[1].imshow(l2_agg.T, cmap='plasma', aspect='auto')
    axes[1].set_xlabel('Layer', fontsize=12)
    axes[1].set_ylabel('Module', fontsize=12)
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(['attn', 'ffn'])
    axes[1].set_title('L2 Difference (averaged across patches, raw)', fontsize=12)
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'oneshot_aggregated_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved aggregated heatmap (raw): {os.path.join(output_dir, 'oneshot_aggregated_heatmap.png')}")
    
    # Aggregated heatmap - normalized
    l2_agg_percentile_min = np.percentile(l2_agg, 5)
    l2_agg_percentile_max = np.percentile(l2_agg, 95)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy aggregated
    im1 = axes[0].imshow(accuracy_agg.T, cmap='viridis', aspect='auto')
    axes[0].set_xlabel('Layer', fontsize=12)
    axes[0].set_ylabel('Module', fontsize=12)
    axes[0].set_yticks([0, 1])
    axes[0].set_yticklabels(['attn', 'ffn'])
    axes[0].set_title('Accuracy (averaged across patches)', fontsize=12)
    plt.colorbar(im1, ax=axes[0])
    
    # L2 aggregated (normalized)
    im2 = axes[1].imshow(l2_agg.T, cmap='plasma', aspect='auto',
                         vmin=l2_agg_percentile_min, vmax=l2_agg_percentile_max)
    axes[1].set_xlabel('Layer', fontsize=12)
    axes[1].set_ylabel('Module', fontsize=12)
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(['attn', 'ffn'])
    axes[1].set_title(f'L2 Difference (normalized 5-95%, range: [{l2_agg_percentile_min:.6f}, {l2_agg_percentile_max:.6f}])', fontsize=12)
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'oneshot_aggregated_heatmap_normalized.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved aggregated heatmap (normalized): {os.path.join(output_dir, 'oneshot_aggregated_heatmap_normalized.png')}")
    
    # Aggregated heatmap - log scale
    l2_agg_log = np.log10(l2_agg + 1e-10)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy aggregated
    im1 = axes[0].imshow(accuracy_agg.T, cmap='viridis', aspect='auto')
    axes[0].set_xlabel('Layer', fontsize=12)
    axes[0].set_ylabel('Module', fontsize=12)
    axes[0].set_yticks([0, 1])
    axes[0].set_yticklabels(['attn', 'ffn'])
    axes[0].set_title('Accuracy (averaged across patches)', fontsize=12)
    plt.colorbar(im1, ax=axes[0])
    
    # L2 aggregated (log scale)
    im2 = axes[1].imshow(l2_agg_log.T, cmap='plasma', aspect='auto')
    axes[1].set_xlabel('Layer', fontsize=12)
    axes[1].set_ylabel('Module', fontsize=12)
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(['attn', 'ffn'])
    axes[1].set_title('L2 Difference (log scale, averaged across patches)', fontsize=12)
    plt.colorbar(im2, ax=axes[1], label='log10(L2)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'oneshot_aggregated_heatmap_log.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved aggregated heatmap (log scale): {os.path.join(output_dir, 'oneshot_aggregated_heatmap_log.png')}")
    
    # Save results to file
    results_path = os.path.join(output_dir, 'oneshot_ablation_results.npz')
    np.savez(
        results_path,
        accuracy_matrix=accuracy_matrix,
        l2_matrix=l2_matrix,
        baseline_accuracy=baseline_acc,
        num_layers=num_layers,
        num_patches=num_patches,
        patch_grid_size=patch_grid_size
    )
    print(f"\nSaved results to: {results_path}")
    print(f"\nOneshot ablation visualization complete!")
    print(f"Baseline accuracy: {baseline_acc:.4f}")
    print(f"Accuracy range: [{accuracy_matrix.min():.4f}, {accuracy_matrix.max():.4f}]")
    print(f"L2 difference range: [{l2_matrix.min():.6f}, {l2_matrix.max():.6f}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier_weights_path", type=str, default="weights/mnist_classifier_weights.pth")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--train_fraction", type=float, default=1.0)
    parser.add_argument("--output_dir", type=str, default="ablation_results/")
    parser.add_argument("--num_images", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--sample_steps", type=int, default=1, help="Number of sampling steps")
    parser.add_argument("--split", type=str, default="valid", choices=["train", "valid"], help="Dataset split to use")
    parser.add_argument("--num_samples_for_l2", type=int, default=50, help="Number of samples to use for L2 computation")
    
    args = parser.parse_args()
    
    # Create experiment-specific output directory
    exp_name = f'oneshot_ablation_train_fraction_{args.train_fraction}-num_images_{args.num_images}-steps_{args.sample_steps}'
    output_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(output_dir, exist_ok=True)
    
    oneshot_ablation_visualization(
        model_path=args.model_path,
        classifier_weights_path=args.classifier_weights_path,
        output_dir=output_dir,
        train_fraction=args.train_fraction,
        num_images=args.num_images,
        batch_size=args.batch_size,
        sample_steps=args.sample_steps,
        split=args.split,
        num_samples_for_l2=args.num_samples_for_l2,
    )
