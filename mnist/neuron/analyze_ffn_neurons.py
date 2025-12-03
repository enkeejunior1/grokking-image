import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import argparse
from datetime import datetime

from model import DiT_Llama
from train import MNISTModularArithmeticDataset, RF


class FFNActivationCollector:
    """Collects FFN intermediate activations (after SiLU gating) for each layer."""
    
    def __init__(self, model):
        self.model = model
        self.activations = []  # List of (layer_idx, activation_tensor)
        self.hooks = []
        
    def register_hooks(self):
        """Register forward hooks to collect FFN activations."""
        for layer_idx, layer in enumerate(self.model.layers):
            hook = self._create_hook(layer_idx)
            handle = layer.feed_forward.register_forward_hook(hook)
            self.hooks.append(handle)
    
    def _create_hook(self, layer_idx):
        def hook(module, input, output):
            # Input to w2 is the activation after SiLU gating: F.silu(w1(x)) * w3(x)
            # We need to recompute this from the input
            x = input[0]  # Shape: [batch_size, num_patches, dim]
            x1 = module.w1(x)  # [batch_size, num_patches, hidden_dim]
            x3 = module.w3(x)  # [batch_size, num_patches, hidden_dim]
            activation = F.silu(x1) * x3  # [batch_size, num_patches, hidden_dim]
            
            self.activations.append((layer_idx, activation.detach().cpu()))
        
        return hook
    
    def clear_activations(self):
        """Clear stored activations."""
        self.activations = []
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []


def collect_activations_and_labels(model, dataloader, target_class=0, max_samples=None):
    """
    Collect FFN activations and corresponding labels.
    
    Returns:
        activations_dict: dict mapping layer_idx -> list of activation tensors
        is_target_class: binary array indicating if sample has target_class
    """
    collector = FFNActivationCollector(model)
    collector.register_hooks()
    
    activations_dict = {}  # layer_idx -> list of [batch_size, num_patches, hidden_dim]
    is_target_class_list = []
    
    model.eval()
    rf = RF(model, ln=True)
    
    total_samples = 0
    
    with torch.no_grad():
        for batch_idx, (_, x_src, label_tgt) in enumerate(tqdm(dataloader, desc="Collecting activations")):
            if max_samples is not None and total_samples >= max_samples:
                break
            
            x_src = x_src.cuda()
            label_tgt = label_tgt.cuda()
            batch_size = x_src.size(0)
            
            # Generate images (we need to run forward pass through the model)
            # We'll just run a single forward pass at a fixed timestep for simplicity
            x_init = torch.randn(batch_size, 1, 32, 32).cuda()
            
            # Run a forward pass to trigger hooks
            t = torch.ones(batch_size).cuda() # Fixed timestep
            _ = model(x_init, t, x_src)
            
            # Collect activations from this batch
            for layer_idx, activation in collector.activations:
                if layer_idx not in activations_dict:
                    activations_dict[layer_idx] = []
                activations_dict[layer_idx].append(activation)
            
            # Record which samples have target_class
            is_target = (label_tgt == target_class).cpu().numpy()
            is_target_class_list.append(is_target)
            
            collector.clear_activations()
            total_samples += batch_size
    
    collector.remove_hooks()
    
    # Concatenate all batches
    for layer_idx in activations_dict:
        activations_dict[layer_idx] = torch.cat(activations_dict[layer_idx], dim=0)
    
    is_target_class = np.concatenate(is_target_class_list)
    
    return activations_dict, is_target_class


def compute_correlations_case1(activations_dict, is_target_class):
    """
    Case 1: Average activations across patches, then compute correlation per neuron.
    
    Returns:
        correlations: dict mapping layer_idx -> [hidden_dim] correlation values
    """
    correlations = {}
    
    for layer_idx, activations in activations_dict.items():
        # activations: [num_samples, num_patches, hidden_dim]
        # Average across patches
        avg_activations = activations.mean(dim=1).numpy()  # [num_samples, hidden_dim]
        
        # Compute correlation for each neuron
        hidden_dim = avg_activations.shape[1]
        neuron_correlations = np.zeros(hidden_dim)
        
        for neuron_idx in range(hidden_dim):
            neuron_acts = avg_activations[:, neuron_idx]
            # Pearson correlation
            corr = np.corrcoef(neuron_acts, is_target_class)[0, 1]
            neuron_correlations[neuron_idx] = corr if not np.isnan(corr) else 0.0
        
        correlations[layer_idx] = neuron_correlations
    
    return correlations


def compute_correlations_case2(activations_dict, is_target_class):
    """
    Case 2: Keep patch dimension, compute correlation per (patch, neuron) pair.
    
    Returns:
        correlations: dict mapping layer_idx -> [num_patches, hidden_dim] correlation values
    """
    correlations = {}
    
    for layer_idx, activations in activations_dict.items():
        # activations: [num_samples, num_patches, hidden_dim]
        num_patches = activations.shape[1]
        hidden_dim = activations.shape[2]
        activations_np = activations.numpy()
        
        patch_neuron_correlations = np.zeros((num_patches, hidden_dim))
        
        for patch_idx in range(num_patches):
            for neuron_idx in range(hidden_dim):
                neuron_acts = activations_np[:, patch_idx, neuron_idx]
                # Pearson correlation
                corr = np.corrcoef(neuron_acts, is_target_class)[0, 1]
                patch_neuron_correlations[patch_idx, neuron_idx] = corr if not np.isnan(corr) else 0.0
        
        correlations[layer_idx] = patch_neuron_correlations
    
    return correlations


def visualize_case1(correlations, output_path, target_class=0):
    """
    Visualize Case 1: heatmap with neurons on x-axis, layers on y-axis (subplots).
    """
    num_layers = len(correlations)
    layer_indices = sorted(correlations.keys())
    
    fig, axes = plt.subplots(num_layers, 1, figsize=(20, 3 * num_layers))
    if num_layers == 1:
        axes = [axes]
    
    for idx, layer_idx in enumerate(layer_indices):
        corr = correlations[layer_idx]
        # Reshape to 2D for heatmap (1 row, many columns)
        corr_2d = corr.reshape(1, -1)
        
        ax = axes[idx]
        im = ax.imshow(corr_2d, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_title(f'Layer {layer_idx} - FFN Neuron Correlations with Target Class {target_class}')
        ax.set_ylabel('Layer')
        ax.set_xlabel('Neuron Index')
        ax.set_yticks([0])
        ax.set_yticklabels([f'L{layer_idx}'])
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Correlation')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Case 1 visualization saved to: {output_path}")


def visualize_case2(correlations, output_path, target_class=0):
    """
    Visualize Case 2: For each layer, create a 16x16 grid of subplots where:
    - Each subplot corresponds to a patch position
    - Each subplot shows the correlation of all neurons for that patch
    
    Args:
        correlations: dict mapping layer_idx -> [num_patches, hidden_dim]
        output_path: path to save the visualization
        target_class: target class being analyzed
    """
    num_layers = len(correlations)
    layer_indices = sorted(correlations.keys())
    
    # Create a figure for each layer
    for layer_idx in layer_indices:
        corr = correlations[layer_idx]  # [num_patches, hidden_dim]
        num_patches, hidden_dim = corr.shape
        
        # Reshape patches to spatial grid
        patch_size = int(np.sqrt(num_patches))
        assert patch_size * patch_size == num_patches, f"num_patches ({num_patches}) must be a perfect square"
        
        print(f"Layer {layer_idx}: Creating {patch_size}x{patch_size} subplot grid, "
              f"each showing {hidden_dim} neurons")
        
        # Create 16x16 subplot grid
        fig, axes = plt.subplots(patch_size, patch_size, figsize=(20, 20))
        
        for i in range(patch_size):
            for j in range(patch_size):
                patch_idx = i * patch_size + j
                ax = axes[i, j]
                
                # Get correlation for this patch across all neurons: [hidden_dim]
                patch_corr = corr[patch_idx, :]
                
                # Reshape to 2D for visualization (1 row, many columns)
                patch_corr_2d = patch_corr.reshape(1, -1)
                
                # Plot heatmap for this patch
                im = ax.imshow(patch_corr_2d, aspect='auto', cmap='RdBu_r', 
                              vmin=-1, vmax=1, interpolation='nearest')
                
                # Remove ticks for cleaner look
                ax.set_xticks([])
                ax.set_yticks([])
                
                # Optionally add patch index as title (can be commented out for cleaner look)
                # ax.set_title(f'{patch_idx}', fontsize=6)
        
        # Add a single colorbar for the entire figure
        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax, label='Correlation')
        
        plt.suptitle(f'Layer {layer_idx} - FFN Neuron Correlations (Target Class {target_class})\n'
                    f'Each cell = one patch position, showing all {hidden_dim} neurons', 
                    fontsize=16, y=0.98)
        
        # Save with layer index in filename
        layer_output_path = output_path.replace('.png', f'_layer_{layer_idx}.png')
        plt.savefig(layer_output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Case 2 Layer {layer_idx} visualization saved to: {layer_output_path}")
    
    print(f"Case 2 visualization complete for all layers")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--train_fraction", type=float, default=0.9)
    parser.add_argument("--num_images", type=int, default=1)
    parser.add_argument("--target_class", type=int, default=0, help="Target class to analyze")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to analyze")
    parser.add_argument("--output_dir", type=str, default="ffn_analysis/")
    args = parser.parse_args()
    
    # Create output directory with datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f'ffn_neurons_train_fraction_{args.train_fraction}-num_images_{args.num_images}-target_class_{args.target_class}-{timestamp}'
    results_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Loading model from: {args.model_path}")
    print(f"Results will be saved to: {results_dir}")
    print(f"Target class: {args.target_class}")
    
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
    print(f"Starting activation collection...")
    
    # Collect activations
    activations_dict, is_target_class = collect_activations_and_labels(
        model, dataloader, target_class=args.target_class, max_samples=args.max_samples
    )
    
    print(f"\nTotal samples collected: {len(is_target_class)}")
    print(f"Samples with target class {args.target_class}: {is_target_class.sum()}")
    print(f"Number of layers: {len(activations_dict)}")
    
    # Print activation shapes
    for layer_idx, acts in activations_dict.items():
        print(f"Layer {layer_idx}: {acts.shape}")
    
    # Compute correlations - Case 1
    print("\n" + "="*50)
    print("Computing Case 1 correlations (averaged across patches)...")
    correlations_case1 = compute_correlations_case1(activations_dict, is_target_class)
    
    # Print statistics for Case 1
    for layer_idx in sorted(correlations_case1.keys()):
        corr = correlations_case1[layer_idx]
        print(f"Layer {layer_idx} - Max corr: {corr.max():.4f}, Min corr: {corr.min():.4f}, "
              f"Mean |corr|: {np.abs(corr).mean():.4f}")
    
    # Visualize Case 1
    case1_output = os.path.join(results_dir, f"case1_target_class_{args.target_class}.png")
    visualize_case1(correlations_case1, case1_output, target_class=args.target_class)
    
    # Compute correlations - Case 2
    print("\n" + "="*50)
    print("Computing Case 2 correlations (per patch)...")
    correlations_case2 = compute_correlations_case2(activations_dict, is_target_class)
    
    # Print statistics for Case 2
    for layer_idx in sorted(correlations_case2.keys()):
        corr = correlations_case2[layer_idx]
        print(f"Layer {layer_idx} - Max corr: {corr.max():.4f}, Min corr: {corr.min():.4f}, "
              f"Mean |corr|: {np.abs(corr).mean():.4f}")
    
    # Visualize Case 2
    case2_output = os.path.join(results_dir, f"case2_target_class_{args.target_class}.png")
    visualize_case2(correlations_case2, case2_output, target_class=args.target_class)
    
    # Save correlation data
    np.savez(
        os.path.join(results_dir, "correlations.npz"),
        **{f"case1_layer_{k}": v for k, v in correlations_case1.items()},
        **{f"case2_layer_{k}": v for k, v in correlations_case2.items()},
        is_target_class=is_target_class
    )
    
    print(f"\n{'='*50}")
    print(f"Analysis complete! Results saved to: {results_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

