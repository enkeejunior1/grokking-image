import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
# import seaborn as sns  # Optional, not used
from collections import defaultdict

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


def ablation_study(
    model_path,
    classifier_weights_path,
    output_dir,
    train_fraction=1.0,
    num_images=1,
    batch_size=256,
    sample_steps=1,
    split="valid",
    k=1,  # Number of modules to remove per iteration
    max_iterations=None,  # Maximum number of iterations (None = remove all)
    num_samples_for_l2=50,  # Number of samples to use for L2 computation
):
    """
    Perform ablation study by iteratively removing modules with smallest L2 change
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
    # Assuming patch_size=2 (default), input_size=32 -> 16x16 = 256 patches
    # We need to check the actual patch size
    patch_size = model.base_model.patch_size
    input_size = model.base_model.input_size
    num_patches = (input_size // patch_size) ** 2
    num_layers = len(model.base_model.layers)
    
    print(f"Model configuration:")
    print(f"  Patch size: {patch_size}")
    print(f"  Input size: {input_size}")
    print(f"  Number of patches: {num_patches}")
    print(f"  Number of layers: {num_layers}")
    print(f"  Module types: ['attn', 'ffn']")
    
    # Initialize: all modules are active
    masked_modules = set()  # Set of (layer_idx, patch_idx, module_type) that are masked
    accuracies = []
    num_masked_list = []
    mask_history = []  # List of masks at each iteration
    
    # Initial accuracy (no masking)
    print("Computing initial accuracy...")
    initial_acc = evaluate_accuracy(model, rf, classifier, dataloader, sample_steps)
    accuracies.append(initial_acc)
    num_masked_list.append(0)
    mask_history.append(np.ones((num_layers, num_patches, 2), dtype=bool))  # (layers, patches, [attn, ffn])
    print(f"Initial accuracy: {initial_acc:.4f}")
    
    # Determine total number of modules
    total_modules = num_layers * num_patches * 2  # layers * patches * (attn + ffn)
    if max_iterations is None:
        max_iterations = total_modules // k
    
    print(f"\nStarting ablation study:")
    print(f"  Total modules: {total_modules}")
    print(f"  Removing {k} module(s) per iteration")
    print(f"  Maximum iterations: {max_iterations}")
    
    # Iteratively remove modules
    for iteration in tqdm(range(max_iterations), desc="Ablation iterations"):
        if len(masked_modules) >= total_modules:
            break
        
        # Compute L2 changes for all remaining modules
        print(f"\nIteration {iteration + 1}: Computing L2 changes for remaining modules...")
        l2_scores = {}
        
        # Create a smaller dataloader for L2 computation
        l2_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        
        remaining_modules = []
        for layer_idx in range(num_layers):
            for patch_idx in range(num_patches):
                for module_type in ['attn', 'ffn']:
                    if (layer_idx, patch_idx, module_type) not in masked_modules:
                        remaining_modules.append((layer_idx, patch_idx, module_type))
        
        # Compute L2 changes
        for layer_idx, patch_idx, module_type in tqdm(remaining_modules, desc="Computing L2", leave=False):
            l2_change = compute_l2_change(
                model, rf, l2_dataloader, layer_idx, patch_idx, module_type, 
                num_samples=num_samples_for_l2
            )
            l2_scores[(layer_idx, patch_idx, module_type)] = l2_change
        
        # Analyze L2 changes by layer (for debugging/understanding)
        if iteration == 0:  # Only print for first iteration to avoid clutter
            layer_avg_l2 = {}
            for layer_idx in range(num_layers):
                layer_l2s = []
                for patch_idx in range(num_patches):
                    for module_type in ['attn', 'ffn']:
                        if (layer_idx, patch_idx, module_type) in l2_scores:
                            layer_l2s.append(l2_scores[(layer_idx, patch_idx, module_type)])
                if layer_l2s:
                    layer_avg_l2[layer_idx] = np.mean(layer_l2s)
            print(f"\nAverage L2 change by layer (first iteration):")
            for layer_idx in sorted(layer_avg_l2.keys()):
                print(f"  Layer {layer_idx}: {layer_avg_l2[layer_idx]:.6f}")
        
        # Select K modules with smallest L2 change
        sorted_modules = sorted(l2_scores.items(), key=lambda x: x[1])
        modules_to_mask = [mod for mod, _ in sorted_modules[:k]]
        
        # Apply masks
        for layer_idx, patch_idx, module_type in modules_to_mask:
            masked_modules.add((layer_idx, patch_idx, module_type))
            model.set_mask(layer_idx, patch_idx, module_type, mask_value=True)
        
        model.register_hooks()
        
        # Evaluate accuracy
        print(f"  Masked {len(modules_to_mask)} modules. Evaluating accuracy...")
        acc = evaluate_accuracy(model, rf, classifier, dataloader, sample_steps)
        accuracies.append(acc)
        num_masked_list.append(len(masked_modules))
        
        # Update mask history
        current_mask = np.ones((num_layers, num_patches, 2), dtype=bool)
        for layer_idx, patch_idx, module_type in masked_modules:
            module_idx = 0 if module_type == 'attn' else 1
            current_mask[layer_idx, patch_idx, module_idx] = False
        mask_history.append(current_mask.copy())
        
        print(f"  Accuracy: {acc:.4f} (masked: {len(masked_modules)}/{total_modules})")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Figure 1: Accuracy vs Number of masked modules
    plt.figure(figsize=(10, 6))
    plt.plot(num_masked_list, accuracies, 'o-', linewidth=2, markersize=6)
    plt.xlabel('Number of Masked Modules', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Ablation Study: Accuracy vs Number of Masked Modules', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_vs_masked_modules.png'), dpi=300)
    plt.close()
    print(f"\nSaved Figure 1: {os.path.join(output_dir, 'accuracy_vs_masked_modules.png')}")
    
    # Figure 2: Visualization of masked modules at each iteration
    # Create a grid showing which patches/layers are masked at each iteration
    num_iterations_to_show = min(len(mask_history), 20)  # Show up to 20 iterations
    iterations_to_show = np.linspace(0, len(mask_history) - 1, num_iterations_to_show, dtype=int)
    
    fig, axes = plt.subplots(2, num_iterations_to_show, figsize=(2 * num_iterations_to_show, 8))
    if num_iterations_to_show == 1:
        axes = axes.reshape(2, 1)
    
    for idx, iter_idx in enumerate(iterations_to_show):
        mask = mask_history[iter_idx]
        
        # Plot for Attention modules
        attn_mask = mask[:, :, 0]  # (layers, patches)
        # Reshape patches to 2D grid (assuming square)
        patch_grid_size = int(np.sqrt(num_patches))
        attn_vis = attn_mask.reshape(num_layers, patch_grid_size, patch_grid_size)
        # Average across layers for visualization
        attn_vis = attn_vis.mean(axis=0)
        
        axes[0, idx].imshow(attn_vis, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        axes[0, idx].set_title(f'Attn (iter {iter_idx})', fontsize=8)
        axes[0, idx].axis('off')
        
        # Plot for FFN modules
        ffn_mask = mask[:, :, 1]  # (layers, patches)
        ffn_vis = ffn_mask.reshape(num_layers, patch_grid_size, patch_grid_size)
        ffn_vis = ffn_vis.mean(axis=0)
        
        axes[1, idx].imshow(ffn_vis, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        axes[1, idx].set_title(f'FFN (iter {iter_idx})', fontsize=8)
        axes[1, idx].axis('off')
    
    plt.suptitle('Masked Modules Across Iterations (Red=Masked, Green=Active)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mask_visualization.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 2: {os.path.join(output_dir, 'mask_visualization.png')}")
    
    # More detailed visualization: layer x patch heatmap for each iteration
    # Create a more detailed view showing layer vs patch for each module type
    num_detailed_iterations = min(len(mask_history), 10)
    detailed_iterations = np.linspace(0, len(mask_history) - 1, num_detailed_iterations, dtype=int)
    
    fig, axes = plt.subplots(2, num_detailed_iterations, figsize=(2 * num_detailed_iterations, 6))
    if num_detailed_iterations == 1:
        axes = axes.reshape(2, 1)
    
    for idx, iter_idx in enumerate(detailed_iterations):
        mask = mask_history[iter_idx]
        
        # Attention: layer (rows) x patch (columns)
        attn_mask = mask[:, :, 0]  # (layers, patches)
        axes[0, idx].imshow(attn_mask, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        axes[0, idx].set_title(f'Attn Iter {iter_idx}\n({num_masked_list[iter_idx]} masked)', fontsize=8)
        axes[0, idx].set_xlabel('Patch', fontsize=7)
        axes[0, idx].set_ylabel('Layer', fontsize=7)
        
        # FFN: layer (rows) x patch (columns)
        ffn_mask = mask[:, :, 1]  # (layers, patches)
        axes[1, idx].imshow(ffn_mask, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        axes[1, idx].set_title(f'FFN Iter {iter_idx}\n({num_masked_list[iter_idx]} masked)', fontsize=8)
        axes[1, idx].set_xlabel('Patch', fontsize=7)
        axes[1, idx].set_ylabel('Layer', fontsize=7)
    
    plt.suptitle('Layer x Patch Mask Visualization (Red=Masked, Green=Active)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'layer_patch_mask_visualization.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved detailed visualization: {os.path.join(output_dir, 'layer_patch_mask_visualization.png')}")
    
    # Save results to file
    results_path = os.path.join(output_dir, 'ablation_results.txt')
    with open(results_path, 'w') as f:
        f.write("Ablation Study Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Split: {split}\n")
        f.write(f"Train fraction: {train_fraction}\n")
        f.write(f"Number of patches: {num_patches}\n")
        f.write(f"Number of layers: {num_layers}\n")
        f.write(f"Total modules: {total_modules}\n")
        f.write(f"K (modules removed per iteration): {k}\n")
        f.write(f"\nNote: Modules are removed based on smallest L2 change.\n")
        f.write(f"If upper layers are removed first, it suggests they have smaller L2 impact,\n")
        f.write(f"possibly due to residual connections allowing lower layers to compensate.\n")
        f.write(f"\nResults:\n")
        f.write(f"{'Iteration':<12} {'Num Masked':<12} {'Accuracy':<12}\n")
        f.write("-" * 50 + "\n")
        for i, (num_masked, acc) in enumerate(zip(num_masked_list, accuracies)):
            f.write(f"{i:<12} {num_masked:<12} {acc:.6f}\n")
    
    print(f"\nSaved results to: {results_path}")
    print(f"\nAblation study complete!")
    print(f"Final accuracy: {accuracies[-1]:.4f}")
    print(f"Total modules masked: {num_masked_list[-1]}/{total_modules}")


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
    parser.add_argument("--k", type=int, default=256, help="Number of modules to remove per iteration")
    parser.add_argument("--max_iterations", type=int, default=None, help="Maximum number of iterations (None = remove all)")
    parser.add_argument("--num_samples_for_l2", type=int, default=50, help="Number of samples to use for L2 computation")
    
    args = parser.parse_args()
    
    # Create experiment-specific output directory
    exp_name = f'ablation_train_fraction_{args.train_fraction}-num_images_{args.num_images}-steps_{args.sample_steps}-k_{args.k}'
    output_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(output_dir, exist_ok=True)
    
    ablation_study(
        model_path=args.model_path,
        classifier_weights_path=args.classifier_weights_path,
        output_dir=output_dir,
        train_fraction=args.train_fraction,
        num_images=args.num_images,
        batch_size=args.batch_size,
        sample_steps=args.sample_steps,
        split=args.split,
        k=args.k,
        max_iterations=args.max_iterations,
        num_samples_for_l2=args.num_samples_for_l2,
    )
