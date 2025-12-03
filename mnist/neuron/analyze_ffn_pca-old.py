import torch
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


@torch.no_grad()
def collect_residual_streams(model, dataloader, max_samples=None):
    """
    Collect residual stream representations after attention and after FFN for each layer.

    Case 2 style: keep patch dimension -> [num_samples, num_patches, dim].

    Returns:
        residual_streams_attn: dict[layer_idx] -> tensor [N, P, D] (after attention, before FFN residual add)
        residual_streams_ffn: dict[layer_idx] -> tensor [N, P, D] (after FFN, final block output)
        labels: np.ndarray [N] (target class labels for each sample)
    """
    # Save original forward methods
    original_forwards = {}
    for layer_idx, layer in enumerate(model.layers):
        original_forwards[layer_idx] = layer.forward

    residual_streams_attn = {}  # layer_idx -> list of [B, P, D]
    residual_streams_ffn = {}   # layer_idx -> list of [B, P, D]
    labels_list = []

    # Wrapper to capture residual streams while reproducing original computation
    def create_wrapper(layer_idx, layer, original_forward):
        def wrapper(x, freqs_cis, adaln_input=None):
            # This re-implements TransformerBlock.forward, but taps into
            # the residual stream after attention and after FFN.
            if adaln_input is not None:
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                    layer.adaLN_modulation(adaln_input).chunk(6, dim=1)
                )

                # Attention branch
                attn_out = layer.attention(
                    modulate(layer.attention_norm(x), shift_msa, scale_msa),
                    freqs_cis,
                )
                x_after_attn = x + gate_msa.unsqueeze(1) * attn_out
            else:
                attn_out = layer.attention(layer.attention_norm(x), freqs_cis)
                x_after_attn = x + attn_out

            # Store residual stream after attention
            if layer_idx not in residual_streams_attn:
                residual_streams_attn[layer_idx] = []
            residual_streams_attn[layer_idx].append(x_after_attn.detach().cpu())

            # FFN branch
            if adaln_input is not None:
                ffn_out = layer.feed_forward(
                    modulate(layer.ffn_norm(x_after_attn), shift_mlp, scale_mlp)
                )
                x_after_ffn = x_after_attn + gate_mlp.unsqueeze(1) * ffn_out
            else:
                ffn_out = layer.feed_forward(layer.ffn_norm(x_after_attn))
                x_after_ffn = x_after_attn + ffn_out

            # Store residual stream after FFN
            if layer_idx not in residual_streams_ffn:
                residual_streams_ffn[layer_idx] = []
            residual_streams_ffn[layer_idx].append(x_after_ffn.detach().cpu())

            return x_after_ffn

        return wrapper

    # Replace forwards with wrappers
    for layer_idx, layer in enumerate(model.layers):
        layer.forward = create_wrapper(layer_idx, layer, original_forwards[layer_idx])

    model.eval()
    total_samples = 0

    for _, x_src, label_tgt in tqdm(dataloader, desc="Collecting residual streams"):
        if max_samples is not None and total_samples >= max_samples:
            break

        x_src = x_src.cuda(non_blocking=True)
        label_tgt = label_tgt.cuda(non_blocking=True)
        batch_size = x_src.size(0)

        # forward diffusion input (we only need a single timestep)
        x_init = torch.randn(batch_size, 1, 32, 32, device=x_src.device)
        t = torch.ones(batch_size, device=x_src.device) * 0.5
        _ = model(x_init, t, x_src)

        labels_list.append(label_tgt.detach().cpu().numpy())
        total_samples += batch_size

        if max_samples is not None and total_samples >= max_samples:
            break

    # Restore original forwards
    for layer_idx, layer in enumerate(model.layers):
        layer.forward = original_forwards[layer_idx]

    # Concatenate across batches
    for layer_idx in residual_streams_attn:
        residual_streams_attn[layer_idx] = torch.cat(
            residual_streams_attn[layer_idx], dim=0
        )
    for layer_idx in residual_streams_ffn:
        residual_streams_ffn[layer_idx] = torch.cat(
            residual_streams_ffn[layer_idx], dim=0
        )

    labels = np.concatenate(labels_list)

    return residual_streams_attn, residual_streams_ffn, labels


def visualize_residual_stream_pca(residual_streams, labels, output_path, stream_type="attn"):
    """
    Visualize residual stream using PCA, patch-wise, with 16x16 subplots.

    For each layer:
      - residual_streams[layer]: [N, P, D]
      - One PCA fit on all (N * P) points -> 4D
      - For each patch index, we plot N points (one per sample) in its own subplot.
      - We make 3 figures per layer: (PCA0,PCA1), (PCA1,PCA2), (PCA2,PCA3).

    Each point is colored by its target class label.
    """
    layer_indices = sorted(residual_streams.keys())
    if not layer_indices:
        print(f"No residual streams to visualize for stream_type={stream_type}.")
        return

    unique_classes = np.unique(labels)
    num_classes = len(unique_classes)
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    class_to_color = {cls: colors[i] for i, cls in enumerate(unique_classes)}

    for layer_idx in layer_indices:
        residual = residual_streams[layer_idx]  # [N, P, D]
        num_samples, num_patches, dim = residual.shape

        patch_size = int(np.sqrt(num_patches))
        assert (
            patch_size * patch_size == num_patches
        ), f"num_patches ({num_patches}) must be a perfect square (e.g., 16x16)"

        # PCA on all patches jointly
        residual_flat = residual.reshape(num_samples * num_patches, dim).numpy()
        pca = PCA(n_components=4)
        pca_result = pca.fit_transform(residual_flat)  # [N*P, 4]
        pca_result = pca_result.reshape(num_samples, num_patches, 4)  # [N, P, 4]

        explained_var = pca.explained_variance_ratio_

        # For coloring per point we still use labels (same for all patches)
        pca_pairs = [
            (0, 1, "pc0_pc1"),
            (1, 2, "pc1_pc2"),
            (2, 3, "pc2_pc3"),
        ]

        for comp_x, comp_y, pair_name in pca_pairs:
            fig, axes = plt.subplots(
                patch_size,
                patch_size,
                figsize=(20, 20),
                squeeze=False,
            )

            for i in range(patch_size):
                for j in range(patch_size):
                    patch_idx = i * patch_size + j
                    ax = axes[i, j]

                    # [N, 4] for this patch
                    patch_pca = pca_result[:, patch_idx, :]

                    for cls in unique_classes:
                        mask = labels == cls
                        if not np.any(mask):
                            continue
                        ax.scatter(
                            patch_pca[mask, comp_x],
                            patch_pca[mask, comp_y],
                            c=[class_to_color[cls]],
                            s=5,
                            alpha=0.6,
                        )

                    ax.set_xticks([])
                    ax.set_yticks([])

            # Global title and legend
            fig.suptitle(
                f"Layer {layer_idx} - {stream_type.upper()} residual\n"
                f"PCA{comp_x} vs PCA{comp_y} "
                f"(explained var: PC{comp_x}={explained_var[comp_x]:.2%}, "
                f"PC{comp_y}={explained_var[comp_y]:.2%})",
                fontsize=14,
                y=0.92,
            )

            handles = [
                plt.Line2D(
                    [], [], marker="o", linestyle="", color=class_to_color[cls], label=f"Class {cls}"
                )
                for cls in unique_classes
            ]
            fig.legend(
                handles=handles,
                loc="upper right",
                bbox_to_anchor=(0.99, 0.995),
                fontsize=10,
            )

            plt.tight_layout(rect=[0, 0, 0.95, 0.90])

            layer_output_path = output_path.replace(
                ".png",
                f"_layer_{layer_idx}_{pair_name}.png",
            )
            plt.savefig(layer_output_path, dpi=150, bbox_inches="tight")
            plt.close()

            print(
                f"Residual stream PCA ({stream_type}, layer {layer_idx}, "
                f"PCA{comp_x} vs PCA{comp_y}) saved to: {layer_output_path}"
            )

    print(f"Residual stream PCA visualization complete for all layers ({stream_type})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model checkpoint")
    parser.add_argument("--train_fraction", type=float, default=0.9)
    parser.add_argument("--num_images", type=int, default=1)
    parser.add_argument(
        "--target_class",
        type=int,
        default=0,
        help="Target class (only used for naming the output directory)",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to analyze",
    )
    parser.add_argument("--output_dir", type=str, default="ffn_analysis/")
    args = parser.parse_args()

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = (
        f"residual_pca_train_fraction_{args.train_fraction}"
        f"-num_images_{args.num_images}"
        f"-target_class_{args.target_class}-{timestamp}"
    )
    results_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(results_dir, exist_ok=True)

    print(f"Loading model from: {args.model_path}")
    print(f"Results will be saved to: {results_dir}")

    # Model
    model = DiT_Llama(
        3,
        32,
        dim=256,
        n_layers=10,
        n_heads=8,
    ).cuda()
    state_dict = torch.load(args.model_path, map_location="cuda")
    model.load_state_dict(state_dict)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params} ({num_params / 1e6:.2f}M)")

    # Dataset (train split)
    dataset = MNISTModularArithmeticDataset(
        p=10,
        split="train",
        train_fraction=args.train_fraction,
        num_images=args.num_images,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"Dataset size: {len(dataset)}")
    print("Starting residual stream collection...")

    # Collect residual streams
    print("\n" + "=" * 50)
    print("Collecting residual stream representations (Case 2)...")
    residual_streams_attn, residual_streams_ffn, labels = collect_residual_streams(
        model, dataloader, max_samples=args.max_samples
    )

    print("\nResidual stream shapes:")
    for layer_idx in sorted(residual_streams_attn.keys()):
        print(
            f"Layer {layer_idx} - Attn: {tuple(residual_streams_attn[layer_idx].shape)}, "
            f"FFN: {tuple(residual_streams_ffn[layer_idx].shape)}"
        )

    # Visualize PCA for attention residual
    print("\n" + "=" * 50)
    print("Visualizing residual stream PCA (after attention)...")
    if residual_streams_attn:
        attn_output_base = os.path.join(results_dir, "residual_pca_attn_case2.png")
        visualize_residual_stream_pca(
            residual_streams_attn, labels, attn_output_base, stream_type="attn"
        )
    else:
        print("WARNING: No attention residual streams to visualize!")

    # Visualize PCA for FFN residual
    print("\n" + "=" * 50)
    print("Visualizing residual stream PCA (after FFN)...")
    if residual_streams_ffn:
        ffn_output_base = os.path.join(results_dir, "residual_pca_ffn_case2.png")
        visualize_residual_stream_pca(
            residual_streams_ffn, labels, ffn_output_base, stream_type="ffn"
        )
    else:
        print("WARNING: No FFN residual streams to visualize!")

    print("\n" + "=" * 50)
    print(f"Analysis complete! Results saved to: {results_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()

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
    """Collects residual stream representations after attn and ffn for each layer."""
    
    def __init__(self, model):
        self.model = model
        self.activations_attn = []  # List of (layer_idx, activation_tensor) after attn
        self.activations_ffn = []   # List of (layer_idx, activation_tensor) after ffn
        self.hooks_attn = []
        self.hooks_ffn = []
        
    def register_hooks(self):
        """Register forward hooks to collect residual stream after attn and ffn."""
        for layer_idx, layer in enumerate(self.model.layers):
            hook_attn = self._create_hook_attn(layer_idx)
            hook_ffn = self._create_hook_ffn(layer_idx)
            
            # Hook after attention (before FFN)
            handle_attn = layer.register_forward_hook(hook_attn)
            self.hooks_attn.append(handle_attn)
            
            # Hook after FFN (at the end of TransformerBlock)
            handle_ffn = layer.register_forward_hook(hook_ffn)
            self.hooks_ffn.append(handle_ffn)
    
    def _create_hook_attn(self, layer_idx):
        def hook(module, input, output):
            # This hook captures the output after attention but before FFN
            # We need to intercept at the right point - actually we'll hook the attention module directly
            pass
        return hook
    
    def _create_hook_ffn(self, layer_idx):
        def hook(module, input, output):
            # This captures the final output of TransformerBlock (after both attn and ffn)
            # We need to get intermediate states, so we'll use a different approach
            pass
        return hook
    
    def clear_activations(self):
        """Clear stored activations."""
        self.activations_attn = []
        self.activations_ffn = []
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self.hooks_attn + self.hooks_ffn:
            handle.remove()
        self.hooks_attn = []
        self.hooks_ffn = []


def collect_residual_streams(model, dataloader, max_samples=None):
    """
    Collect residual stream representations after attn and ffn for each layer.
    Uses Case 2 approach: keeps patch dimension.
    
    Returns:
        residual_streams_attn: dict mapping layer_idx -> [num_samples, num_patches, dim]
        residual_streams_ffn: dict mapping layer_idx -> [num_samples, num_patches, dim]
        labels: array of target class labels [num_samples]
    """
    # We'll manually track residual streams by modifying the forward pass
    # Store original forward method
    original_forwards = {}
    for layer_idx, layer in enumerate(model.layers):
        original_forwards[layer_idx] = layer.forward
    
    residual_streams_attn = {}  # layer_idx -> list of [batch_size, num_patches, dim]
    residual_streams_ffn = {}   # layer_idx -> list of [batch_size, num_patches, dim]
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
    
    labels = np.concatenate(labels_list)
    
    return residual_streams_attn, residual_streams_ffn, labels


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
    residual_streams_attn, residual_streams_ffn, labels = collect_residual_streams(
        model, dataloader, max_samples=args.max_samples
    )
    
    print(f"\nResidual stream shapes:")
    for layer_idx in sorted(residual_streams_attn.keys()):
        print(f"Layer {layer_idx} - Attn: {residual_streams_attn[layer_idx].shape}, "
              f"FFN: {residual_streams_ffn[layer_idx].shape}")
    
    # Visualize residual stream PCA - Case 1 (Global PCA) for attn
    print("\n" + "="*50)
    print("Visualizing residual stream PCA Case 1 (Global PCA) - after attn...")
    residual_pca_attn_case1_output = os.path.join(results_dir, f"residual_pca_attn_case1.png")
    visualize_residual_stream_pca_case1(residual_streams_attn, labels, residual_pca_attn_case1_output, stream_type="attn")
    
    # Visualize residual stream PCA - Case 2 (Patch-wise PCA) for attn
    print("\n" + "="*50)
    print("Visualizing residual stream PCA Case 2 (Patch-wise PCA) - after attn...")
    residual_pca_attn_case2_output = os.path.join(results_dir, f"residual_pca_attn_case2.png")
    visualize_residual_stream_pca_case2(residual_streams_attn, labels, residual_pca_attn_case2_output, stream_type="attn")
    
    # Visualize residual stream PCA - Case 1 (Global PCA) for ffn
    print("\n" + "="*50)
    print("Visualizing residual stream PCA Case 1 (Global PCA) - after ffn...")
    residual_pca_ffn_case1_output = os.path.join(results_dir, f"residual_pca_ffn_case1.png")
    visualize_residual_stream_pca_case1(residual_streams_ffn, labels, residual_pca_ffn_case1_output, stream_type="ffn")
    
    # Visualize residual stream PCA - Case 2 (Patch-wise PCA) for ffn
    print("\n" + "="*50)
    print("Visualizing residual stream PCA Case 2 (Patch-wise PCA) - after ffn...")
    residual_pca_ffn_case2_output = os.path.join(results_dir, f"residual_pca_ffn_case2.png")
    visualize_residual_stream_pca_case2(residual_streams_ffn, labels, residual_pca_ffn_case2_output, stream_type="ffn")
    
    print(f"\n{'='*50}")
    print(f"Analysis complete! Results saved to: {results_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

