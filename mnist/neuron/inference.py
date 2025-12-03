import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import os

import torch.optim as optim
import torchvision.utils as tvu
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tqdm import tqdm
from train import RF, MNISTModularArithmeticDataset

if __name__ == "__main__":
    from model import DiT_Llama
    from classifier import MNISTClassifier
    import argparse

    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--classifier_weights_path", type=str, default="weights/mnist_classifier_weights.pth")
        parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint")
        parser.add_argument("--train_fraction", type=float, default=1.0)
        parser.add_argument("--output_dir", type=str, default="inference_results/")
        parser.add_argument("--num_images", type=int, default=1)
        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--sample_steps", type=int, default=1, help="Number of sampling steps")
        parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to generate")
        parser.add_argument("--num_repeats", type=int, default=1, help="Number of times to repeat through the dataset")
        parser.add_argument("--split", type=str, default="train", choices=["train", "valid"], help="Dataset split to use")
        parser.add_argument("--save_all_images", action="store_true", help="Save all generated images")
        return parser.parse_args()
    
    args = parse_args()
    
    # Create output directory
    exp_name = 'inference_train_fraction_{}-num_images_{}-steps_{}'.format(args.train_fraction, args.num_images, args.sample_steps)
    results_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(results_dir, exist_ok=True)
    
    # Create subdirectory for generated images
    images_dir = os.path.join(results_dir, "generated_images")
    os.makedirs(images_dir, exist_ok=True)
    
    print(f"Loading model from: {args.model_path}")
    print(f"Results will be saved to: {results_dir}")

    # Initialize model
    model = DiT_Llama(
        3, 32, dim=256, n_layers=10, n_heads=8,
    ).cuda()
    
    # Load trained weights
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    
    # Load classifier for evaluation
    classifier = MNISTClassifier().cuda()
    classifier.load_state_dict(torch.load(args.classifier_weights_path))
    classifier.eval()

    model_size = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {model_size}, {model_size / 1e6:.2f}M")

    rf = RF(model, ln=True)
    
    # Load dataset
    dataset = MNISTModularArithmeticDataset(
        p=10, 
        split=args.split, 
        train_fraction=args.train_fraction, 
        num_images=args.num_images
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    
    print(f"Dataset split: {args.split}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of repeats: {args.num_repeats}")
    print(f"Starting inference with {args.sample_steps} sampling steps...")

    all_accuracies = []
    total_samples = 0
    global_idx = 0
    
    with torch.no_grad():
        for repeat_idx in range(args.num_repeats):
            print(f"\nRepeat {repeat_idx + 1}/{args.num_repeats}")
            
            for batch_idx, (_, x_src, label_tgt) in enumerate(tqdm(dataloader, desc=f"Repeat {repeat_idx + 1}")):
                x_src = x_src.cuda()
                label_tgt = label_tgt.cuda()
                batch_size = x_src.size(0)
                
                # Initialize random noise
                x_init = torch.randn(batch_size, 1, 32, 32).cuda()
                
                # Generate images
                images = rf.sample(x_init, x_src, sample_steps=args.sample_steps)
                generated_imgs = images[-1]  # Take the final generated image
                
                # Evaluate with classifier
                preds = classifier(generated_imgs)
                preds = preds.argmax(dim=1)
                
                # Calculate accuracy
                accuracy = (preds == label_tgt).float().mean().item()
                all_accuracies.append(accuracy)
                
                # Save all generated images
                if args.save_all_images:
                    # Save batch as grid visualization
                    batch_result = torch.cat(
                        [x_src, generated_imgs], dim=1
                    ).reshape(-1, 1, 32, 32)
                    tvu.save_image(
                        batch_result,
                        f"{images_dir}/batch_{batch_idx:04d}_repeat_{repeat_idx:02d}.png",
                        nrow=batch_size,
                        normalize=True
                    )
                    
                    # Also save individual images
                    for i in range(batch_size):
                        img_pair = torch.cat([x_src[i:i+1], generated_imgs[i:i+1]], dim=1)
                        img_pair = img_pair.reshape(-1, 1, 32, 32)
                        tvu.save_image(
                            img_pair,
                            f"{images_dir}/img_{global_idx:06d}_src_gen.png",
                            nrow=2,
                            normalize=True
                        )
                        global_idx += 1
                else:
                    # Save sample visualizations (first batch only)
                    if repeat_idx == 0 and batch_idx == 0:
                        num_to_save = min(16, batch_size)
                        result = torch.cat(
                            [x_src[:num_to_save], generated_imgs[:num_to_save]], dim=1
                        ).reshape(-1, 1, 32, 32)
                        tvu.save_image(
                            result, 
                            f"{results_dir}/samples_{args.split}.png", 
                            nrow=3,
                            normalize=True
                        )
                
                total_samples += batch_size
    
    # Print final statistics
    mean_accuracy = np.mean(all_accuracies)
    std_accuracy = np.std(all_accuracies)
    
    print(f"\n{'='*50}")
    print(f"Inference Results on {args.split} set:")
    print(f"{'='*50}")
    print(f"Total samples generated: {total_samples}")
    print(f"Mean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Results saved to: {results_dir}")
    print(f"{'='*50}")
    
    # Save statistics to file
    stats_path = os.path.join(results_dir, f"statistics_{args.split}.txt")
    with open(stats_path, 'w') as f:
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Split: {args.split}\n")
        f.write(f"Train fraction: {args.train_fraction}\n")
        f.write(f"Number of repeats: {args.num_repeats}\n")
        f.write(f"Total samples: {total_samples}\n")
        f.write(f"Sample steps: {args.sample_steps}\n")
        f.write(f"Mean Accuracy: {mean_accuracy:.4f}\n")
        f.write(f"Std Accuracy: {std_accuracy:.4f}\n")
        f.write(f"Save all images: {args.save_all_images}\n")
        if args.save_all_images:
            f.write(f"Images saved to: {images_dir}\n")
    
    print(f"Statistics saved to: {stats_path}")