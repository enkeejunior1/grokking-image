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
import sys
import os

import torch.optim as optim
import torchvision.utils as tvu
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tqdm import tqdm
from datetime import datetime

from perturbed_model import MNISTModularArithmeticDataset, PerturbedRF
from model import DiT_Llama
from classifier import MNISTClassifier
import argparse


def layer_level_perturbation_test(rf, x_gen, x_src, sample_steps, save_image=False):
    images_by_perturbed_layers = rf.perturbed_sample_layer_level(x_gen, x_src, sample_steps)
    
    for l, images in enumerate(images_by_perturbed_layers):
        final_image = images[-1]
    
        classifier_outputs = classifier(final_image)
        classifier_predictions = torch.argmax(classifier_outputs, dim=1)
        failed_mask = (label_tgt != classifier_predictions)    
    
        # final_image = 1 - torch.relu(final_image)  # Check inverting MNIST
        final_image[failed_mask] = 1 - torch.relu(final_image[failed_mask])  # Reverse the failed cases
        accuracy = 1 - failed_mask.float().mean().item()

        if save_image:
            num_vis = 100
            result = torch.cat(
                [x_src[:num_vis], final_image[:num_vis]], dim=1
            ).reshape(-1, 1, 32, 32)
            tvu.save_image(result, f"{results_dir}/sample_{i+1}_attention_layer_{l+1}_perturb (accuracy {accuracy:.2f}).png", nrow=3)
    
    print(f"Sampling completed for batch {i+1}/{dl_all.__len__()}")  


def head_level_perturbation_test(rf, x_gen, x_src, n_heads, sample_steps, save_image=False):
    images_by_perturbed_heads = rf.perturbed_sample_head_level(x_gen, x_src, n_heads=n_heads, sample_steps=sample_steps)
    
    for l, layer in enumerate(images_by_perturbed_heads):
        print(f"Processing layer {l+1}/{len(images_by_perturbed_heads)}")
        for h, final_image in enumerate(layer):
            print(f"  Processing head {h+1}/{n_heads}", end="")
            classifier_outputs = classifier(final_image)            
            max_prob, max_idx = torch.max(F.softmax(classifier_outputs, dim=1), dim=1)
            mean_confidence = max_prob.mean().item()
            print(f" (softmax output: {mean_confidence:.2f})", end="")   
            
            classifier_predictions = torch.argmax(classifier_outputs, dim=1)
            failed_mask = (label_tgt != classifier_predictions)    
        
            # final_image = 1 - torch.relu(final_image)  # Check inverting MNIST
            final_image[failed_mask] = 1 - torch.relu(final_image[failed_mask])  # Reverse the failed cases
            accuracy = 1 - failed_mask.float().mean().item()
            
            print(f" (accuracy {accuracy:.2f})")

            if save_image:
                num_vis = 100
                result = torch.cat(
                    [x_src[:num_vis], final_image[:num_vis]], dim=1
                ).reshape(-1, 1, 32, 32)
                tvu.save_image(result, f"{results_dir}/sample_{i+1}_attention_head_layer_{l+1}_head_{h+1}_perturb (accuracy {accuracy:.2f}).png", nrow=3) 
    

if __name__ == "__main__":    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--classifier_weights_path", type=str, default="weights/mnist_classifier_weights.pth")
        parser.add_argument("--generative_model_path", type=str, default="weights/")
        parser.add_argument("--train_fraction", type=float, default=0.9)    # As sugested by Yonghyun
        parser.add_argument("--output_dir", type=str, default="results/")
        parser.add_argument("--num_images", type=int, default=1)    # As sugested by Yonghyun
        return parser.parse_args()
    args = parse_args()
    
    now = datetime.now()
    formatted_time = now.strftime("%m.%d.%H.%M")
    
    pretrained_name = 'train_fraction_{}-num_images_{}'.format(args.train_fraction, args.num_images)
    experiment_name = 'test_attention_head_{}'.format(formatted_time)
    weights_dir = os.path.join(args.generative_model_path, pretrained_name)
    results_dir = os.path.join(args.output_dir, experiment_name)
    os.makedirs(weights_dir, exist_ok=True)
    # os.makedirs(results_dir, exist_ok=True)

    n_heads = 8
    model = DiT_Llama(
        3, 32, dim=256, n_layers=10, n_heads=n_heads,
    ).to(device)
    
    classifier = MNISTClassifier().to(device)
    weights = torch.load(args.classifier_weights_path, map_location=device) # For cpu compatibility
    classifier.load_state_dict(weights)

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {model_size}, {model_size / 1e6}M")

    rf = PerturbedRF(model)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    
    ds_all = MNISTModularArithmeticDataset(p=10, split='all', train_fraction=args.train_fraction, num_images=args.num_images)
    dl_all = DataLoader(ds_all, batch_size=256, shuffle=False, drop_last=False)

    tol = 0
        
    # Load a pre-trained checkpoint
    epoch_to_load = 29700
    checkpoint_path = os.path.join(weights_dir, f"model_epoch_{epoch_to_load}.pth")
    
    if not checkpoint_path:
        raise ValueError(f"Checkpoint path {checkpoint_path} does not exist.")
    
    state_dict = torch.load(checkpoint_path, map_location=device)    
    model.load_state_dict(state_dict)
    
    rf.model.eval()
    print("Starting sampling {}".format(dl_all.__len__()))
    
    for i, (x_tgt, x_src, label_tgt) in enumerate(dl_all):
        x_src, label_tgt = x_src.to(device), label_tgt.to(device)
        batch_size_train = x_src.size(0)
        x_gen = torch.randn(batch_size_train, 1, 32, 32).to(device)
        
        with torch.no_grad():
            # # Layer-level perturbation
            # layer_level_perturbation_test(rf, x_gen, x_src, sample_steps=1, save_image=True) # T=1
            
            # Head-level perturbation
            head_level_perturbation_test(rf, x_gen, x_src, n_heads=n_heads, sample_steps=1, save_image=False) # T=1