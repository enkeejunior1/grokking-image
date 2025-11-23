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
from datetime import datetime


class MNISTModularArithmeticDataset(Dataset):
    def __init__(self, p=9, split='train', train_fraction=0.3, num_images=None):
        """
        MNIST-based modular arithmetic dataset.
        
        Args:
            p: Prime modulus (should be <= 10 for MNIST digits)
            split: 'train' or 'val'
            train_fraction: Fraction of data to use for training
        """
        self.p = p
        
        # Load MNIST dataset
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        mnist_dataset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform
        )
        
        # Group MNIST images by digit
        self.digit_images = {i: [] for i in range(10)}
        for img, label in mnist_dataset:
            if label < p:  # Only use digits 0 to p-1
                self.digit_images[label].append(img)
        if num_images is not None:
            self.digit_images = {i: self.digit_images[i][:num_images] for i in range(10)}
            
        # Generate all possible pairs for modular addition
        all_pairs = []
        for a in range(p):
            for b in range(p):
                result = (a + b) % p
                all_pairs.append((a, b, result))
        
        # Split into train/val
        np.random.seed(42)
        indices = np.random.permutation(len(all_pairs))
        train_size = int(train_fraction * len(all_pairs))
        
        if split == 'train':
            self.data = [all_pairs[i] for i in indices[:train_size]]
        elif split == 'all':
            self.data = all_pairs
        else:
            self.data = [all_pairs[i] for i in indices[train_size:]]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        a, b, result = self.data[idx]
        
        # Randomly select MNIST images for each digit
        img_a = self.digit_images[a][np.random.randint(len(self.digit_images[a]))]
        img_b = self.digit_images[b][np.random.randint(len(self.digit_images[b]))]
        img_result = self.digit_images[result][np.random.randint(len(self.digit_images[result]))]
        assert img_a.shape == img_b.shape == img_result.shape == (1, 32, 32)
        
        img_tgt = img_result
        img_src = torch.cat([img_a, img_b], dim=0)
        label_tgt = torch.tensor(result, dtype=torch.long)
        return img_tgt, img_src, label_tgt

class RF:
    def __init__(self, model, ln=True):
        self.model = model
        self.ln = ln

    def forward(self, x, cond):
        b = x.size(0)
        if self.ln:
            nt = torch.randn((b,)).to(x.device)
            t = torch.sigmoid(nt)
        else:
            t = torch.rand((b,)).to(x.device)
        texp = t.view([b, *([1] * len(x.shape[1:]))])
        z1 = torch.randn_like(x)
        zt = (1 - texp) * x + texp * z1
        vtheta = self.model(zt, t, cond)
        loss = ((z1 - x - vtheta) ** 2).mean(dim=list(range(1, len(x.shape)))).mean()
        return loss

    @torch.no_grad()
    def sample(self, z, cond, sample_steps=50):
        b = z.size(0)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * b).to(z.device).view([b, *([1] * len(z.shape[1:]))])
        images = [z]
        for i in range(sample_steps, 0, -1):
            t = i / sample_steps
            t = torch.tensor([t] * b).to(z.device)

            vc = self.model(z, t, cond)
            # if null_cond is not None:
            #     vu = self.model(z, t, null_cond)
            #     vc = vu + cfg * (vc - vu)

            z = z - dt * vc
            images.append(z)
        return images
    

if __name__ == "__main__":
    from model import DiT_Llama
    from classifier import MNISTClassifier
    import argparse

    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--classifier_weights_path", type=str, default="weights/mnist_classifier_weights.pth")
        parser.add_argument("--generative_model_path", type=str, default="weights/")
        parser.add_argument("--train_fraction", type=float, default=0.7)
        parser.add_argument("--output_dir", type=str, default="results/")
        parser.add_argument("--num_images", type=int, default=1)
        return parser.parse_args()
    args = parse_args()
    
    now = datetime.now()
    formatted_time = now.strftime("%m.%d.%H.%M")
    
    pretrained_name = 'train_fraction_{}-num_images_{}'.format(args.train_fraction, args.num_images)
    experiment_name = 'test_attention_head_{}'.format(formatted_time)
    weights_dir = os.path.join(args.generative_model_path, pretrained_name)
    results_dir = os.path.join(args.output_dir, experiment_name)
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    model = DiT_Llama(
        3, 32, dim=256, n_layers=10, n_heads=8,
    ).cuda()
    
    classifier = MNISTClassifier().cuda()
    classifier.load_state_dict(torch.load(args.classifier_weights_path))

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {model_size}, {model_size / 1e6}M")

    rf = RF(model)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    
    # ds_train = MNISTModularArithmeticDataset(p=10, split='train', train_fraction=args.train_fraction, num_images=args.num_images)
    # ds_valid = MNISTModularArithmeticDataset(p=10, split='valid', train_fraction=args.train_fraction, num_images=args.num_images)
    # dl_train = DataLoader(ds_train, batch_size=256, shuffle=True, drop_last=False)
    # dl_valid = DataLoader(ds_valid, batch_size=256, shuffle=False, drop_last=False)
    
    ds_all = MNISTModularArithmeticDataset(p=10, split='all', train_fraction=args.train_fraction, num_images=args.num_images)
    dl_all = DataLoader(ds_all, batch_size=256, shuffle=False, drop_last=False)

    tol = 0
    
    
    # Load a pre-trained checkpoint
    weights_dir = "weights/train_fraction_0.9-num_images_16"
    epoch_to_load = 234900
    checkpoint_path = os.path.join(weights_dir, f"model_epoch_{epoch_to_load}.pth")
    
    if not checkpoint_path:
        raise ValueError(f"Checkpoint path {checkpoint_path} does not exist.")
    
    state_dict = torch.load(checkpoint_path)
    
    model.load_state_dict(state_dict)
    
    rf.model.eval()
    print("Starting sampling {}".format(dl_all.__len__()))
    
    print(dl_all.size())
    
    # for i, (x_tgt, x_src, label_tgt) in enumerate(dl_all):
    #     _, x_src, label_tgt = x_tgt.cuda(), x_src.cuda(), label_tgt.cuda()
    #     batch_size_train = x_src.size(0)
    #     x_tgt = torch.randn(batch_size_train, 1, 32, 32).cuda()
        
    #     with torch.no_grad():
    #         images = rf.sample(x_tgt, x_src)

    #         num_vis = 4
    #         result = torch.cat(
    #             [x_src[:num_vis], images[-1][:num_vis]], dim=1
    #         ).reshape(-1, 1, 32, 32)
    #         tvu.save_image(result, f"{results_dir}/sample_{i+1}_attention.png", nrow=3)
    
    print("Sampling completed and images saved.")   
    