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
    import wandb
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
    
    exp_name = 'train_fraction_{}-num_images_{}'.format(args.train_fraction, args.num_images)
    weights_dir = os.path.join(args.generative_model_path, exp_name)
    results_dir = os.path.join(args.output_dir, exp_name)
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
    
    ds_train = MNISTModularArithmeticDataset(p=10, split='train', train_fraction=args.train_fraction, num_images=args.num_images)
    ds_valid = MNISTModularArithmeticDataset(p=10, split='valid', train_fraction=args.train_fraction, num_images=args.num_images)
    dl_train = DataLoader(ds_train, batch_size=256, shuffle=True, drop_last=False)
    dl_valid = DataLoader(ds_valid, batch_size=256, shuffle=False, drop_last=False)

    wandb.init(project=f"mnist_grokking", name=exp_name)
    tol = 0
    for epoch in tqdm(range(50000)):
        for i, (x_tgt, x_src, _) in enumerate(dl_train):
            x_tgt, x_src = x_tgt.cuda(), x_src.cuda()
            optimizer.zero_grad()
            loss = rf.forward(x_tgt, x_src)
            loss.backward()
            optimizer.step()

            wandb.log({"train_loss": loss.item()})

        if epoch % 1000 == 0:
            torch.save(model.state_dict(), os.path.join(weights_dir, f"model_epoch_{epoch}.pth"))

        if epoch % 10 != 0:
            continue

        rf.model.eval()
        with torch.no_grad():
            _, x_src_train, label_tgt_train = next(iter(dl_train))
            _, x_src_valid, label_tgt_valid = next(iter(dl_valid))
            batch_size_train = x_src_train.size(0)
            batch_size_valid = x_src_valid.size(0)
            x_tgt_train = torch.randn(batch_size_train, 1, 32, 32).cuda()
            x_tgt_valid = torch.randn(batch_size_valid, 1, 32, 32).cuda()
            x_src_train, label_tgt_train = x_src_train.cuda(), label_tgt_train.cuda()
            x_src_valid, label_tgt_valid = x_src_valid.cuda(), label_tgt_valid.cuda()
            images_train = rf.sample(x_tgt_train, x_src_train)
            images_valid = rf.sample(x_tgt_valid, x_src_valid)

            num_vis = 4
            result_train = torch.cat(
                [x_src_train[:num_vis], images_train[-1][:num_vis]], dim=1
            ).reshape(-1, 1, 32, 32)
            result_valid = torch.cat(
                [x_src_valid[:num_vis], images_valid[-1][:num_vis]], dim=1
            ).reshape(-1, 1, 32, 32)
            tvu.save_image(result_train, f"{results_dir}/sample_{epoch}_train.png", nrow=3)
            tvu.save_image(result_valid, f"{results_dir}/sample_{epoch}_valid.png", nrow=3)

            # gif = []
            # for image in images_train:
            #     # unnormalize
            #     image = image * 0.5 + 0.5
            #     image = image.clamp(0, 1)
            #     x_as_image = make_grid(image.float(), nrow=4)
            #     img = x_as_image.permute(1, 2, 0).cpu().numpy()
            #     img = (img * 255).astype(np.uint8)
            #     gif.append(Image.fromarray(img))

            # gif[0].save(
            #     f"{results_dir}/sample_{epoch}.gif",
            #     save_all=True,
            #     append_images=gif[1:],
            #     duration=100,
            #     loop=0,
            # )

            # last_img = gif[-1]
            # last_img.save(f"{results_dir}/sample_{epoch}_last.png")

            # ------------------------------------------------------------
            # Classifier
            # ------------------------------------------------------------
            classifier.eval()
            with torch.no_grad():
                preds = classifier(images_train[-1].detach())
                preds = preds.argmax(dim=1)
                preds = preds.cpu().numpy()
                label_tgt = label_tgt_train.cpu().numpy()
                accuracy = (preds == label_tgt).mean()
                wandb.log({"Train Accuracy": accuracy})

                preds = classifier(images_valid[-1].detach())
                preds = preds.argmax(dim=1)
                preds = preds.cpu().numpy()
                label_tgt = label_tgt_valid.cpu().numpy()
                accuracy = (preds == label_tgt).mean()
                wandb.log({"Valid Accuracy": accuracy})
                
                # if accuracy > 0.99:
                #     tol += 1
                #     if tol > 25:
                #         torch.save(model.state_dict(), os.path.join(weights_dir, f"model_final.pth"))
                #         exit()
                # else:
                #     tol = 0
            classifier.train()
        rf.model.train()
    torch.save(model.state_dict(), os.path.join(weights_dir, f"model_final.pth"))