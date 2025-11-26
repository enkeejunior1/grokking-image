import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np

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
        
        if split == 'all':
            self.data = all_pairs   # Evaluation only, so no permutation required.
        elif split == 'train':
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


from train import RF

class PerturbedRF(RF):    
    @torch.no_grad()
    def perturbed_sample_layer_level(self, z, cond, sample_steps=1): # T=1
        b = z.size(0)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * b).to(z.device).view([b, *([1] * len(z.shape[1:]))])
        perturbed_images = [[] for _ in range(len(self.model.layers))]
        
        handler = None
        for l, layer in enumerate(self.model.layers):
            # Remove the previously registered handler
            if handler is not None:
                handler.remove()
            
            # Register a new handler for the current layer
            handler = layer.attention.register_forward_hook(attention_hook_layer_level)
            
            # Create a copy of the input for perturbation
            z_copy = torch.clone(z)

            # Sampling loop
            for i in range(sample_steps, 0, -1):
                t = i / sample_steps
                t = torch.tensor([t] * b).to(z_copy.device)

                vc = self.model(z, t, cond)
                
                z_copy = z_copy - dt * vc
                
                # Append all the images for now
                perturbed_images[l].append(z)
                
        return perturbed_images
    
    @torch.no_grad()
    def perturbed_sample_head_level(self, z, cond, n_heads, sample_steps=1): # T=1
        b = z.size(0)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * b).to(z.device).view([b, *([1] * len(z.shape[1:]))])
        perturbed_images = [[None for _ in range(n_heads)] for _ in range(len(self.model.layers))]
        
        handler = None
        for l, layer in enumerate(self.model.layers):            
            for h in range(n_heads):
                # Remove the previously registered handler
                if handler is not None:
                    handler.remove()
                    
                # Create hook
                curr_hook = make_head_level_attention_hook(l, h, zero_out=False)
                
                # Register a new handler for the current head
                handler = layer.attention.register_forward_hook(curr_hook)
                
                # Create a copy of the input for perturbation
                z_copy = torch.clone(z)

                # Sampling loop
                for i in range(sample_steps, 0, -1):
                    t = i / sample_steps
                    t = torch.tensor([t] * b).to(z_copy.device)

                    vc = self.model(z, t, cond)
                    
                    z_copy = z_copy - dt * vc
                    
                    # Append all the images for now
                    perturbed_images[l][h] = z_copy
                
        return perturbed_images



def attention_hook_layer_level(module, input, output):
    '''
        Layer level perturbation : A = I
        A @ v = I @ v = v
    '''
    x, freqs_cis = input
    xv = module.wv(x)

    return module.wo(xv)


def make_head_level_attention_hook(layer_id, head_id, zero_out=False):
    def attention_hook_head_level(module, input, output):
        '''
            Head level perturbation
            Two options:
            1. Zero out the output of the head : 100% accuracy for every case
            2. Replace the output of the head with v of the head : works for most cases
        '''
        x, freqs_cis = input
        bsz, seqlen, _ = x.shape

        xq, xk, xv = module.wq(x), module.wk(x), module.wv(x)

        dtype = xq.dtype

        xq = module.q_norm(xq)
        xk = module.k_norm(xk)

        xq = xq.view(bsz, seqlen, module.n_heads, module.head_dim)
        xk = xk.view(bsz, seqlen, module.n_heads, module.head_dim)
        xv = xv.view(bsz, seqlen, module.n_heads, module.head_dim)

        xq, xk = module.apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        xq, xk = xq.to(dtype), xk.to(dtype)

        output = F.scaled_dot_product_attention(
            xq.permute(0, 2, 1, 3),
            xk.permute(0, 2, 1, 3),
            xv.permute(0, 2, 1, 3),
            dropout_p=0.0,
            is_causal=False,
        ).permute(0, 2, 1, 3)
        
        # Turn off the target head
        if zero_out:
            # Option 1: Zero out the output of the head
            output[:, :, head_id, :] = 0.0
        else:        
            # Option 2: Replace the output of the head with v of the head
            output[:, :, head_id, :] = xv[:, :, head_id, :]
        
        output = output.flatten(-2)

        return module.wo(output)

    return attention_hook_head_level