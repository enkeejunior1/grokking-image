import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np

import pydot
import colorsys
import os

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
    def __init__(self, model, ln=True):
        super().__init__(model, ln=True)
        self.deactivated_heads = set()
        self.head_handlers = dict()
    
    def add_deactivated_head(self, layer_id, head_id):
        if (layer_id, head_id) in self.deactivated_heads:
            print(f"Head {head_id} in layer {layer_id} is already deactivated.")
            return
        
        curr_hook = make_head_level_attention_hook(layer_id, head_id, zero_out=False)
        curr_layer = self.model.layers[layer_id]
        handler = curr_layer.attention.register_forward_hook(curr_hook)
        
        self.deactivated_heads.add((layer_id, head_id))
        self.head_handlers[(layer_id, head_id)] = handler
        
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
                # Pass if this head is already deactivated
                if (l, h) in self.deactivated_heads:
                    # print(f"Skipping deactivated head ({l},{h}).")
                    continue
                
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

def generate_rainbow_hex_colors(N):
    """
    N개의 레이어에 고르게 분포된 무지개색 HEX 코드 리스트를 생성합니다.
    """
    hex_colors = []
    MAX_HUE = 0.85 
    
    for i in range(N):
        if N == 1:
            hue_fraction = 0.0
        else:
            hue_fraction = i / (N - 1) * MAX_HUE
        
        rgb_float = colorsys.hsv_to_rgb(hue_fraction, 1.0, 1.0)
        
        r = int(rgb_float[0] * 255)
        g = int(rgb_float[1] * 255)
        b = int(rgb_float[2] * 255)
        
        hex_code = f"#{r:02x}{g:02x}{b:02x}".upper()
        hex_colors.append(hex_code)
        
    return hex_colors


def draw_network(n_layers, n_heads, deactivated_heads, results_dir):
    graph_attn = pydot.Dot("transformer_flow", graph_type="digraph", rankdir="LR", splines="line") 
    layer_colors = generate_rainbow_hex_colors(n_layers)
    rad = "0.2"
    for l in range(n_layers):
        curr_cluster = pydot.Cluster(f"cluster_L{l+1}", label=f"Layer {l+1}", color="lightgrey", style="filled", fillcolor="#F9F9F9")
        
        for h in range(n_heads):
            if (l, h) in deactivated_heads:
                head_node = pydot.Node(f"L{l+1}_H{h+1}", label="", width=rad, height=rad, fixed_size="true", shape="circle", style="filled", fillcolor="grey", penwidth="0")
            else:
                head_node = pydot.Node(f"L{l+1}_H{h+1}", label="", width=rad, height=rad, fixed_size="true", shape="circle", style="filled", fillcolor=layer_colors[l], penwidth="0")
            curr_cluster.add_node(head_node)
        
        graph_attn.add_subgraph(curr_cluster)
        
        if l > 0:
            for h_prev in range(n_heads):
                for h_curr in range(n_heads):
                    if (l-1, h_prev) not in deactivated_heads and (l, h_curr) not in deactivated_heads:
                        edge = pydot.Edge(f"L{l}_H{h_prev+1}", f"L{l+1}_H{h_curr+1}", style="solid", color="black", arrowhead="normal", arrowsize="0.2")
                        graph_attn.add_edge(edge)
    
    graph_attn.write_png(f"{results_dir}/transformer_structure.png")
    # print("Graph structure with clusters defined.")
    

from PIL import Image

def stack_images_vertically(image_paths, output_path, trial_num, alignment='center'):
    """
    여러 이미지 파일을 불러와 수직으로 병합하고, 너비를 가장 넓은 이미지에 맞춥니다.
    
    Args:
        image_paths (list): 이미지 파일 경로 리스트
        alignment (str): 'left', 'center', 'right' 중 하나로 수평 정렬 지정
        
    Returns:
        Image: 병합된 단일 Image 객체
    """
    if not image_paths:
        return None

    # 1. 모든 이미지 로드 및 치수 계산
    images = [Image.open(path).convert("RGB") for path in image_paths]
    
    # 최대 너비와 총 높이 계산
    max_width = max(img.width for img in images)
    total_height = sum(img.height for img in images)
    
    # 2. 새로운 캔버스 생성 (흰색 배경)
    # RGB 모드, 최대 너비, 총 높이
    stacked_image = Image.new('RGB', (max_width, total_height), color='white')
    
    # 3. 이미지 순서대로 붙여넣기
    y_offset = 0
    for img in images:
        
        # 수평 정렬에 따른 x 좌표 계산
        if alignment == 'center':
            x_offset = (max_width - img.width) // 2
        elif alignment == 'right':
            x_offset = max_width - img.width
        else: # 'left' 또는 기본값
            x_offset = 0
            
        stacked_image.paste(img, (x_offset, y_offset))
        y_offset += img.height # 다음 이미지를 위해 높이 업데이트
    
    image_name = f"stacked_result_trial_{trial_num}.png"
    full_path = os.path.join(output_path, image_name)
    try:
        stacked_image.save(full_path)
    except Exception as e:
        print(f"이미지 저장 오류 발생: {e}")