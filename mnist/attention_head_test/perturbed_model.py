from model import Attention

class PerturbedAttention(Attention):
    def __init__(self, *args, noise_level=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_level = noise_level

    def forward(self, x, context=None):
        # Original attention computation
        attn_output = super().forward(x, context)
        # Add perturbation noise
        noise = torch.randn_like(attn_output) * self.noise_level
        return attn_output + noise