import torch
from torch import nn


class LazyLayerNorm(nn.Module):
    def __init__(self, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.weight = None
        self.bias = None
        
    def _initialize_parameters(self, normalized_shape):
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        
    def forward(self, x):
        if self.weight is None and self.elementwise_affine:
            self._initialize_parameters(x.shape[-1])
            if x.is_cuda:
                self.weight = self.weight.cuda()
                self.bias = self.bias.cuda()
        
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        if self.elementwise_affine:
            x_norm = x_norm * self.weight + self.bias
            
        return x_norm

def normalize_pos_diff(pos_diff, eps=1e-8):
    return pos_diff / (pos_diff.norm(dim=2, keepdim=True) + eps)

def get_activation_function(activation_type: str) -> nn.Module:
    """
    Get activation function from string type.
    
    Args:
        activation_type: String name of the activation function
        
    Returns:
        PyTorch activation module
        
    Raises:
        ValueError: If activation type is not supported
    """
    activation_type = activation_type.lower()
    
    activation_map = {
        'relu': nn.ReLU(),
        'leaky_relu': nn.LeakyReLU(),
        'gelu': nn.GELU(),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
        'swish': nn.SiLU(),  # SiLU is the same as Swish
        'silu': nn.SiLU(),
        'elu': nn.ELU(),
        'prelu': nn.PReLU(),
        'softplus': nn.Softplus(),
        'none': nn.Identity(),
        'identity': nn.Identity(),
    }
    
    if activation_type not in activation_map:
        available_activations = ', '.join(activation_map.keys())
        raise ValueError(f"Unsupported activation type '{activation_type}'. "
                       f"Available options: {available_activations}")
    
    return activation_map[activation_type]
