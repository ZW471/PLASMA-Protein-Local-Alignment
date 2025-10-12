from typing import Tuple, List

import torch
from torch import nn

from .utils import get_activation_function, LazyLayerNorm

def masked_out(batch_c: torch.Tensor, batch_q: torch.Tensor, out: torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor]:
    mask = batch_q.unsqueeze(1) == batch_c.unsqueeze(0)
    out = out.masked_fill(~mask, float('-inf'))
    return out, mask


def dot_product(
        H_q: torch.Tensor, 
        H_c: torch.Tensor,
        batch_q: torch.Tensor,
        batch_c: torch.Tensor
    ) -> torch.Tensor:
    return masked_out(batch_c, batch_q, torch.matmul(H_q, H_c.T))

# 20x faster
def hinge_non_linearity(
        H_q: torch.Tensor,
        H_c: torch.Tensor,
        batch_q: torch.Tensor,
        batch_c: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    # H_q: [n, f], H_c: [m, f]
    # 1) compute all pairwise L1 distances in a single C++/CUDA kernel
    l1 = torch.cdist(H_q, H_c, p=1)  # -> [n, m]
    # 2) row‐ and col‐ sums
    sum_q = H_q.sum(dim=1)  # -> [n]
    sum_c = H_c.sum(dim=1)  # -> [m]
    # 3) positive-part sum via the identity above
    #    pos[i,j] = (l1[i,j] + sum_q[i] - sum_c[j]) / 2
    pos = (l1 + sum_q.unsqueeze(1) - sum_c.unsqueeze(0)) * 0.5
    # 4) negate
    out = -pos

    # 5) mask out different‐batch entries → set to -inf
    #    mask[i,j] = True  iff  batch_q[i] == batch_c[j]
    return masked_out(batch_c, batch_q, out)

class DotProduct(nn.Module):
    def __init__(self, normalize=False):
        super().__init__()
        self.normalize = normalize
        if self.normalize:
            self.norm = LazyLayerNorm(elementwise_affine=False)

    def forward(self, H_q: torch.Tensor, H_c: torch.Tensor, batch_q: torch.Tensor, batch_c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for dot product similarity.
        
        Args:
            H_q: Query embeddings [n, f]
            H_c: Candidate embeddings [m, f]
            batch_q: Batch indices for queries [n]
            batch_c: Batch indices for candidates [m]
            
        Returns:
            Tuple of (similarity_scores, mask)
        """
        if self.normalize:
            H_q = self.norm(H_q)
            H_c = self.norm(H_c)
        return dot_product(H_q, H_c, batch_q, batch_c)

class Hinge(nn.Module):
    def __init__(self, normalize=False):
        super().__init__()
        self.normalize = normalize
        if self.normalize:
            self.norm = LazyLayerNorm(elementwise_affine=False)

    def forward(self,
        H_q: torch.Tensor,
        H_c: torch.Tensor, 
        batch_q: torch.Tensor,
        batch_c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the hinge non-linearity.
        
        Args:
            H_q: Query embeddings [n, f]
            H_c: Candidate embeddings [m, f]
            batch_q: Batch indices for queries [n]
            batch_c: Batch indices for candidates [m]
            
        Returns:
            Tuple of (similarity_scores, mask)
        """
        if self.normalize:
            H_q = self.norm(H_q)
            H_c = self.norm(H_c)
        return hinge_non_linearity(H_q, H_c, batch_q, batch_c)

class LRL(nn.Module):
    def __init__(self, hidden_dim=128, out_method="hinge"):
        super().__init__()
        self.out_method = out_method
        self.lrl = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self,
        H_q: torch.Tensor,
        H_c: torch.Tensor,
        batch_q: torch.Tensor,
        batch_c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        H_q = self.norm(self.lrl(H_q))
        H_c = self.norm(self.lrl(H_c))

        if self.out_method == "hinge":
            return hinge_non_linearity(H_q, H_c, batch_q, batch_c)
        elif self.out_method == "dot":
            return dot_product(H_q, H_c, batch_q, batch_c)
        else:
            raise ValueError(f"Unknown out_method {self.out_method}")


class MLP(nn.Module):
    def __init__(self, hidden_dims: List[int], activation_type: str = "relu", out_method: str = "hinge"):
        """
        Multi-layer perceptron similar to LRL but with configurable layer dimensions and activation.
        
        Args:
            hidden_dims: List of hidden layer dimensions. The number of layers will be len(hidden_dims) + 1
            activation_type: String name of activation function (e.g., 'relu', 'gelu', 'tanh')
            out_method: Output method for similarity computation (default: 'hinge')
        """
        super().__init__()
        self.out_method = out_method
        self.hidden_dims = hidden_dims
        
        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims must contain at least one dimension")
        
        # Build the MLP layers
        layers = []
        
        # First layer (lazy to handle input dimension automatically)
        layers.append(nn.LazyLinear(hidden_dims[0]))
        layers.append(get_activation_function(activation_type))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(get_activation_function(activation_type))
        
        self.mlp = nn.Sequential(*layers)
        
        # Batch normalization on the final output dimension
        self.bn = nn.BatchNorm1d(hidden_dims[-1])

    def forward(self,
        H_q: torch.Tensor,
        H_c: torch.Tensor,
        batch_q: torch.Tensor,
        batch_c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the MLP.
        
        Args:
            H_q: Query embeddings [n, input_dim]
            H_c: Candidate embeddings [m, input_dim]
            batch_q: Batch indices for queries [n]
            batch_c: Batch indices for candidates [m]
            
        Returns:
            Tuple of (similarity_scores, mask)
        """
        H_q = self.mlp(H_q)
        H_c = self.mlp(H_c)

        if self.out_method == "hinge":
            return hinge_non_linearity(H_q, H_c, batch_q, batch_c)
        else:
            raise ValueError(f"Unknown out_method {self.out_method}")


if __name__ == "__main__":
    H_q = torch.randn([4, 5])
    H_c = torch.randn([6, 5])
    batch_q = torch.tensor([0, 0, 1, 2]).long()
    batch_c = torch.tensor([0, 1, 1, 1, 2, 2], dtype=torch.long)

    # Test hinge_non_linearity function
    print("Testing hinge_non_linearity:")
    result = hinge_non_linearity(H_q, H_c, batch_q, batch_c)
    print(result)

    result = hinge_non_linearity(H_c, H_q, batch_c, batch_q)
    print(result)
    
    # Test LRL class
    print("\nTesting LRL:")
    lrl = LRL(hidden_dim=128)
    result = lrl(H_q, H_c, batch_q, batch_c)
    print(f"LRL output shape: {result[0].shape}")
    
    # Test MLP class
    print("\nTesting MLP:")
    # Example 1: 2-layer MLP with ReLU
    mlp1 = MLP(hidden_dims=[64, 32], activation_type="relu")
    result1 = mlp1(H_q, H_c, batch_q, batch_c)
    print(f"MLP (64->32, ReLU) output shape: {result1[0].shape}")
    
    # Example 2: 3-layer MLP with GELU
    mlp2 = MLP(hidden_dims=[128, 64, 32], activation_type="gelu")
    result2 = mlp2(H_q, H_c, batch_q, batch_c)
    print(f"MLP (128->64->32, GELU) output shape: {result2[0].shape}")
    
    # Example 3: Single layer MLP with Tanh
    mlp3 = MLP(hidden_dims=[256], activation_type="tanh")
    result3 = mlp3(H_q, H_c, batch_q, batch_c)
    print(f"MLP (256, Tanh) output shape: {result3[0].shape}")