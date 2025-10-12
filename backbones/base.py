"""
Base backbone model abstract class for protein embedding models.
"""

from abc import ABC, abstractmethod
import torch.nn as nn
from typing import Dict, Any


class BaseBackboneModel(nn.Module, ABC):
    """
    Abstract base class for protein backbone embedding models.
    
    All backbone models should inherit from this class and implement
    the required abstract methods.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the backbone model.
        
        Args:
            config: Configuration dictionary containing model parameters
        """
        super().__init__()
        self.config = config
    
    @abstractmethod
    def forward(self, sequences: list) -> Dict[str, Dict[str, Any]]:
        """
        Forward pass to generate embeddings.
        
        Args:
            sequences: List of protein sequences (strings)
            
        Returns:
            Dictionary with structure:
            {
                "model_name_1": {
                    "AA": tensor,  # Amino acid level embeddings
                    "PR": tensor   # Protein level embeddings
                },
                "model_name_2": {
                    "AA": tensor,
                    "PR": tensor
                }
            }
        """
        pass