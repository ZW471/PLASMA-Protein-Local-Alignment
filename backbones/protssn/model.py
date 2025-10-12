"""
ProtSSN backbone model for protein embedding generation.

This module provides a wrapper around the ProtSSN model to make it compatible
with the embed.py system and follow the BaseBackboneModel interface.
"""

import os
import sys
import torch
import yaml
from pathlib import Path
from typing import Dict, Any, List
from torch_geometric.data import Data

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from ..base import BaseBackboneModel
from .protssn import ProtSSN
from .models import PLM_model, GNN_model
from .dataset_utils import NormalizeProtein


class ProtSSNModel(BaseBackboneModel):
    """
    ProtSSN backbone model for generating protein embeddings from PDB structures.
    
    This model uses protein structure information from PDB files to generate
    both amino acid level and protein level embeddings.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # ProtSSN specific parameters
        self.gnn_config_path = config.get('gnn_config', 'backbones/protssn/egnn.yaml')
        self.gnn_model_path = config.get('gnn_model_path', 'weights/protssn_k20_h512.pt')
        self.plm_model_name = config.get('plm_model', 'facebook/esm2_t33_650M_UR50D')
        self.plm_hidden_size = config.get('plm_hidden_size', 1280)
        self.gnn_hidden_dim = config.get('gnn_hidden_dim', 512)
        self.c_alpha_max_neighbors = config.get('c_alpha_max_neighbors', 10)
        self.gnn_type = config.get('gnn', 'egnn')
        
        # Weight directory for ESM model storage
        self.cache_dir = config.get('cache_dir', f"weights/{self.plm_model_name.split('/')[-1]}")
        
        # Load configuration
        self._load_config()
        
        # Initialize models
        self._load_models()
        
    def _load_config(self):
        """Load GNN configuration from YAML file."""
        gnn_config_path = Path(self.gnn_config_path)
        if not gnn_config_path.is_absolute():
            # Try relative to project root first
            project_root = Path(__file__).parent.parent.parent
            gnn_config_path = project_root / self.gnn_config_path
            if not gnn_config_path.exists():
                # Try relative to this module
                gnn_config_path = Path(__file__).parent / self.gnn_config_path.split('/')[-1]
        
        if gnn_config_path.exists():
            with open(gnn_config_path, 'r') as f:
                gnn_configs = yaml.load(f, Loader=yaml.FullLoader)
                self.gnn_config = gnn_configs[self.gnn_type]
                self.gnn_config["hidden_channels"] = self.gnn_hidden_dim
        else:
            # Default EGNN configuration
            self.gnn_config = {
                "hidden_channels": self.gnn_hidden_dim,
                "edge_attr_dim": 93,
                "output_dim": 20,
                "dropout": 0.1,
                "n_layers": 3,
                "residual": False,
                "embedding": False,
                "embedding_dim": 64,
                "mlp_num": 2
            }
    
    def _load_models(self):
        """Initialize PLM and GNN models."""
        # Create args object for compatibility with existing ProtSSN code
        class Args:
            def __init__(self, config):
                self.plm = config.plm_model_name
                self.plm_hidden_size = config.plm_hidden_size
                self.gnn = config.gnn_type
                self.gnn_config = config.gnn_config
                self.gnn_hidden_dim = config.gnn_hidden_dim
                self.c_alpha_max_neighbors = config.c_alpha_max_neighbors
                self.cache_dir = config.cache_dir
        
        args = Args(self)
        
        # Initialize PLM model (ESM)
        self.plm_model = PLM_model(args)
        
        # Initialize GNN model
        self.gnn_model = GNN_model(args)
        
        # Load pre-trained GNN weights if available
        gnn_model_path = Path(self.gnn_model_path)
        if not gnn_model_path.is_absolute():
            # Try relative to project root first, then to this module
            project_root = Path(__file__).parent.parent.parent
            gnn_model_path = project_root / self.gnn_model_path
            if not gnn_model_path.exists():
                gnn_model_path = Path(__file__).parent / self.gnn_model_path
        
        if gnn_model_path.exists():
            self.gnn_model.load_state_dict(torch.load(gnn_model_path, map_location=self.device))
        else:
            print(f"Warning: GNN model weights not found at {gnn_model_path}")
        
        # Initialize ProtSSN wrapper
        normalize_path = Path(__file__).parent / f'cath_k{self.c_alpha_max_neighbors}_mean_attr.pt'
        self.protssn = ProtSSN(
            c_alpha_max_neighbors=self.c_alpha_max_neighbors,
            pre_transform=NormalizeProtein(filename=str(normalize_path)) if normalize_path.exists() else None,
            plm_model=self.plm_model,
            gnn_model=self.gnn_model
        )
        
    def forward(self, pdb_paths: List[str]) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Generate embeddings from PDB files.
        
        Args:
            pdb_paths: List of paths to PDB files
            
        Returns:
            Dictionary with ProtSSN embeddings at AA and PR levels
        """
        model_key = "ProtSSN"
        
        aa_embeddings = []
        pr_embeddings = []
        
        with torch.no_grad():
            for pdb_path in pdb_paths:
                # Generate protein graph from PDB
                graph = self.protssn.generate_protein_graph(pdb_path)
                if graph is None:
                    raise RuntimeError(f" Failed to generate graph for {pdb_path}. ")
                # Process through PLM and GNN
                batch_graph = self.plm_model([graph])
                logits, embeds = self.gnn_model(batch_graph)
                
                # Store embeddings
                # AA level: residue-level embeddings (embeds)
                # PR level: protein-level embeddings (mean pooling of residues)
                aa_embedding = embeds.detach().cpu()
                pr_embedding = embeds.mean(dim=0).detach().cpu()  # Remove keepdim=True for [1280] shape
                
                aa_embeddings.append(aa_embedding)
                pr_embeddings.append(pr_embedding)
                
                # Clean up GPU memory
                del embeds, logits, batch_graph, graph
                torch.cuda.empty_cache()
        
        # For single PDB file, return without list wrapping
        if len(pdb_paths) == 1:
            return {
                model_key: {
                    "AA": aa_embeddings[0],
                    "PR": pr_embeddings[0]
                }
            }
        
        # For multiple PDB files, stack tensors
        return {
            model_key: {
                "AA": torch.stack([emb.squeeze(0) if emb.dim() > 1 else emb for emb in aa_embeddings]),
                "PR": torch.stack([emb.squeeze(0) if emb.dim() > 1 else emb for emb in pr_embeddings])
            }
        }
    
    def compute_embedding(self, pdb_path: str, reduction: str = None) -> torch.Tensor:
        """
        Compute embeddings for a single PDB file (compatibility method).
        
        Args:
            pdb_path: Path to PDB file
            reduction: Reduction method ('mean', 'sum', 'max', or None)
            
        Returns:
            Embedding tensor
        """
        return self.protssn.compute_embedding(pdb_path, reduction)
    
    def compute_logits(self, pdb_path: str) -> torch.Tensor:
        """
        Compute logits for a single PDB file (compatibility method).
        
        Args:
            pdb_path: Path to PDB file
            
        Returns:
            Logits tensor
        """
        return self.protssn.compute_logits(pdb_path)