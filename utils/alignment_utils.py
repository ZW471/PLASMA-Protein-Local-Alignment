"""
Alignment utilities for scoring and evaluation.
"""

import torch
import json
from pathlib import Path
from torch.nn.functional import cosine_similarity
from torch_scatter import scatter_sum
from torchmetrics.classification import BinaryAUROC
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from typing import Dict, Any, Tuple
from loguru import logger
import torch.nn.functional as F
from omegaconf import DictConfig

def max_main_diag_score(x: torch.Tensor, T: int = 10) -> float:
    if T > 0:
        kernel = torch.eye(T, device=x.device, dtype=torch.float32)  # (T,T)
        kernel = kernel.view(1, 1, T, T) 
        x_f = x.to(torch.float32).unsqueeze(0).unsqueeze(0) 
        return (F.conv2d(x_f, kernel).max().clamp(max=T) / T).item()
    else:
        return 1.0

def alignment_score(H_q: torch.Tensor, H_c: torch.Tensor, alignment_matrix: torch.Tensor, batch_c: torch.Tensor, threshold: float = 0.5, only_pos: bool = True, K: int = 10) -> torch.Tensor:
    """Compute alignment score between query and candidate using alignment matrix."""
    c_matched = (alignment_matrix.max(dim=0).values > threshold).unsqueeze(1)
    q_matched = (alignment_matrix.max(dim=1).values > threshold).unsqueeze(1)
    H_q_agg = scatter_sum(c_matched * alignment_matrix.T @ (q_matched * H_q), batch_c, dim=0)
    H_c_agg = scatter_sum(c_matched * H_c, batch_c, dim=0)
    result = cosine_similarity(H_q_agg, H_c_agg, dim=-1) * max_main_diag_score(alignment_matrix, T=K)
    if only_pos:
        return torch.clamp(result, min=0.0)
    else:
        return result

def label_match_loss(l1, l2, alignment, is_match=True):
    """
    Calculate the label match loss based on the alignment and labels.
    
    Args:
        l1 (torch.Tensor): Labels for the first protein.
        l2 (torch.Tensor): Labels for the second protein.
        alignment (torch.Tensor): Alignment matrix.
        is_match (bool): If True, calculate loss for matching labels; otherwise, for non-matching labels.
    
    Returns:
        float: The calculated label match loss.
    """
    if is_match:
        return torch.clamp(l2 - l1 @ alignment, min=0).sum() / l2.sum()
    else:
        return torch.tensor(0.0)   # No loss for non-matching labels


def load_sequence_embedding(seq_id: str, embeddings_dir: Path, device: torch.device) -> torch.Tensor:
    """Load embedding for a specific sequence ID."""
    embedding_file = embeddings_dir / f"{seq_id}.pt"
    if not embedding_file.exists():
        raise FileNotFoundError(f"Embedding file not found: {embedding_file}")
    
    embedding = torch.load(embedding_file, map_location=device, weights_only=False)
    
    # Handle batch dimension if present
    if embedding.dim() == 3 and embedding.shape[0] == 1:
        embedding = embedding.squeeze(0)
    
    return embedding


def setup_model(cfg: DictConfig, device: torch.device) -> Tuple[Any, str]:
    """Setup model based on configuration."""
    from alignment.alignment import Alignment
    
    model_type = cfg.model.type.lower()
    
    if model_type == "plasma":
        # Load plasma alignment model with trained weights
        model_path = Path(cfg.model.model_cfg.model_path)
        config_path = model_path / "config.json"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            model_config = json.load(f)
        
        # Find the model checkpoint file
        model_files = list(model_path.glob("*_best.pt"))
        if not model_files:
            raise FileNotFoundError(f"No model checkpoint (*_best.pt) found in: {model_path}")
        
        model_checkpoint = model_files[0]
        
        # Extract ALL model settings from config.json
        model_info = model_config.get('model_info', {})
        
        # Get eta configuration
        eta_config = model_info.get('eta_config', {})
        eta_type = eta_config.get('type', 'lrl')
        eta_kwargs = {k: v for k, v in eta_config.items() if k != 'type'}
        
        # Get omega configuration  
        omega_config = model_info.get('omega_config', {})
        omega_type = omega_config.get('type', 'sinkhorn')
        omega_kwargs = {k: v for k, v in omega_config.items() if k != 'type'}
        
        # Create and load alignment model
        alignment_model = Alignment(
            eta=eta_type,
            omega=omega_type,
            omega_kwargs=omega_kwargs,
            eta_kwargs=eta_kwargs,
        ).to(device)
        
        alignment_model.load_state_dict(torch.load(model_checkpoint, map_location=device, weights_only=False))
        alignment_model.eval()
        
        backbone_model = model_info.get('backbone_model', 'ProstT5')
        logger.success(f"Loaded {model_type} model from {model_checkpoint}")
        logger.info(f"Model settings: eta={eta_type}, omega={omega_type}, backbone={backbone_model}")
        
        return alignment_model, backbone_model
        
    elif model_type == "plasma-pf":
        # Parameter-free plasma model - no checkpoint loading, uses hinge eta
        eta = "hinge"  # plasma-pf always uses hinge
        omega = cfg.model.model_cfg.get('omega', 'sinkhorn')
        backbone_model = cfg.model.model_cfg.get('model_name', 'ProstT5')
        
        # Create alignment model without loading weights
        alignment_model = Alignment(
            eta=eta,
            omega=omega,
            omega_kwargs=cfg.model.model_cfg.get('omega_kwargs', {}),
            eta_kwargs=cfg.model.model_cfg.get('eta_kwargs', {}),
        ).to(device)
        
        alignment_model.eval()
        
        logger.success(f"Created parameter-free plasma model with eta={eta}, omega={omega}")
        
        return alignment_model, backbone_model
        
    elif model_type == "backbone":
        # For backbone models, no model loading needed - just use cosine similarity
        backbone_model = cfg.model.model_cfg.get('model_name', 'ProstT5')
        logger.info(f"Using backbone model: {backbone_model} with cosine similarity")
        
        return None, backbone_model
        
    else:
        raise ValueError(f"Unknown model type: {model_type}. Must be 'plasma', 'plasma-pf', or 'backbone'")


def compute_alignment_score_single(query_emb: torch.Tensor, 
                                 candidate_emb: torch.Tensor,
                                 model: Any,
                                 model_type: str,
                                 device: torch.device,
                                 threshold: float = 0.5,
                                 K: int = 10) -> Tuple[float, torch.Tensor]:
    """Compute alignment score between two proteins and return score + matrix."""
    if model_type.lower() == "backbone":
        # Use cosine similarity for backbone models
        query_flat = query_emb.flatten()
        candidate_flat = candidate_emb.flatten()
        
        # Compute cosine similarity
        cos_sim = cosine_similarity(query_flat.unsqueeze(0), candidate_flat.unsqueeze(0))
        score = (cos_sim.item() + 1) / 2  # Convert from [-1, 1] to [0, 1]
        
        # Create dummy alignment matrix for backbone models
        alignment_matrix = torch.ones(query_emb.shape[0], candidate_emb.shape[0], device=device) * score
        
    else:
        # Use alignment model for plasma models
        query_batch = torch.zeros(query_emb.shape[0], dtype=torch.long, device=device)
        candidate_batch = torch.zeros(candidate_emb.shape[0], dtype=torch.long, device=device)
        
        with torch.no_grad():
            alignment_matrix = model(
                query_emb,
                candidate_emb,
                query_batch,
                candidate_batch
            )
            
            # Remove batch dimension if present
            if alignment_matrix.dim() == 3:
                alignment_matrix = alignment_matrix.squeeze(0)
            
            # Compute alignment score
            score = alignment_score(
                query_emb,
                candidate_emb,
                alignment_matrix,
                candidate_batch,
                threshold=threshold,
                K=K
            ).item()
    
    return score, alignment_matrix


def compute_alignment_score_batch(query_emb: torch.Tensor, 
                                candidate_emb: torch.Tensor,
                                model: Any,
                                model_type: str,
                                device: torch.device,
                                threshold: float = 0.5,
                                K: int = 10) -> float:
    """Compute alignment score between two proteins (batch version, returns only score)."""
    score, _ = compute_alignment_score_single(query_emb, candidate_emb, model, model_type, device, threshold, K)
    return score

# def evaluate_dataset(data_loader: DataLoader, 
    #                 dataset_name: str, 
    #                 alignment_model: torch.nn.Module,
    #                 alignment_no_learning: torch.nn.Module,
    #                 embeddings_dict: Dict[str, torch.Tensor],
    #                 device: torch.device) -> Dict[str, Any]:
    # """Evaluate a dataset and return AUC scores for different methods."""
    # from .data_utils import get_batch_embeddings
    
    # auc_alignment = BinaryAUROC()
    # auc_no_learning = BinaryAUROC()
    # auc_tm = BinaryAUROC()
    # auc_prostt5 = BinaryAUROC()
    
    # skip_count = 0
    
    # with torch.no_grad():
    #     for batch in tqdm(data_loader, desc=f"Evaluating {dataset_name}"):
    #         try:
    #             batch = batch.to(device)
                
    #             # Get embeddings on-the-fly from CPU dictionary
    #             query_emb, candidate_emb, query_full_emb, candidate_full_emb, query_emb_batch, candidate_emb_batch = get_batch_embeddings(batch, embeddings_dict, device)
                
    #             # Skip if no embeddings found
    #             if query_emb.numel() == 0 or candidate_emb.numel() == 0:
    #                 skip_count += 1
    #                 continue
                
    #             # Our trained alignment model
    #             M = alignment_model(query_emb, candidate_emb, query_emb_batch, candidate_emb_batch)
    #             sim_alignment = alignment_score(query_emb, candidate_emb, M, candidate_emb_batch)
                
    #             # Alignment without learning
    #             M_no_learning = alignment_no_learning(query_emb, candidate_emb, query_emb_batch, candidate_emb_batch)
    #             sim_no_learning = alignment_score(query_emb, candidate_emb, M_no_learning, candidate_emb_batch)
                
    #             # ProST-T5 baseline (simple averaging)
    #             sim_prostt5 = distance_score(query_emb, candidate_emb, query_emb_batch, candidate_emb_batch)
                
    #             # TM-Vec baseline
    #             sim_tm = cosine_similarity(query_full_emb, candidate_full_emb, dim=-1)
                
    #             # Update metrics
    #             auc_alignment.update(sim_alignment.detach().cpu(), batch.y.cpu())
    #             auc_no_learning.update(sim_no_learning.detach().cpu(), batch.y.cpu())
    #             auc_prostt5.update(sim_prostt5.detach().cpu(), batch.y.cpu())
    #             auc_tm.update(sim_tm.detach().cpu(), batch.y.cpu())
                
    #             # Memory cleanup
    #             del M, sim_alignment, M_no_learning, sim_no_learning, sim_prostt5, sim_tm
    #             del query_emb, candidate_emb, query_full_emb, candidate_full_emb, query_emb_batch, candidate_emb_batch
    #             torch.cuda.empty_cache()
                
    #         except RuntimeError as e:
    #             logger.error(f"Error processing batch: {e}")
    #             skip_count += 1
    #             continue
    
    # # Compute final scores
    # results = {
    #     'alignment_trained': auc_alignment.compute().item(),
    #     'no_learning': auc_no_learning.compute().item(),
    #     'prostt5': auc_prostt5.compute().item(),
    #     'tm_vec': auc_tm.compute().item(),
    #     'skipped': skip_count
    # }
    
    # return results