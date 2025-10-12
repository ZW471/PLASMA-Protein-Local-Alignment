"""
Data utilities for processing embeddings and creating datasets.
"""

import torch
import pandas as pd
import json
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from typing import List, Tuple, Dict, Any


class PairGraphDataset(Dataset):
    """PyTorch Geometric dataset for protein pairs."""
    
    def __init__(self, data_list):
        super().__init__()
        self.data_list = data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        return self.data_list[idx]


def convert_pairs_to_dataset(pairs_list: List[Tuple[Tuple[int, int], int]], 
                           df_full: pd.DataFrame) -> PairGraphDataset:
    """Convert list of (pair, label) to PyG dataset without embedding tensors."""
    data_list = []
    
    for (idx1, idx2), label in pairs_list:
        # Get the actual row data
        row1 = df_full.iloc[idx1]
        row2 = df_full.iloc[idx2]
        
        # Get UIDs for embedding lookup
        query_uid = row1['uid']
        candidate_uid = row2['uid']
        
        # Create composite keys for tracking purposes
        query_key = f"{row1['interpro_id']}_{query_uid}"
        candidate_key = f"{row2['interpro_id']}_{candidate_uid}"
        
        data = Data(
            y=torch.tensor([label], dtype=torch.long),
            query_uid=query_uid,
            candidate_uid=candidate_uid,
            frag_key=query_key,
            seq_key=candidate_key,
            query_idx=idx1,
            candidate_idx=idx2
        )
        data_list.append(data)
    
    return PairGraphDataset(data_list)


def get_batch_embeddings(batch: Data, embeddings_dict: Dict[str, torch.Tensor], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get embeddings for a batch on-the-fly from CPU dictionary."""
    batch_size = len(batch.query_uid)
    
    # Collect all embeddings
    query_embs = []
    candidate_embs = []
    query_batch_indices = []
    candidate_batch_indices = []
    
    for i in range(batch_size):
        query_uid = batch.query_uid[i]
        candidate_uid = batch.candidate_uid[i]
        
        # Get AA-level embeddings using uid directly
        if query_uid in embeddings_dict and candidate_uid in embeddings_dict:
            query_emb = embeddings_dict[query_uid].to(device)
            candidate_emb = embeddings_dict[candidate_uid].to(device)
            
            query_embs.append(query_emb)
            candidate_embs.append(candidate_emb)
            
            # Create batch indices
            query_batch_indices.extend([i] * query_emb.shape[0])
            candidate_batch_indices.extend([i] * candidate_emb.shape[0])
    
    # Concatenate embeddings
    batched_query_emb = torch.cat(query_embs, dim=0) if query_embs else torch.empty(0, device=device)
    batched_candidate_emb = torch.cat(candidate_embs, dim=0) if candidate_embs else torch.empty(0, device=device)
    
    query_emb_batch = torch.tensor(query_batch_indices, device=device, dtype=torch.long)
    candidate_emb_batch = torch.tensor(candidate_batch_indices, device=device, dtype=torch.long)
    
    # Return the same number of values as before for compatibility, but reuse the AA-level embeddings
    return batched_query_emb, batched_candidate_emb, batched_query_emb, batched_candidate_emb, query_emb_batch, candidate_emb_batch


def add_target_labels_batch(batch: Data, df_full: pd.DataFrame, query_emb_batch: torch.Tensor, candidate_emb_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Add target labels to batch for the multi-component loss - handles batch_size > 1."""
    batch_size = len(batch.frag_key)
    all_query_targets = []
    all_candidate_targets = []
    
    for i in range(batch_size):
        # Extract keys from batch
        query_key = batch.frag_key[i]
        candidate_key = batch.seq_key[i]
        
        # Extract UIDs and InterPro IDs
        query_uid = query_key.split('_')[1]
        query_interpro = query_key.split('_')[0]
        candidate_uid = candidate_key.split('_')[1]
        candidate_interpro = candidate_key.split('_')[0]
        
        # Get target labels from dataset
        query_targets = df_full[(df_full['uid'] == query_uid) & (df_full['interpro_id'] == query_interpro)]['label'].values
        candidate_targets = df_full[(df_full['uid'] == candidate_uid) & (df_full['interpro_id'] == candidate_interpro)]['label'].values
        
        if len(query_targets) > 0 and len(candidate_targets) > 0:
            query_targets = torch.tensor(json.loads(query_targets[0]), dtype=torch.float32)
            candidate_targets = torch.tensor(json.loads(candidate_targets[0]), dtype=torch.float32)
            
            # For negative pairs, set targets to 0
            if batch.y[i].item() == 0:  # Check individual sample label
                query_targets = torch.zeros_like(query_targets)
                candidate_targets = torch.zeros_like(candidate_targets)
        else:
            # If no targets found, create zero tensors with appropriate size
            # Use the actual sequence lengths from the batch
            query_len = (query_emb_batch == i).sum().item()
            candidate_len = (candidate_emb_batch == i).sum().item()
            query_targets = torch.zeros(query_len, dtype=torch.float32)
            candidate_targets = torch.zeros(candidate_len, dtype=torch.float32)
        
        all_query_targets.append(query_targets)
        all_candidate_targets.append(candidate_targets)
    
    # Concatenate all targets to match the batched embedding structure
    batched_query_targets = torch.cat(all_query_targets, dim=0)
    batched_candidate_targets = torch.cat(all_candidate_targets, dim=0)
    
    return batched_query_targets, batched_candidate_targets


def count_pos_neg(pairs_list: List[Tuple[Tuple[int, int], int]]) -> Tuple[int, int]:
    """Count positive and negative pairs in a pairs list."""
    pos_count = sum(1 for _, label in pairs_list if label == 1)
    neg_count = sum(1 for _, label in pairs_list if label == 0)
    return pos_count, neg_count