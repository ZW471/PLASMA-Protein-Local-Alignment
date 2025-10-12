"""
Protein embedding generation script using configurable backbone models.

This script generates protein embeddings using various backbone models,
with support for both amino acid (residue) level and protein level embeddings.
"""

import os
import sys
from pathlib import Path
import importlib
import pandas as pd
import torch
from typing import Dict, Any, List
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
import logging

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_device(device_config: str) -> str:
    """Get appropriate device based on configuration."""
    if device_config == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_config


def load_model_class(target_path: str):
    """Dynamically load model class from target path."""
    module_path, class_name = target_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def load_and_deduplicate_data(data_path: str, deduplicate_column: str) -> pd.DataFrame:
    """Load full.csv and deduplicate by specified column."""
    csv_path = Path(data_path)
    
    logger.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    
    logger.info(f"Total records before deduplication: {len(df)}")
    df_deduplicated = df.drop_duplicates(subset=[deduplicate_column])
    logger.info(f"Total records after deduplication: {len(df_deduplicated)}")
    logger.info(f"Removed {len(df) - len(df_deduplicated)} duplicate records")
    
    return df_deduplicated


def get_pdb_path(uid: str, pdb_base_dir: str) -> str:
    """Get PDB file path for a given UID."""
    pdb_path = Path(pdb_base_dir) / f"{uid}.pdb"
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")
    return str(pdb_path)


def filter_sequences_by_length(df: pd.DataFrame, max_length: int) -> pd.DataFrame:
    """Filter sequences by maximum length."""
    sequence_col = 'seq_full'
    
    logger.info(f"Filtering sequences longer than {max_length} residues")
    df['sequence_length'] = df[sequence_col].str.len()
    df_filtered = df[df['sequence_length'] <= max_length].copy()
    df_filtered = df_filtered.drop('sequence_length', axis=1)
    
    logger.info(f"Filtered out {len(df) - len(df_filtered)} sequences")
    return df_filtered


def get_existing_embeddings(output_dirs: List[Path]) -> set:
    """Check which embeddings already exist in all output directories."""
    existing_files = []
    
    for output_dir in output_dirs:
        if output_dir.exists():
            files_in_dir = {
                f.stem for f in output_dir.glob("*.pt")
            }
            existing_files.append(files_in_dir)
        else:
            existing_files.append(set())
    
    # Only include files that exist in ALL directories
    if existing_files:
        return set.intersection(*existing_files)
    return set()


def create_output_directories(base_path: Path, model_names: List[str], levels: List[str]) -> Dict[str, Path]:
    """Create output directories for embeddings."""
    output_dirs = {}
    
    for model_name in model_names:
        for level in levels:
            dir_key = f"{model_name}_{level}"
            dir_path = base_path / f"{model_name}" / f"{level}_embeddings"
            dir_path.mkdir(parents=True, exist_ok=True)
            output_dirs[dir_key] = dir_path
            logger.info(f"Created output directory: {dir_path}")
    
    return output_dirs


def save_embeddings(embeddings: Dict[str, Dict[str, torch.Tensor]], 
                   uid: str, 
                   output_dirs: Dict[str, Path]):
    """Save embeddings to appropriate directories."""
    for model_name, model_embeddings in embeddings.items():
        for level, embedding_tensor in model_embeddings.items():
            dir_key = f"{model_name}_{level}"
            output_path = output_dirs[dir_key] / f"{uid}.pt"
            torch.save(embedding_tensor, output_path)


@hydra.main(version_base=None, config_path="configs", config_name="embed")
def main(cfg: DictConfig) -> None:
    """Main embedding generation function."""
    logger.info("Starting embedding generation")
    logger.info(f"Configuration: {cfg}")
    
    # Set device
    device = get_device(cfg.device)
    logger.info(f"Using device: {device}")
    
    # Load and process data
    df = load_and_deduplicate_data(cfg.data_path, cfg.deduplicate_column)
    df_filtered = filter_sequences_by_length(df, cfg.max_sequence_length)
    
    # Set up backbone configuration
    backbone_cfg = cfg.backbone
    backbone_cfg.device = device
    
    # Load model class and instantiate
    logger.info(f"Loading model: {cfg.model}")
    model_class = load_model_class(backbone_cfg.target)
    model = model_class(backbone_cfg)
    
    # Create output directories
    base_output_path = Path(cfg.output_base_path)
    model_names = backbone_cfg.embedding_types
    levels = backbone_cfg.levels
    output_dirs = create_output_directories(base_output_path, model_names, levels)
    
    # Check existing embeddings
    existing_embeddings = get_existing_embeddings(list(output_dirs.values()))
    logger.info(f"Found {len(existing_embeddings)} existing complete embeddings")
    
    # Check if model needs PDB files
    need_pdb = backbone_cfg.get('need_pdb', False)
    pdb_base_dir = cfg.get('pdb_base_dir', 'data/pdb/raw') if need_pdb else None
    
    # Prepare sequences to process
    sequences_to_process = []
    for _, row in df_filtered.iterrows():
        uid = row[cfg.deduplicate_column]
        if cfg.overwrite_existing or uid not in existing_embeddings:
            if need_pdb:
                # Check if PDB file exists
                try:
                    pdb_path = get_pdb_path(uid, pdb_base_dir)
                    sequences_to_process.append({
                        'uid': uid,
                        'sequence': row['seq_full'],
                        'pdb_path': pdb_path
                    })
                except FileNotFoundError:
                    logger.warning(f"PDB file not found for UID {uid}, skipping")
                    continue
            else:
                sequences_to_process.append({
                    'uid': uid,
                    'sequence': row['seq_full']
                })
    
    logger.info(f"Will process {len(sequences_to_process)} sequences")
    
    # Process sequences
    with torch.no_grad():
        for seq_info in tqdm(sequences_to_process, desc="Generating embeddings"):
            uid = seq_info['uid']
            
            if need_pdb:
                # For PDB-based models, pass the PDB path
                pdb_path = seq_info['pdb_path']
                embeddings = model.forward([pdb_path])
            else:
                # For sequence-based models, pass the sequence
                sequence = seq_info['sequence']
                embeddings = model.forward([sequence])
            
            # Save embeddings
            save_embeddings(embeddings, uid, output_dirs)
            
            # Clean up memory
            del embeddings
            torch.cuda.empty_cache()
    
    logger.info("Embedding generation completed successfully")


if __name__ == "__main__":
    main()