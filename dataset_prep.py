"""
Dataset preparation script refactored from data_split.ipynb.

This script creates balanced positive/negative pairs from CSV data,
excludes 1/10 of InterPro IDs for test_hard, and splits the remaining
pairs into train/val/test with test_hard having similar size as test.
"""

import pandas as pd
import random
import torch
import numpy as np
from pathlib import Path
from loguru import logger
from typing import List, Tuple, Dict
import hydra
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv
import subprocess
import shutil
import os
from utils.visualization_utils import (
    create_combined_distribution_plot,
    create_sequence_length_analysis
)

# Get the absolute path to the plasma directory
load_dotenv(Path(__file__).parent / ".env")


def merge_local_csvs(task: str, plasma_path: Path) -> str:
    """
    Merge all CSV files in data/raw folder (except full.csv) when task is 'full'.
    
    Args:
        task: Name of the task (should be 'full')
        plasma_path: Path to the plasma directory
        
    Returns:
        Path to the merged CSV file
    """
    data_path = plasma_path / "data"
    raw_path = data_path / "raw"
    
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_path}")
    
    merged_csv_path = raw_path / "full.csv"
    
    try:
        logger.info(f"Merging all CSV files in {raw_path} to create full.csv")
        
        # Find all CSV files in raw directory, excluding full.csv
        csv_files = [f for f in raw_path.glob("*.csv") if f.name != "full.csv"]
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {raw_path} (excluding full.csv)")
        
        logger.info(f"Found {len(csv_files)} CSV files to merge: {[f.name for f in csv_files]}")
        
        # Read and merge all CSV files
        dataframes = []
        for csv_file in csv_files:
            logger.info(f"Reading {csv_file.name}")
            df = pd.read_csv(csv_file)
            logger.info(f"  - Shape: {df.shape}")
            dataframes.append(df)
        
        # Concatenate all dataframes
        merged_df = pd.concat(dataframes, ignore_index=True)
        logger.info(f"Merged dataset shape: {merged_df.shape}")
        
        # Save merged dataset
        merged_df.to_csv(merged_csv_path, index=False)
        logger.success(f"Merged dataset saved to {merged_csv_path}")
        
        return str(merged_csv_path)
        
    except Exception as e:
        logger.error(f"Local CSV merge failed: {e}")
        raise


def download_and_merge_dataset(dataset_name: str, download_url: str, plasma_path: Path) -> str:
    """
    Download dataset from HuggingFace and merge CSV files.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'motif')
        download_url: HuggingFace URL to clone
        plasma_path: Path to the plasma directory
        
    Returns:
        Path to the merged CSV file
    """
    data_path = plasma_path / "data"
    raw_path = data_path / "raw"
    tmp_path = data_path / "_tmp"
    
    # Create directories if they don't exist
    raw_path.mkdir(parents=True, exist_ok=True)
    tmp_path.mkdir(parents=True, exist_ok=True)
    
    merged_csv_path = raw_path / f"{dataset_name}.csv"
    
    try:
        logger.info(f"Downloading dataset {dataset_name} from {download_url}")
        
        # Clone the repository into _tmp folder
        repo_name = download_url.split("/")[-1]
        clone_path = tmp_path / repo_name
        
        # Remove existing clone if it exists
        if clone_path.exists():
            shutil.rmtree(clone_path)
        
        # Clone the repository
        subprocess.run(
            ["git", "clone", download_url, str(clone_path)],
            check=True,
            capture_output=True,
            text=True
        )
        
        # Find all CSV files in the cloned repository
        csv_files = list(clone_path.glob("*.csv"))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {clone_path}")
        
        logger.info(f"Found {len(csv_files)} CSV files: {[f.name for f in csv_files]}")
        
        # Read and merge all CSV files
        dataframes = []
        for csv_file in csv_files:
            logger.info(f"Reading {csv_file.name}")
            df = pd.read_csv(csv_file)
            logger.info(f"  - Shape: {df.shape}")
            dataframes.append(df)
        
        # Concatenate all dataframes
        merged_df = pd.concat(dataframes, ignore_index=True)
        logger.info(f"Merged dataset shape: {merged_df.shape}")
        
        # Save merged dataset
        merged_df.to_csv(merged_csv_path, index=False)
        logger.success(f"Merged dataset saved to {merged_csv_path}")
        
        # Clean up temporary directory
        shutil.rmtree(tmp_path)
        logger.info("Temporary directory cleaned up")
        
        return str(merged_csv_path)
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Git clone failed: {e}")
        logger.error(f"Command output: {e.stdout}")
        logger.error(f"Command error: {e.stderr}")
        raise
    except Exception as e:
        logger.error(f"Dataset download and merge failed: {e}")
        raise
    finally:
        # Always clean up temporary directory
        if tmp_path.exists():
            shutil.rmtree(tmp_path)


def setup_logging(output_dir: Path):
    """Setup logging configuration."""
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # Add file logging
    log_file = output_dir / "data_split.log"
    logger.add(log_file, rotation="10 MB", level="INFO")
    logger.info("Starting data split process")


def filter_and_analyze_data(data_path: str, max_sequence_length: int = 1024) -> pd.DataFrame:
    """Load and filter the dataset, removing sequences longer than max_sequence_length."""
    logger.info(f"Loading dataset from {data_path}")
    data = pd.read_csv(data_path)
    logger.info(f"Original dataset size: {len(data)}")
    
    # Filter out sequences with length > max_sequence_length (if limit is set)
    if max_sequence_length > 0:
        logger.info(f"Filtering out sequences with length > {max_sequence_length}...")
        data['seq_length'] = data['seq_full'].str.len()
        data_filtered = data[data['seq_length'] <= max_sequence_length].copy()
        
        logger.info(f"After filtering (seq_length <= {max_sequence_length}): {len(data_filtered)}")
        logger.info(f"Removed {len(data) - len(data_filtered)} sequences")
    else:
        logger.info("No sequence length filtering applied (max_sequence_length = -1)")
        data['seq_length'] = data['seq_full'].str.len()
        data_filtered = data.copy()
    
    # Analyze InterPro ID frequencies
    interpro_counts = data_filtered['interpro_id'].value_counts()
    logger.info(f"Total InterPro IDs after filtering: {len(interpro_counts)}")
    
    return data_filtered


def exclude_interpro_ids_for_test_hard(data: pd.DataFrame, exclusion_ratio: float = 0.1) -> Tuple[pd.DataFrame, List[str]]:
    """
    Exclude 1/10 of InterPro IDs for test_hard dataset.
    These excluded IDs will be used only for test_hard.
    """
    interpro_counts = data['interpro_id'].value_counts()
    frequent_interpro_ids = interpro_counts.index.tolist()

    # Exclude 1/10 of frequent InterPro IDs for test_hard
    num_to_exclude = max(1, int(len(frequent_interpro_ids) * exclusion_ratio))
    random.shuffle(frequent_interpro_ids)
    
    excluded_interpro_ids = frequent_interpro_ids[:num_to_exclude]
    remaining_interpro_ids = frequent_interpro_ids[num_to_exclude:]
    
    logger.info(f"Selected InterPro IDs: {len(frequent_interpro_ids)}")
    logger.info(f"Excluded for test_hard: {num_to_exclude} InterPro IDs")
    logger.info(f"Remaining for train/val/test: {len(remaining_interpro_ids)} InterPro IDs")
    logger.info(f"Excluded InterPro IDs: {excluded_interpro_ids}")
    
    # Create datasets
    main_data = data[data['interpro_id'].isin(remaining_interpro_ids)].copy()
    test_hard_data = data[data['interpro_id'].isin(excluded_interpro_ids)].copy()
    
    logger.info(f"Main dataset size: {len(main_data)}")
    logger.info(f"Test hard dataset size: {len(test_hard_data)}")
    
    return main_data, excluded_interpro_ids


def get_interpro_rows(data: pd.DataFrame) -> Dict[str, List[int]]:
    """Group rows by InterPro ID for efficient sampling."""
    interpro_counts = data['interpro_id'].value_counts()
    frequent_interpro_ids = interpro_counts.index
    
    interpro_rows = {}
    for interpro_id in frequent_interpro_ids:
        rows_with_id = data[data['interpro_id'] == interpro_id]
        if len(rows_with_id) >= 2:
            interpro_rows[interpro_id] = rows_with_id.index.tolist()
    
    logger.info(f"InterPro IDs with ≥2 rows available for sampling: {len(interpro_rows)}")
    return interpro_rows


def sample_positive_pairs_exact(interpro_rows: Dict[str, List[int]], data: pd.DataFrame, target_pairs: int) -> List[Tuple[int, int]]:
    """Sample exactly target_pairs positive pairs with uniform distribution across InterPro IDs."""
    interpro_list = list(interpro_rows.keys())
    num_interpros = len(interpro_list)
    
    if num_interpros == 0:
        logger.warning("No InterPro IDs available for positive pair sampling")
        return []
    
    # Calculate exactly how many pairs each InterPro ID should have
    pairs_per_interpro = target_pairs // num_interpros
    remaining_pairs = target_pairs % num_interpros
    
    logger.info(f"Target positive pairs: {target_pairs}")
    logger.info(f"Pairs per InterPro ID: {pairs_per_interpro}")
    logger.info(f"Remaining pairs to distribute: {remaining_pairs}")
    
    sampled_pairs = []
    
    # First pass: try to get the base amount for each InterPro ID
    for i, interpro_id in enumerate(interpro_list):
        pairs_needed = pairs_per_interpro
        if i < remaining_pairs:  # Distribute remaining pairs to first few InterPro IDs
            pairs_needed += 1
        
        rows_list = interpro_rows[interpro_id]
        pairs_sampled = 0
        attempts = 0
        max_attempts = pairs_needed * 50
        sampled_pairs_for_interpro = set()
        
        while pairs_sampled < pairs_needed and attempts < max_attempts:
            # Randomly select two different indices
            if len(rows_list) >= 2:
                idx1, idx2 = random.sample(rows_list, 2)
                
                # Check if UIDs are different
                uid1 = data.loc[idx1, 'uid']
                uid2 = data.loc[idx2, 'uid']
                
                if uid1 != uid2:
                    pair = (idx1, idx2) if idx1 < idx2 else (idx2, idx1)  # Consistent ordering
                    if pair not in sampled_pairs_for_interpro:
                        sampled_pairs_for_interpro.add(pair)
                        pairs_sampled += 1
            
            attempts += 1
        
        sampled_pairs.extend(list(sampled_pairs_for_interpro))
    
    # Second pass: if we don't have exactly target_pairs, adjust
    current_count = len(sampled_pairs)
    logger.info(f"After first pass: {current_count} pairs")
    
    if current_count < target_pairs:
        # Need more pairs - sample from any InterPro ID
        needed = target_pairs - current_count
        logger.info(f"Need {needed} more pairs")
        
        sampled_pairs_set = set(sampled_pairs)
        attempts = 0
        max_attempts = needed * 100
        
        while len(sampled_pairs) < target_pairs and attempts < max_attempts:
            # Randomly select an InterPro ID
            interpro_id = random.choice(interpro_list)
            rows_list = interpro_rows[interpro_id]
            
            if len(rows_list) >= 2:
                idx1, idx2 = random.sample(rows_list, 2)
                
                # Check if UIDs are different
                uid1 = data.loc[idx1, 'uid']
                uid2 = data.loc[idx2, 'uid']
                
                if uid1 != uid2:
                    pair = (idx1, idx2) if idx1 < idx2 else (idx2, idx1)
                    if pair not in sampled_pairs_set:
                        sampled_pairs.append(pair)
                        sampled_pairs_set.add(pair)
            
            attempts += 1
    
    elif current_count > target_pairs:
        # Too many pairs - randomly remove excess
        excess = current_count - target_pairs
        logger.info(f"Removing {excess} excess pairs")
        sampled_pairs = random.sample(sampled_pairs, target_pairs)
    
    return sampled_pairs


def sample_negative_pairs_exact(interpro_rows: Dict[str, List[int]], data: pd.DataFrame, target_pairs: int) -> List[Tuple[int, int]]:
    """Sample exactly target_pairs negative pairs with uniform distribution across InterPro IDs."""
    interpro_list = list(interpro_rows.keys())
    num_interpros = len(interpro_list)
    
    if num_interpros < 2:
        logger.warning("Need at least 2 InterPro IDs for negative pair sampling")
        return []
    
    # Calculate how many times each InterPro should appear in first position
    pairs_per_interpro_pos = target_pairs // num_interpros
    remaining_pairs = target_pairs % num_interpros
    
    logger.info(f"Target negative pairs: {target_pairs}")
    logger.info(f"Pairs per InterPro ID per position: {pairs_per_interpro_pos}")
    logger.info(f"Remaining pairs to distribute: {remaining_pairs}")
    
    sampled_pairs = []
    
    # First pass: try to get the base amount for each InterPro ID in first position
    for i, interpro_id1 in enumerate(interpro_list):
        pairs_needed = pairs_per_interpro_pos
        if i < remaining_pairs:
            pairs_needed += 1
        
        pairs_sampled = 0
        attempts = 0
        max_attempts = pairs_needed * 50
        sampled_pairs_for_interpro = set()
        
        while pairs_sampled < pairs_needed and attempts < max_attempts:
            # Choose a different InterPro ID for second position
            other_interpros = [iid for iid in interpro_list if iid != interpro_id1]
            interpro_id2 = random.choice(other_interpros)
            
            # Sample one row from each InterPro ID
            idx1 = random.choice(interpro_rows[interpro_id1])
            idx2 = random.choice(interpro_rows[interpro_id2])
            
            # Check if UIDs are different
            uid1 = data.loc[idx1, 'uid']
            uid2 = data.loc[idx2, 'uid']
            
            if uid1 != uid2:
                pair = (idx1, idx2)
                if pair not in sampled_pairs_for_interpro:
                    sampled_pairs_for_interpro.add(pair)
                    pairs_sampled += 1
            
            attempts += 1
        
        sampled_pairs.extend(list(sampled_pairs_for_interpro))
    
    # Second pass: if we don't have exactly target_pairs, adjust
    current_count = len(sampled_pairs)
    logger.info(f"After first pass: {current_count} pairs")
    
    if current_count < target_pairs:
        # Need more pairs
        needed = target_pairs - current_count
        logger.info(f"Need {needed} more pairs")
        
        sampled_pairs_set = set(sampled_pairs)
        attempts = 0
        max_attempts = needed * 100
        
        while len(sampled_pairs) < target_pairs and attempts < max_attempts:
            # Randomly select two different InterPro IDs
            interpro_id1, interpro_id2 = random.sample(interpro_list, 2)
            
            # Sample one row from each InterPro ID
            idx1 = random.choice(interpro_rows[interpro_id1])
            idx2 = random.choice(interpro_rows[interpro_id2])
            
            # Check if UIDs are different
            uid1 = data.loc[idx1, 'uid']
            uid2 = data.loc[idx2, 'uid']
            
            if uid1 != uid2:
                pair = (idx1, idx2)
                if pair not in sampled_pairs_set:
                    sampled_pairs.append(pair)
                    sampled_pairs_set.add(pair)
            
            attempts += 1
    
    elif current_count > target_pairs:
        # Too many pairs - randomly remove excess
        excess = current_count - target_pairs
        logger.info(f"Removing {excess} excess pairs")
        sampled_pairs = random.sample(sampled_pairs, target_pairs)
    
    return sampled_pairs


def split_pairs(all_pairs: List[Tuple[Tuple[int, int], int]], train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple[List, List, List]:
    """Split pairs into train/validation/test sets."""
    # Shuffle the pairs
    random.shuffle(all_pairs)
    
    # Calculate split sizes
    total = len(all_pairs)
    train_size = int(train_ratio * total)
    val_size = int(val_ratio * total)
    test_size = total - train_size - val_size
    
    logger.info(f"Split sizes - Train: {train_size}, Validation: {val_size}, Test: {test_size}")
    
    # Split the data
    train_pairs = all_pairs[:train_size]
    val_pairs = all_pairs[train_size:train_size + val_size]
    test_pairs = all_pairs[train_size + val_size:]
    
    return train_pairs, val_pairs, test_pairs


def count_pos_neg(pairs_list: List[Tuple[Tuple[int, int], int]]) -> Tuple[int, int]:
    """Count positive and negative pairs in a pairs list."""
    pos_count = sum(1 for _, label in pairs_list if label == 1)
    neg_count = sum(1 for _, label in pairs_list if label == 0)
    return pos_count, neg_count




def create_balanced_dataset(cfg: DictConfig, seed: int) -> Dict:
    """
    Create a balanced dataset following the data_split.ipynb logic.
    
    Args:
        cfg: Hydra configuration object
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with dataset statistics
    """
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Validate inputs
    if cfg.total_samples % 2 != 0:
        raise ValueError("total_samples must be even (half positive, half negative)")
    
    if abs(cfg.train_ratio + cfg.val_ratio + (1 - cfg.train_ratio - cfg.val_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    
    # Setup output directory and logging
    output_path = Path(cfg.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    setup_logging(output_path)
    
    logger.success(f"Dataset preparation started")
    logger.info(f"Data CSV: {cfg.data_csv_path}")
    logger.info(f"Total samples: {cfg.total_samples}")
    logger.info(f"Output directory: {cfg.output_dir}")
    logger.info(f"Seed: {seed}")
    
    # Load and filter data
    data = filter_and_analyze_data(cfg.data_csv_path, cfg.max_sequence_length)
    
    # Exclude InterPro IDs for test_hard
    main_data, excluded_interpro_ids = exclude_interpro_ids_for_test_hard(data, cfg.exclusion_ratio)
    test_hard_data = data[data['interpro_id'].isin(excluded_interpro_ids)].copy()
    
    # Get InterPro rows for main dataset
    main_interpro_rows = get_interpro_rows(main_data)
    
    # Calculate target pairs (half positive, half negative for main dataset)
    target_pairs_per_type = cfg.total_samples // 2
    
    # Sample positive and negative pairs from main dataset
    logger.info("=== SAMPLING FROM MAIN DATASET (for train/val/test) ===")
    sampled_positive_pairs = sample_positive_pairs_exact(main_interpro_rows, main_data, target_pairs_per_type)
    sampled_negative_pairs = sample_negative_pairs_exact(main_interpro_rows, main_data, target_pairs_per_type)
    
    logger.info(f"Sampled positive pairs: {len(sampled_positive_pairs)}")
    logger.info(f"Sampled negative pairs: {len(sampled_negative_pairs)}")
    
    # Combine and split main dataset pairs
    all_main_pairs = [(pair, 1) for pair in sampled_positive_pairs] + [(pair, 0) for pair in sampled_negative_pairs]
    train_pairs, val_pairs, test_pairs = split_pairs(all_main_pairs, cfg.train_ratio, cfg.val_ratio)
    
    # Create test_hard dataset with same size as test
    target_test_hard_size = len(test_pairs)
    target_test_hard_pairs_per_type = target_test_hard_size // 2
    
    logger.info(f"=== SAMPLING FOR TEST_HARD ===")
    logger.info(f"Target test_hard size: {target_test_hard_size}")
    
    # Get InterPro rows for test_hard dataset
    test_hard_interpro_rows = get_interpro_rows(test_hard_data)
    
    # Sample from excluded InterPro IDs for test_hard
    test_hard_positive_pairs = sample_positive_pairs_exact(test_hard_interpro_rows, test_hard_data, target_test_hard_pairs_per_type)
    test_hard_negative_pairs = sample_negative_pairs_exact(test_hard_interpro_rows, test_hard_data, target_test_hard_pairs_per_type)
    
    # Create test_hard pairs
    test_hard_pairs = [(pair, 1) for pair in test_hard_positive_pairs] + [(pair, 0) for pair in test_hard_negative_pairs]
    
    logger.info(f"Test hard positive pairs: {len(test_hard_positive_pairs)}")
    logger.info(f"Test hard negative pairs: {len(test_hard_negative_pairs)}")
    logger.info(f"Total test hard pairs: {len(test_hard_pairs)}")
    
    # Save splits
    logger.info("Saving data splits...")
    torch.save(train_pairs, output_path / "train.pt")
    torch.save(val_pairs, output_path / "validation.pt")
    torch.save(test_pairs, output_path / "test.pt")
    torch.save(test_hard_pairs, output_path / "test_hard.pt")
    
    # Count positive/negative in each split
    train_pos, train_neg = count_pos_neg(train_pairs)
    val_pos, val_neg = count_pos_neg(val_pairs)
    test_pos, test_neg = count_pos_neg(test_pairs)
    test_hard_pos, test_hard_neg = count_pos_neg(test_hard_pairs)
    
    logger.info(f"=== FINAL SPLIT STATISTICS ===")
    logger.info(f"Train: {len(train_pairs)} pairs (pos: {train_pos}, neg: {train_neg})")
    logger.info(f"Validation: {len(val_pairs)} pairs (pos: {val_pos}, neg: {val_neg})")
    logger.info(f"Test: {len(test_pairs)} pairs (pos: {test_pos}, neg: {test_neg})")
    logger.info(f"Test hard: {len(test_hard_pairs)} pairs (pos: {test_hard_pos}, neg: {test_hard_neg})")
    
    # Create metadata
    metadata = {
        'total_samples': int(cfg.total_samples),
        'dataset_sizes': {
            'original': len(data),
            'filtered': len(data),
            'main': len(main_data),
            'test_hard_pool': len(test_hard_data)
        },
        'split_sizes': {
            'train': len(train_pairs),
            'validation': len(val_pairs),
            'test': len(test_pairs),
            'test_hard': len(test_hard_pairs)
        },
        'split_distribution': {
            'train': {'pos': train_pos, 'neg': train_neg},
            'validation': {'pos': val_pos, 'neg': val_neg},
            'test': {'pos': test_pos, 'neg': test_neg},
            'test_hard': {'pos': test_hard_pos, 'neg': test_hard_neg}
        },
        'interpro_info': {
            'total_interpro_ids': data['interpro_id'].nunique(),
            'main_interpro_ids': len(main_interpro_rows),
            'excluded_interpro_ids': len(excluded_interpro_ids),
            'excluded_ids_list': excluded_interpro_ids
        },
        'parameters': {
            'train_ratio': float(cfg.train_ratio),
            'val_ratio': float(cfg.val_ratio),
            'test_ratio': float(1 - cfg.train_ratio - cfg.val_ratio),
            'exclusion_ratio': float(cfg.exclusion_ratio),
            'max_sequence_length': int(cfg.max_sequence_length),
            'seed': seed
        }
    }
    
    # Save metadata
    import json
    with open(output_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create plots
    logger.info("Creating visualization plots...")
    
    # 1. Combined distribution plot (InterPro + positive/negative pairs)
    create_combined_distribution_plot(data, sampled_positive_pairs, sampled_negative_pairs, output_path)
    
    # 2. Sequence length analysis with KDE
    create_sequence_length_analysis(data, train_pairs, val_pairs, test_pairs, test_hard_pairs, output_path)
    
    logger.success("Dataset preparation and visualization completed successfully!")
    
    return metadata


@hydra.main(version_base=None, config_path="configs", config_name="dataset_prep")
def main(cfg: DictConfig) -> None:
    from hydra.core.hydra_config import HydraConfig
    
    # Get run directory from Hydra
    if HydraConfig.initialized():
        hydra_cfg = HydraConfig.get()
        run_dir = Path(hydra_cfg.runtime.output_dir)
    else:
        # Fallback for non-Hydra execution
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plasma_path = Path(__file__).parent
        run_dir = plasma_path / "runs" / f"dataset_prep_{cfg.task}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        all_metadata = []
        
        # Setup config paths (runtime-inferable)
        plasma_path = Path(__file__).parent
        cfg.data_csv_path = str(plasma_path / "data" / "raw" / f"{cfg.task}.csv") if cfg.data_csv_path is None else cfg.data_csv_path
        cfg.output_dir = str(plasma_path / "data" / "processed") if cfg.output_dir is None else cfg.output_dir
        
        # Special handling for "full" task - merge all CSVs in data/raw except full.csv
        if cfg.task == "full":
            logger.info("Task is 'full' - merging all CSV files in data/raw folder")
            cfg.data_csv_path = merge_local_csvs(cfg.task, plasma_path)
            logger.success(f"Local CSV files merged successfully: {cfg.data_csv_path}")
        else:
            # Check if CSV file exists, if not download and merge it
            csv_path = Path(cfg.data_csv_path)
            if not csv_path.exists():
                logger.info(f"CSV file not found at {cfg.data_csv_path}")
                
                # Check if we have a download link for this dataset
                if hasattr(cfg, 'links') and cfg.task in cfg.links:
                    download_url = cfg.links[cfg.task]
                    logger.info(f"Found download link for {cfg.task}: {download_url}")
                    
                    # Download and merge dataset
                    cfg.data_csv_path = download_and_merge_dataset(
                        cfg.task, 
                        download_url, 
                        plasma_path
                    )
                    logger.success(f"Dataset downloaded and merged successfully: {cfg.data_csv_path}")
                else:
                    raise FileNotFoundError(
                        f"CSV file not found at {cfg.data_csv_path} and no download link "
                        f"available for dataset '{cfg.task}' in config.links"
                    )
            else:
                logger.info(f"Using existing CSV file: {cfg.data_csv_path}")

        for i, seed in enumerate(cfg.seeds):
            logger.info(f"\n{'='*60}")
            logger.success(f"Creating split {i} with seed {seed}")
            logger.info(f"{'='*60}")
            
            # Create split-specific output directory
            split_output_dir = f"{cfg.output_dir}/{cfg.task}/split_{i}"
            
            # Create a copy of config for this split
            split_cfg = OmegaConf.create(OmegaConf.to_yaml(cfg))
            split_cfg.output_dir = split_output_dir
            
            metadata = create_balanced_dataset(split_cfg, seed)
            
            metadata['split_index'] = i
            all_metadata.append(metadata)
            
            print(f"\n=== Split {i} Creation Summary (seed={seed}) ===")
            print(f"Total samples: {metadata['total_samples']}")
            print(f"Dataset sizes:")
            for split, size in metadata['split_sizes'].items():
                dist = metadata['split_distribution'][split]
                print(f"  {split.capitalize()}: {size} (pos: {dist['pos']}, neg: {dist['neg']})")
            print(f"InterPro IDs - Main: {metadata['interpro_info']['main_interpro_ids']}, "
                  f"Excluded: {metadata['interpro_info']['excluded_interpro_ids']}")
            print(f"Output directory: {split_output_dir}")
        
        # Save combined metadata
        combined_metadata = {
            'total_splits': len(cfg.seeds),
            'seeds_used': str(cfg.seeds),
            'splits_metadata': all_metadata,
            'common_parameters': {
            'task': str(cfg.task),
            'total_samples': int(cfg.total_samples),
            'train_ratio': float(cfg.train_ratio),
            'val_ratio': float(cfg.val_ratio),
            'exclusion_ratio': float(cfg.exclusion_ratio),
            'max_sequence_length': int(cfg.max_sequence_length)
            }
        }
        
        import json
        combined_metadata_path = Path(cfg.output_dir) / cfg.task / "combined_metadata.json"
        with open(combined_metadata_path, 'w') as f:
            json.dump(combined_metadata, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"All {len(cfg.seeds)} splits created successfully!")
        print(f"Seeds used: {cfg.seeds}")
        print(f"Combined metadata saved to: {combined_metadata_path}")
        print(f"{'='*60}")
        
    except Exception as e:
        logger.error(f"Dataset creation failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    main()