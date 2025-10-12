"""
Protein Function evaluation script using no-learning alignment + backbone embeddings.

This script evaluates no-learning alignment models across all datasets and splits,
using AA-level embeddings and the Alignment model with specified parameters.
"""

import torch
import sys
import pandas as pd
import numpy as np
import json
import random
import time
from pathlib import Path
from torch_geometric.loader import DataLoader
from loguru import logger
import hydra
from omegaconf import DictConfig
from alignment.alignment import Alignment
from utils import (
    convert_pairs_to_dataset,
    get_batch_embeddings,
    add_target_labels_batch,
    count_pos_neg,
    label_match_loss,
    setup_logging
)
from utils.time_utils import TimeTracker

# EBA imports
try:
    from EBA.eba import methods as eba_methods
    from EBA.eba import score_matrices as eba_sm
    EBA_AVAILABLE = True
except ImportError:
    EBA_AVAILABLE = False


def compute_f1_max(scores, targets):
    """Compute F1Max by finding the threshold that maximizes F1 score."""
    from sklearn.metrics import f1_score
    import numpy as np
    
    scores_np = scores.detach().cpu().numpy() if hasattr(scores, 'detach') else scores
    targets_np = targets.detach().cpu().numpy() if hasattr(targets, 'detach') else targets
    
    # Generate threshold range based on score distribution
    thresholds = np.linspace(scores_np.min(), scores_np.max(), 1000)
    f1_scores = []
    
    for threshold in thresholds:
        predictions = (scores_np >= threshold).astype(int)
        if len(np.unique(predictions)) > 1:  # Avoid division by zero
            f1 = f1_score(targets_np, predictions, zero_division=0)
        else:
            f1 = 0.0
        f1_scores.append(f1)
    
    return max(f1_scores)


def create_alignment_model(cfg, device):
    """Create alignment model from config."""
    # Check if EBA alignment is requested (EBA config only has 'score' field)
    is_eba = hasattr(cfg, 'score') and isinstance(cfg.score, str) and cfg.score in ['raw', 'max', 'min']
    if is_eba:
        if not EBA_AVAILABLE:
            raise ImportError("EBA library not available. Please install EBA to use EBA alignment.")
        return None  # EBA doesn't use the Alignment class
    
    # Build eta_kwargs based on configuration
    eta_kwargs = {}
    if cfg.eta.type == 'lrl':
        eta_kwargs['hidden_dim'] = cfg.eta.hidden_dim
    elif cfg.eta.type == 'hinge' and hasattr(cfg.eta, 'normalize'):
        eta_kwargs['normalize'] = cfg.eta.normalize
    
    # Build omega_kwargs based on configuration  
    omega_kwargs = {}
    if cfg.omega.type == 'sinkhorn' and hasattr(cfg.omega, 'temperature'):
        omega_kwargs['temperature'] = cfg.omega.temperature
    
    alignment = Alignment(
        eta=cfg.eta.type, 
        omega=cfg.omega.type, 
        eta_kwargs=eta_kwargs,
        omega_kwargs=omega_kwargs
    ).to(device)
    
    return alignment


def evaluate_alignment_model(data_loader, dataset_name, seq_embeddings, device, df_full, cfg, timer=None):
    """Evaluate alignment model on dataset."""
    from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
    from tqdm import tqdm
    
    # Initialize metrics
    auc = BinaryAUROC()
    pr_auc = BinaryAveragePrecision()
    scores_list = []
    targets_list = []
    
    # Initialize label match score tracking (positive samples only)
    label_match_scores_positive = []
    
    # Create alignment model
    alignment_model = create_alignment_model(cfg, device)
    use_eba = hasattr(cfg, 'score') and isinstance(cfg.score, str) and cfg.score in ['raw', 'max', 'min']
    if not use_eba:
        alignment_model.eval()
    
    skip_count = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Evaluating {dataset_name}"):
            try:
                # Start timing for this batch if timer is provided
                batch_start_time = time.time() if timer is not None else None
                
                batch = batch.to(device)
                
                # Get embeddings on-the-fly from CPU dictionary
                query_emb, candidate_emb, _, _, query_emb_batch, candidate_emb_batch = get_batch_embeddings(batch, seq_embeddings, device)
                
                # Skip if no embeddings found
                if query_emb.numel() == 0 or candidate_emb.numel() == 0:
                    skip_count += 1
                    # Reset batch timing since we're skipping this batch
                    batch_start_time = None
                    continue
                
                # Compute alignment matrix and score
                if use_eba:
                    # Use EBA for alignment
                    similarity_matrix = eba_sm.compute_similarity_matrix(query_emb, candidate_emb)
                    eba_results = eba_methods.compute_eba(similarity_matrix)
                    
                    # Select the appropriate EBA score based on config
                    if cfg.score == 'raw':
                        eba_score = eba_results['EBA_raw']
                    elif cfg.score == 'max':
                        eba_score = eba_results['EBA_max']
                    elif cfg.score == 'min':
                        eba_score = eba_results['EBA_min']
                    else:
                        raise ValueError('Invalid score type: ', cfg.score)
                    
                    sim = torch.tensor([eba_score], device=device)
                    # For EBA, we need to normalize the similarity matrix for label match calculation
                    M = similarity_matrix
                    # Min-max normalization for EBA alignment matrix
                    M_min = M.min()
                    M_max = M.max() 
                    if M_max > M_min:
                        M = (M - M_min) / (M_max - M_min)
                else:
                    # Use standard alignment model
                    from utils.alignment_utils import alignment_score
                    M = alignment_model(query_emb, candidate_emb, query_emb_batch, candidate_emb_batch)
                    sim = alignment_score(query_emb, candidate_emb, M, candidate_emb_batch, threshold=cfg.score.threshold, K=cfg.score.K)
                
                # Update metrics
                auc.update(sim.detach().cpu(), batch.y.cpu())
                pr_auc.update(sim.detach().cpu(), batch.y.cpu())
                
                # Store scores and targets for F1Max calculation
                scores_list.extend(sim.detach().cpu().tolist())
                targets_list.extend(batch.y.cpu().tolist())
                
                # Calculate label match score if target labels are available
                try:
                    # Get target labels for this batch
                    query_targets, candidate_targets = add_target_labels_batch(batch, df_full, query_emb_batch, candidate_emb_batch)
                    query_targets = query_targets.to(device)
                    candidate_targets = candidate_targets.to(device)
                    
                    # Calculate label match loss and convert to score (1 - loss)
                    is_match = batch.y.bool().item() if batch.y.numel() == 1 else True
                    label_match_loss_val = label_match_loss(query_targets, candidate_targets, M, is_match=is_match)
                    label_match_score = 1.0 - label_match_loss_val.item()
                    
                    # Only store label match score for positive classes (since negative always returns 0)
                    if is_match:
                        label_match_scores_positive.append(label_match_score)
                        
                except Exception as e:
                    # Skip label match calculation if target labels are not available or other error
                    pass
                
                # End timing for this batch if timer is provided
                if timer is not None and batch_start_time is not None:
                    batch_end_time = time.time()
                    batch_time = batch_end_time - batch_start_time
                    batch_size = len(batch.y)  # Number of samples in this batch
                    timer.add_sample_time(batch_time, batch_size)
                
                # Memory cleanup
                del M, sim, query_emb, candidate_emb, query_emb_batch, candidate_emb_batch
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                logger.error(f"Error processing batch: {e}")
                skip_count += 1
                continue
    
    # Compute final scores with "no_learning" key structure to match train evaluation format
    results = {
        'metrics': {
            'rocauc': {
                'no_learning': auc.compute().item()
            },
            'pr_auc': {
                'no_learning': pr_auc.compute().item()
            },
            'f1_max': {
                'no_learning': compute_f1_max(torch.tensor(scores_list), torch.tensor(targets_list)) if scores_list else 0.0
            },
            'label_match_score': {
                'no_learning': sum(label_match_scores_positive) / len(label_match_scores_positive) if label_match_scores_positive else 0.0
            }
        },
        'skipped': skip_count
    }
    
    return results




@hydra.main(version_base=None, config_path="configs", config_name="evaluate_pf")
def main(cfg: DictConfig) -> None:
    from hydra.core.hydra_config import HydraConfig
    from datetime import datetime
    import os
    
    # Check if qsub launcher is configured
    if hasattr(cfg, 'launcher') and cfg.launcher is not None:
        # Import launcher utilities  
        import sys  # Re-import to ensure scope
        from utils.launcher_utils import create_launcher
        
        # Get the current job's configuration to build the correct command
        if HydraConfig.initialized():
            hydra_cfg = HydraConfig.get()
            # Get the overrides for this specific job
            job_overrides = hydra_cfg.job.override_dirname.split(',') if hydra_cfg.job.override_dirname else []
            
            # Remove launcher-specific overrides
            filtered_overrides = []
            for override in job_overrides:
                if (not override.startswith("launcher=") and 
                    not override.startswith("hydra/launcher=") and 
                    not override.startswith("launcher.")):
                    filtered_overrides.append(override)
            
            # Get the Hydra output directory for this job
            job_output_dir = Path(hydra_cfg.runtime.output_dir)
            
            # Add hydra output directory override to preserve sweep structure
            hydra_override = f"hydra.run.dir={job_output_dir}"
            filtered_overrides.append(hydra_override)
            
            # Construct the command using the filtered overrides
            script_name = sys.argv[0]
            if filtered_overrides:
                command = f"python {script_name} {' '.join(filtered_overrides)}"
            else:
                command = f"python {script_name}"
        else:
            # Fallback: reconstruct from sys.argv
            original_args = []
            if len(sys.argv) > 1:
                original_args = sys.argv[1:]
            
            # Remove launcher-specific arguments and multirun flag
            filtered_args = []
            for arg in original_args:
                if (not arg.startswith("launcher=") and 
                    not arg.startswith("hydra/launcher=") and 
                    not arg.startswith("launcher.") and
                    arg != "--multirun"):
                    filtered_args.append(arg)
            
            script_name = sys.argv[0]
            command = f"python {script_name} {' '.join(filtered_args)}"
        
        # Create launcher config with Hydra directory info
        from omegaconf import OmegaConf
        launcher_config = OmegaConf.to_container(cfg, resolve=True)
        if HydraConfig.initialized():
            # Add Hydra runtime info to launcher config
            launcher_config['hydra_output_dir'] = str(job_output_dir)
        
        launcher = create_launcher(launcher_config)
        job_id = launcher.submit_job(command, launcher_config)
        
        if job_id:
            print(f"Successfully submitted job: {job_id}")
        else:
            print("Job submission completed (debug mode or script generation)")
        
        return  # Exit without running evaluation locally
    
    # Initialize time tracker
    timer = TimeTracker("evaluation")
    
    # Get run directory from Hydra or create manually
    if HydraConfig.initialized():
        hydra_cfg = HydraConfig.get()
        run_dir = Path(hydra_cfg.runtime.output_dir)
    else:
        # Fallback to manual directory creation following train.py structure
        from utils.run_utils import create_run_directory
        run_dir = create_run_directory(f"{cfg.task}_split{cfg.split}")
    
    # Setup device
    if cfg.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else: 
        device = torch.device(cfg.device)
    
    # Define paths
    plasma_dir = Path(__file__).parent
    data_dir = plasma_dir / "data"
    splits_dir = plasma_dir / "data" / "processed" / cfg.task / f"split_{cfg.split}"
    
    logger.success(f"Task: {cfg.task}")
    logger.success(f"Backbone model: {cfg.backbone_model}")
    logger.success(f"Split index: {cfg.split}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Splits directory: {splits_dir}")
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Device: {device}")
    
    # Check if using EBA (simplified config) or standard alignment
    use_eba = hasattr(cfg, 'score') and isinstance(cfg.score, str) and cfg.score in ['raw', 'max', 'min']
    if use_eba:
        logger.info(f"Using EBA alignment with score type: {cfg.score}")
    else:
        logger.info(f"Eta config: {cfg.eta}")
        logger.info(f"Omega config: {cfg.omega}")
        logger.info(f"Score config: {cfg.score}")
    
    # Setup logging
    setup_logging(run_dir, f"{cfg.task}_split{cfg.split}_pf_evaluation")
    
    # Load original dataset for metadata and get unique UIDs
    logger.success("Loading dataset...")
    df_full = pd.read_csv(plasma_dir / "data" / "raw" / f"{cfg.task}.csv")
    logger.info(f"Loaded dataset with {len(df_full)} rows")
    
    # Get unique UIDs from the dataset for memory-efficient embedding loading
    unique_uids = set(df_full['uid'].unique())
    logger.info(f"Found {len(unique_uids)} unique UIDs in dataset")
    
    # Load embeddings from plasma embeddings directory (only needed ones)
    logger.info("Loading embeddings...")
    embeddings_dir = plasma_dir / "data" / "embeddings" / cfg.backbone_model / "AA_embeddings"
    logger.info(f"Embeddings directory: {embeddings_dir}")
    
    # Load only the AA-level embedding files we need
    seq_embeddings = {}
    if embeddings_dir.exists():
        missing_embeddings = []
        for uid in unique_uids:
            emb_file = embeddings_dir / f"{uid}.pt"
            if emb_file.exists():
                try:
                    embedding = torch.load(emb_file, weights_only=False)
                    # For AA-level embeddings, use uid as key and squeeze if needed
                    if embedding.dim() == 3 and embedding.shape[0] == 1:
                        seq_embeddings[uid] = embedding.squeeze(0)  # Remove batch dimension
                    else:
                        seq_embeddings[uid] = embedding
                except Exception as e:
                    logger.warning(f"Failed to load {emb_file}: {e}")
                    missing_embeddings.append(uid)
            else:
                missing_embeddings.append(uid)
        
        logger.info(f"Successfully loaded AA-level embeddings for {len(seq_embeddings)} entries")
        if missing_embeddings:
            logger.warning(f"Missing embeddings for {len(missing_embeddings)} UIDs: {missing_embeddings[:10]}{'...' if len(missing_embeddings) > 10 else ''}")
    else:
        logger.error(f"Embeddings directory not found: {embeddings_dir}")
        raise FileNotFoundError(f"Embeddings directory not found: {embeddings_dir}")
        
    # Load dataset splits
    logger.success("Loading dataset splits...")
    test_pairs = torch.load(splits_dir / "test.pt", weights_only=False)
    test_hard_pairs = torch.load(splits_dir / "test_hard.pt", weights_only=False)
            
    logger.info(f"Test: {len(test_pairs)} pairs")
    logger.info(f"Test hard: {len(test_hard_pairs)} pairs")
    
    # Count positive/negative in each split
    test_pos, test_neg = count_pos_neg(test_pairs)
    test_hard_pos, test_hard_neg = count_pos_neg(test_hard_pairs)
    
    logger.info(f"Test: {test_pos} pos, {test_neg} neg")
    logger.info(f"Test hard: {test_hard_pos} pos, {test_hard_neg} neg")
    
    # Convert splits to datasets (without embedding tensors)
    logger.success("Converting splits to PyG datasets...")
    test_dataset = convert_pairs_to_dataset(test_pairs, df_full)
    test_hard_dataset = convert_pairs_to_dataset(test_hard_pairs, df_full)
    
    logger.info(f"Dataset sizes after conversion:")
    logger.info(f"  Test: {len(test_dataset)} pairs")
    logger.info(f"  Test hard: {len(test_hard_dataset)} pairs")
    
    # Create data loaders
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)
    test_hard_loader = DataLoader(test_hard_dataset, batch_size=cfg.batch_size, shuffle=False)
    
    logger.success(f"Created data loaders with batch size {cfg.batch_size}")
    
    # Start timing
    timer.start()
            
    # Evaluate test set (frequent InterPro IDs) - track timing per batch
    logger.info("Evaluating test set (frequent InterPro IDs)...")
    test_results = evaluate_alignment_model(test_loader, "Test_Frequent", seq_embeddings, 
                                          device, df_full, cfg, timer)
    
    # Evaluate test hard set (less frequent InterPro IDs) - no timing tracking
    logger.info("Evaluating test hard set (less frequent InterPro IDs)...")
    test_hard_results = evaluate_alignment_model(test_hard_loader, "Test_Hard", seq_embeddings, 
                                               device, df_full, cfg)
    
    # End timing
    timer.end()
            
    # Log results
    logger.info("=== RESULTS ===")
    logger.info("Test Set (Frequent InterPro IDs):")
    logger.info(f"  ALIGNMENT: ROCAUC: {test_results['metrics']['rocauc']['no_learning']:.4f}, F1_max: {test_results['metrics']['f1_max']['no_learning']:.4f}, PR_AUC: {test_results['metrics']['pr_auc']['no_learning']:.4f}")
    logger.info(f"  Skipped batches: {test_results['skipped']}")
    
    logger.info("Test Hard Set (Less Frequent InterPro IDs):")
    logger.info(f"  ALIGNMENT: ROCAUC: {test_hard_results['metrics']['rocauc']['no_learning']:.4f}, F1_max: {test_hard_results['metrics']['f1_max']['no_learning']:.4f}, PR_AUC: {test_hard_results['metrics']['pr_auc']['no_learning']:.4f}")
    logger.info(f"  Skipped batches: {test_hard_results['skipped']}")
            
    # Save config for reproduction
    import json
    from datetime import datetime
    from omegaconf import OmegaConf
    
    config = {
        'model_info': {
            'task': cfg.task,
            'backbone_model': cfg.backbone_model,
            'split': cfg.split,
            'eta_config': OmegaConf.to_container(cfg.eta, resolve=True) if hasattr(cfg, 'eta') else None,
            'omega_config': OmegaConf.to_container(cfg.omega, resolve=True) if hasattr(cfg, 'omega') else None,
            'score_config': OmegaConf.to_container(cfg.score, resolve=True) if type(cfg.score) != str else cfg.score
        },
        'evaluation_args': OmegaConf.to_container(cfg, resolve=True),
        'paths': {
            'splits_dir': str(splits_dir),
            'plasma_dir': str(plasma_dir),
            'embeddings_dir': str(embeddings_dir),
            'run_dir': str(run_dir)
        },
        'dataset_info': {
            'dataset_sizes': {
                'test': len(test_dataset),
                'test_hard': len(test_hard_dataset)
            },
            'split_distribution': {
                'test': {'pos': test_pos, 'neg': test_neg},
                'test_hard': {'pos': test_hard_pos, 'neg': test_hard_neg}
            }
        },
        'results': {
            'test_frequent': test_results,
            'test_hard': test_hard_results
        },
        'created_at': datetime.now().isoformat(),
        'evaluation_status': 'completed'
    }
    
    config_path = run_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Config saved to {config_path}")
    
    # Save results file following same naming convention as train.py
    results_file = run_dir / f"{cfg.task}_split{cfg.split}_pf_evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(config['results'], f, indent=2)
    
    logger.success(f"Saved results to {results_file}")
    
    # Save timing data
    timing_path = run_dir / "inference_time.json"
    timer.save_timing_data(timing_path)
    
    logger.success("=== PROTEIN FUNCTION EVALUATION COMPLETED ===")
    logger.success(f"Results directory: {run_dir}")
    logger.success(f"Results saved at: {results_file}")
    logger.success(f"Config saved at: {config_path}")
    logger.success(f"Timing data saved at: {timing_path}")
    
    return 0


if __name__ == "__main__":
    main()