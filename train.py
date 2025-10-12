"""
Training script for alignment model using dataset splits.
"""

import torch
import sys
import pandas as pd
from pathlib import Path
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from torch_geometric.loader import DataLoader
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from tqdm import tqdm
from loguru import logger
import hydra
from omegaconf import DictConfig, OmegaConf
import random
import time

from alignment.alignment import Alignment
from utils import (
    convert_pairs_to_dataset,
    get_batch_embeddings,
    add_target_labels_batch,
    count_pos_neg,
    alignment_score,
    label_match_loss,
    create_run_directory,
    setup_logging
)
from utils.time_utils import TimeTracker

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


@hydra.main(version_base=None, config_path="configs", config_name="train")
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
        
        return  # Exit without running training locally
    
    # Get run directory from Hydra or create manually
    if HydraConfig.initialized():
        hydra_cfg = HydraConfig.get()
        run_dir = Path(hydra_cfg.runtime.output_dir)
    else:
        # Fallback to manual directory creation
        from utils.run_utils import create_run_directory
        run_dir = create_run_directory(f"{cfg.task}_split{cfg.split}")
    
    # Set random seeds
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    torch.cuda.empty_cache()
    
    # Setup device
    if cfg.device == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(cfg.device)
    
    # Setup logging
    setup_logging(run_dir, f"{cfg.task}_split{cfg.split}")
    
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
    train_pairs = torch.load(splits_dir / "train.pt", weights_only=False)
    val_pairs = torch.load(splits_dir / "validation.pt", weights_only=False)
    test_pairs = torch.load(splits_dir / "test.pt", weights_only=False)
    test_hard_pairs = torch.load(splits_dir / "test_hard.pt", weights_only=False)
    
    logger.info(f"Train: {len(train_pairs)} pairs")
    logger.info(f"Validation: {len(val_pairs)} pairs")
    logger.info(f"Test: {len(test_pairs)} pairs")
    logger.info(f"Test hard: {len(test_hard_pairs)} pairs")
    
    # Sample training dataset if dataset_fraction < 1.0
    if cfg.dataset_fraction < 1.0:
        original_train_size = len(train_pairs)
        target_train_size = int(original_train_size * cfg.dataset_fraction)
        
        # Use random seed for reproducible sampling
        random.seed(cfg.seed)
        train_pairs = random.sample(train_pairs, target_train_size)
        
        logger.info(f"Sampled training dataset from {original_train_size} to {len(train_pairs)} pairs (fraction: {cfg.dataset_fraction})")
    else:
        logger.info(f"Using full training dataset (dataset_fraction: {cfg.dataset_fraction})")
    
    # Count positive/negative in each split
    train_pos, train_neg = count_pos_neg(train_pairs)
    val_pos, val_neg = count_pos_neg(val_pairs)
    test_pos, test_neg = count_pos_neg(test_pairs)
    test_hard_pos, test_hard_neg = count_pos_neg(test_hard_pairs)
    
    logger.info(f"Train: {train_pos} pos, {train_neg} neg")
    logger.info(f"Validation: {val_pos} pos, {val_neg} neg")
    logger.info(f"Test: {test_pos} pos, {test_neg} neg")
    logger.info(f"Test hard: {test_hard_pos} pos, {test_hard_neg} neg")
    
    # Convert splits to datasets (without embedding tensors)
    logger.success("Converting splits to PyG datasets...")
    train_dataset = convert_pairs_to_dataset(train_pairs, df_full)
    val_dataset = convert_pairs_to_dataset(val_pairs, df_full)
    test_dataset = convert_pairs_to_dataset(test_pairs, df_full)
    test_hard_dataset = convert_pairs_to_dataset(test_hard_pairs, df_full)
    
    logger.info(f"Dataset sizes after conversion:")
    logger.info(f"  Train: {len(train_dataset)} pairs")
    logger.info(f"  Validation: {len(val_dataset)} pairs")
    logger.info(f"  Test: {len(test_dataset)} pairs")
    logger.info(f"  Test hard: {len(test_hard_dataset)} pairs")
    
    # Create data loaders (no follow_batch needed since we don't have embedding tensors)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
    
    logger.success("Created data loaders with batch size {cfg.batch_size}")
    
    # Save config for reproduction BEFORE training starts
    import json
    from datetime import datetime
    
    config = {
        'model_info': {
            'model_path': str(run_dir / f"{cfg.task}_split{cfg.split}_best.pt"),
            'task': cfg.task,
            'backbone_model': cfg.backbone_model,
            'split': cfg.split,
            'eta_config': OmegaConf.to_container(cfg.eta, resolve=True),
            'omega_config': OmegaConf.to_container(cfg.omega, resolve=True),
            'score_config': OmegaConf.to_container(cfg.score, resolve=True)
        },
        'training_args': OmegaConf.to_container(cfg, resolve=True),
        'paths': {
            'splits_dir': str(splits_dir),
            'plasma_dir': str(plasma_dir),
            'embeddings_dir': str(embeddings_dir),
            'run_dir': str(run_dir)
        },
        'dataset_info': {
            'dataset_sizes': {
                'train': len(train_dataset),
                'validation': len(val_dataset),
                'test': len(test_dataset),
                'test_hard': len(test_hard_dataset)
            },
            'split_distribution': {
                'train': {'pos': train_pos, 'neg': train_neg},
                'validation': {'pos': val_pos, 'neg': val_neg},
                'test': {'pos': test_pos, 'neg': test_neg},
                'test_hard': {'pos': test_hard_pos, 'neg': test_hard_neg}
            },
            'dataset_fraction': cfg.dataset_fraction
        },
        'created_at': datetime.now().isoformat(),
        'training_status': 'started'
    }
    
    config_path = run_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Config saved to {config_path}")
    
    # Initialize alignment model
    logger.success("Initializing alignment model...")
    
    # Build eta_kwargs based on configuration
    eta_kwargs = {}
    if cfg.eta.type == 'lrl':
        eta_kwargs['hidden_dim'] = cfg.eta.hidden_dim
    elif cfg.eta.type == 'hinge' and hasattr(cfg.eta, 'normalize'):
        eta_kwargs['normalize'] = cfg.eta.normalize
    
    # Build omega_kwargs based on configuration  
    omega_kwargs = {}
    if cfg.omega.type == 'sinkhorn':
        if hasattr(cfg.omega, 'temperature'):
            omega_kwargs['temperature'] = cfg.omega.temperature
        if hasattr(cfg.omega, 'n_iters'):
            omega_kwargs['n_iters'] = cfg.omega.n_iters
    
    alignment_model = Alignment(
        eta=cfg.eta.type, 
        omega=cfg.omega.type, 
        eta_kwargs=eta_kwargs,
        omega_kwargs=omega_kwargs
    ).to(device)
    
    # Setup optimizer and loss
    optimizer = Adam(alignment_model.parameters(), lr=cfg.learning_rate)
    bce_with_logits_criterion = BCEWithLogitsLoss()
    
    # Training variables
    if cfg.supervise_metric == "loss":
        best_val_metric = float('inf')  # For loss, we want to minimize
    else:
        best_val_metric = 0.0  # For other metrics, we want to maximize
    patience_counter = 0
    
    logger.success("Starting training...")
    logger.info(f"Learning rate: {cfg.learning_rate}")
    logger.info(f"Number of epochs: {cfg.num_epochs}")
    logger.info(f"Patience: {cfg.patience}")
    logger.info(f"Min delta: {cfg.min_delta}")
    logger.info(f"Target loss weight: {cfg.target_loss_weight}")
    logger.info(f"Supervise metric: {cfg.supervise_metric}")
    
    # Initialize timing variables
    training_start_time = time.time()
    epoch_times = []
    
    # Initialize time trackers for training and validation
    training_timer = TimeTracker("training")
    validation_timer = TimeTracker("validation")
    
    # Training loop
    for epoch in range(cfg.num_epochs):
        epoch_start_time = time.time()
        epoch_display = epoch + 1  # Display as 1-indexed for logging
        
        # Start timing for first epoch only
        if epoch == 0:
            training_timer.start()
        
        # Training phase
        alignment_model.train()
        total_loss, total_alignment_loss, total_label_loss_positive = 0.0, 0.0, 0.0
        positive_label_count = 0  # Count positive classes for averaging
        skip_count = 0
        
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch_display}"):
            try:
                batch_start_time = time.time() if epoch == 0 else None
                
                batch = batch.to(device)
                optimizer.zero_grad()
                
                # Get embeddings on-the-fly from CPU dictionary
                query_emb, candidate_emb, query_full_emb, candidate_full_emb, query_emb_batch, candidate_emb_batch = get_batch_embeddings(batch, seq_embeddings, device)
                
                # Skip if no embeddings found
                if query_emb.numel() == 0 or candidate_emb.numel() == 0:
                    skip_count += 1
                    continue
                
                # Forward pass
                M = alignment_model(query_emb, candidate_emb, query_emb_batch, candidate_emb_batch)
                sim_scores = alignment_score(query_emb, candidate_emb, M, candidate_emb_batch, threshold=cfg.score.threshold, K=cfg.score.K)
                
                # Main alignment loss
                alignment_targets = batch.y.float().to(device)
                alignment_loss = bce_with_logits_criterion(sim_scores, alignment_targets)
                
                # Get target labels for this batch
                query_targets, candidate_targets = add_target_labels_batch(batch, df_full, query_emb_batch, candidate_emb_batch)
                query_targets = query_targets.to(device)
                candidate_targets = candidate_targets.to(device)
                
                # Label match loss for targets using alignment matrix (weighted)
                if cfg.target_loss_weight > 0:
                    # Use label_match_loss with is_match based on alignment targets
                    is_match = batch.y.bool().item() if batch.y.numel() == 1 else True  # Default to True for batch
                    try:
                        label_loss = label_match_loss(query_targets, candidate_targets, M, is_match=is_match) * cfg.target_loss_weight
                    except Exception as e:
                        logger.warning(f"Failed to calculate label match loss: {e}")
                        label_loss = torch.tensor(0.0, device=device)
                    # Combine losses
                    total_batch_loss = alignment_loss + label_loss
                else:
                    # Only use alignment loss if target weight is 0
                    label_loss = torch.tensor(0.0, device=device)
                    total_batch_loss = alignment_loss
                
                total_batch_loss.backward()
                optimizer.step()
                
                total_loss += total_batch_loss.item()
                total_alignment_loss += alignment_loss.item()
                
                # Only accumulate label loss for positive classes
                if cfg.target_loss_weight > 0:
                    is_match = batch.y.bool().item() if batch.y.numel() == 1 else True
                    if is_match:
                        total_label_loss_positive += label_loss.item()
                        positive_label_count += 1
                
                # Track sample timing for first epoch only
                if epoch == 0 and batch_start_time is not None:
                    batch_end_time = time.time()
                    batch_time = batch_end_time - batch_start_time
                    batch_size = len(batch.y)  # Number of samples in this batch
                    training_timer.add_sample_time(batch_time, batch_size)
                
                # Memory cleanup (if enabled)
                if cfg.enable_cleanup:
                    del M, sim_scores, alignment_loss, total_batch_loss
                    if cfg.target_loss_weight > 0:
                        del label_loss
                    del query_emb, candidate_emb, query_full_emb, candidate_full_emb, query_emb_batch, candidate_emb_batch
                    torch.cuda.empty_cache()
                
            except RuntimeError as e:
                logger.error(f"Training error: {e}")
                skip_count += 1
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                continue
        
        effective_batches = len(train_loader) - skip_count
        avg_loss = total_loss / max(effective_batches, 1)
        avg_alignment_loss = total_alignment_loss / max(effective_batches, 1)
        avg_label_loss_positive = total_label_loss_positive / max(positive_label_count, 1) if cfg.target_loss_weight > 0 else 0.0
        
        if cfg.target_loss_weight > 0 and positive_label_count > 0:
            logger.success(f"Epoch {epoch_display} → Train Total Loss: {avg_loss:.4f}, Alignment: {avg_alignment_loss:.4f}, "
                        f"Label Match (positive only): {avg_label_loss_positive:.4f}, Pos Count: {positive_label_count}, Skipped: {skip_count}")
        else:
            logger.success(f"Epoch {epoch_display} → Train Total Loss: {avg_loss:.4f}, Alignment: {avg_alignment_loss:.4f}, "
                        f"Skipped: {skip_count}")
        
        # End training timing for first epoch only
        if epoch == 0:
            training_timer.end()
        
        # Validation phase
        alignment_model.eval()
        val_auc = BinaryAUROC()
        val_pr_auc = BinaryAveragePrecision()
        
        # Start validation timing for first epoch only
        if epoch == 0:
            validation_timer.start()
        
        # Store scores and targets for F1Max calculation
        val_scores = []
        val_targets = []
        val_label_match_scores_positive = []  # Store 1 - label_match_loss for positive classes only
        val_total_loss = 0.0
        skip_val = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch_display}"):
                try:
                    val_batch_start_time = time.time() if epoch == 0 else None
                    batch = batch.to(device)
                    
                    # Get embeddings on-the-fly from CPU dictionary
                    query_emb, candidate_emb, query_full_emb, candidate_full_emb, query_emb_batch, candidate_emb_batch = get_batch_embeddings(batch, seq_embeddings, device)
                    
                    # Skip if no embeddings found
                    if query_emb.numel() == 0 or candidate_emb.numel() == 0:
                        skip_val += 1
                        continue
                    
                    M = alignment_model(query_emb, candidate_emb, query_emb_batch, candidate_emb_batch)
                    sim_scores = alignment_score(query_emb, candidate_emb, M, candidate_emb_batch, threshold=cfg.score.threshold, K=cfg.score.K)
                    
                    # Calculate validation loss
                    alignment_targets = batch.y.float().to(device)
                    val_loss = bce_with_logits_criterion(sim_scores, alignment_targets)
                    val_total_loss += val_loss.item()
                    
                    # Update alignment metrics
                    val_auc.update(sim_scores.detach().cpu(), batch.y.cpu())
                    val_pr_auc.update(sim_scores.detach().cpu(), batch.y.cpu())
                    # Store scores and targets for F1Max calculation
                    val_scores.extend(sim_scores.detach().cpu().tolist())
                    val_targets.extend(batch.y.cpu().tolist())
                    
                    # Calculate label match score if target_loss_weight > 0
                    if cfg.target_loss_weight > 0:
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
                                val_label_match_scores_positive.append(label_match_score)
                        except Exception as e:
                            logger.warning(f"Failed to calculate label match metrics in validation: {e}")
                    
                    # Track validation timing for first epoch only
                    if epoch == 0 and val_batch_start_time is not None:
                        val_batch_end_time = time.time()
                        val_batch_time = val_batch_end_time - val_batch_start_time
                        val_batch_size = len(batch.y)  # Number of samples in this batch
                        validation_timer.add_sample_time(val_batch_time, val_batch_size)
                    
                    # Memory cleanup (if enabled)
                    if cfg.enable_cleanup:
                        del M, sim_scores
                        del query_emb, candidate_emb, query_full_emb, candidate_full_emb, query_emb_batch, candidate_emb_batch
                        torch.cuda.empty_cache()
                    
                except RuntimeError as e:
                    logger.error(f"Validation error: {e}")
                    skip_val += 1
                    torch.cuda.empty_cache()
                    continue
        
        # End validation timing for first epoch only
        if epoch == 0:
            validation_timer.end()
        
        val_rocauc = val_auc.compute().item()
        val_pr_auc_score = val_pr_auc.compute().item()
        # Compute F1Max using proper threshold optimization
        val_f1_max = compute_f1_max(torch.tensor(val_scores), torch.tensor(val_targets)) if val_scores else 0.0
        val_effective_batches = len(val_loader) - skip_val
        val_avg_loss = val_total_loss / max(val_effective_batches, 1)
        
        # Compute label match metrics if available (only for positive classes)
        if cfg.target_loss_weight > 0 and val_label_match_scores_positive:
            val_label_match_avg_positive = sum(val_label_match_scores_positive) / len(val_label_match_scores_positive)
            
            # Log progress with all metrics
            logger.info(f"Epoch {epoch_display} → Val Alignment - ROCAUC: {val_rocauc:.4f}, F1_max: {val_f1_max:.4f}, PR_AUC: {val_pr_auc_score:.4f}, Loss: {val_avg_loss:.4f}")
            logger.info(f"Epoch {epoch_display} → Val Label Match Score (positive only): {val_label_match_avg_positive:.4f}, Skipped: {skip_val}")
        else:
            # Log progress without label match metrics
            logger.info(f"Epoch {epoch_display} → Val Alignment - ROCAUC: {val_rocauc:.4f}, F1_max: {val_f1_max:.4f}, PR_AUC: {val_pr_auc_score:.4f}, Loss: {val_avg_loss:.4f}, Skipped: {skip_val}")
        
        # Early stopping check based on selected metric
        if cfg.supervise_metric == "rocauc":
            current_val_metric = val_rocauc
        elif cfg.supervise_metric == "prauc":
            current_val_metric = val_pr_auc_score
        elif cfg.supervise_metric == "loss":
            current_val_metric = val_avg_loss  # Use actual loss value
        elif cfg.supervise_metric == "label_match_score":
            current_val_metric = val_label_match_avg_positive if (cfg.target_loss_weight > 0 and val_label_match_scores_positive) else 0.0
        else:  # f1max
            current_val_metric = val_f1_max
        
        # Check improvement based on metric type
        if cfg.supervise_metric == "loss":
            improved = current_val_metric < best_val_metric - cfg.min_delta  # For loss, lower is better
        else:
            improved = current_val_metric > best_val_metric + cfg.min_delta  # For other metrics, higher is better
            
        if improved:
            best_val_metric = current_val_metric
            patience_counter = 0
            # Save best model
            torch.save(alignment_model.state_dict(), run_dir / f"{cfg.task}_split{cfg.split}_best.pt")
            logger.success(f"New best model saved with {cfg.supervise_metric.upper()} = {best_val_metric:.4f}")
        else:
            patience_counter += 1
            
        
        # Record epoch time
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)
        
        if patience_counter >= cfg.patience:
            logger.warning(f"Early stopping triggered after {epoch_display} epochs")
            break
    
    # Calculate overall training time statistics
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time
    average_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0.0
    
    # Get detailed timing from first epoch
    training_timing = training_timer.get_timing_summary()
    validation_timing = validation_timer.get_timing_summary()
    
    training_avg_time_per_sample = training_timing.get('average_time_per_sample_seconds', 0.0)
    validation_avg_time_per_sample = validation_timing.get('average_time_per_sample_seconds', 0.0)
    training_samples_processed = training_timing.get('total_samples_processed', 0)
    validation_samples_processed = validation_timing.get('total_samples_processed', 0)
    
    logger.success(f"Training completed. Best validation {cfg.supervise_metric.upper()}: {best_val_metric:.4f}")
    logger.info(f"Total training time: {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes)")
    logger.info(f"Average time per epoch: {average_epoch_time:.2f} seconds")
    logger.info(f"First epoch training average time per sample: {training_avg_time_per_sample*1000:.2f} ms")
    logger.info(f"First epoch validation average time per sample: {validation_avg_time_per_sample*1000:.2f} ms")
    logger.info(f"Total samples processed in first epoch training: {training_samples_processed}")
    logger.info(f"Total samples processed in first epoch validation: {validation_samples_processed}")
    
    # Save training time data (overall training statistics)
    additional_training_data = {
        'average_epoch_time_seconds': average_epoch_time,
        'num_epochs_completed': len(epoch_times),
        'epoch_times_seconds': epoch_times,
        'overall_training_start_time': training_start_time,
        'overall_training_end_time': training_end_time
    }
    
    # Combine training timing with additional data
    training_time_data = training_timing.copy()
    training_time_data.update(additional_training_data)
    
    training_time_path = run_dir / "training_time.json"
    with open(training_time_path, 'w') as f:
        json.dump(training_time_data, f, indent=2)
    
    # Save validation time data (inference time)
    validation_time_path = run_dir / "inference_time.json"
    with open(validation_time_path, 'w') as f:
        json.dump(validation_timing, f, indent=2)
    
    logger.info(f"Training time data saved to {training_time_path}")
    logger.info(f"Inference time data saved to {validation_time_path}")
    
    # Update config with training results
    # Create results dictionary with final metrics
    results_dict = {
        'best_val_rocauc': val_rocauc,
        'best_val_f1_max': val_f1_max,
        'best_val_pr_auc': val_pr_auc_score,
        'best_val_loss': val_avg_loss,
        'final_epoch': epoch_display,
        'early_stopped': patience_counter >= cfg.patience,
        'supervise_metric': cfg.supervise_metric,
        'best_supervise_metric_value': best_val_metric
    }
    
    # Add label match score if available (only for positive classes)
    if cfg.target_loss_weight > 0 and val_label_match_scores_positive:
        results_dict.update({
            'best_val_label_match_score_positive': val_label_match_avg_positive
        })
    
    config['results'] = results_dict
    config['training_status'] = 'completed'
    config['completed_at'] = datetime.now().isoformat()
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.success(f"Config updated with training results")
    
    logger.success(f"Training completed successfully!")
    logger.success(f"Best model saved at: {run_dir / f'{cfg.task}_split{cfg.split}_best.pt'}")
    logger.success(f"Config saved at: {config_path}")
    
    # Run evaluation if requested
    if cfg.test:
        logger.success("Starting automatic evaluation...")
        try:
            import subprocess
            import sys
            
            # Get the path to evaluate.py
            evaluate_script = Path(__file__).parent / "evaluate.py"
            
            # Run evaluation using the run directory with no_learning disabled
            cmd = [sys.executable, str(evaluate_script), f"model_path={str(run_dir)}", "skip_no_learning=true"]
            
            logger.success(f"Running evaluation command: {' '.join(cmd)}")
            
            # Run evaluation and capture output
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)
            
            if result.returncode == 0:
                logger.success(f"Evaluation completed successfully!")
                logger.success("Evaluation output:")
                for line in result.stdout.split('\n'):
                    if line.strip():
                        logger.info(f"EVAL: {line}")
            else:
                logger.error("Evaluation failed!")
                logger.error("Error output:")
                for line in result.stderr.split('\n'):
                    if line.strip():
                        logger.error(f"EVAL ERROR: {line}")
                
        except Exception as e:
            logger.error(f"Failed to run evaluation: {e}")
            logger.info("You can manually run evaluation with:")
            logger.info(f"python evaluate.py model_path={run_dir}")
    else:
        logger.success("Skipping evaluation (--no-test was specified)")
        logger.info("To evaluate later, run:")
        logger.info(f"python evaluate.py model_path={run_dir}")

    return 0


if __name__ == "__main__":
    main()