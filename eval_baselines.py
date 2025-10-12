"""
Baseline evaluation script for comparing baseline models across all datasets and splits.

This script evaluates baseline models (prostt5, tm_vec, esm2, protbert, prott5, ankh) and 
optionally no-learning alignment baseline across all available datasets and splits.
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

def evaluate_baselines_only(data_loader, dataset_name, baseline_embeddings, device, df_full, 
                           baseline_models, include_no_learning=False, seq_embeddings=None, timer=None):
    """Evaluate only baseline models without trained alignment model."""
    from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
    from tqdm import tqdm
    from torch.nn.functional import cosine_similarity
    
    # Initialize baseline metrics dynamically
    baseline_aucs = {model: BinaryAUROC() for model in baseline_models}
    baseline_pr_aucs = {model: BinaryAveragePrecision() for model in baseline_models}
    # Store scores and targets for F1Max calculation
    baseline_scores = {model: [] for model in baseline_models}
    baseline_targets = {model: [] for model in baseline_models}
    
    # Initialize no-learning metrics if requested
    if include_no_learning:
        auc_no_learning = BinaryAUROC()
        pr_auc_no_learning = BinaryAveragePrecision()
        no_learning_scores = []
        no_learning_targets = []
        
        # Initialize label match score tracking for no-learning (positive samples only)
        label_match_scores_no_learning_positive = []
        
        # Create no-learning alignment model
        alignment_no_learning = Alignment(eta='hinge', omega='sinkhorn').to(device)
        alignment_no_learning.eval()
    
    skip_count = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Evaluating {dataset_name}"):
            try:
                # Start timing for this batch if timer is provided
                batch_start_time = time.time() if timer is not None else None
                
                batch = batch.to(device)
                
                # Get baseline embeddings
                query_uid = batch.frag_key[0].split('_')[1]
                candidate_uid = batch.seq_key[0].split('_')[1]
                
                # Calculate similarities for each baseline model
                baseline_sims = {}
                for model in baseline_models:
                    if model in baseline_embeddings:
                        embeddings = baseline_embeddings[model]
                        if query_uid in embeddings and candidate_uid in embeddings:
                            query_emb = embeddings[query_uid].to(device)
                            candidate_emb = embeddings[candidate_uid].to(device)
                            sim = cosine_similarity(query_emb.unsqueeze(0), candidate_emb.unsqueeze(0), dim=-1)
                            baseline_sims[model] = sim
                        else:
                            baseline_sims[model] = None
                    else:
                        baseline_sims[model] = None

                # End timing for this batch if timer is provided
                if timer is not None and batch_start_time is not None:
                    batch_end_time = time.time()
                    batch_time = batch_end_time - batch_start_time
                    batch_size = len(batch.y)  # Number of samples in this batch
                    timer.add_sample_time(batch_time / len(baseline_models), batch_size)
                
                # Update baseline metrics if embeddings were found
                for model in baseline_models:
                    if baseline_sims[model] is not None:
                        baseline_aucs[model].update(baseline_sims[model].detach().cpu(), batch.y.cpu())
                        baseline_pr_aucs[model].update(baseline_sims[model].detach().cpu(), batch.y.cpu())
                        # Store scores and targets for F1Max calculation
                        baseline_scores[model].extend(baseline_sims[model].detach().cpu().tolist())
                        baseline_targets[model].extend(batch.y.cpu().tolist())

                # Memory cleanup
                for model in baseline_models:
                    if baseline_sims[model] is not None:
                        del baseline_sims[model]
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                logger.error(f"Error processing batch: {e}")
                skip_count += 1
                continue
    
    # Compute final scores
    results = {
        'metrics': {
            'rocauc': {},
            'f1_max': {},
            'pr_auc': {},
            'label_match_score': {}
        },
        'skipped': skip_count
    }
    
    # Add baseline model results dynamically
    for model in baseline_models:
        results['metrics']['rocauc'][model] = baseline_aucs[model].compute().item()
        results['metrics']['pr_auc'][model] = baseline_pr_aucs[model].compute().item()
        # Compute F1Max using proper threshold optimization
        if baseline_scores[model]:  # Only if we have scores
            results['metrics']['f1_max'][model] = compute_f1_max(torch.tensor(baseline_scores[model]), torch.tensor(baseline_targets[model]))
        else:
            results['metrics']['f1_max'][model] = 0.0
    
    # Add no-learning results if included
    if include_no_learning:
        results['metrics']['rocauc']['no_learning'] = auc_no_learning.compute().item()
        results['metrics']['pr_auc']['no_learning'] = pr_auc_no_learning.compute().item()
        # Compute F1Max using proper threshold optimization
        if no_learning_scores:  # Only if we have scores
            results['metrics']['f1_max']['no_learning'] = compute_f1_max(torch.tensor(no_learning_scores), torch.tensor(no_learning_targets))
        else:
            results['metrics']['f1_max']['no_learning'] = 0.0
        # Add label match score for no-learning
        results['metrics']['label_match_score']['no_learning'] = sum(label_match_scores_no_learning_positive) / len(label_match_scores_no_learning_positive) if label_match_scores_no_learning_positive else 0.0
    
    return results


def discover_datasets_and_splits(dataset_path, dataset_names=None):
    """Discover all datasets and their splits in the dataset_path."""
    dataset_path = Path(dataset_path)
    datasets_info = {}
    
    if not dataset_path.exists():
        logger.error(f"Dataset path not found: {dataset_path}")
        return datasets_info
    
    # Iterate through all directories in dataset_path
    for dataset_dir in dataset_path.iterdir():
        if dataset_dir.is_dir():
            dataset_name = dataset_dir.name
            if dataset_names and dataset_name not in dataset_names:
                continue
            splits = []
            
            # Look for split directories
            for split_dir in dataset_dir.iterdir():
                if split_dir.is_dir() and split_dir.name.startswith('split_'):
                    try:
                        split_num = int(split_dir.name.split('_')[1])
                        splits.append(split_num)
                    except (IndexError, ValueError):
                        continue
            
            if splits:
                datasets_info[dataset_name] = sorted(splits)
                logger.info(f"Found dataset '{dataset_name}' with splits: {splits}")
    
    return datasets_info


@hydra.main(version_base=None, config_path="configs", config_name="baselines")
def main(cfg: DictConfig) -> None:
    from hydra.core.hydra_config import HydraConfig
    
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
    
    # Get run directory from Hydra
    if HydraConfig.initialized():
        hydra_cfg = HydraConfig.get()
        run_dir = Path(hydra_cfg.runtime.output_dir)
    else:
        # Fallback for non-Hydra execution
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plasma_path = Path(__file__).parent
        run_dir = plasma_path / "runs" / f"baselines_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    if cfg.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(cfg.device)
    
    # Define paths
    plasma_dir = Path(__file__).parent
    dataset_path = plasma_dir / cfg.dataset_path
    
    logger.success("=== BASELINE EVALUATION STARTING ===")
    logger.info(f"Device: {device}")
    logger.info(f"Dataset path: {dataset_path}")
    logger.info(f"Baseline models: {cfg.baseline_models}")
    logger.info(f"Include no-learning: {cfg.include_no_learning}")
    logger.info(f"Run directory: {run_dir}")
    
    # Discover all datasets and splits
    if cfg.task == "all":
        datasets_info = discover_datasets_and_splits(dataset_path, dataset_names=cfg.datasets)
        
        # Filter by configured datasets if specified
        if cfg.datasets:
            filtered_info = {}
            for dataset in cfg.datasets:
                if dataset in datasets_info:
                    filtered_info[dataset] = datasets_info[dataset]
                else:
                    logger.warning(f"Configured dataset '{dataset}' not found in {dataset_path}")
            datasets_info = filtered_info
    else:
        # Single task mode
        datasets_info = discover_datasets_and_splits(dataset_path)
        if cfg.task in datasets_info:
            datasets_info = {cfg.task: datasets_info[cfg.task]}
        else:
            logger.error(f"Task '{cfg.task}' not found in {dataset_path}")
            return 1
    
    if not datasets_info:
        logger.error("No datasets found to evaluate")
        return 1
    
    logger.success(f"Will evaluate {len(datasets_info)} datasets: {list(datasets_info.keys())}")
    
    # Mapping from baseline names to actual directories
    baseline_dirs = {
        'prostt5': "ProstT5",
        'tm_vec': "TM-Vec",
        'esm2': "esm2_t33_650M_UR50D",
        'protbert': "prot_bert", 
        'prott5': "prot_t5_xl_half_uniref50-enc",
        'ankh': "ankh-base",
        'protssn': "ProtSSN"
    }
    
    # Store all results for final summary
    all_results = {}
    
    # Process each dataset and split
    for dataset_name, splits in datasets_info.items():
        logger.success(f"\n=== Processing dataset: {dataset_name} ===")
        
        # Load original dataset for metadata and get unique UIDs
        df_full = pd.read_csv(plasma_dir / "data" / "raw" / f"{dataset_name}.csv")
        logger.info(f"Loaded dataset with {len(df_full)} rows")
        
        # Get unique UIDs from the dataset for memory-efficient embedding loading
        unique_uids = set(df_full['uid'].unique())
        logger.info(f"Found {len(unique_uids)} unique UIDs in dataset")
        
        # Load protein-level embeddings for baselines (only needed ones)
        logger.info("Loading baseline protein-level embeddings...")
        baseline_embeddings = {}
        for model in cfg.baseline_models:
            if model in baseline_dirs:
                model_dir = baseline_dirs[model]
                embeddings_dir = plasma_dir / "data" / "embeddings" / model_dir / "PR_embeddings"
                logger.info(f"Loading {model} from: {embeddings_dir}")
                
                model_embeddings = {}
                if embeddings_dir.exists():
                    missing_embeddings = []
                    for uid in unique_uids:
                        emb_file = embeddings_dir / f"{uid}.pt"
                        if emb_file.exists():
                            try:
                                embedding = torch.load(emb_file, weights_only=False)
                                model_embeddings[uid] = embedding
                            except Exception as e:
                                logger.warning(f"Failed to load {model} {emb_file}: {e}")
                                missing_embeddings.append(uid)
                        else:
                            missing_embeddings.append(uid)
                    
                    logger.info(f"Successfully loaded {model} protein embeddings for {len(model_embeddings)} entries")
                    if missing_embeddings:
                        logger.warning(f"Missing {model} embeddings for {len(missing_embeddings)} UIDs")
                else:
                    logger.warning(f"{model} protein embeddings directory not found: {embeddings_dir}")
                
                baseline_embeddings[model] = model_embeddings
            else:
                logger.warning(f"Unknown baseline model: {model}. Skipping...")
        
        # Load AA-level embeddings for no-learning baseline if needed
        seq_embeddings = None
        if cfg.include_no_learning:
            # Use first available backbone model for no-learning baseline
            available_backbones = ["ProstT5", "TM-Vec", "esm2_t33_650M_UR50D", "prot_bert", "prot_t5_xl_half_uniref50-enc", "ankh-base", "ProtSSN"]
            for backbone in available_backbones:
                embeddings_dir = plasma_dir / "data" / "embeddings" / backbone / "AA_embeddings"
                if embeddings_dir.exists():
                    logger.info(f"Loading AA-level embeddings for no-learning from: {embeddings_dir}")
                    seq_embeddings = {}
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
                    break
            
            if seq_embeddings is None:
                logger.warning("No AA-level embeddings found for no-learning baseline. Disabling no-learning evaluation.")
                cfg.include_no_learning = False
        
        dataset_results = {}
        
        # Process each split
        for split in splits:
            logger.info(f"\n--- Processing split {split} ---")
            
            splits_dir = dataset_path / dataset_name / f"split_{split}"
            
            # Create task/split directory structure
            task_split_dir = run_dir / dataset_name / str(split)
            task_split_dir.mkdir(parents=True, exist_ok=True)
            
            # Setup logging for this split in the split directory
            setup_logging(task_split_dir, f"{dataset_name}_split{split}_evaluation")
            
            # Initialize time tracker for this split
            timer = TimeTracker("evaluation")
            timer.start()
            
            # Load dataset splits
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
            test_dataset = convert_pairs_to_dataset(test_pairs, df_full)
            test_hard_dataset = convert_pairs_to_dataset(test_hard_pairs, df_full)
            
            logger.info(f"Dataset sizes after conversion:")
            logger.info(f"  Test: {len(test_dataset)} pairs")
            logger.info(f"  Test hard: {len(test_hard_dataset)} pairs")
            
            # Create data loaders
            test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)
            test_hard_loader = DataLoader(test_hard_dataset, batch_size=cfg.batch_size, shuffle=False)
            
            # Evaluate test set (frequent InterPro IDs) - track timing per batch
            logger.info("Evaluating test set (frequent InterPro IDs)...")
            test_results = evaluate_baselines_only(test_loader, "Test_Frequent", baseline_embeddings, 
                                                 device, df_full, cfg.baseline_models, cfg.include_no_learning, seq_embeddings, timer)
            
            # Evaluate test hard set (less frequent InterPro IDs) - no timing tracking
            logger.info("Evaluating test hard set (less frequent InterPro IDs)...")
            test_hard_results = evaluate_baselines_only(test_hard_loader, "Test_Hard", baseline_embeddings, 
                                                      device, df_full, cfg.baseline_models, cfg.include_no_learning, seq_embeddings)
            
            # End timing
            timer.end()
            
            # Log results
            logger.info("=== RESULTS ===")
            logger.info("Test Set (Frequent InterPro IDs):")
            for model in cfg.baseline_models:
                logger.info(f"  {model.upper()}: ROCAUC: {test_results['metrics']['rocauc'][model]:.4f}, F1_max: {test_results['metrics']['f1_max'][model]:.4f}, PR_AUC: {test_results['metrics']['pr_auc'][model]:.4f}")
            if cfg.include_no_learning:
                logger.info(f"  NO_LEARNING: ROCAUC: {test_results['metrics']['rocauc']['no_learning']:.4f}, F1_max: {test_results['metrics']['f1_max']['no_learning']:.4f}, PR_AUC: {test_results['metrics']['pr_auc']['no_learning']:.4f}")
            logger.info(f"  Skipped batches: {test_results['skipped']}")
            
            logger.info("Test Hard Set (Less Frequent InterPro IDs):")
            for model in cfg.baseline_models:
                logger.info(f"  {model.upper()}: ROCAUC: {test_hard_results['metrics']['rocauc'][model]:.4f}, F1_max: {test_hard_results['metrics']['f1_max'][model]:.4f}, PR_AUC: {test_hard_results['metrics']['pr_auc'][model]:.4f}")
            if cfg.include_no_learning:
                logger.info(f"  NO_LEARNING: ROCAUC: {test_hard_results['metrics']['rocauc']['no_learning']:.4f}, F1_max: {test_hard_results['metrics']['f1_max']['no_learning']:.4f}, PR_AUC: {test_hard_results['metrics']['pr_auc']['no_learning']:.4f}")
            logger.info(f"  Skipped batches: {test_hard_results['skipped']}")
            
            # Store results
            split_results = {
                'test_frequent': test_results,
                'test_hard': test_hard_results,
                'model_info': {
                    'task': dataset_name,
                    'split': split,
                    'baseline_models': list(cfg.baseline_models),  # Convert ListConfig to list
                    'include_no_learning': bool(cfg.include_no_learning)  # Convert to regular bool
                },
                'evaluation_config': {
                    'batch_size': int(cfg.batch_size),  # Convert to regular int
                    'device': str(device)
                },
                'dataset_sizes': {
                    'test': len(test_dataset),
                    'test_hard': len(test_hard_dataset)
                },
                'split_distribution': {
                    'test': {'pos': test_pos, 'neg': test_neg},
                    'test_hard': {'pos': test_hard_pos, 'neg': test_hard_neg}
                },
                'evaluation_timestamp': json.loads(json.dumps(pd.Timestamp.now(), default=str))
            }
            
            dataset_results[f"split_{split}"] = split_results
            
            # Save individual split results in hierarchical structure: {task}/{split}/{task}_split{split}_evaluation.json
            results_file = task_split_dir / f"{dataset_name}_split{split}_evaluation.json"
            with open(results_file, 'w') as f:
                json.dump(split_results, f, indent=2)
            
            # Save timing data
            timing_path = task_split_dir / "inference_time.json"
            timer.save_timing_data(timing_path)
            
            logger.success(f"Saved results to {results_file}")
            logger.success(f"Saved timing data to {timing_path}")
        
        all_results[dataset_name] = dataset_results
    
    # Save combined results (optional, for backwards compatibility)
    combined_results_file = run_dir / "all_evaluation_results.json"
    with open(combined_results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.success(f"Saved combined results to {combined_results_file}")
    
    
    logger.success("=== BASELINE EVALUATION COMPLETED ===")
    logger.success(f"Results directory: {run_dir}")
    logger.success(f"Evaluated {len(all_results)} datasets with {sum(len(splits) for splits in datasets_info.values())} total splits")
    
    return 0


if __name__ == "__main__":
    main()