"""
Alignment-based baseline evaluation script for structural methods like TM-align and Foldseek.

This script evaluates alignment-based methods (TM-align, Foldseek) that require protein sequences 
or structures and perform alignment operations via command-line tools.
"""

import torch
import pandas as pd
import json
import subprocess
import tempfile
import time
import datetime
from pathlib import Path
from torch_geometric.loader import DataLoader
from loguru import logger
import hydra
from omegaconf import DictConfig, OmegaConf
from utils import (
    convert_pairs_to_dataset,
    count_pos_neg,
    setup_logging
)


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


def get_pdb_path(uid, dataset_name, plasma_dir):
    """Get PDB file path for a given UID and dataset."""
    pdb_path = plasma_dir / "data" / "pdb" / dataset_name / f"{uid}.pdb"
    if pdb_path.exists():
        return pdb_path
    return None


def build_uid_to_pdb_mapping(plasma_dir, unique_uids):
    """Build a mapping from UID to PDB file path for all UIDs in the dataset."""
    pdb_base_dir = plasma_dir / "data" / "pdb"
    uid_to_pdb = {}
    
    if not pdb_base_dir.exists():
        logger.warning(f"PDB directory not found: {pdb_base_dir}")
        return uid_to_pdb
    
    logger.info(f"Building PDB file mapping for {len(unique_uids)} unique UIDs...")
    
    # Iterate through all subdirectories once
    for subdir in pdb_base_dir.iterdir():
        if subdir.is_dir():
            logger.debug(f"Scanning PDB directory: {subdir.name}")
            # Check for each UID we need in this subdirectory
            for uid in unique_uids:
                if uid not in uid_to_pdb:  # Only check if not already found
                    pdb_file = subdir / f"{uid}.pdb"
                    if pdb_file.exists():
                        uid_to_pdb[uid] = pdb_file
    
    logger.info(f"Found PDB files for {len(uid_to_pdb)} out of {len(unique_uids)} UIDs")
    missing_count = len(unique_uids) - len(uid_to_pdb)
    if missing_count > 0:
        logger.warning(f"Missing PDB files for {missing_count} UIDs")
    
    return uid_to_pdb


def batch_create_protein_databases(uid_to_pdb, foldseek_executable, plasma_dir):
    """
    Batch create foldseek databases for proteins that don't have cached databases yet.
    This is more efficient than creating them one by one during evaluation.
    
    Args:
        uid_to_pdb (dict): Mapping from UID to PDB file path
        foldseek_executable (str): Path to foldseek executable
        plasma_dir (Path): Path to plasma directory
        
    Returns:
        int: Number of databases created
    """
    threedi_cache_dir = plasma_dir / "data" / "threeDi"
    threedi_cache_dir.mkdir(exist_ok=True)
    
    missing_uids = []
    
    # Find UIDs that don't have cached databases
    for uid, pdb_path in uid_to_pdb.items():
        db_path = threedi_cache_dir / f"{uid}_db"
        if not db_path.exists():
            missing_uids.append((uid, pdb_path))
    
    if not missing_uids:
        logger.info("All protein databases already cached")
        return 0
    
    logger.info(f"Creating foldseek databases for {len(missing_uids)} proteins...")
    
    created_count = 0
    for uid, pdb_path in missing_uids:
        db_path = threedi_cache_dir / f"{uid}_db"
        
        try:
            cmd = [
                foldseek_executable, "createdb",
                str(pdb_path), str(db_path),
                "--threads", "1",
                "-v", "0"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                created_count += 1
                if created_count % 100 == 0:  # Progress indicator
                    logger.info(f"Created {created_count}/{len(missing_uids)} databases...")
            else:
                logger.debug(f"Failed to create database for {uid}: {result.stderr}")
                
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            logger.debug(f"Error creating database for {uid}: {e}")
    
    logger.success(f"Created {created_count} new protein databases")
    return created_count


def run_foldseek_alignment(query_pdb, target_pdb, temp_dir, foldseek_executable=None, use_gpu=False, gpu_id=0):
    """
    Run Foldseek alignment between two PDB structures using cached databases.
    Times only the alignment step, excluding 3di conversion time.
    Assumes databases have already been created by batch_create_protein_databases().
    
    Args:
        query_pdb (Path): Path to query PDB file
        target_pdb (Path): Path to target PDB file
        temp_dir (Path): Temporary directory for intermediate files
        foldseek_executable (str): Path to foldseek executable
        use_gpu (bool): Enable GPU acceleration
        gpu_id (int): GPU device ID to use
        
    Returns:
        tuple: (alignment_score, alignment_time_seconds) or (None, 0.0) if alignment failed
    """
    if not query_pdb or not target_pdb:
        return None, 0.0
        
    if not query_pdb.exists() or not target_pdb.exists():
        logger.debug(f"PDB files missing: {query_pdb.exists()=}, {target_pdb.exists()=}")
        return None, 0.0
    
    if not foldseek_executable:
        logger.debug("No foldseek executable provided")
        return None, 0.0
    
    try:
        # Get cached database paths
        plasma_dir = Path(__file__).parent
        threedi_cache_dir = plasma_dir / "data" / "threeDi"
        
        query_uid = query_pdb.stem
        target_uid = target_pdb.stem
        
        query_db = threedi_cache_dir / f"{query_uid}_db"
        target_db = threedi_cache_dir / f"{target_uid}_db"
        
        # Check if databases exist (should have been created in batch)
        if not query_db.exists() or not target_db.exists():
            logger.debug(f"Database missing: query={query_db.exists()}, target={target_db.exists()}")
            return None, 0.0
        
        # Step 2: Run alignment on pre-converted databases (timed)
        aln_db = temp_dir / f"alignment_{query_pdb.stem}_{target_pdb.stem}"
        
        cmd_align = [
            foldseek_executable, "search",
            str(query_db), str(target_db), str(aln_db), str(temp_dir),
            "--max-seqs", "1",  # Only return the best match
            "--exhaustive-search", "1",  # Bypass prefiltering for pairwise alignment
            "--threads", "1",  # Single thread for simple pairwise alignment
            "-v", "0"  # Reduce verbosity
        ]
        
        # Add GPU options if enabled
        if use_gpu:
            cmd_align.extend(["--gpu", "1"])
            if gpu_id != 0:
                cmd_align.extend(["--gpu-id", str(gpu_id)])
        
        # Time only the alignment step
        alignment_start = time.time()
        result = subprocess.run(cmd_align, capture_output=True, text=True, timeout=5)
        alignment_end = time.time()
        alignment_time = alignment_end - alignment_start
        
        if result.returncode == 0:
            # Convert alignment database to readable format
            aln_file = temp_dir / f"alignment_{query_pdb.stem}_{target_pdb.stem}.tsv"
            
            cmd_convert = [
                foldseek_executable, "convertalis",
                str(query_db), str(target_db), str(aln_db), str(aln_file),
                "--format-output", "bits",
                "-v", "0"
            ]
            
            convert_result = subprocess.run(cmd_convert, capture_output=True, text=True, timeout=5)
            
            if convert_result.returncode == 0 and aln_file.exists():
                # Parse alignment results
                with open(aln_file, 'r') as f:
                    lines = f.readlines()
                
                if lines:
                    # With --format-output "bits", we get just the bit score
                    try:
                        bit_score = float(lines[0].strip())
                        return bit_score, alignment_time
                    except ValueError:
                        pass
        
        logger.debug(f"Foldseek alignment failed: {result.stderr}")
        return None, alignment_time  # Return timing even if alignment failed
        
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError) as e:
        logger.debug(f"Foldseek alignment error: {e}")
        return None, 0.0
    except Exception as e:
        logger.debug(f"Unexpected foldseek error: {e}")
        return None, 0.0


def run_tm_align(query_pdb, target_pdb, temp_dir, tm_align_executable=None):
    """
    Run TM-align between two PDB structures.
    
    Args:
        query_pdb (Path): Path to query PDB file
        target_pdb (Path): Path to target PDB file  
        temp_dir (Path): Temporary directory for intermediate files
        tm_align_executable (str): Path to TM-align executable
        
    Returns:
        float or None: TM-score (None if alignment failed)
    """
    if not query_pdb or not target_pdb:
        return None
        
    if not query_pdb.exists() or not target_pdb.exists():
        logger.debug(f"PDB files missing: {query_pdb.exists()=}, {target_pdb.exists()=}")
        return None
    
    if not tm_align_executable:
        logger.debug("No TM-align executable provided")
        return None
    
    try:
        # Run TM-align
        cmd = [tm_align_executable, str(query_pdb), str(target_pdb)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)  # Reduced timeout
        
        if result.returncode == 0:
            # Parse TM-score from output (use Chain_1 normalized score)
            lines = result.stdout.split('\n')
            for line in lines:
                if line.startswith('TM-score=') and 'Chain_1' in line:
                    # Extract TM-score value: "TM-score= 0.18430 (if normalized by length of Chain_1..."
                    try:
                        score_part = line.split('=')[1].strip().split()[0]
                        tm_score = float(score_part)
                        return tm_score
                    except (IndexError, ValueError):
                        continue
        
        logger.debug(f"TM-align failed: {result.stderr}")
        return None
        
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError) as e:
        logger.debug(f"TM-align error: {e}")
        return None
    except Exception as e:
        logger.debug(f"Unexpected TM-align error: {e}")
        return None


def evaluate_alignment_method(data_loader, dataset_name, alignment_method, foldseek_executable=None, tm_align_executable=None, use_gpu=False, gpu_id=0, uid_to_pdb=None):
    """Evaluate single alignment-based method."""
    from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
    from tqdm import tqdm
    
    # Initialize metrics for the single method
    method_auc = BinaryAUROC()
    method_pr_auc = BinaryAveragePrecision()
    method_scores = []
    method_targets = []
    
    # Initialize timing variables
    sample_times = []
    total_samples = len(data_loader)
    start_time = time.time()
    
    skip_count = 0
    
    # Create temporary directory for alignment files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader, desc=f"Evaluating {dataset_name}")):
                try:
                    sample_start_time = time.time()
                    # Get UID information
                    query_uid = batch.frag_key[0].split('_')[1]
                    candidate_uid = batch.seq_key[0].split('_')[1]
                    
                    # Get PDB file paths from prebuilt mapping
                    query_pdb = uid_to_pdb.get(query_uid) if uid_to_pdb else None
                    candidate_pdb = uid_to_pdb.get(candidate_uid) if uid_to_pdb else None
                    
                    # Run single alignment method
                    alignment_score = None
                    alignment_time = 0.0
                    
                    if alignment_method == 'foldseek':
                        alignment_score, alignment_time = run_foldseek_alignment(
                            query_pdb, candidate_pdb, temp_path, foldseek_executable, use_gpu, gpu_id
                        )
                    elif alignment_method == 'tm_align':
                        alignment_score = run_tm_align(
                            query_pdb, candidate_pdb, temp_path, tm_align_executable
                        )
                        alignment_time = 0.0  # TM-align timing not modified
                    
                    # Update metrics
                    if alignment_score is not None:
                        score = torch.tensor([alignment_score])  # Make it a 1D tensor
                        target = batch.y.cpu()
                        
                        method_auc.update(score, target)
                        method_pr_auc.update(score, target)
                        method_scores.extend([alignment_score])
                        method_targets.extend(target.tolist())
                    else:
                        # Skip this pair (no structure files available)
                        skip_count += 1
                    
                    # Record sample time
                    sample_end_time = time.time()
                    if alignment_method == 'foldseek' and alignment_score is not None:
                        # For foldseek, record only the alignment time (excluding 3di conversion)
                        sample_times.append(alignment_time)
                    else:
                        # For other methods or failed alignments, record total sample time
                        sample_times.append(sample_end_time - sample_start_time)
                    
                    # No temporary files to clean up (using existing PDB files)
                    
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
                    skip_count += 1
                    # Still record time for failed samples
                    sample_end_time = time.time()
                    sample_times.append(sample_end_time - sample_start_time)
                    continue
    
    # Compute timing statistics
    end_time = time.time()
    total_time = end_time - start_time
    avg_sample_time = sum(sample_times) / len(sample_times) if sample_times else 0.0
    
    # Compute final scores
    results = {
        'metrics': {
            'rocauc': 0.0,
            'f1_max': 0.0,
            'pr_auc': 0.0
        },
        'method': alignment_method,
        'skipped': skip_count,
        'timing': {
            'total_evaluation_time_seconds': total_time,
            'total_evaluation_time_minutes': total_time / 60.0,
            'total_evaluation_time_hours': total_time / 3600.0,
            'average_sample_time_seconds': avg_sample_time,
            'total_samples_processed': len(sample_times),
            'samples_per_second': len(sample_times) / total_time if total_time > 0 else 0.0,
            'sample_times_seconds': sample_times
        }
    }
    
    try:
        results['metrics']['rocauc'] = method_auc.compute().item()
        results['metrics']['pr_auc'] = method_pr_auc.compute().item()
        
        # Compute F1Max
        if method_scores:
            results['metrics']['f1_max'] = compute_f1_max(
                torch.tensor(method_scores), 
                torch.tensor(method_targets)
            )
        else:
            results['metrics']['f1_max'] = 0.0
    except Exception as e:
        logger.warning(f"Failed to compute metrics for {alignment_method}: {e}")
        results['metrics']['rocauc'] = 0.0
        results['metrics']['pr_auc'] = 0.0
        results['metrics']['f1_max'] = 0.0
    
    return results


def get_dataset_splits(dataset_path, dataset_name):
    """Get available splits for a specific dataset."""
    dataset_path = Path(dataset_path)
    dataset_dir = dataset_path / dataset_name
    splits = []
    
    if not dataset_dir.exists():
        logger.error(f"Dataset directory not found: {dataset_dir}")
        return splits
    
    # Look for split directories
    for split_dir in dataset_dir.iterdir():
        if split_dir.is_dir() and split_dir.name.startswith('split_'):
            try:
                split_num = int(split_dir.name.split('_')[1])
                splits.append(split_num)
            except (IndexError, ValueError):
                continue
    
    if splits:
        logger.info(f"Found dataset '{dataset_name}' with splits: {sorted(splits)}")
    
    return sorted(splits)


@hydra.main(version_base=None, config_path="configs", config_name="align_baselines")
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
    
    # Get run directory from Hydra or create organized structure
    if HydraConfig.initialized():
        hydra_cfg = HydraConfig.get()
        run_dir = Path(hydra_cfg.runtime.output_dir)
    else:
        # Create organized directory structure following train.py pattern
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plasma_path = Path(__file__).parent
        
        # Create organized structure following train.py pattern: runs/{job_name}/{task}/{timestamp}/
        run_dir = plasma_path / "runs" / "eval_align_baselines" / cfg.task / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    if cfg.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(cfg.device)
    
    # Define paths
    plasma_dir = Path(__file__).parent
    dataset_path = plasma_dir / cfg.dataset_path
    
    logger.success("=== ALIGNMENT BASELINE EVALUATION STARTING ===")
    logger.info(f"Device: {device}")
    logger.info(f"Dataset path: {dataset_path}")
    logger.info(f"Task: {cfg.task}")
    logger.info(f"Split: {cfg.split}")
    logger.info(f"Alignment method: {cfg.alignment_method}")
    logger.info(f"GPU acceleration: {'Enabled' if cfg.use_gpu else 'Disabled'}")
    if cfg.use_gpu:
        logger.info(f"GPU device ID: {cfg.gpu_id}")
    logger.info(f"Run directory: {run_dir}")
    
    # Check if the requested alignment tool is available
    foldseek_executable = None
    tm_align_executable = None
    method_available = False
    
    if cfg.alignment_method == 'foldseek':
        # Try to find foldseek in common locations
        foldseek_paths = [
            '/home/drizer/miniconda3/pkgs/foldseek-10.941cd33-h5021889_1/bin/foldseek',
            '/home/drizer/miniconda3/envs/venusx/bin/foldseek',
            'foldseek'  # If it's in PATH
        ]
        for foldseek_path in foldseek_paths:
            try:
                # Foldseek doesn't have --version, so just run without args to test if it exists
                result = subprocess.run([foldseek_path], 
                             capture_output=True, timeout=5)
                # If foldseek runs and produces output, it's available (even if exit code is non-zero)
                if result.stdout or result.stderr:
                    if b"foldseek" in result.stderr.lower() or b"foldseek" in result.stdout.lower():
                        logger.info(f"✓ Foldseek is available at: {foldseek_path}")
                        # Store the working path for later use
                        foldseek_executable = foldseek_path
                        method_available = True
                        break
            except (FileNotFoundError, subprocess.TimeoutExpired) as e:
                logger.debug(f"Failed to find foldseek at {foldseek_path}: {e}")
                continue
            except subprocess.CalledProcessError:
                # This is expected - foldseek without args returns non-zero exit code
                pass
        
        if not method_available:
            logger.error("✗ Foldseek not found")
            return 1
            
    elif cfg.alignment_method == 'tm_align':
        # Try to find TM-align in common locations
        tm_align_paths = [
            '/home/drizer/miniconda3/envs/venusx/bin/TMalign',
            './tmalign/TMalign',  # Local tmalign directory
            'TMalign'  # If it's in PATH
        ]
        for tm_align_path in tm_align_paths:
            try:
                # Test if TMalign is available by running with -h flag
                result = subprocess.run([tm_align_path, '-h'], 
                             capture_output=True, timeout=5)
                # TMalign with -h should produce help output
                if result.stdout or result.stderr:
                    if b"TMalign" in result.stdout or b"TM-align" in result.stdout or b"TMalign" in result.stderr or b"TM-align" in result.stderr:
                        logger.info(f"✓ TM-align is available at: {tm_align_path}")
                        tm_align_executable = tm_align_path
                        method_available = True
                        break
            except (FileNotFoundError, subprocess.TimeoutExpired) as e:
                logger.debug(f"Failed to find TM-align at {tm_align_path}: {e}")
                continue
            except subprocess.CalledProcessError:
                # This might be expected depending on TM-align version
                pass
        
        if not method_available:
            logger.error("✗ TM-align not found")
            return 1
    
    # Get splits for the specific task
    available_splits = get_dataset_splits(dataset_path, cfg.task)
    
    if not available_splits:
        logger.error(f"No splits found for task '{cfg.task}'")
        return 1
    
    if cfg.split not in available_splits:
        logger.error(f"Split {cfg.split} not found for task '{cfg.task}'. Available splits: {available_splits}")
        return 1
    
    logger.success(f"Will evaluate task '{cfg.task}' split {cfg.split} with method '{cfg.alignment_method}'")
    
    # Process single dataset and split
    logger.success(f"\n=== Processing dataset: {cfg.task} split {cfg.split} ===")
    
    # Load original dataset for sequences
    df_full = pd.read_csv(plasma_dir / "data" / "raw" / f"{cfg.task}.csv")
    logger.info(f"Loaded dataset with {len(df_full)} rows")
    
    # Get unique UIDs from the dataset for PDB file mapping
    unique_uids = set(df_full['uid'].unique())
    logger.info(f"Found {len(unique_uids)} unique UIDs in dataset")
    
    # Build UID to PDB file mapping once for the entire dataset
    uid_to_pdb = build_uid_to_pdb_mapping(plasma_dir, unique_uids)
    
    # Batch create foldseek databases for all proteins (if using foldseek)
    if cfg.alignment_method == 'foldseek' and foldseek_executable:
        logger.info("Pre-creating foldseek databases for all proteins...")
        batch_create_protein_databases(uid_to_pdb, foldseek_executable, plasma_dir)
    
    splits_dir = dataset_path / cfg.task / f"split_{cfg.split}"
    
    # Setup logging for this run
    setup_logging(run_dir, f"{cfg.task}_split{cfg.split}_{cfg.alignment_method}_evaluation")
    
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
    
    # Convert splits to datasets
    test_dataset = convert_pairs_to_dataset(test_pairs, df_full)
    test_hard_dataset = convert_pairs_to_dataset(test_hard_pairs, df_full)
    
    # Sample datasets if dataset_fraction < 1.0
    if cfg.dataset_fraction < 1.0:
        import random
        random.seed(42)  # For reproducible sampling
        
        original_test_size = len(test_dataset)
        original_test_hard_size = len(test_hard_dataset)
        
        # Sample test dataset
        test_sample_size = max(1, int(len(test_dataset) * cfg.dataset_fraction))
        test_indices = random.sample(range(len(test_dataset)), test_sample_size)
        test_dataset = torch.utils.data.Subset(test_dataset, test_indices)
        
        # Sample test hard dataset
        test_hard_sample_size = max(1, int(len(test_hard_dataset) * cfg.dataset_fraction))
        test_hard_indices = random.sample(range(len(test_hard_dataset)), test_hard_sample_size)
        test_hard_dataset = torch.utils.data.Subset(test_hard_dataset, test_hard_indices)
        
        logger.info(f"Dataset sampling (fraction={cfg.dataset_fraction}):")
        logger.info(f"  Test: {original_test_size} → {len(test_dataset)} pairs ({len(test_dataset)/original_test_size*100:.1f}%)")
        logger.info(f"  Test hard: {original_test_hard_size} → {len(test_hard_dataset)} pairs ({len(test_hard_dataset)/original_test_hard_size*100:.1f}%)")
    
    logger.info(f"Dataset sizes after conversion:")
    logger.info(f"  Test: {len(test_dataset)} pairs")
    logger.info(f"  Test hard: {len(test_hard_dataset)} pairs")
    
    # Create data loaders with batch size 1 for sequence-by-sequence processing
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    test_hard_loader = DataLoader(test_hard_dataset, batch_size=1, shuffle=False)
    
    # Evaluate test set
    logger.info("Evaluating test set (frequent InterPro IDs)...")
    test_results = evaluate_alignment_method(
        test_loader, "Test_Frequent", cfg.alignment_method, foldseek_executable, tm_align_executable, cfg.use_gpu, cfg.gpu_id, uid_to_pdb
    )
    
    # Evaluate test hard set
    logger.info("Evaluating test hard set (less frequent InterPro IDs)...")
    test_hard_results = evaluate_alignment_method(
        test_hard_loader, "Test_Hard", cfg.alignment_method, foldseek_executable, tm_align_executable, cfg.use_gpu, cfg.gpu_id, uid_to_pdb
    )
    
    # Log results
    logger.info("=== RESULTS ===")
    logger.info("Test Set (Frequent InterPro IDs):")
    logger.info(f"  {cfg.alignment_method.upper()}: ROCAUC: {test_results['metrics']['rocauc']:.4f}, F1_max: {test_results['metrics']['f1_max']:.4f}, PR_AUC: {test_results['metrics']['pr_auc']:.4f}")
    logger.info(f"  Skipped batches: {test_results['skipped']}")
    logger.info(f"  Timing: {test_results['timing']['total_evaluation_time_seconds']:.2f}s ({test_results['timing']['average_sample_time_seconds']:.3f}s/sample)")
    
    logger.info("Test Hard Set (Less Frequent InterPro IDs):")
    logger.info(f"  {cfg.alignment_method.upper()}: ROCAUC: {test_hard_results['metrics']['rocauc']:.4f}, F1_max: {test_hard_results['metrics']['f1_max']:.4f}, PR_AUC: {test_hard_results['metrics']['pr_auc']:.4f}")
    logger.info(f"  Skipped batches: {test_hard_results['skipped']}")
    logger.info(f"  Timing: {test_hard_results['timing']['total_evaluation_time_seconds']:.2f}s ({test_hard_results['timing']['average_sample_time_seconds']:.3f}s/sample)")
    
    # Store results
    results = {
        'test_frequent': test_results,
        'test_hard': test_hard_results,
        'model_info': {
            'task': cfg.task,
            'split': cfg.split,
            'alignment_method': cfg.alignment_method
        },
        'evaluation_config': {
            'batch_size': 1,  # Fixed for alignment methods
            'device': str(device),
            'use_gpu': cfg.use_gpu,
            'gpu_id': cfg.gpu_id if cfg.use_gpu else None,
            'dataset_fraction': cfg.dataset_fraction
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
    # Create evaluation timing summary
    evaluation_timing = {
        'total_evaluation_time_seconds': test_results['timing']['total_evaluation_time_seconds'] + test_hard_results['timing']['total_evaluation_time_seconds'],
        'total_evaluation_time_minutes': (test_results['timing']['total_evaluation_time_seconds'] + test_hard_results['timing']['total_evaluation_time_seconds']) / 60.0,
        'total_evaluation_time_hours': (test_results['timing']['total_evaluation_time_seconds'] + test_hard_results['timing']['total_evaluation_time_seconds']) / 3600.0,
        'test_frequent_time_seconds': test_results['timing']['total_evaluation_time_seconds'],
        'test_hard_time_seconds': test_hard_results['timing']['total_evaluation_time_seconds'],
        'average_sample_time_test_frequent': test_results['timing']['average_sample_time_seconds'],
        'average_sample_time_test_hard': test_hard_results['timing']['average_sample_time_seconds'],
        'total_samples_processed': test_results['timing']['total_samples_processed'] + test_hard_results['timing']['total_samples_processed'],
        'samples_per_second_overall': (test_results['timing']['total_samples_processed'] + test_hard_results['timing']['total_samples_processed']) / (test_results['timing']['total_evaluation_time_seconds'] + test_hard_results['timing']['total_evaluation_time_seconds']) if (test_results['timing']['total_evaluation_time_seconds'] + test_hard_results['timing']['total_evaluation_time_seconds']) > 0 else 0.0,
        'evaluation_start_time': time.time() - (test_results['timing']['total_evaluation_time_seconds'] + test_hard_results['timing']['total_evaluation_time_seconds']),
        'evaluation_end_time': time.time(),
    }
    
    # Save results
    results_file = run_dir / f"{cfg.task}_split{cfg.split}_{cfg.alignment_method}_evaluation.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save evaluation timing separately (similar to training_time.json)
    timing_file = run_dir / "evaluation_time.json"
    with open(timing_file, 'w') as f:
        json.dump(evaluation_timing, f, indent=2)
    
    logger.success(f"Saved results to {results_file}")
    logger.success(f"Saved timing info to {timing_file}")
    
    logger.success("=== ALIGNMENT BASELINE EVALUATION COMPLETED ===")
    logger.success(f"Results directory: {run_dir}")
    logger.success(f"Evaluated task '{cfg.task}' split {cfg.split} with method '{cfg.alignment_method}'")
    
    return 0


if __name__ == "__main__":
    main()