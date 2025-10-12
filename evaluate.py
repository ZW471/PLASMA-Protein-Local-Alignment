"""
Evaluation script for alignment model comparing with tm-vec and prostt5 baselines.
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





def plot_alignment_example(batch, alignment_matrix, similarity_score, df_full, examples_dir, dataset_name):
    """Plot alignment heatmap with target annotations."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Convert alignment matrix to numpy
        M_np = alignment_matrix.detach().cpu().numpy()
        
        # Get target labels for query and candidate
        query_key = batch.frag_key[0]
        candidate_key = batch.seq_key[0]
        
        query_uid = query_key.split('_')[1]
        query_interpro = query_key.split('_')[0]
        candidate_uid = candidate_key.split('_')[1]
        candidate_interpro = candidate_key.split('_')[0]
        
        query_targets = df_full[(df_full['uid'] == query_uid) & (df_full['interpro_id'] == query_interpro)]['label'].values[0]
        candidate_targets = df_full[(df_full['uid'] == candidate_uid) & (df_full['interpro_id'] == candidate_interpro)]['label'].values[0]
        
        # Parse and normalize target values to [0, 1] range
        query_targets_parsed = np.array(json.loads(query_targets), dtype=float)
        candidate_targets_parsed = np.array(json.loads(candidate_targets), dtype=float)
        
        # Normalize to [0, 1] range for visualization
        query_targets_norm = (query_targets_parsed - query_targets_parsed.min()) / (query_targets_parsed.max() - query_targets_parsed.min() + 1e-8)
        candidate_targets_norm = (candidate_targets_parsed - candidate_targets_parsed.min()) / (candidate_targets_parsed.max() - candidate_targets_parsed.min() + 1e-8)
        
        # Build a 2×2 grid where
        # (2,1) is the alignment heatmap,
        # (1,1) is a 1×N bar of candidate targets,
        # (2,2) is an N×1 bar of query targets,
        # (1,2) we'll leave empty.
        fig = make_subplots(
            rows=2, cols=2,
            column_widths=[0.8, 0.2],
            row_heights=[0.2, 0.8],
            specs=[[{"type": "heatmap"}, {"type": "heatmap"}],
                   [{"type": "heatmap"}, {"type": "heatmap"}]],
            horizontal_spacing=0.02,
            vertical_spacing=0.02,
        )
        
        # Top row: candidate targets as a 1×L heatmap (normalized to [0,1])
        fig.add_trace(
            go.Heatmap(
                z=[candidate_targets_norm],
                colorscale="Plasma",
                showscale=False,
                zmin=0, zmax=1
            ),
            row=1, col=1
        )
        
        # Right col: query targets as an L×1 heatmap (normalized to [0,1])
        fig.add_trace(
            go.Heatmap(
                z=[[v] for v in query_targets_norm],
                colorscale="Plasma",
                showscale=False,
                zmin=0, zmax=1
            ),
            row=2, col=2
        )
        
        # Main alignment matrix
        fig.add_trace(
            go.Heatmap(
                z=M_np,
                colorscale="Plasma",
                colorbar=dict(title="Alignment Score"),
                zmin=0, zmax=1
            ),
            row=2, col=1
        )
        
        # Hide the unused subplot
        fig.update_xaxes(visible=False, row=1, col=2)
        fig.update_xaxes(visible=False, row=2, col=2)
        fig.update_yaxes(visible=False, row=1, col=2)
        fig.update_yaxes(visible=False, row=1, col=1)
        
        # Label axes of the main heatmap
        fig.update_xaxes(title_text=f"Candidate ({candidate_key}) tokens", row=2, col=1)
        fig.update_yaxes(title_text=f"Query ({query_key}) tokens", row=2, col=1)
        
        # Add title with similarity score
        label = "Positive" if batch.y[0].item() == 1 else "Negative"
        fig.update_layout(
            title=f"{dataset_name} - {label} Example: {query_key} vs {candidate_key} (similarity: {similarity_score:.4f})",
        )
        
        # Save as HTML
        if not examples_dir.exists():
            examples_dir.mkdir(parents=True)
        
        filename = f"{dataset_name}_{label}_{query_key}_{candidate_key}_alignment.html"
        fig.write_html(examples_dir / filename)
        
        return True
        
    except Exception as e:
        logger.warning(f"Failed to plot example: {e}")
        return False


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

def evaluate_dataset_with_examples(data_loader, dataset_name, alignment_model, alignment_no_learning, 
                                  embeddings_dict, baseline_embeddings, device, df_full, examples_dir, example_num, no_target_label=False, baseline_models=None, skip_no_learning=False, score_config=None, timer=None):
    """Evaluate a dataset and return AUC, F1, and PR AUC scores, collecting examples for visualization."""
    from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
    from tqdm import tqdm
    from torch.nn.functional import cosine_similarity
    from utils.alignment_utils import alignment_score
    from utils.data_utils import add_target_labels_batch
    
    # Set default score_config if not provided
    if score_config is None:
        score_config = {'K': 10, 'threshold': 0.5}
    
    auc_alignment = BinaryAUROC()
    pr_auc_alignment = BinaryAveragePrecision()
    
    # Store scores and targets for F1Max calculation
    alignment_scores = []
    alignment_targets = []
    
    # Initialize no_learning metrics only if not skipped
    if not skip_no_learning:
        auc_no_learning = BinaryAUROC()
        pr_auc_no_learning = BinaryAveragePrecision()
        no_learning_scores = []
        no_learning_targets = []
    
    # Initialize baseline metrics dynamically
    baseline_models = baseline_models or []
    baseline_aucs = {model: BinaryAUROC() for model in baseline_models}
    baseline_pr_aucs = {model: BinaryAveragePrecision() for model in baseline_models}
    # Store scores and targets for F1Max calculation
    baseline_scores = {model: [] for model in baseline_models}
    baseline_targets = {model: [] for model in baseline_models}
    
    # Label match score tracking for alignment_trained and no_learning models (only if enabled)
    if not no_target_label:
        # Store label match scores for positive samples only
        label_match_scores_alignment_positive = []
        
        # Initialize no_learning label match scores only if not skipped
        if not skip_no_learning:
            label_match_scores_no_learning_positive = []
    
    skip_count = 0
    
    # For collecting examples
    positive_examples = []
    negative_examples = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Evaluating {dataset_name}"):
            try:
                # Start timing for this batch if timer is provided
                batch_start_time = time.time() if timer is not None else None
                
                batch = batch.to(device)
                
                # Get embeddings on-the-fly from CPU dictionary
                query_emb, candidate_emb, _, _, query_emb_batch, candidate_emb_batch = get_batch_embeddings(batch, embeddings_dict, device)
                
                # Skip if no embeddings found
                if query_emb.numel() == 0 or candidate_emb.numel() == 0:
                    skip_count += 1
                    # Reset batch timing since we're skipping this batch
                    batch_start_time = None
                    continue
                
                # Our trained alignment model
                M = alignment_model(query_emb, candidate_emb, query_emb_batch, candidate_emb_batch)
                sim_alignment = alignment_score(query_emb, candidate_emb, M, candidate_emb_batch, threshold=score_config['threshold'], K=score_config['K'])
                
                # Alignment without learning (only if not skipped)
                if not skip_no_learning:
                    M_no_learning = alignment_no_learning(query_emb, candidate_emb, query_emb_batch, candidate_emb_batch)
                    sim_no_learning = alignment_score(query_emb, candidate_emb, M_no_learning, candidate_emb_batch, threshold=score_config['threshold'], K=score_config['K'])
                
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
                
                # Update AUC metrics
                auc_alignment.update(sim_alignment.detach().cpu(), batch.y.cpu())
                
                # Store scores and targets for F1Max calculation
                alignment_scores.extend(sim_alignment.detach().cpu().tolist())
                alignment_targets.extend(batch.y.cpu().tolist())
                
                # Update PR AUC metrics
                pr_auc_alignment.update(sim_alignment.detach().cpu(), batch.y.cpu())
                
                # Update no_learning metrics only if not skipped
                if not skip_no_learning:
                    auc_no_learning.update(sim_no_learning.detach().cpu(), batch.y.cpu())
                    no_learning_scores.extend(sim_no_learning.detach().cpu().tolist())
                    no_learning_targets.extend(batch.y.cpu().tolist())
                    pr_auc_no_learning.update(sim_no_learning.detach().cpu(), batch.y.cpu())
                
                # Calculate label match scores for alignment_trained and no_learning models (only if enabled)
                if not no_target_label:
                    try:
                        # Get target labels for this batch
                        query_targets, candidate_targets = add_target_labels_batch(batch, df_full, query_emb_batch, candidate_emb_batch)
                        query_targets = query_targets.to(device)
                        candidate_targets = candidate_targets.to(device)
                        
                        # Calculate label match loss and convert to score (1 - loss) for alignment_trained
                        is_match = batch.y.bool().item() if batch.y.numel() == 1 else True
                        label_match_loss_alignment = label_match_loss(query_targets, candidate_targets, M, is_match=is_match)
                        label_match_score_alignment = 1.0 - label_match_loss_alignment.item()
                        
                        # Only store label match score for positive classes (since negative always returns 0)
                        if is_match:
                            label_match_scores_alignment_positive.append(label_match_score_alignment)
                        
                        # Calculate no_learning label match score only if not skipped
                        if not skip_no_learning:
                            label_match_loss_no_learning = label_match_loss(query_targets, candidate_targets, M_no_learning, is_match=is_match)
                            label_match_score_no_learning = 1.0 - label_match_loss_no_learning.item()
                            
                            # Only store label match score for positive classes
                            if is_match:
                                label_match_scores_no_learning_positive.append(label_match_score_no_learning)
                        
                    except Exception as e:
                        logger.warning(f"Failed to calculate label match metrics: {e}")
                
                # Update baseline metrics if embeddings were found
                for model in baseline_models:
                    if baseline_sims[model] is not None:
                        baseline_aucs[model].update(baseline_sims[model].detach().cpu(), batch.y.cpu())
                        baseline_pr_aucs[model].update(baseline_sims[model].detach().cpu(), batch.y.cpu())
                        # Store scores and targets for F1Max calculation
                        baseline_scores[model].extend(baseline_sims[model].detach().cpu().tolist())
                        baseline_targets[model].extend(batch.y.cpu().tolist())
                
                # Collect examples for plotting (with random selection)
                if example_num > 0:
                    is_positive = batch.y[0].item() == 1
                    
                    if is_positive:
                        if len(positive_examples) < example_num:
                            positive_examples.append((batch, M.clone(), sim_alignment[0].item()))
                        elif random.random() < 0.1:  # 10% chance to replace existing example
                            idx = random.randint(0, len(positive_examples) - 1)
                            positive_examples[idx] = (batch, M.clone(), sim_alignment[0].item())
                    else:
                        if len(negative_examples) < example_num:
                            negative_examples.append((batch, M.clone(), sim_alignment[0].item()))
                        elif random.random() < 0.1:  # 10% chance to replace existing example
                            idx = random.randint(0, len(negative_examples) - 1)
                            negative_examples[idx] = (batch, M.clone(), sim_alignment[0].item())
                
                # End timing for this batch if timer is provided
                if timer is not None and batch_start_time is not None:
                    batch_end_time = time.time()
                    batch_time = batch_end_time - batch_start_time
                    batch_size = len(batch.y)  # Number of samples in this batch
                    timer.add_sample_time(batch_time, batch_size)
                
                # Memory cleanup
                del M, sim_alignment
                if not skip_no_learning:
                    del M_no_learning, sim_no_learning
                for model in baseline_models:
                    if baseline_sims[model] is not None:
                        del baseline_sims[model]
                del query_emb, candidate_emb, query_emb_batch, candidate_emb_batch
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                logger.error(f"Error processing batch: {e}")
                skip_count += 1
                continue
    
    # Plot examples if requested
    if example_num > 0 and examples_dir:
        logger.info(f"Plotting {len(positive_examples)} positive and {len(negative_examples)} negative examples for {dataset_name}")
        
        all_examples = positive_examples + negative_examples
        for i, (example_batch, alignment_matrix, similarity_score) in enumerate(all_examples):
            try:
                plot_alignment_example(example_batch, alignment_matrix, similarity_score, 
                                     df_full, examples_dir, dataset_name)
            except Exception as e:
                logger.warning(f"Failed to plot example {i}: {e}")
    
    # Compute final scores with new structure
    results = {
        'metrics': {
            'rocauc': {
                'alignment_trained': auc_alignment.compute().item()
            },
            'f1_max': {
                'alignment_trained': compute_f1_max(torch.tensor(alignment_scores), torch.tensor(alignment_targets)) if alignment_scores else 0.0
            },
            'pr_auc': {
                'alignment_trained': pr_auc_alignment.compute().item()
            },
            'label_match_score': {
                'alignment_trained': sum(label_match_scores_alignment_positive) / len(label_match_scores_alignment_positive) if not no_target_label and label_match_scores_alignment_positive else 0.0
            }
        },
        'skipped': skip_count
    }
    
    # Add no_learning results only if not skipped
    if not skip_no_learning:
        results['metrics']['rocauc']['no_learning'] = auc_no_learning.compute().item()
        results['metrics']['f1_max']['no_learning'] = compute_f1_max(torch.tensor(no_learning_scores), torch.tensor(no_learning_targets)) if no_learning_scores else 0.0
        results['metrics']['pr_auc']['no_learning'] = pr_auc_no_learning.compute().item()
        results['metrics']['label_match_score']['no_learning'] = sum(label_match_scores_no_learning_positive) / len(label_match_scores_no_learning_positive) if not no_target_label and label_match_scores_no_learning_positive else 0.0
    
    # Add baseline model results dynamically
    for model in baseline_models:
        results['metrics']['rocauc'][model] = baseline_aucs[model].compute().item()
        results['metrics']['pr_auc'][model] = baseline_pr_aucs[model].compute().item()
        # Compute F1Max using proper threshold optimization
        if baseline_scores[model]:  # Only if we have scores
            results['metrics']['f1_max'][model] = compute_f1_max(torch.tensor(baseline_scores[model]), torch.tensor(baseline_targets[model]))
        else:
            results['metrics']['f1_max'][model] = 0.0
    
    # Label match scores are already included in the main metrics section
    
    return results


def plot_combined_metrics_visualization(test_freq_results, test_hard_results, save_dir):
    """Create a combined 2x3 subplot visualization comparing Test_Frequent and Test_Hard metrics."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Check if target metrics are available
        has_target_metrics = 'q_target_metrics' in test_freq_results and 'c_target_metrics' in test_freq_results
        
        if has_target_metrics:
            # Create subplot layout: 2 rows x 3 cols
            subplot_titles = [
                "Alignment ROC-AUC", "Alignment PR-AUC", "Alignment F1 Max",
                "Target ROC-AUC", "Target PR-AUC", "Target F1 Max"
            ]
            rows = 2
        else:
            # Create subplot layout: 1 row x 3 cols (only alignment metrics)
            subplot_titles = [
                "Alignment ROC-AUC", "Alignment PR-AUC", "Alignment F1 Max"
            ]
            rows = 1
        
        if has_target_metrics:
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=subplot_titles,
                specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
            )
        else:
            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=subplot_titles,
                specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
            )
        
        # Color scheme for Test_Frequent (darker) and Test_Hard (lighter)
        freq_colors = {
            'alignment_trained': '#1f77b4',
            'no_learning': '#ff7f0e', 
            'prostt5': '#2ca02c',
            'tm_vec': '#d62728',
            'esm2': '#17becf',
            'protbert': '#bcbd22',
            'prott5': '#e377c2',
            'ankh': '#8c564b'
        }
        hard_colors = {
            'alignment_trained': '#74a9cf',
            'no_learning': '#ffb366', 
            'prostt5': '#66c266',
            'tm_vec': '#ff6b6b',
            'esm2': '#7dd3d3',
            'protbert': '#d4d466',
            'prott5': '#f0a3d1',
            'ankh': '#c49c94'
        }
        
        model_names = {
            'alignment_trained': 'Ours',
            'no_learning': 'Ours (w/o LP)',
            'prostt5': 'ProST-T5',
            'tm_vec': 'TM-Vec',
            'esm2': 'ESM2',
            'protbert': 'ProtBERT',
            'prott5': 'ProtT5',
            'ankh': 'Ankh'
        }
        
        # Row 1: Alignment metrics comparison
        for col, metric in enumerate(['rocauc', 'pr_auc', 'f1_max'], 1):
            models = list(test_freq_results['metrics'][metric].keys())
            
            # Test_Frequent values
            freq_values = [test_freq_results['metrics'][metric][model] for model in models]
            # Test_Hard values
            hard_values = [test_hard_results['metrics'][metric][model] for model in models]
            
            # Add Test_Frequent bars with fallback for unknown models
            model_labels = [model_names.get(m, m.upper()) for m in models]
            freq_colors_list = [freq_colors.get(m, '#7f7f7f') for m in models]  # default gray
            hard_colors_list = [hard_colors.get(m, '#bfbfbf') for m in models]  # default light gray
            
            fig.add_trace(
                go.Bar(
                    x=model_labels,
                    y=freq_values,
                    marker_color=freq_colors_list,
                    showlegend=False,
                    text=[f"N: {v:.3f}" for v in freq_values],
                    textposition='auto',
                    offsetgroup=1,
                    width=0.35
                ),
                row=1, col=col
            )
            
            # Add Test_Hard bars
            fig.add_trace(
                go.Bar(
                    x=model_labels,
                    y=hard_values,
                    marker_color=hard_colors_list,
                    showlegend=False,
                    text=[f"H: {v:.3f}" for v in hard_values],
                    textposition='auto',
                    offsetgroup=2,
                    width=0.35
                ),
                row=1, col=col
            )
            
            # Set y-axis range for better visibility
            fig.update_yaxes(range=[0, 1], row=1, col=col)
        
        # Row 2: Target metrics comparison (only if target metrics are available)
        if has_target_metrics:
            # Colors for query vs candidate within each test set
            freq_q_colors = {'alignment_trained': '#1f77b4', 'no_learning': '#ff7f0e'}
            freq_c_colors = {'alignment_trained': '#4dabf7', 'no_learning': '#ffa94d'}
            hard_q_colors = {'alignment_trained': '#74a9cf', 'no_learning': '#ffb366'}
            hard_c_colors = {'alignment_trained': '#a8d4f0', 'no_learning': '#ffc999'}
            
            target_metrics = ['rocauc', 'pr_auc', 'f1_max']
            for col, metric in enumerate(target_metrics, 1):
                models = ['alignment_trained', 'no_learning']
                
                # Test_Frequent query values
                freq_q_values = [test_freq_results['q_target_metrics'][metric][model] for model in models]
                # Test_Frequent candidate values
                freq_c_values = [test_freq_results['c_target_metrics'][metric][model] for model in models]
                # Test_Hard query values
                hard_q_values = [test_hard_results['q_target_metrics'][metric][model] for model in models]
                # Test_Hard candidate values
                hard_c_values = [test_hard_results['c_target_metrics'][metric][model] for model in models]
                
                # Create x-axis labels for grouped bars
                x_labels = []
                for model in models:
                    x_labels.extend([f"{model_names[model]} (Q)", f"{model_names[model]} (C)"])
                
                # Combine values for plotting
                freq_values = []
                hard_values = []
                colors_freq = []
                colors_hard = []
                
                for i, model in enumerate(models):
                    # Query values
                    freq_values.append(freq_q_values[i])
                    hard_values.append(hard_q_values[i])
                    colors_freq.append(freq_q_colors[model])
                    colors_hard.append(hard_q_colors[model])
                    
                    # Candidate values
                    freq_values.append(freq_c_values[i])
                    hard_values.append(hard_c_values[i])
                    colors_freq.append(freq_c_colors[model])
                    colors_hard.append(hard_c_colors[model])
                
                # Add Test_Frequent target bars
                fig.add_trace(
                    go.Bar(
                        x=x_labels,
                        y=freq_values,
                        marker_color=colors_freq,
                        showlegend=False,
                        text=[f"N: {v:.3f}" for v in freq_values],
                        textposition='auto',
                        offsetgroup=3,
                        width=0.35
                    ),
                    row=2, col=col
                )
                
                # Add Test_Hard target bars
                fig.add_trace(
                    go.Bar(
                        x=x_labels,
                        y=hard_values,
                        marker_color=colors_hard,
                        showlegend=False,
                        text=[f"H: {v:.3f}" for v in hard_values],
                        textposition='auto',
                        offsetgroup=4,
                        width=0.35
                    ),
                    row=2, col=col
                )
                
                # Set y-axis range
                fig.update_yaxes(range=[0, 1], row=2, col=col)
                # Rotate x-axis labels for better readability
                fig.update_xaxes(tickangle=45, row=2, col=col)
        
        # Update layout
        title = "Combined Evaluation Metrics: Test Frequent vs Test Hard"
        if not has_target_metrics:
            title += " (Alignment Only)"
        
        fig.update_layout(
            title=title,
            height=900 if has_target_metrics else 500,
            showlegend=False,
            barmode='group'
        )
        
        # Save plot
        if not save_dir.exists():
            save_dir.mkdir(parents=True)
        
        plot_file = save_dir / "combined_metrics_overview.html"
        fig.write_html(plot_file)
        
        return True, str(plot_file)
        
    except Exception as e:
        logger.warning(f"Failed to create combined metrics visualization: {e}")
        return False, None


def load_config_from_path(model_path):
    """Load configuration from config.json in the same directory as the model."""
    model_path = Path(model_path)
    
    # If model_path is a directory, look for config.json inside
    if model_path.is_dir():
        config_path = model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"config.json not found in {model_path}")
    else:
        # If model_path is a file, look for config.json in the same directory
        config_path = model_path.parent / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"config.json not found in {model_path.parent}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config, config_path


def find_all_configs(base_path):
    """Recursively find all config.json files in subdirectories."""
    base_path = Path(base_path)
    config_files = []
    
    if base_path.is_file():
        # Single file provided
        if base_path.name == "config.json":
            config_files.append(base_path.parent)
        else:
            # Assume it's a model file, look for config in same directory
            config_path = base_path.parent / "config.json"
            if config_path.exists():
                config_files.append(base_path.parent)
    else:
        # Directory provided, search recursively
        for config_path in base_path.rglob("config.json"):
            config_files.append(config_path.parent)
    
    return sorted(config_files)


def evaluate_single_model(model_dir, cfg):
    """Evaluate a single model given its directory containing config.json."""
    # Initialize time tracker
    timer = TimeTracker("evaluation")
    timer.start()
    
    try:
        # Load configuration from config.json
        config, config_path = load_config_from_path(model_dir)
        logger.success(f"Processing: {model_dir}")
        logger.info(f"Loaded configuration from: {config_path}")
        
        # Extract configuration values
        model_info = config['model_info']
        training_args = config['training_args']
        paths = config['paths']
        
        task = model_info['task'] if cfg.task == 'any' else cfg.task
        dataset = task if cfg.dataset is None else cfg.dataset
        split = model_info['split']
        # Extract alignment config from model_info (new structure)
        eta_config = model_info.get('eta_config', {'type': 'lrl', 'hidden_dim': 512})
        omega_config = model_info.get('omega_config', {'type': 'sinkhorn', 'temperature': 0.1})
        # Extract score config with backward compatibility
        score_config = model_info.get('score_config', {'K': 10, 'threshold': 0.5})
        if 'score_config' not in model_info:
            logger.warning(f"score_config not found in {config_path}, using default values: K=10, threshold=0.5")
        backbone_model = cfg.backbone_model if cfg.backbone_model is not None else model_info.get('backbone_model', 'ProstT5')
        
        # Use command line overrides if provided, otherwise use config values
        batch_size = cfg.batch_size if cfg.batch_size is not None else training_args.get('batch_size', 2)
        device_arg = cfg.device if cfg.device is not None else training_args.get('device', 'auto')
        seed = training_args.get('seed', 42)
        
        # Set random seeds
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.empty_cache()
        
        # Setup device
        if device_arg == "auto":
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device_arg)
        
        # Determine paths - use the model directory as run_dir for output
        run_dir = model_dir
        model_path = Path(model_info['model_path'])
        
        setup_logging(run_dir, f"{task}_split{split}_evaluation")
        
        # Define paths using config information
        plasma_dir = Path(paths['plasma_dir'])
        splits_dir = plasma_dir / "data" / "processed" / task / f"split_{split}"
        
        logger.info(f"Task: {task}")
        logger.info(f"Backbone model: {backbone_model}")
        logger.info(f"Split: {split}")
        logger.info(f"Eta config: {eta_config}")
        logger.info(f"Omega config: {omega_config}")
        logger.info(f"Score config: {score_config}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Seed: {seed}")
        logger.info(f"Device: {device}")
        logger.info(f"Model path: {model_path}")
        
        # Load original dataset for metadata and get unique UIDs
        logger.info("Loading dataset...")
        df_full = pd.read_csv(plasma_dir / "data" / "raw" / f"{dataset}.csv")
        logger.info(f"Loaded dataset with {len(df_full)} rows")
        
        # Get unique UIDs from the dataset for memory-efficient embedding loading
        unique_uids = set(df_full['uid'].unique())
        logger.info(f"Found {len(unique_uids)} unique UIDs in dataset")
        
        # Load embeddings from plasma embeddings directory (only needed ones)
        logger.info("Loading embeddings...")
        embeddings_dir = plasma_dir / "data" / "embeddings" / backbone_model / "AA_embeddings"
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
                logger.warning(f"Missing embeddings for {len(missing_embeddings)} UIDs")
        else:
            logger.error(f"Embeddings directory not found: {embeddings_dir}")
            raise FileNotFoundError(f"Embeddings directory not found: {embeddings_dir}")
        
        # Load protein-level embeddings for baselines (only needed ones)
        logger.info("Loading baseline protein-level embeddings...")
        baseline_models = cfg.baseline_models if cfg.baseline_models is not None else []
        if baseline_models:
            logger.info(f"Loading baseline models: {baseline_models}")
        else:
            logger.info("No baseline models specified - will only evaluate alignment models")
        
        # Mapping from baseline names to actual directories
        baseline_dirs = {
            'prostt5': "ProstT5",
            'tm_vec': "TM-Vec",
            'esm2': "esm2_t33_650M_UR50D",
            'protbert': "prot_bert", 
            'prott5': "prot_t5_xl_half_uniref50-enc",
            'ankh': "ankh-base"
        }
        
        baseline_embeddings = {}
        for model in baseline_models:
            if model in baseline_dirs:
                model_dir_name = baseline_dirs[model]
                embeddings_dir_baseline = plasma_dir / "data" / "embeddings" / model_dir_name / "PR_embeddings"
                
                model_embeddings = {}
                if embeddings_dir_baseline.exists():
                    missing_embeddings = []
                    for uid in unique_uids:
                        emb_file = embeddings_dir_baseline / f"{uid}.pt"
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
                    logger.warning(f"{model} protein embeddings directory not found: {embeddings_dir_baseline}")
                
                baseline_embeddings[model] = model_embeddings
            else:
                logger.warning(f"Unknown baseline model: {model}. Skipping...")
        
        # Load dataset splits
        logger.info("Loading dataset splits...")
        train_pairs = torch.load(splits_dir / "train.pt", weights_only=False)
        val_pairs = torch.load(splits_dir / "validation.pt", weights_only=False)
        test_pairs = torch.load(splits_dir / "test.pt", weights_only=False)
        test_hard_pairs = torch.load(splits_dir / "test_hard.pt", weights_only=False)
        
        # Count positive/negative in each split
        train_pos, train_neg = count_pos_neg(train_pairs)
        val_pos, val_neg = count_pos_neg(val_pairs)
        test_pos, test_neg = count_pos_neg(test_pairs)
        test_hard_pos, test_hard_neg = count_pos_neg(test_hard_pairs)
        
        # Convert splits to datasets (without embedding tensors)
        logger.info("Converting splits to PyG datasets...")
        train_dataset = convert_pairs_to_dataset(train_pairs, df_full)
        val_dataset = convert_pairs_to_dataset(val_pairs, df_full)
        test_dataset = convert_pairs_to_dataset(test_pairs, df_full)
        test_hard_dataset = convert_pairs_to_dataset(test_hard_pairs, df_full)
        
        # Create data loaders (no follow_batch needed since we don't have embedding tensors)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        test_hard_loader = DataLoader(test_hard_dataset, batch_size=batch_size, shuffle=False)
        
        logger.info(f"Created data loaders with batch size {batch_size}")
        
        # Load trained alignment model
        logger.info("Loading trained alignment model...")
        
        # Build eta_kwargs based on configuration
        eta_kwargs = {}
        if eta_config['type'] == 'lrl':
            eta_kwargs['hidden_dim'] = eta_config['hidden_dim']
        elif eta_config['type'] == 'hinge' and eta_config.get('normalize'):
            eta_kwargs['normalize'] = eta_config['normalize']
        
        # Build omega_kwargs based on configuration  
        omega_kwargs = {}
        if omega_config['type'] == 'sinkhorn':
            if omega_config.get('temperature'):
                omega_kwargs['temperature'] = omega_config['temperature']
            if omega_config.get('n_iters'):
                omega_kwargs['n_iters'] = omega_config['n_iters']
        
        alignment_trained = Alignment(
            eta=eta_config['type'], 
            omega=omega_config['type'], 
            eta_kwargs=eta_kwargs,
            omega_kwargs=omega_kwargs
        ).to(device)
        alignment_trained.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        alignment_trained.eval()
        logger.info(f"Loaded trained model from {model_path}")
        
        # No learning baseline (only if not skipped)
        alignment_no_learning = None
        if not cfg.skip_no_learning:
            logger.info("Loading no-learning baseline...")
            alignment_no_learning = Alignment(eta='hinge', omega='sinkhorn').to(device)
            alignment_no_learning.eval()
        else:
            logger.info("Skipping no-learning baseline (skip_no_learning=true)")
        
        # Start evaluation
        logger.info("Starting evaluation...")
        if cfg.no_target_label:
            logger.info("Target classification metrics disabled (no_target_label=true)")
        else:
            logger.info("Target classification metrics enabled")
        
        # Setup examples directory if requested
        examples_dir = None
        if cfg.example_num > 0:
            examples_dir = run_dir / "examples"
            logger.info(f"Examples will be saved to: {examples_dir}")
        
        # Evaluate test set (frequent InterPro IDs) - track timing per batch
        logger.info("Evaluating test set (frequent InterPro IDs)...")
        test_results = evaluate_dataset_with_examples(test_loader, "Test_Frequent", alignment_trained, 
                                                     alignment_no_learning, seq_embeddings, baseline_embeddings, 
                                                     device, df_full, examples_dir, cfg.example_num, cfg.no_target_label, baseline_models, cfg.skip_no_learning, score_config, timer)
        
        # Evaluate test hard set (less frequent InterPro IDs) - no timing tracking
        logger.info("Evaluating test hard set (less frequent InterPro IDs)...")
        test_hard_results = evaluate_dataset_with_examples(test_hard_loader, "Test_Hard", alignment_trained,
                                                           alignment_no_learning, seq_embeddings, baseline_embeddings,
                                                           device, df_full, examples_dir, cfg.example_num, cfg.no_target_label, baseline_models, cfg.skip_no_learning, score_config)
        
        # Log and display results
        logger.success("=== EVALUATION RESULTS ===")
        
        print(f"\n=== Results for {model_dir} ===")
        print(f"Test Set - Alignment (trained): ROCAUC: {test_results['metrics']['rocauc']['alignment_trained']:.4f}, F1_max: {test_results['metrics']['f1_max']['alignment_trained']:.4f}, PR_AUC: {test_results['metrics']['pr_auc']['alignment_trained']:.4f}")
        print(f"Test Hard - Alignment (trained): ROCAUC: {test_hard_results['metrics']['rocauc']['alignment_trained']:.4f}, F1_max: {test_hard_results['metrics']['f1_max']['alignment_trained']:.4f}, PR_AUC: {test_hard_results['metrics']['pr_auc']['alignment_trained']:.4f}")
        
        # Save results with restructured format
        results_summary = {
            'test_frequent': test_results,
            'test_hard': test_hard_results,
            'model_info': {
                'model_path': str(model_path),
                'task': task,
                'backbone_model': backbone_model,
                'split': split,
                'eta_config': eta_config,
                'omega_config': omega_config,
                'score_config': score_config
            },
            'evaluation_config': {
                'batch_size': batch_size,
                'seed': seed,
                'device': str(device),
                'embeddings_dir': str(embeddings_dir)
            },
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
            'config_path': str(config_path),
            'evaluation_timestamp': json.loads(json.dumps(pd.Timestamp.now(), default=str))
        }
        
        # Save as JSON in the same directory as the model
        results_file = run_dir / f"{task}_split{split}_evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        logger.success(f"Saved evaluation results to {results_file}")
        
        # Create combined metrics visualization
        logger.info("Creating combined metrics visualization...")
        
        # Create combined plot for both test sets
        plot_dir = run_dir / "plots"
        
        success, plot_path = plot_combined_metrics_visualization(test_results, test_hard_results, plot_dir)
        if success:
            logger.success(f"Combined metrics plot saved to: {plot_path}")
        
        logger.success(f"Evaluation completed successfully for {model_dir}!")
        
        # End timing and save inference_time.json
        timer.end()
        timing_path = run_dir / "inference_time.json"
        timer.save_timing_data(timing_path)
        
        # Clean up GPU memory
        del alignment_trained, seq_embeddings, baseline_embeddings
        if alignment_no_learning is not None:
            del alignment_no_learning
        torch.cuda.empty_cache()
        
        return True, results_summary
        
    except Exception as e:
        logger.error(f"Failed to evaluate {model_dir}: {e}")
        return False, None



@hydra.main(version_base=None, config_path="configs", config_name="evaluate")
def main(cfg: DictConfig) -> None:
    from hydra.core.hydra_config import HydraConfig
    
    # Validate required parameters
    if cfg.model_path is None:
        logger.error("model_path is required. Specify it in config or override with model_path=<path>")
        return 1
    
    # Find all config files in the provided path
    model_dirs = find_all_configs(cfg.model_path)
    
    if not model_dirs:
        logger.error(f"No config.json files found in {cfg.model_path}")
        return 1
    
    logger.success(f"Found {len(model_dirs)} model directories to evaluate:")
    for model_dir in model_dirs:
        logger.info(f"  - {model_dir}")
    
    # Batch evaluation
    successful_evaluations = 0
    failed_evaluations = 0
    all_results = []
    
    for i, model_dir in enumerate(model_dirs, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating model {i}/{len(model_dirs)}: {model_dir}")
        logger.info(f"{'='*60}")
        
        success, results = evaluate_single_model(model_dir, cfg)
        
        if success:
            successful_evaluations += 1
            all_results.append((model_dir, results))
            logger.success(f"✓ Successfully evaluated {model_dir}")
        else:
            failed_evaluations += 1
            logger.error(f"✗ Failed to evaluate {model_dir}")
        
        # Clear GPU memory between evaluations
        torch.cuda.empty_cache()
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("BATCH EVALUATION SUMMARY")
    logger.info(f"{'='*60}")
    logger.success(f"Successfully evaluated: {successful_evaluations}/{len(model_dirs)} models")
    if failed_evaluations > 0:
        logger.error(f"Failed evaluations: {failed_evaluations}/{len(model_dirs)} models")
    
    # Create summary table
    if all_results:
        logger.info("\n=== SUMMARY TABLE ===")
        print(f"{'Model Directory':<50} {'Test ROCAUC':<12} {'Test Hard ROCAUC':<15} {'Test F1':<10} {'Test Hard F1':<13}")
        print("-" * 110)
        
        for model_dir, results in all_results:
            model_name = str(model_dir).split('/')[-1] if len(str(model_dir).split('/')) > 1 else str(model_dir)
            test_auc = results['test_frequent']['metrics']['rocauc']['alignment_trained']
            test_hard_auc = results['test_hard']['metrics']['rocauc']['alignment_trained']
            test_f1 = results['test_frequent']['metrics']['f1_max']['alignment_trained']
            test_hard_f1 = results['test_hard']['metrics']['f1_max']['alignment_trained']
            
            print(f"{model_name:<50} {test_auc:<12.4f} {test_hard_auc:<15.4f} {test_f1:<10.4f} {test_hard_f1:<13.4f}")
        
        # Best performers
        best_test_auc = max(all_results, key=lambda x: x[1]['test_frequent']['metrics']['rocauc']['alignment_trained'])
        best_test_hard_auc = max(all_results, key=lambda x: x[1]['test_hard']['metrics']['rocauc']['alignment_trained'])
        
        logger.info(f"\n=== BEST PERFORMERS ===")
        logger.info(f"Best Test ROCAUC: {best_test_auc[0]} ({best_test_auc[1]['test_frequent']['metrics']['rocauc']['alignment_trained']:.4f})")
        logger.info(f"Best Test Hard ROCAUC: {best_test_hard_auc[0]} ({best_test_hard_auc[1]['test_hard']['metrics']['rocauc']['alignment_trained']:.4f})")
    
    return 0 if failed_evaluations == 0 else 1


if __name__ == "__main__":
    main()
