"""
Visualization utilities for creating plots and charts from dataset analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from typing import List, Tuple, Dict
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats


def create_combined_distribution_plot(data: pd.DataFrame,
                                    sampled_positive_pairs: List[Tuple[int, int]], 
                                    sampled_negative_pairs: List[Tuple[int, int]], 
                                    output_path: Path) -> None:
    """Create and save combined distribution plots with 3 subplots."""
    logger.info("Creating combined distribution plots")
    
    # Get the distribution of InterPro IDs
    interpro_counts = data['interpro_id'].value_counts()
    
    # Analyze positive pair distribution
    positive_interpro_distribution = {}
    for idx1, idx2 in sampled_positive_pairs:
        interpro_id = data.loc[idx1, 'interpro_id']
        positive_interpro_distribution[interpro_id] = positive_interpro_distribution.get(interpro_id, 0) + 1
    
    # Analyze negative pair distribution by InterPro ID position
    negative_interpro_first_pos = {}
    negative_interpro_second_pos = {}
    
    for idx1, idx2 in sampled_negative_pairs:
        id1 = data.loc[idx1, 'interpro_id']
        id2 = data.loc[idx2, 'interpro_id']
        
        # Count InterPro IDs in first position
        negative_interpro_first_pos[id1] = negative_interpro_first_pos.get(id1, 0) + 1
        
        # Count InterPro IDs in second position
        negative_interpro_second_pos[id2] = negative_interpro_second_pos.get(id2, 0) + 1
    
    # Create subplot figure with 3 subplots
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            "InterPro ID Distribution (Original Dataset)",
            "Positive Pairs Distribution by InterPro ID", 
            "Negative Pairs Distribution by InterPro ID Position"
        ],
        specs=[
            [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}]
        ],
        horizontal_spacing=0.08
    )
    
    # 1. InterPro ID distribution
    fig.add_trace(
        go.Bar(
            x=interpro_counts.index[:20],  # Top 20 for readability
            y=interpro_counts.values[:20],
            name="InterPro Distribution",
            marker_color="steelblue",
            showlegend=False
        ),
        row=1, col=1
    )
    
    # 2. Positive pairs distribution
    if positive_interpro_distribution:
        pos_interpro_ids = list(positive_interpro_distribution.keys())
        pos_counts = list(positive_interpro_distribution.values())
        fig.add_trace(
            go.Bar(
                x=pos_interpro_ids,
                y=pos_counts,
                name="Positive Pairs",
                marker_color="green",
                showlegend=False
            ),
            row=1, col=2
        )
    
    # 3. Negative pairs distribution
    all_interpro_ids = sorted(set(list(negative_interpro_first_pos.keys()) + list(negative_interpro_second_pos.keys())))
    first_pos_counts = [negative_interpro_first_pos.get(iid, 0) for iid in all_interpro_ids]
    second_pos_counts = [negative_interpro_second_pos.get(iid, 0) for iid in all_interpro_ids]
    
    fig.add_trace(
        go.Bar(
            x=all_interpro_ids,
            y=first_pos_counts,
            name='First Position',
            marker_color='lightblue'
        ),
        row=1, col=3
    )
    
    fig.add_trace(
        go.Bar(
            x=all_interpro_ids,
            y=second_pos_counts,
            name='Second Position',
            marker_color='lightcoral'
        ),
        row=1, col=3
    )
    
    # Update layout
    fig.update_layout(
        title_text="Dataset Distribution Analysis",
        title_x=0.5,
        height=500,
        showlegend=True,
        template='plotly_white',
        barmode='group'  # This applies to the third subplot
    )
    
    # Update axis labels
    fig.update_xaxes(title_text="InterPro ID (Top 20)", row=1, col=1, tickangle=45)
    fig.update_yaxes(title_text="Number of Sequences", row=1, col=1)
    
    fig.update_xaxes(title_text="InterPro ID", row=1, col=2, tickangle=45)
    fig.update_yaxes(title_text="Number of Positive Pairs", row=1, col=2)
    
    fig.update_xaxes(title_text="InterPro ID", row=1, col=3, tickangle=45)
    fig.update_yaxes(title_text="Number of Negative Pairs", row=1, col=3)
    
    # Save the plot
    fig.write_html(output_path / "combined_distribution_analysis.html")
    logger.info(f"Saved combined_distribution_analysis.html")
    
    # Log summary statistics
    logger.info(f"Total InterPro IDs: {len(interpro_counts)}")
    logger.info(f"Mean sequences per InterPro ID: {interpro_counts.mean():.2f}")
    logger.info(f"Median sequences per InterPro ID: {interpro_counts.median():.2f}")
    logger.info(f"Max sequences per InterPro ID: {interpro_counts.max()}")
    logger.info(f"Min sequences per InterPro ID: {interpro_counts.min()}")


def create_sequence_length_analysis(data: pd.DataFrame,
                                  train_pairs: List[Tuple[Tuple[int, int], int]],
                                  val_pairs: List[Tuple[Tuple[int, int], int]],
                                  test_pairs: List[Tuple[Tuple[int, int], int]],
                                  test_hard_pairs: List[Tuple[Tuple[int, int], int]],
                                  output_path: Path) -> None:
    """Create comprehensive sequence length distribution analysis."""
    logger.info("Creating sequence length distribution analysis")
    
    # Calculate sequence lengths for original dataset
    original_seq_lengths = data['seq_full'].str.len()
    
    # Helper functions
    def get_pair_seq_lengths(pairs_list, label_name):
        """Extract query and candidate sequence lengths from pairs"""
        query_lengths = []
        candidate_lengths = []
        
        for pair, label in pairs_list:
            idx1, idx2 = pair
            
            # Get sequence lengths
            query_len = len(data.loc[idx1, 'seq_full'])
            candidate_len = len(data.loc[idx2, 'seq_full'])
            
            query_lengths.append(query_len)
            candidate_lengths.append(candidate_len)
        
        return query_lengths, candidate_lengths
    
    def get_single_seq_lengths_from_pairs(pairs_list, label_name):
        """Extract all sequence lengths from pairs (for histograms)"""
        seq_lengths = []
        
        for pair, label in pairs_list:
            idx1, idx2 = pair
            
            # Get sequence lengths
            seq1_len = len(data.loc[idx1, 'seq_full'])
            seq2_len = len(data.loc[idx2, 'seq_full'])
            
            seq_lengths.extend([seq1_len, seq2_len])
        
        return seq_lengths
    
    # Get sequence lengths for each split
    train_pos_pairs = [(pair, 1) for pair, label in train_pairs if label == 1]
    train_neg_pairs = [(pair, 0) for pair, label in train_pairs if label == 0]
    val_pos_pairs = [(pair, 1) for pair, label in val_pairs if label == 1]
    val_neg_pairs = [(pair, 0) for pair, label in val_pairs if label == 0]
    test_pos_pairs = [(pair, 1) for pair, label in test_pairs if label == 1]
    test_neg_pairs = [(pair, 0) for pair, label in test_pairs if label == 0]
    test_hard_pos_pairs = [(pair, 1) for pair, label in test_hard_pairs if label == 1]
    test_hard_neg_pairs = [(pair, 0) for pair, label in test_hard_pairs if label == 0]
    
    # Extract sequence lengths for histograms (positive pairs)
    train_pos_lengths = get_single_seq_lengths_from_pairs(train_pos_pairs, "Train Positive")
    val_pos_lengths = get_single_seq_lengths_from_pairs(val_pos_pairs, "Val Positive")
    test_pos_lengths = get_single_seq_lengths_from_pairs(test_pos_pairs, "Test Positive")
    test_hard_pos_lengths = get_single_seq_lengths_from_pairs(test_hard_pos_pairs, "Test Hard Positive")
    
    # Extract query vs candidate lengths for scatter plots (negative pairs)
    train_neg_query, train_neg_candidate = get_pair_seq_lengths(train_neg_pairs, "Train")
    val_neg_query, val_neg_candidate = get_pair_seq_lengths(val_neg_pairs, "Val")
    test_neg_query, test_neg_candidate = get_pair_seq_lengths(test_neg_pairs, "Test")
    test_hard_neg_query, test_hard_neg_candidate = get_pair_seq_lengths(test_hard_neg_pairs, "Test Hard")
    
    # Create subplot figure
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            "Original Dataset (Histogram)", "Positive Splits (Histograms)",
            "Negative Splits: Train & Val (Scatter)", "Negative Splits: Test & Test Hard (Scatter)",
            "Length Distribution Summary", "InterPro ID Frequency in Samples"
        ],
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # Helper function to create KDE curve
    def create_kde_curve(data_points, name, color):
        """Create KDE curve for given data points"""
        if len(data_points) > 1:
            # Create KDE
            kde = stats.gaussian_kde(data_points)
            x_range = np.linspace(min(data_points), max(data_points), 100)
            kde_values = kde(x_range)
            return go.Scatter(
                x=x_range,
                y=kde_values,
                mode='lines',
                name=f"{name} KDE",
                line=dict(color=color, width=2)
            )
        return None
    
    # 1. Original dataset histogram with KDE
    fig.add_trace(
        go.Histogram(
            x=original_seq_lengths,
            nbinsx=50,
            name="Original Dataset",
            marker_color="lightblue",
            opacity=0.7,
            histnorm='probability density'
        ),
        row=1, col=1
    )
    
    # Add KDE for original dataset
    kde_original = create_kde_curve(original_seq_lengths.tolist(), "Original", "blue")
    if kde_original:
        fig.add_trace(kde_original, row=1, col=1)
    
    # 2. Positive splits histograms (overlaid) with KDE
    fig.add_trace(
        go.Histogram(
            x=train_pos_lengths,
            nbinsx=30,
            name="Train Positive",
            marker_color="green",
            opacity=0.6,
            histnorm='probability density'
        ),
        row=1, col=2
    )
    
    kde_train = create_kde_curve(train_pos_lengths, "Train", "darkgreen")
    if kde_train:
        fig.add_trace(kde_train, row=1, col=2)
    
    fig.add_trace(
        go.Histogram(
            x=val_pos_lengths,
            nbinsx=30,
            name="Val Positive",
            marker_color="orange",
            opacity=0.6,
            histnorm='probability density'
        ),
        row=1, col=2
    )
    
    kde_val = create_kde_curve(val_pos_lengths, "Val", "darkorange")
    if kde_val:
        fig.add_trace(kde_val, row=1, col=2)
    
    fig.add_trace(
        go.Histogram(
            x=test_pos_lengths,
            nbinsx=30,
            name="Test Positive",
            marker_color="red",
            opacity=0.6,
            histnorm='probability density'
        ),
        row=1, col=2
    )
    
    kde_test = create_kde_curve(test_pos_lengths, "Test", "darkred")
    if kde_test:
        fig.add_trace(kde_test, row=1, col=2)
    
    fig.add_trace(
        go.Histogram(
            x=test_hard_pos_lengths,
            nbinsx=30,
            name="Test Hard Positive",
            marker_color="purple",
            opacity=0.6,
            histnorm='probability density'
        ),
        row=1, col=2
    )
    
    kde_test_hard = create_kde_curve(test_hard_pos_lengths, "Test Hard", "darkviolet")
    if kde_test_hard:
        fig.add_trace(kde_test_hard, row=1, col=2)
    
    # 3. Negative splits scatter plots: Train & Val (Query vs Candidate lengths)
    fig.add_trace(
        go.Scatter(
            x=train_neg_candidate,
            y=train_neg_query,
            mode='markers',
            marker=dict(
                color="blue",
                size=4,
                opacity=0.6
            ),
            name="Train Negative",
            text=[f"Train<br>Query: {q}<br>Candidate: {c}" 
                  for q, c in zip(train_neg_query, train_neg_candidate)],
            hovertemplate="%{text}<extra></extra>"
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=val_neg_candidate,
            y=val_neg_query,
            mode='markers',
            marker=dict(
                color="orange",
                size=4,
                opacity=0.6
            ),
            name="Val Negative",
            text=[f"Validation<br>Query: {q}<br>Candidate: {c}" 
                  for q, c in zip(val_neg_query, val_neg_candidate)],
            hovertemplate="%{text}<extra></extra>"
        ),
        row=2, col=1
    )
    
    # 4. Negative splits scatter plots: Test & Test Hard (Query vs Candidate lengths)
    fig.add_trace(
        go.Scatter(
            x=test_neg_candidate,
            y=test_neg_query,
            mode='markers',
            marker=dict(
                color="red",
                size=4,
                opacity=0.6
            ),
            name="Test Negative",
            text=[f"Test<br>Query: {q}<br>Candidate: {c}" 
                  for q, c in zip(test_neg_query, test_neg_candidate)],
            hovertemplate="%{text}<extra></extra>"
        ),
        row=2, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=test_hard_neg_candidate,
            y=test_hard_neg_query,
            mode='markers',
            marker=dict(
                color="purple",
                size=4,
                opacity=0.6
            ),
            name="Test Hard Negative",
            text=[f"Test Hard<br>Query: {q}<br>Candidate: {c}" 
                  for q, c in zip(test_hard_neg_query, test_hard_neg_candidate)],
            hovertemplate="%{text}<extra></extra>"
        ),
        row=2, col=2
    )
    
    # 5. Length distribution summary (violin plots)
    all_splits_data = [
        train_pos_lengths + train_neg_query + train_neg_candidate,
        val_pos_lengths + val_neg_query + val_neg_candidate,
        test_pos_lengths + test_neg_query + test_neg_candidate,
        test_hard_pos_lengths + test_hard_neg_query + test_hard_neg_candidate
    ]
    split_names = ["Train", "Validation", "Test", "Test Hard"]
    
    for i, (data_split, name) in enumerate(zip(all_splits_data, split_names)):
        fig.add_trace(
            go.Violin(
                y=data_split,
                name=name,
                box_visible=True,
                meanline_visible=True
            ),
            row=3, col=1
        )
    
    # 6. InterPro ID frequency in samples
    # Count InterPro ID frequencies across all samples
    interpro_freq = {}
    all_pair_lists = [train_pos_pairs, train_neg_pairs, val_pos_pairs, val_neg_pairs,
                     test_pos_pairs, test_neg_pairs, test_hard_pos_pairs, test_hard_neg_pairs]
    
    for pair_list in all_pair_lists:
        for pair, label in pair_list:
            idx1, idx2 = pair
            interpro1 = data.loc[idx1, 'interpro_id']
            interpro2 = data.loc[idx2, 'interpro_id']
            interpro_freq[interpro1] = interpro_freq.get(interpro1, 0) + 1
            interpro_freq[interpro2] = interpro_freq.get(interpro2, 0) + 1
    
    # Sort by frequency and take top 20
    top_interpros = sorted(interpro_freq.items(), key=lambda x: x[1], reverse=True)[:20]
    interpro_names, interpro_counts = zip(*top_interpros) if top_interpros else ([], [])
    
    if top_interpros:
        fig.add_trace(
            go.Bar(
                x=list(interpro_names),
                y=list(interpro_counts),
                name="InterPro Frequency",
                marker_color="lightgreen",
                showlegend=False
            ),
            row=3, col=2
        )
    
    # Update layout
    fig.update_layout(
        title_text="Sequence Length Distribution Analysis: Original vs Sampled Data (Max Length: 1024)",
        title_x=0.5,
        height=1200,
        showlegend=True,
        template='plotly_white'
    )
    
    # Update axis labels
    fig.update_xaxes(title_text="Sequence Length", row=1, col=1)
    fig.update_yaxes(title_text="Probability Density", row=1, col=1)
    
    fig.update_xaxes(title_text="Sequence Length", row=1, col=2)
    fig.update_yaxes(title_text="Probability Density", row=1, col=2)
    
    fig.update_xaxes(title_text="Candidate Sequence Length", row=2, col=1)
    fig.update_yaxes(title_text="Query Sequence Length", row=2, col=1)
    
    fig.update_xaxes(title_text="Candidate Sequence Length", row=2, col=2)
    fig.update_yaxes(title_text="Query Sequence Length", row=2, col=2)
    
    fig.update_xaxes(title_text="Split", row=3, col=1)
    fig.update_yaxes(title_text="Sequence Length", row=3, col=1)
    
    fig.update_xaxes(title_text="InterPro ID (Top 20)", row=3, col=2)
    fig.update_yaxes(title_text="Frequency in Samples", row=3, col=2)
    
    # Rotate x-axis labels for better readability
    fig.update_xaxes(tickangle=45, row=3, col=2)
    
    # Add diagonal reference lines to scatter plots
    if train_neg_query and train_neg_candidate:
        # Combine all lengths to find the maximum
        all_lengths = (train_neg_query + train_neg_candidate + val_neg_query + val_neg_candidate +
                       test_neg_query + test_neg_candidate + test_hard_neg_query + test_hard_neg_candidate)
        max_len = max(all_lengths)
        
        # Add diagonal line to first scatter plot
        fig.add_shape(
            type="line",
            x0=0, y0=0, x1=max_len, y1=max_len,
            line=dict(color="gray", width=1, dash="dash"),
            row=2, col=1
        )
        
        # Add diagonal line to second scatter plot
        fig.add_shape(
            type="line",
            x0=0, y0=0, x1=max_len, y1=max_len,
            line=dict(color="gray", width=1, dash="dash"),
            row=2, col=2
        )
    
    # Save the plot
    fig.write_html(output_path / "sequence_length_distribution_analysis.html")
    logger.info(f"Saved sequence_length_distribution_analysis.html")
