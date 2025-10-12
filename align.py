"""
Protein sequence alignment visualization script.
Generates alignment matrix visualization and summary for two protein sequences.
"""

import torch
import pandas as pd
import json
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from loguru import logger
import hydra
from omegaconf import DictConfig, ListConfig
from utils import setup_logging
from utils.alignment_utils import (
    alignment_score, 
    load_sequence_embedding, 
    setup_model, 
    compute_alignment_score_single
)
import numpy as np
from typing import Union, List

# EBA imports
try:
    from EBA.eba import methods as eba_methods
    from EBA.eba import score_matrices as eba_sm
    EBA_AVAILABLE = True
except ImportError:
    EBA_AVAILABLE = False


# load_sequence_embedding is now imported from utils.alignment_utils


def load_datasets(dataset_paths: Union[str, List[str]], plasma_dir: Path) -> pd.DataFrame:
    """Load and concatenate multiple CSV datasets."""
    
    # Convert single path to list
    if isinstance(dataset_paths, str):
        dataset_paths = [dataset_paths]
    elif isinstance(dataset_paths, ListConfig):
        dataset_paths = list(dataset_paths)
    
    dataframes = []
    
    for dataset_path in dataset_paths:
        # Handle both absolute and relative paths
        if Path(dataset_path).is_absolute():
            full_path = Path(dataset_path)
        else:
            # Try relative to plasma_dir/data/raw first
            full_path = plasma_dir / "data" / "raw" / dataset_path
            if not full_path.exists():
                # Try relative to plasma_dir
                full_path = plasma_dir / dataset_path
        
        if not full_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path} (tried {full_path})")
        
        logger.info(f"Loading dataset: {full_path}")
        df = pd.read_csv(full_path)
        dataframes.append(df)
    
    # Concatenate all dataframes
    df_combined = pd.concat(dataframes, ignore_index=True)
    
    # Remove duplicates based on 'uid' column if it exists
    if 'uid' in df_combined.columns:
        original_len = len(df_combined)
        df_combined = df_combined.drop_duplicates(subset=['uid'], keep='first')
        duplicates_removed = original_len - len(df_combined)
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate entries based on 'uid'")
    
    logger.info(f"Combined dataset loaded with {len(df_combined)} total sequences from {len(dataset_paths)} files")
    
    return df_combined


def get_sequence_info(seq_id: str, df_full: pd.DataFrame) -> dict:
    """Get sequence information from the full dataset."""
    seq_info = df_full[df_full['uid'] == seq_id]
    if seq_info.empty:
        raise ValueError(f"Sequence ID {seq_id} not found in dataset")
    
    return {
        'uid': seq_id,
        'interpro_id': seq_info['interpro_id'].iloc[0] if 'interpro_id' in seq_info.columns else None,
        'seq_full': seq_info['seq_full'].iloc[0] if 'seq_full' in seq_info.columns else None,
        'length': len(seq_info['seq_full'].iloc[0]) if 'seq_full' in seq_info.columns else None,
        'label': seq_info['label'].iloc[0] if 'label' in seq_info.columns else None
    }


def create_alignment_visualization(alignment_matrix: torch.Tensor, 
                                 query_info: dict, 
                                 candidate_info: dict,
                                 alignment_score_val: float,
                                 output_path: Path,
                                 use_eba: bool = False,
                                 font_size: int = 12,
                                 width: int = 800,
                                 height: int = 600) -> tuple:
    """Create and save alignment matrix visualization as HTML and PNG."""
    
    # Convert to numpy for visualization
    if hasattr(alignment_matrix, 'detach'):
        M_np = alignment_matrix.detach().cpu().numpy()
    else:
        M_np = alignment_matrix.cpu().numpy() if hasattr(alignment_matrix, 'cpu') else alignment_matrix
    
    # Create figure
    fig = go.Figure()
    
    # Add main alignment heatmap
    heatmap_kwargs = {
        'z': M_np,
        'colorscale': "Plasma",
        'colorbar': dict(title=""),
        'showscale': True
    }
    
    # Only set zmin/zmax for non-EBA matrices
    if not use_eba:
        heatmap_kwargs.update({'zmin': 0, 'zmax': 1})
    
    fig.add_trace(go.Heatmap(**heatmap_kwargs))
    
    # Update layout with configurable font size and dimensions
    fig.update_layout(
        title=f"Score: {alignment_score_val:.4f}",
        xaxis_title=f"Candidate ({candidate_info['uid']}) - Length: {candidate_info['length']}",
        yaxis_title=f"Query ({query_info['uid']}) - Length: {query_info['length']}",
        font=dict(size=font_size),
        width=width,
        height=height
    )
    
    # Save as HTML
    output_file_html = output_path / f"alignment_{query_info['uid']}_{candidate_info['uid']}.html"
    fig.write_html(output_file_html)
    
    # Save as PNG with explicit width, height, and font size
    output_file_png = output_path / f"alignment_{query_info['uid']}_{candidate_info['uid']}.png"
    fig.write_image(output_file_png, width=width, height=height)
    
    logger.success(f"Alignment visualization saved to: {output_file_html}")
    logger.success(f"Alignment visualization saved to: {output_file_png}")
    return str(output_file_html), str(output_file_png)


def compute_reduced_alignment_matrix(alignment_matrix: torch.Tensor) -> tuple:
    """Compute reduced alignment matrices by taking row and column maxima."""
    
    # Row max (for query): max alignment score for each query position
    query_reduced = alignment_matrix.max(dim=1).values.cpu().numpy().tolist()
    
    # Column max (for candidate): max alignment score for each candidate position  
    candidate_reduced = alignment_matrix.max(dim=0).values.cpu().numpy().tolist()
    
    return query_reduced, candidate_reduced


@hydra.main(version_base=None, config_path="configs", config_name="align")
def main(cfg: DictConfig) -> None:
    """Main alignment function."""
    
    # Validate required parameters
    if not cfg.query_id:
        raise ValueError("query_id is required")
    if not cfg.candidate_id:
        raise ValueError("candidate_id is required")
    if cfg.model.type == "plasma" and not cfg.model.model_cfg.model_path:
        raise ValueError("model.model_cfg.model_path is required")
    
    # Setup paths
    plasma_dir = Path(__file__).parent
    
    # Use Hydra output directory if output_dir is null
    if cfg.output_dir is None:
        from hydra.core.hydra_config import HydraConfig
        if HydraConfig.initialized():
            hydra_cfg = HydraConfig.get()
            output_dir = Path(hydra_cfg.runtime.output_dir)
        else:
            output_dir = Path("./alignment_results")
    else:
        output_dir = Path(cfg.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(output_dir, f"align_{cfg.query_id}_{cfg.candidate_id}")
    
    # Setup device
    if cfg.get('device', 'auto') == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(cfg.device)
    
    logger.info(f"Using device: {device}")
    logger.info(f"Query ID: {cfg.query_id}")
    logger.info(f"Candidate ID: {cfg.candidate_id}")
    logger.info(f"Model type: {cfg.model.type}")
    logger.info(f"Output directory: {output_dir}")
    
    # Setup model
    model, backbone_model = setup_model(cfg, device)
    
    # Determine embeddings directory and load datasets
    if cfg.model.type.lower() == "plasma":
        embeddings_dir = plasma_dir / "data" / "embeddings" / backbone_model / "AA_embeddings"
    elif cfg.model.type.lower() == "plasma-pf":
        embeddings_dir = plasma_dir / "data" / "embeddings" / backbone_model / "AA_embeddings"
    else:
        # For backbone models
        embeddings_dir = plasma_dir / "data" / "embeddings" / backbone_model / "PR_embeddings"
    
    # Load dataset(s) for sequence information
    if cfg.get('dataset_paths'):
        # Use custom dataset paths (can be multiple)
        df_full = load_datasets(cfg.dataset_paths, plasma_dir)
    elif cfg.model.type.lower() == "plasma":
        # For plasma models, get dataset from config.json
        model_path = Path(cfg.model.model_cfg.model_path)
        config_path = model_path / "config.json"
        with open(config_path, 'r') as f:
            model_config = json.load(f)
        dataset = model_config.get('model_info', {}).get('task', 'motif')
        df_full = load_datasets(f"{dataset}.csv", plasma_dir)
    else:
        # Use task from config for plasma-pf and backbone
        dataset = cfg.get('task', 'motif')
        df_full = load_datasets(f"{dataset}.csv", plasma_dir)
    
    # Get sequence information
    query_info = get_sequence_info(cfg.query_id, df_full)
    candidate_info = get_sequence_info(cfg.candidate_id, df_full)
    
    logger.info(f"Query: {query_info['uid']} (length: {query_info['length']})")
    logger.info(f"Candidate: {candidate_info['uid']} (length: {candidate_info['length']})")
    
    # Load embeddings
    if not embeddings_dir.exists():
        raise FileNotFoundError(f"Embeddings directory not found: {embeddings_dir}")
    
    logger.info(f"Loading embeddings from: {embeddings_dir}")
    
    query_emb = load_sequence_embedding(cfg.query_id, embeddings_dir, device)
    candidate_emb = load_sequence_embedding(cfg.candidate_id, embeddings_dir, device)
    
    logger.info(f"Query embedding shape: {query_emb.shape}")
    logger.info(f"Candidate embedding shape: {candidate_emb.shape}")
    
    # Check if using EBA alignment
    use_eba = hasattr(cfg, 'alignment_type') and cfg.alignment_type == 'eba'
    
    if use_eba:
        if not EBA_AVAILABLE:
            raise ImportError("EBA library not available. Please install EBA to use EBA alignment.")
        
        logger.info("Computing EBA alignment matrix...")
        # Use EBA for alignment
        similarity_matrix = eba_sm.compute_similarity_matrix(query_emb, candidate_emb)
        eba_results = eba_methods.compute_eba(similarity_matrix)
        
        # Select the appropriate EBA score based on config
        if cfg.eba.score == 'raw':
            score = eba_results['EBA_raw']
        elif cfg.eba.score == 'max':
            score = eba_results['EBA_max']
        elif cfg.eba.score == 'min':
            score = eba_results['EBA_min']
        else:
            raise ValueError(f'Invalid EBA score type: {cfg.eba.score}')
        
        alignment_matrix = similarity_matrix
        logger.info(f"Using EBA alignment with score type: {cfg.eba.score}")
    else:
        # Compute alignment matrix and score using standard method
        logger.info("Computing alignment matrix...")
        score, alignment_matrix = compute_alignment_score_single(
            query_emb, candidate_emb, model, cfg.model.type, device
        )
    
    logger.info(f"Alignment matrix shape: {alignment_matrix.shape}")
    logger.info(f"Alignment score: {score:.4f}")
    
    # Create visualization
    logger.info("Creating alignment visualization...")
    viz_config = cfg.get('visualization', {})
    font_size = viz_config.get('font_size', 12)
    width = viz_config.get('width', 800)
    height = viz_config.get('height', 600)
    html_path, png_path = create_alignment_visualization(
        alignment_matrix, query_info, candidate_info, score, output_dir, use_eba, font_size, width, height
    )
    
    # Compute reduced alignment matrices
    query_reduced, candidate_reduced = compute_reduced_alignment_matrix(alignment_matrix)
    
    # Create alignment summary
    alignment_summary = {
        'query_id': cfg.query_id,
        'candidate_id': cfg.candidate_id,
        'alignment_score': score,
        'query_info': query_info,
        'candidate_info': candidate_info,
        'alignment_matrix_shape': list(alignment_matrix.shape),
        'reduced_alignment_matrix': {
            'query': query_reduced,
            'candidate': candidate_reduced
        },
        'model_info': {
            'model_type': cfg.model.type,
            'backbone_model': backbone_model
        },
        'visualization_paths': {
            'html': html_path,
            'png': png_path
        }
    }
    
    # Save alignment summary
    summary_path = output_dir / "alignment_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(alignment_summary, f, indent=2)
    
    logger.success(f"Alignment summary saved to: {summary_path}")
    
    # Display results
    print(f"\n=== Alignment Results ===")
    print(f"Query: {cfg.query_id} (length: {query_info['length']})")
    print(f"Candidate: {cfg.candidate_id} (length: {candidate_info['length']})")
    print(f"Alignment Score: {score:.4f}")
    print(f"Alignment Matrix Shape: {alignment_matrix.shape}")
    print(f"HTML Visualization: {html_path}")
    print(f"PNG Visualization: {png_path}")
    print(f"Summary: {summary_path}")
    
    return 0


if __name__ == "__main__":
    main()