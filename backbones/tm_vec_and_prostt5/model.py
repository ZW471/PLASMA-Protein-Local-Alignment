import gc
import os
import sys
sys.path.append(os.getcwd())
import torch
import argparse
from transformers import T5EncoderModel, T5Tokenizer
from tqdm import tqdm
import pandas as pd
from typing import Dict, Any, List
import urllib.request
from pathlib import Path

from .embed_structure_model import (
    trans_basic_block,
    trans_basic_block_Config
)
from .tm_vec_utils import (
    featurize_prottrans,
    embed_tm_vec, encode
)
from ..base import BaseBackboneModel


def download_file_if_not_exists(url: str, destination: str) -> None:
    """
    Download a file from a URL if it doesn't already exist at the destination.

    Args:
        url: URL to download from
        destination: Path where the file should be saved
    """
    dest_path = Path(destination)

    if dest_path.exists():
        print(f"File already exists at {destination}, skipping download.")
        return

    # Create parent directory if it doesn't exist
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {url} to {destination}...")
    try:
        urllib.request.urlretrieve(url, destination)
        print(f"Successfully downloaded to {destination}")
    except Exception as e:
        raise RuntimeError(f"Failed to download {url}: {str(e)}")


class TMVecAndProstT5Model(BaseBackboneModel):
    """
    Combined TM-Vec and ProstT5 backbone model for protein embeddings.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

        # Load ProstT5 model
        prostt5_path = config['model_paths']['prostt5_model']
        self.tokenizer = T5Tokenizer.from_pretrained(prostt5_path, do_lower_case=False, cache_dir='weights/ProstT5')
        self.prostt5_model = T5EncoderModel.from_pretrained(prostt5_path, cache_dir='weights/ProstT5', use_safetensors=True)
        self.prostt5_model = self.prostt5_model.to(self.device)
        self.prostt5_model.eval()

        # Download TM-Vec files if they don't exist
        tm_vec_checkpoint = config['model_paths']['tm_vec_checkpoint']
        tm_vec_config_path = config['model_paths']['tm_vec_config']

        # URLs for TM-Vec files
        TM_VEC_CONFIG_URL = "https://figshare.com/ndownloader/files/46296310"
        TM_VEC_CHECKPOINT_URL = "https://figshare.com/ndownloader/files/46296322"

        # Download files if they don't exist
        download_file_if_not_exists(TM_VEC_CONFIG_URL, tm_vec_config_path)
        download_file_if_not_exists(TM_VEC_CHECKPOINT_URL, tm_vec_checkpoint)

        # Load TM-Vec model
        tm_vec_config = trans_basic_block_Config.from_json(tm_vec_config_path)
        self.tm_vec_model = trans_basic_block.load_from_checkpoint(
            tm_vec_checkpoint, config=tm_vec_config
        )
        self.tm_vec_model = self.tm_vec_model.to(self.device)
        self.tm_vec_model.eval()
    
    def forward(self, sequences: List[str]) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Generate embeddings for protein sequences.
        
        Args:
            sequences: List of protein sequences
            
        Returns:
            Dictionary with ProstT5 and TM-Vec embeddings at AA and PR levels
        """
        results = {}
        
        with torch.no_grad():
            for sequence in sequences:
                # Get ProstT5 embeddings (residue-level)
                protrans_sequence = featurize_prottrans(
                    [sequence], self.prostt5_model, self.tokenizer, self.device
                )
                prostt5_aa = protrans_sequence.squeeze().detach().cpu()
                prostt5_pr = protrans_sequence.mean(dim=1).squeeze().detach().cpu()
                
                # Get TM-Vec embeddings
                tm_vec_aa, tm_vec_pr = embed_tm_vec(
                    protrans_sequence, self.tm_vec_model, self.device
                )
                tm_vec_aa = tm_vec_aa.squeeze().detach().cpu()
                tm_vec_pr = tm_vec_pr.squeeze().detach().cpu()
                
                # Clean up GPU memory
                del protrans_sequence
                torch.cuda.empty_cache()
        
        return {
            "ProstT5": {
                "AA": prostt5_aa,
                "PR": prostt5_pr
            },
            "TM-Vec": {
                "AA": tm_vec_aa,
                "PR": tm_vec_pr
            }
        }

def load_cath_data_from_csv(file_path):
    """Load CATH data which only supports sequences (no fragments) and uses different column names."""
    import csv
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'File {file_path} not found.')
        
    data = []
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append({
                'name': row['uid'],
                'sequence': row['seq_full']
            })
    return data
