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

from .embed_structure_model import (
    trans_basic_block, 
    trans_basic_block_Config
)
from .tm_vec_utils import (
    featurize_prottrans, 
    embed_tm_vec, encode
)
from ..base import BaseBackboneModel


class TMVecAndProstT5Model(BaseBackboneModel):
    """
    Combined TM-Vec and ProstT5 backbone model for protein embeddings.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load ProstT5 model
        prostt5_path = config['model_paths']['prostt5_model']
        self.tokenizer = T5Tokenizer.from_pretrained(prostt5_path)
        self.prostt5_model = T5EncoderModel.from_pretrained(prostt5_path)
        self.prostt5_model = self.prostt5_model.to(self.device)
        self.prostt5_model.eval()
        
        # Load TM-Vec model
        tm_vec_checkpoint = config['model_paths']['tm_vec_checkpoint']
        tm_vec_config_path = config['model_paths']['tm_vec_config']
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

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_folder', type=str, default='../data')
    parser.add_argument('--model_dir', type=str, default='../weights/tm-vec')
    parser.add_argument('--out_folder', type=str, default='../outputs')
    parser.add_argument('--out_batch_size', type=int, default=100, help='Batch size for processing sequences')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files if they exist')
    args = parser.parse_args()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    print('>>> Output folder:', args.out_folder)
    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)
    
    # Create parent folder named as task
    task_folder = os.path.join(args.out_folder)
    if not os.path.exists(task_folder):
        os.makedirs(task_folder)
    
    # Create separate folders for each data type
    protrans_pr_folder = os.path.join(task_folder, 'ProstT5_PR_embeddings')
    protrans_res_folder = os.path.join(task_folder, 'ProstT5_AA_embeddings')
    tm_vec_pr_folder = os.path.join(task_folder, 'TM-Vec_PR_embeddings')
    tm_vec_res_folder = os.path.join(task_folder, 'TM-Vec_AA_embeddings')
    
    for folder in [protrans_pr_folder, protrans_res_folder, tm_vec_pr_folder, tm_vec_res_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    tokenizer = T5Tokenizer.from_pretrained("/home/drizer/PycharmProjects/VenusX/weights/tm-vec/models--Rostlab--ProstT5_fp16/snapshots/07a6547d51de603f1be84fd9f2db4680ee535a86")
    model = T5EncoderModel.from_pretrained("/home/drizer/PycharmProjects/VenusX/weights/tm-vec/models--Rostlab--ProstT5_fp16/snapshots/07a6547d51de603f1be84fd9f2db4680ee535a86")

    model = model.to(device)
    model = model.eval()
    print('>>> T5 Model loaded')
    
    #TM-Vec model paths
    tm_vec_model_cpnt = f"{args.model_dir}/tm_vec_cath_model.ckpt"
    tm_vec_model_config = f"{args.model_dir}/tm_vec_cath_model_params.json"
    #Load the TM-Vec model
    tm_vec_model_config = trans_basic_block_Config.from_json(tm_vec_model_config)
    model_deep = trans_basic_block.load_from_checkpoint(tm_vec_model_cpnt, config=tm_vec_model_config)
    model_deep = model_deep.to(device)
    model_deep = model_deep.eval()
    print('>>> tmvec Model loaded')
    
    # Load data using pandas and filter directly
    data_file_path = os.path.join(args.dataset_folder, "full.csv")
    df = pd.read_csv(data_file_path)
    
    print('>>> Processing sequences only')
    
    # Use seq_full column for all dataset types
    sequence_col = 'seq_full'
    name_col = 'uid'
    
    # Filter out sequences longer than 1024 residues
    print(f'>>> Total sequences before filtering: {len(df)}')
    df['sequence_length'] = df[sequence_col].str.len()
    df_filtered = df[df['sequence_length'] <= 1024].copy()
    df_filtered = df_filtered.drop('sequence_length', axis=1)
    
    print(f'>>> Total sequences after filtering: {len(df_filtered)}')
    print(f'>>> Filtered out {len(df) - len(df_filtered)} sequences longer than 1024 residues')

    # Save filtered data to CSV file in task folder maintaining original format
    filtered_csv_path = os.path.join(task_folder, 'data.csv')
    df_filtered.to_csv(filtered_csv_path, index=False)
    print(f'>>> Filtered data saved to: {filtered_csv_path}')
    
    # Check what files already exist in ALL 4 folders
    folder_files = []
    for folder in [protrans_pr_folder, protrans_res_folder, tm_vec_pr_folder, tm_vec_res_folder]:
        if os.path.exists(folder):
            files_in_folder = {filename[:-3] for filename in os.listdir(folder) if filename.endswith('.pt')}
            folder_files.append(files_in_folder)
        else:
            folder_files.append(set())
    
    # Only include files that exist in ALL 4 folders
    existing_files = set.intersection(*folder_files) if folder_files else set()
    
    print(f'>>> Found {len(existing_files)} sequences with complete embeddings in all 4 folders')
    
    # Remove duplicated names and filter out existing files
    df_unique = df_filtered.drop_duplicates(subset=[name_col])
    print(f'>>> Removed {len(df_filtered) - len(df_unique)} duplicate names')
    
    # Convert to the format expected by the rest of the code
    seq_data = []
    for _, row in df_unique.iterrows():
        seq_name = row[name_col]
        if args.overwrite or seq_name not in existing_files:
            seq_data.append({
                'name': seq_name,
                'sequence': row[sequence_col]
            })
    
    print(f'>>> Will process {len(seq_data)} sequences (skipped {len(df_unique) - len(seq_data)} existing files)')


    with torch.no_grad():
        for idx, seq_info in enumerate(tqdm(seq_data)):
            
            sequence = seq_info['sequence']
            seq_name = seq_info['name']
            
            # Define file paths for all embedding types
            protrans_pr_file = os.path.join(protrans_pr_folder, f'{seq_name}.pt')
            protrans_res_file = os.path.join(protrans_res_folder, f'{seq_name}.pt')
            tm_vec_pr_file = os.path.join(tm_vec_pr_folder, f'{seq_name}.pt')
            tm_vec_res_file = os.path.join(tm_vec_res_folder, f'{seq_name}.pt')
            
            # this is the residue-level encoding
            protrans_sequence = featurize_prottrans([sequence], model, tokenizer, device)
            protrans_res_embedding = protrans_sequence.squeeze().detach().cpu()
            protrans_pr_embedding = protrans_sequence.mean(dim=1).squeeze().detach().cpu()
            
            # this is the protein-level encoding
            embedded_residues, embedded_sequence = embed_tm_vec(protrans_sequence, model_deep, device)
            tm_vec_res_embedding = embedded_residues.squeeze().detach().cpu()
            tm_vec_pr_embedding = embedded_sequence.squeeze().detach().cpu()
            
            # Save each embedding type to its respective folder
            torch.save(protrans_pr_embedding, protrans_pr_file)
            torch.save(protrans_res_embedding, protrans_res_file)
            torch.save(tm_vec_pr_embedding, tm_vec_pr_file)
            torch.save(tm_vec_res_embedding, tm_vec_res_file)
            
            del protrans_sequence, embedded_sequence, protrans_pr_embedding, protrans_res_embedding, tm_vec_res_embedding, tm_vec_pr_embedding
            torch.cuda.empty_cache()

    print('>>> All embeddings saved to individual files in separate folders')

