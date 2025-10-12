import re
import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any, List
from transformers import (
    BertModel,
    BertTokenizer,
    EsmModel,
    AutoTokenizer,
    T5Tokenizer, 
    T5EncoderModel
)

import pandas as pd
from ..base import BaseBackboneModel


class PLModels(BaseBackboneModel):
    """
    Protein Language Models backbone supporting ESM2, ProtBERT, ProtT5, and Ankh.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = config['model_name']
        self.model_type = config['model_type']
        self.cache_dir = config.get('cache_dir', f"weights/{self.model_name.split('/')[-1]}")
        self.max_length = config.get('max_length', 1024)
        
        # Load model and tokenizer
        self._load_model_and_tokenizer()
        
    def _load_model_and_tokenizer(self):
        """Load the appropriate model and tokenizer based on model_type."""
        if self.model_type == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir)
            self.model = BertModel.from_pretrained(self.model_name, cache_dir=self.cache_dir)
        elif self.model_type == 'esm':
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir)
            self.model = EsmModel.from_pretrained(self.model_name, cache_dir=self.cache_dir)
        elif self.model_type == 't5':
            self.tokenizer = T5Tokenizer.from_pretrained(
                self.model_name, do_lower_case=False, cache_dir=self.cache_dir
            )
            self.model = T5EncoderModel.from_pretrained(self.model_name, cache_dir=self.cache_dir)
        elif self.model_type == 'ankh':
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir)
            self.model = T5EncoderModel.from_pretrained(self.model_name, cache_dir=self.cache_dir)
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")
        
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def _preprocess_sequence(self, sequence: str) -> str:
        """Preprocess sequence based on model type."""
        if self.model_type == 'bert':
            return " ".join(sequence)
        elif self.model_type == 't5':
            return " ".join(list(re.sub(r"[UZOB]", "X", sequence)))
        else:
            return sequence
    
    def forward(self, sequences: List[str]) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Generate embeddings for protein sequences.
        
        Args:
            sequences: List of protein sequences
            
        Returns:
            Dictionary with model embeddings at AA and PR levels
        """
        model_key = self.model_name.split('/')[-1]  # Use model name as key
        
        aa_embeddings = []
        pr_embeddings = []
        
        with torch.no_grad():
            for sequence in sequences:
                # Preprocess sequence
                processed_seq = self._preprocess_sequence(sequence)
                
                # Tokenize
                inputs = self.tokenizer(
                    processed_seq,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=self.model_type == 't5',
                    max_length=self.max_length,
                    truncation=True
                )
                
                # Move to device
                input_ids = inputs["input_ids"].to(self.device)
                attention_mask = inputs["attention_mask"].to(self.device)
                
                # Get model outputs
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                features = outputs.last_hidden_state
                
                # For ProtT5 models, remove the EOS token (last token) from embeddings
                if self.model_type == 't5':
                    features = features[:, :-1, :]  # Remove last token (EOS)
                    attention_mask = attention_mask[:, :-1]  # Adjust attention mask accordingly
                
                # Mask features and calculate protein-level embedding
                masked_features = features * attention_mask.unsqueeze(2)
                sum_features = torch.sum(masked_features, dim=1)
                mean_pooled_features = sum_features / attention_mask.sum(dim=1, keepdim=True)
                
                # Store embeddings
                aa_embeddings.append(features.squeeze(0).detach().cpu())  # Remove batch dimension
                pr_embeddings.append(mean_pooled_features.squeeze(0).detach().cpu())  # Remove batch dimension
                
                # Clean up GPU memory
                del features, masked_features, sum_features, mean_pooled_features
                torch.cuda.empty_cache()
        
        # For single sequence, return without list wrapping
        if len(sequences) == 1:
            return {
                model_key: {
                    "AA": aa_embeddings[0],
                    "PR": pr_embeddings[0]
                }
            }
        
        # For multiple sequences, stack tensors
        return {
            model_key: {
                "AA": torch.stack(aa_embeddings),
                "PR": torch.stack(pr_embeddings)
            }
        }

def get_embedding(model_name,
                  model_type,
                  data,
                  batch_size,
                  out_file,
                  seq_fragment=False,
                  seq_type='motif'):

    if model_type == 'bert':
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)
    elif model_type == 'esm':
        model = EsmModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    elif model_type == 't5':
        tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
        model = T5EncoderModel.from_pretrained(model_name)
    elif model_type == 'ankh':
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = T5EncoderModel.from_pretrained(model_name)

    model.cuda()
    model.eval()

    def collate_fn(batch):
        if seq_fragment:
            sequences = [example["fragment"] for example in batch]
        else:
            sequences = [example["sequence"] for example in batch]
        if model_type == 'bert': sequences = [" ".join(seq) for seq in sequences]
        if model_type == 't5': sequences = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequences]
        names = [example["name"] for example in batch]

        if seq_fragment:
            if seq_type == 'motif': max_len = 128
            elif seq_type == 'active_site': max_len = 128
            elif seq_type == 'binding_site': max_len = 128
            elif seq_type == 'conserved_site': max_len = 128
            elif seq_type == 'domain': max_len = 512
            else: raise ValueError("Invalid seq_type")
        else: max_len = 1024
        results = tokenizer(sequences, return_tensors="pt", padding=True, add_special_tokens=model_type == 't5',
                            max_length=max_len, truncation=True)
        results["name"] = names
        results["sequence"] = sequences
        return results

    res_data = {}
    eval_loader = DataLoader(data, batch_size=batch_size,
                             shuffle=False, collate_fn=collate_fn, num_workers=12)

    with torch.no_grad():
        for batch in tqdm(eval_loader):
            if isinstance(batch["input_ids"], list):
                batch["input_ids"] = torch.stack(batch["input_ids"])
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            features = outputs.last_hidden_state
            
            # For ProtT5 models, remove the EOS token (last token) from embeddings
            if model_type == 't5':
                features = features[:, :-1, :]  # Remove last token (EOS)
                attention_mask = attention_mask[:, :-1]  # Adjust attention mask accordingly
            
            masked_features = features * attention_mask.unsqueeze(2)
            sum_features = torch.sum(masked_features, dim=1)
            mean_pooled_features = sum_features / attention_mask.sum(dim=1, keepdim=True)
            for name, feature in zip(batch["name"], mean_pooled_features):
                res_data[name] = feature.detach().cpu()
            torch.cuda.empty_cache()

    torch.save(res_data, out_file)
    
if __name__ == '__main__':

    
    parser = argparse.ArgumentParser()

    parser.add_argument('--seq_fragment', action='store_true', default=False)
    parser.add_argument('--seq_type', choices=['motif', 'domain', 'active_site', 'binding_site', 'conserved_site'])
    parser.add_argument('--dataset_file', type=str)
    parser.add_argument('--model_name', type=str, default='facebook/esm2_t33_650M_UR50D')
    parser.add_argument('--model_type', type=str, default='esm')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--out_file', type=str)
   
    args = parser.parse_args()
    
    out_dir = os.path.dirname(args.out_file)
    os.makedirs(out_dir, exist_ok=True)
    
    # Load data using pandas like in TM-Vec model
    df = pd.read_csv(args.dataset_file)
    
    # Convert to the format expected by get_embedding function
    data = []
    for _, row in df.iterrows():
        if args.seq_fragment:
            data.append({
                'name': row['uid'],
                'fragment': row['seq_fragment'] if 'seq_fragment' in row.columns else row['seq_full'],
                'sequence': row['seq_full']
            })
        else:
            data.append({
                'name': row['uid'],
                'sequence': row['seq_full']
            })

    get_embedding(
        args.model_name, 
        args.model_type, 
        data, 
        args.batch_size, 
        args.out_file, 
        args.seq_fragment, 
        args.seq_type)


    