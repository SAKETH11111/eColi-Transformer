#!/usr/bin/env python3
"""
Data retokenization script for converting datasets to tokenizer v2.

Usage:
    python tools/retokenize_dataset.py --input data/processed --output data_v2
"""

import argparse
import torch
from pathlib import Path
import sys
from typing import List, Dict, Any
import pandas as pd
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from ecoli_transformer.tokenizer_v2 import CodingTokenizerV2


def retokenize_fasta_file(fasta_path: Path, tokenizer: CodingTokenizerV2) -> List[Dict[str, Any]]:
    """
    Retokenize a FASTA file with the new tokenizer.
    
    Returns list of dictionaries with tokenized data.
    """
    sequences = []
    
    with open(fasta_path, 'r') as f:
        current_id = None
        current_seq = ""
        
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # Save previous sequence
                if current_id and current_seq:
                    try:
                        token_ids, codons = tokenizer.encode_cds(current_seq, add_special_tokens=True)
                        sequences.append({
                            'gene_id': current_id,
                            'sequence': current_seq,
                            'input_ids': torch.tensor(token_ids, dtype=torch.long),
                            'labels': torch.tensor(token_ids, dtype=torch.long),  # For MLM, labels = inputs
                            'organism_id': torch.tensor(0, dtype=torch.long),  # E. coli = 0
                            'length': len(current_seq)
                        })
                    except Exception as e:
                        print(f"Warning: Skipping sequence {current_id}: {e}")
                
                # Start new sequence
                current_id = line[1:].split()[0]  # Take first part of header
                current_seq = ""
            else:
                current_seq += line.upper()
        
        # Don't forget the last sequence
        if current_id and current_seq:
            try:
                token_ids, codons = tokenizer.encode_cds(current_seq, add_special_tokens=True)
                sequences.append({
                    'gene_id': current_id,
                    'sequence': current_seq,
                    'input_ids': torch.tensor(token_ids, dtype=torch.long),
                    'labels': torch.tensor(token_ids, dtype=torch.long),
                    'organism_id': torch.tensor(0, dtype=torch.long),
                    'length': len(current_seq)
                })
            except Exception as e:
                print(f"Warning: Skipping sequence {current_id}: {e}")
    
    return sequences


def retokenize_csv_with_metrics(csv_path: Path, tokenizer: CodingTokenizerV2) -> List[Dict[str, Any]]:
    """
    Retokenize CSV file that contains sequences with metrics (CAI, MFE, etc).
    """
    df = pd.read_csv(csv_path)
    sequences = []
    
    # Expected columns: sequence, cai, mfe, etc.
    required_cols = ['sequence']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV must contain columns: {required_cols}")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {csv_path.name}"):
        seq = str(row['sequence']).upper()
        
        try:
            token_ids, codons = tokenizer.encode_cds(seq, add_special_tokens=True)
            
            data = {
                'gene_id': f"seq_{idx}",
                'sequence': seq,
                'input_ids': torch.tensor(token_ids, dtype=torch.long),
                'labels': torch.tensor(token_ids, dtype=torch.long),
                'organism_id': torch.tensor(0, dtype=torch.long),
                'length': len(seq)
            }
            
            # Add metrics if available
            if 'cai' in df.columns and pd.notna(row['cai']):
                data['cai'] = torch.tensor(float(row['cai']), dtype=torch.float)
            if 'mfe' in df.columns and pd.notna(row['mfe']):
                data['mfe'] = torch.tensor(float(row['mfe']), dtype=torch.float)
            if 'gc_content' in df.columns and pd.notna(row['gc_content']):
                data['gc_content'] = torch.tensor(float(row['gc_content']), dtype=torch.float)
                
            sequences.append(data)
            
        except Exception as e:
            print(f"Warning: Skipping sequence {idx}: {e}")
    
    return sequences


def save_dataset_split(sequences: List[Dict[str, Any]], output_path: Path, split_name: str):
    """Save a dataset split as a .pt file."""
    if not sequences:
        print(f"Warning: No sequences to save for {split_name}")
        return
        
    # Organize data by keys
    dataset = {}
    for key in sequences[0].keys():
        if key in ['gene_id', 'sequence']:
            # Keep as lists for string data
            dataset[key] = [seq[key] for seq in sequences]
        elif key in ['input_ids', 'labels']:
            # Handle variable-length sequences - store as list of tensors
            dataset[key] = [seq[key] for seq in sequences]
        elif key == 'length':
            # Convert ints to tensor
            dataset[key] = torch.tensor([seq[key] for seq in sequences], dtype=torch.long)
        else:
            # Stack scalar tensors (organism_id, cai, mfe, etc.)
            values = [seq[key] for seq in sequences]
            if torch.is_tensor(values[0]):
                dataset[key] = torch.stack(values)
            else:
                # Convert to tensor if not already
                dataset[key] = torch.tensor(values)
    
    output_file = output_path / f"{split_name}.pt"
    torch.save(dataset, output_file)
    print(f"Saved {len(sequences)} sequences to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Retokenize dataset with tokenizer v2")
    parser.add_argument("--input", type=str, required=True, help="Input directory with data")
    parser.add_argument("--output", type=str, required=True, help="Output directory for retokenized data")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], 
                       help="Dataset splits to process")
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    print(f"Retokenizing data from {input_dir} to {output_dir}")
    print(f"Processing splits: {args.splits}")
    
    # Initialize tokenizer v2
    tokenizer = CodingTokenizerV2()
    print(f"Using tokenizer v2 with vocab size: {tokenizer.vocab_size}")
    
    total_sequences = 0
    
    # Process each split
    for split in args.splits:
        print(f"\n=== Processing {split} split ===")
        
        # Look for FASTA files first
        fasta_file = input_dir / f"{split}.fasta"
        csv_file = input_dir / f"{split}.csv"
        
        sequences = []
        
        if fasta_file.exists():
            print(f"Found FASTA file: {fasta_file}")
            sequences = retokenize_fasta_file(fasta_file, tokenizer)
            
        elif csv_file.exists():
            print(f"Found CSV file: {csv_file}")
            sequences = retokenize_csv_with_metrics(csv_file, tokenizer)
            
        else:
            print(f"Warning: No data file found for {split} split")
            print(f"Looked for: {fasta_file} or {csv_file}")
            continue
        
        if sequences:
            save_dataset_split(sequences, output_dir, split)
            total_sequences += len(sequences)
        
    print(f"\n✅ Retokenization complete!")
    print(f"Total sequences processed: {total_sequences}")
    print(f"Output directory: {output_dir}")
    
    # Create a simple README
    readme_path = output_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(f"""# Retokenized Dataset (Tokenizer V2)

Generated with tokenizer v2 (68-token vocabulary).

## Splits
{chr(10).join(f"- {split}.pt" for split in args.splits)}

## Vocabulary
- Size: {tokenizer.vocab_size} tokens
- Special tokens: [PAD], [MASK], [CLS], [SEP]
- Codon tokens: 64 raw DNA codons

## Total sequences: {total_sequences}
""")
    
    print(f"README written to {readme_path}")


if __name__ == "__main__":
    main()