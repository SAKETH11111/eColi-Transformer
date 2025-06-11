import argparse
from pathlib import Path
import sys
import os
import re
import torch
from typing import Dict, List, Optional, Tuple
import numpy as np
import RNA
import time
import concurrent.futures
import random

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from ecoli_transformer.tokenizer import CodonTokenizer, CODON_TO_AA

FASTA_HEADER_PATTERN = re.compile(r">(?P<gene_id>[^|]+)(?:\|len=(?P<length>\d+))?(?:\|cai=(?P<cai>[\d\.]+))?")

def parse_fasta(filepath: Path) -> List[Tuple[str, str, Dict[str, Optional[str]]]]:
    sequences = []
    current_seq = ""
    seq_id = None
    metadata = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if seq_id is not None and current_seq:
                    sequences.append((seq_id, current_seq, metadata))
                current_seq = ""
                match = FASTA_HEADER_PATTERN.match(line)
                if match:
                    metadata = match.groupdict()
                    seq_id = metadata.get('gene_id')
                else:
                    seq_id = line[1:].split()[0]
                    metadata = {'gene_id': seq_id}
            elif seq_id is not None:
                current_seq += line.upper().replace(' ', '')
        if seq_id is not None and current_seq:
            sequences.append((seq_id, current_seq, metadata))
    return sequences

def run_rnafold(sequence: str) -> Optional[float]:
    if not sequence or len(sequence) > 10000:
        return None
    try:
        _, mfe = RNA.fold(sequence)
        return mfe
    except Exception:
        return None

def cds_to_protein(cds: str) -> Optional[str]:
    """Translates a coding sequence (CDS) to a protein sequence."""
    codons = [cds[i:i+3] for i in range(0, len(cds), 3)]
    protein_seq = []
    for codon in codons:
        if len(codon) == 3:
            aa = CODON_TO_AA.get(codon)
            if aa is None: return None # Invalid codon
            if aa == '*': break # Stop codon
            protein_seq.append(aa)
    return "".join(protein_seq)

def apply_stream_masking(tokens: List[str], tokenizer: CodonTokenizer, mask_prob=0.15, mask_ratio=0.8, random_ratio=0.1):
    """
    Applies CodonTransformer's masking strategy.
    Example: 'M_ATG' -> 'M_UNK' (80% of selected tokens)
    """
    masked_tokens = list(tokens)
    labels = ['[PAD]'] * len(tokens)
    
    num_to_mask = int(len(tokens) * mask_prob)
    if num_to_mask == 0:
        return masked_tokens, labels

    indices_to_mask = sorted(random.sample(range(len(tokens)), num_to_mask))
    
    for i in indices_to_mask:
        labels[i] = tokens[i] # The ground truth is the original AA_CODON token
        
        rand_val = random.random()
        if rand_val < mask_ratio:
            # 80% chance: mask to corresponding AA_UNK token
            aa = tokens[i].split('_')[0]
            masked_tokens[i] = f"{aa}_UNK"
        elif rand_val < mask_ratio + random_ratio:
            # 10% chance: replace with a random AA_CODON token
            random_token_idx = random.randint(0, len(tokenizer.vocab) - 1)
            masked_tokens[i] = tokenizer.id_to_token[random_token_idx]
        # 10% chance: keep original token
        
    return masked_tokens, labels

def create_stream_training_example(sequence_data: Tuple[str, str, Dict[str, Optional[str]]], tokenizer: CodonTokenizer) -> Optional[Dict]:
    seq_id, cds, metadata = sequence_data
    
    protein_sequence = cds_to_protein(cds)
    if not protein_sequence:
        return None

    aa_codon_tokens = tokenizer.cds_to_aa_codon_tokens(cds)
    if len(aa_codon_tokens) != len(protein_sequence):
        return None # Mismatch, likely due to internal stop codons

    masked_tokens, labels = apply_stream_masking(aa_codon_tokens, tokenizer)

    input_ids = tokenizer.encode(['[CLS]'] + masked_tokens + ['[SEP]'])
    label_ids = [-100] + [tokenizer.token_to_id.get(lbl, -100) for lbl in labels] + [-100]
    
    cai_str = metadata.get('cai')
    cai = float(cai_str) if cai_str else np.nan
    mfe = run_rnafold(cds) or np.nan

    return {
        'gene_id': seq_id,
        'input_ids': torch.LongTensor(input_ids),
        'labels': torch.LongTensor(label_ids),
        'organism_id': 0, # Default to E. coli
        'cai': cai,
        'mfe': mfe,
        'length': len(protein_sequence)
    }

def process_sequence_legacy(sequence_data: Tuple[str, str, Dict[str, Optional[str]]], tokenizer: CodonTokenizer) -> Optional[Dict]:
    seq_id, sequence, metadata = sequence_data
    token_ids_list, _ = tokenizer.encode_cds(sequence)
    codon_length = len(token_ids_list) - 2
    if codon_length <= 0: return None
    cai_str = metadata.get('cai')
    cai = float(cai_str) if cai_str else np.nan
    mfe = run_rnafold(sequence) or np.nan
    return {
        'gene_id': seq_id,
        'token_ids': torch.LongTensor(token_ids_list),
        'cai': cai,
        'mfe': mfe,
        'length': codon_length
    }

def main():
    parser = argparse.ArgumentParser(description="Build encoded PyTorch tensors from FASTA files.")
    parser.add_argument("--fasta", type=str, required=True, help="Input FASTA file path.")
    parser.add_argument("--out", type=str, required=True, help="Output PyTorch tensor file path.")
    parser.add_argument("--stream", action="store_true", help="Enable STREAM (CodonTransformer) data generation.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N sequences.")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes.")
    args = parser.parse_args()

    if not Path(args.fasta).is_file():
        print(f"Error: Input FASTA file not found: {args.fasta}")
        sys.exit(1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    tokenizer = CodonTokenizer()
    sequences = parse_fasta(Path(args.fasta))
    if args.limit:
        sequences = sequences[:args.limit]

    print(f"Found {len(sequences)} sequences. Using {'STREAM' if args.stream else 'legacy'} processing.")
    
    process_func = create_stream_training_example if args.stream else process_sequence_legacy
    
    all_results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_seq = {executor.submit(process_func, seq_data, tokenizer): i for i, seq_data in enumerate(sequences)}
        for i, future in enumerate(concurrent.futures.as_completed(future_to_seq)):
            try:
                result = future.result()
                if result:
                    all_results.append(result)
            except Exception as e:
                print(f"Error processing sequence: {e}")
            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(sequences)} sequences...")

    valid_results = [r for r in all_results if r is not None]
    print(f"Successfully processed {len(valid_results)}/{len(sequences)} sequences.")

    if not valid_results:
        print("Error: No sequences were successfully processed.")
        sys.exit(1)

    if args.stream:
        data_dict = {
            'gene_id': [r['gene_id'] for r in valid_results],
            'input_ids': [r['input_ids'] for r in valid_results],
            'labels': [r['labels'] for r in valid_results],
            'organism_id': torch.LongTensor([r['organism_id'] for r in valid_results]),
            'cai': torch.FloatTensor([r['cai'] for r in valid_results]),
            'mfe': torch.FloatTensor([r['mfe'] for r in valid_results]),
            'length': torch.LongTensor([r['length'] for r in valid_results])
        }
    else: # Legacy
        data_dict = {
            'gene_id': [r['gene_id'] for r in valid_results],
            'token_ids': [r['token_ids'] for r in valid_results],
            'cai': torch.FloatTensor([r['cai'] for r in valid_results]),
            'mfe': torch.FloatTensor([r['mfe'] for r in valid_results]),
            'length': torch.LongTensor([r['length'] for r in valid_results])
        }

    torch.save(data_dict, args.out)
    print(f"✔️  Encoded {len(valid_results)} sequences | saved to {args.out}")

if __name__ == "__main__":
    main()
