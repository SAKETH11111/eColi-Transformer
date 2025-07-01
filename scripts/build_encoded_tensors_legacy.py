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
from functools import partial
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from ecoli_transformer.tokenizer_v2 import CodingTokenizerV2 as CodonTokenizer, GENETIC_CODE

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

def run_rnafold(sequence: str, timeout: int = 30) -> Optional[float]:
    """Calculates MFE with a timeout to prevent getting stuck on long sequences."""
    if not sequence:
        return None
    
    # For extremely long sequences (>5000), skip RNA folding to avoid excessive computation
    # but still allow longer sequences than before
    if len(sequence) > 5000:
        return None
    
    try:
        # Direct call - ViennaRNA is generally thread-safe for fold()
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
            aa = GENETIC_CODE.get(codon)
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

def create_stream_training_example(sequence_data: Tuple[str, str, Dict[str, Optional[str]]], tokenizer: CodonTokenizer, skip_rna: bool = False) -> Optional[Dict]:
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
    mfe = np.nan if skip_rna else (run_rnafold(cds) or np.nan)

    return {
        'gene_id': seq_id,
        'input_ids': torch.LongTensor(input_ids),
        'labels': torch.LongTensor(label_ids),
        'organism_id': 0, # Default to E. coli
        'cai': cai,
        'mfe': mfe,
        'length': len(protein_sequence)
    }

def process_sequence_legacy(sequence_data: Tuple[str, str, Dict[str, Optional[str]]], tokenizer: CodonTokenizer, skip_rna: bool = False) -> Optional[Dict]:
    seq_id, sequence, metadata = sequence_data
    token_ids_list, _ = tokenizer.encode_cds(sequence)
    codon_length = len(token_ids_list) - 2
    if codon_length <= 0: return None
    cai_str = metadata.get('cai')
    cai = float(cai_str) if cai_str else np.nan
    mfe = np.nan if skip_rna else (run_rnafold(sequence) or np.nan)
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
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of worker processes.")
    parser.add_argument("--max_len", type=int, default=3000, help="Maximum sequence length to process.")
    parser.add_argument("--batch_size", type=int, default=500, help="Batch size for processing.")
    parser.add_argument("--no_rna", action="store_true", help="Skip RNA folding calculations.")
    args = parser.parse_args()

    if not Path(args.fasta).is_file():
        print(f"Error: Input FASTA file not found: {args.fasta}")
        sys.exit(1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    tokenizer = CodonTokenizer()
    sequences = parse_fasta(Path(args.fasta))
    if args.limit:
        sequences = sequences[:args.limit]

    print(f"Found {len(sequences)} sequences.")
    print(f"Processing ALL {len(sequences)} sequences using {'STREAM' if args.stream else 'legacy'} method.")
    print(f"Using {args.workers} workers with batch size {args.batch_size}")
    if args.no_rna:
        print("RNA folding calculations disabled.")
    
    # Process all sequences at once with chunking for better performance
    all_results = []
    
    process_func = create_stream_training_example if args.stream else process_sequence_legacy
    p_process_func = partial(process_func, tokenizer=tokenizer, skip_rna=args.no_rna)
    
    # Calculate optimal chunksize for maximum throughput
    chunksize = max(1, len(sequences) // (args.workers * 8))
    print(f"Using chunksize {chunksize} for optimal load balancing...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Submit all work at once for maximum parallelization
        futures = []
        batch_size = args.batch_size
        
        # Submit in batches to avoid overwhelming memory
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i+batch_size]
            batch_futures = [executor.submit(p_process_func, seq_data) for seq_data in batch]
            futures.extend(batch_futures)
        
        print(f"Submitted {len(futures)} tasks across {args.workers} workers")
        
        # Collect results as they complete
        completed_count = 0
        with tqdm(total=len(futures), desc="Processing sequences") as pbar:
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result(timeout=120)  # 2 minute timeout per sequence
                    if result:
                        all_results.append(result)
                    completed_count += 1
                    pbar.update(1)
                    
                    # Progress updates every 500 completions
                    if completed_count % 500 == 0:
                        print(f"Completed {completed_count}/{len(futures)} sequences, {len(all_results)} successful")
                        
                except Exception as e:
                    # Don't let individual failures stop the whole process
                    completed_count += 1
                    pbar.update(1)
                    if "timeout" in str(e).lower():
                        print(f"Warning: Sequence timed out (likely long RNA folding)")
                    else:
                        print(f"Warning: Failed to process sequence: {e}")
                    continue

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
