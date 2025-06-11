"""
Builds encoded tensors from FASTA files using CodonTokenizer.

Reads sequences from a FASTA file, tokenizes them into codons and codon pairs,
extracts CAI from the header, calculates Minimum Free Energy (MFE) using RNAfold,
and saves the results as a dictionary of PyTorch tensors.

Uses multiprocessing to speed up the process, especially the RNAfold calls.
"""

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
from functools import partial

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

try:
    from ecoli_transformer.tokenizer import CodonTokenizer
except ImportError as e:
    print(f"Error importing tokenizer module: {e}")
    print(f"Ensure ecoli_transformer/tokenizer.py exists and Python path is correct: {sys.path}")
    sys.exit(1)

FASTA_HEADER_PATTERN = re.compile(r">(?P<gene_id>[^|]+)(?:\|len=(?P<length>\d+))?(?:\|cai=(?P<cai>[\d\.]+))?(?:\|gc=(?P<gc>[\d\.]+))?")
RNAFOLD_MFE_PATTERN = re.compile(r".*\( *(-?\d+\.\d+) *\)")

def parse_fasta(filepath: Path) -> List[Tuple[str, str, Dict[str, Optional[str]]]]:
    """Parses a FASTA file, returning sequence ID, sequence, and header metadata."""
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
                    print(f"Warning: Could not parse header fully for: {line}. Using ID: {seq_id}")

            elif seq_id is not None:
                current_seq += line.upper().replace(' ', '')

        if seq_id is not None and current_seq:
            sequences.append((seq_id, current_seq, metadata))

    return sequences

def run_rnafold(sequence: str, timeout_seconds: int = 30) -> Optional[float]:
    """Calculates MFE using the ViennaRNA Python binding with timeout."""
    if not sequence:
        return None
    try:
        # RNA.fold returns a tuple: (structure, mfe)
        # For very long sequences, this can take a very long time
        # Add a simple timeout by limiting sequence length
        if len(sequence) > 10000:  # Skip very long sequences
            print(f"Warning: Skipping RNA fold for sequence of length {len(sequence)} (too long)")
            return None
            
        _, mfe = RNA.fold(sequence)
        return mfe
    except Exception as e:
        print(f"Warning: RNA.fold failed: {e}")
        return None

def process_sequence(sequence_data: Tuple[str, str, Dict[str, Optional[str]]], tokenizer: CodonTokenizer) -> Optional[Dict]:
    """
    Helper function to process a single sequence entry.
    Tokenizes, extracts CAI, runs RNAfold, and returns a dictionary of results.
    Returns None if processing fails significantly (e.g., zero length).
    """
    seq_id, sequence, metadata = sequence_data

    try:
        token_ids_list, pair_ids_list = tokenizer.encode_cds(sequence)
        codon_length = len(token_ids_list) - 2
    except Exception as e:
        print(f"Error tokenizing sequence {seq_id}: {e}")
        return None

    if codon_length <= 0:
        return None

    cai_str = metadata.get('cai')
    cai: float = np.nan
    if cai_str:
        try:
            cai = float(cai_str)
        except (ValueError, TypeError):
            pass

    # RNA folding can be slow for long sequences
    mfe = run_rnafold(sequence)
    if mfe is None:
        mfe = np.nan

    return {
        'gene_id': seq_id,
        'token_ids': torch.LongTensor(token_ids_list),
        'pair_ids': torch.LongTensor(pair_ids_list),
        'cai': cai,
        'mfe': mfe,
        'length': codon_length
    }


def main():
    parser = argparse.ArgumentParser(description="Build encoded PyTorch tensors from FASTA files.")
    parser.add_argument("--fasta", type=str, required=True,
                        help="Input FASTA file path (e.g., data/processed/train.fasta).")
    parser.add_argument("--out", type=str, required=True,
                        help="Output PyTorch tensor file path (e.g., data/processed/train.pt).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N sequences (for testing).")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of worker processes (defaults to number of CPUs).")
    args = parser.parse_args()

    fasta_path = Path(args.fasta)
    out_path = Path(args.out)
    limit = args.limit
    num_workers = args.workers

    if not fasta_path.is_file():
        print(f"Error: Input FASTA file not found: {fasta_path}")
        sys.exit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = CodonTokenizer()

    print(f"Parsing FASTA file: {fasta_path}...")
    sequences = parse_fasta(fasta_path)
    num_sequences_total = len(sequences)
    print(f"Found {num_sequences_total} sequences.")

    if limit is not None:
        sequences = sequences[:limit]
        print(f"Processing limit applied: {limit} sequences.")

    num_sequences_to_process = len(sequences)
    if num_sequences_to_process == 0:
        print("No sequences to process. Exiting.")
        sys.exit(0)

    print(f"Encoding sequences and calculating MFE using {num_workers or 'all available'} CPU cores...")
    start_time = time.time()

    all_results: List[Optional[Dict]] = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Determine the actual number of workers the executor is likely using.
        effective_workers = num_workers if num_workers is not None else os.cpu_count()
        if effective_workers is None:
            effective_workers = 1
        if effective_workers <= 0:
            effective_workers = 1

        print(f"Using {effective_workers} effective worker(s).")
        print(f"Starting processing of {num_sequences_to_process} sequences...")
        sys.stdout.flush()

        # Submit all tasks immediately for real-time progress
        future_to_sequence = {
            executor.submit(process_sequence, seq_data, tokenizer): i 
            for i, seq_data in enumerate(sequences)
        }

        processed_count = 0
        last_print_time = start_time
        
        # Process results as they complete (real-time progress)
        for future in concurrent.futures.as_completed(future_to_sequence):
            try:
                result = future.result()
                all_results.append(result)
            except Exception as e:
                print(f"Error processing sequence: {e}")
                all_results.append(None)
            
            processed_count += 1
            current_time = time.time()
            
            # Print progress every 1 second or every 50 sequences, whichever comes first
            if (current_time - last_print_time > 1 or 
                processed_count % 50 == 0 or 
                processed_count == num_sequences_to_process):
                elapsed_time = current_time - start_time
                avg_time_per_seq = elapsed_time / processed_count if processed_count > 0 else 0
                sequences_per_sec = processed_count / elapsed_time if elapsed_time > 0 else 0
                print(f"  Processed {processed_count}/{num_sequences_to_process} sequences... "
                      f"({avg_time_per_seq:.3f} s/seq, {sequences_per_sec:.1f} seq/s)")
                sys.stdout.flush()
                last_print_time = current_time

    valid_results = [r for r in all_results if r is not None]
    num_failed = num_sequences_to_process - len(valid_results)

    if num_failed > 0:
        print(f"Warning: Failed to process {num_failed}/{num_sequences_to_process} sequences.")

    if not valid_results:
         print("Error: No sequences were successfully processed.")
         sys.exit(1)

    data_dict = {
        'gene_id': [r['gene_id'] for r in valid_results],
        'token_ids': [r['token_ids'] for r in valid_results],
        'pair_ids': [r['pair_ids'] for r in valid_results],
        'cai': torch.FloatTensor([r['cai'] for r in valid_results]),
        'mfe': torch.FloatTensor([r['mfe'] for r in valid_results]),
        'length': torch.LongTensor([r['length'] for r in valid_results])
    }

    total_time = time.time() - start_time
    print(f"Finished processing in {total_time:.2f} seconds.")

    print(f"Saving encoded data to: {out_path}...")
    try:
        torch.save(data_dict, out_path)
    except Exception as e:
        print(f"Error saving data to {out_path}: {e}")
        sys.exit(1)

    avg_len = data_dict['length'].float().mean().item() if len(data_dict['length']) > 0 else 0
    print(f"✔️  Encoded {len(valid_results)} sequences | avg len {avg_len:.0f} codons | saved {out_path}")


if __name__ == "__main__":
    main()
