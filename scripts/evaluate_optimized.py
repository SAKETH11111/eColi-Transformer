import argparse
import json
import sys
from pathlib import Path
import torch
import numpy as np
import RNA
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from ecoli_transformer.decode import generate_optimized
from ecoli_transformer.tokenizer import RESTRICTION_SITES

def parse_fasta(fasta_file: str) -> dict:
    """Parses a FASTA file into a dictionary of headers to sequences."""
    sequences = {}
    with open(fasta_file, 'r') as f:
        header = None
        seq = []
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if header:
                    sequences[header] = "".join(seq)
                header = line[1:]
                seq = []
            else:
                seq.append(line)
        if header:
            sequences[header] = "".join(seq)
    return sequences

def calculate_cai(sequence: str, weights: dict) -> float:
    """Calculates Codon Adaptation Index (CAI) using a weight table."""
    codons = [sequence[i:i+3] for i in range(0, len(sequence), 3)]
    cai_values = [weights.get(codon, 0) for codon in codons]
    
    # Geometric mean of the weights
    cai_values = [v for v in cai_values if v > 0]
    if not cai_values:
        return 0.0
    
    return np.exp(np.mean(np.log(cai_values)))

def calculate_mfe(sequence: str) -> float:
    """Calculates Minimum Free Energy (MFE) using ViennaRNA."""
    try:
        _, mfe = RNA.fold(sequence)
        return mfe
    except Exception:
        return 0.0

def count_restriction_sites(sequence: str, sites: dict) -> int:
    """Counts the number of occurrences of restriction sites."""
    return sum(sequence.count(site) for site in sites.values())

def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare wild-type vs. optimized gene sequences.")
    parser.add_argument("--input_fasta", type=str, required=True, help="FASTA file of wild-type sequences.")
    parser.add_argument("--output_json", type=str, required=True, help="Output JSON file for the results.")
    parser.add_argument("--model_path", type=str, default="checkpoints/multitask_long.pt", help="Path to the trained model.")
    parser.add_argument("--cai_csv", type=str, default="data/raw/CAI.csv", help="Path to CAI weights CSV.")
    parser.add_argument("--beam_size", type=int, default=5, help="Beam size for optimization.")
    args = parser.parse_args()

    # Load CAI weights
    cai_weights = pd.read_csv(args.cai_csv).set_index('Codon')['Weight'].to_dict()

    # Parse input sequences
    wild_type_sequences = parse_fasta(args.input_fasta)
    
    results = {}
    for header, wt_seq in wild_type_sequences.items():
        print(f"Processing: {header}")
        
        # --- Evaluate Wild-Type ---
        wt_metrics = {
            "cai": calculate_cai(wt_seq, cai_weights),
            "mfe": calculate_mfe(wt_seq),
            "restriction_sites": count_restriction_sites(wt_seq, RESTRICTION_SITES),
            "sequence": wt_seq
        }
        
        # --- Generate Optimized Sequence ---
        # Mask the sequence for the decoder
        codons = [wt_seq[i:i+3] for i in range(0, len(wt_seq), 3)]
        if len(codons) > 2:
            masked_seq = codons[0] + ("NNN" * (len(codons) - 2)) + codons[-1]
        else:
            masked_seq = wt_seq

        optimized_results = generate_optimized(
            sequence=masked_seq,
            model_path=args.model_path,
            beam_size=args.beam_size
        )
        
        if optimized_results:
            opt_seq, _ = optimized_results[0]
            # --- Evaluate Optimized ---
            opt_metrics = {
                "cai": calculate_cai(opt_seq, cai_weights),
                "mfe": calculate_mfe(opt_seq),
                "restriction_sites": count_restriction_sites(opt_seq, RESTRICTION_SITES),
                "sequence": opt_seq
            }
        else:
            print(f"  -> Optimization failed for {header}")
            opt_metrics = {"error": "Optimization failed", "sequence": ""}

        results[header] = {
            "wild_type": wt_metrics,
            "optimized": opt_metrics
        }

    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"✅ Evaluation complete. Results saved to {args.output_json}")

if __name__ == "__main__":
    main()
