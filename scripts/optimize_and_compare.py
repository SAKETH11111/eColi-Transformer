import argparse
import sys
from pathlib import Path
import RNA
import pandas as pd
import numpy as np

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from ecoli_transformer.decode import generate_optimized

def calculate_cai(sequence: str, weights: dict) -> float:
    """Calculates Codon Adaptation Index (CAI) using a weight table."""
    codons = [sequence[i:i+3] for i in range(0, len(sequence), 3) if len(sequence[i:i+3]) == 3]
    cai_values = [weights.get(codon, 0) for codon in codons]
    
    cai_values = [v for v in cai_values if v > 0]
    if not cai_values:
        return 0.0
    
    return np.exp(np.mean(np.log(cai_values)))

def calculate_gc_content(sequence: str) -> float:
    """Calculates the GC content of a DNA sequence."""
    gc_count = sequence.count('G') + sequence.count('C')
    return (gc_count / len(sequence)) * 100 if len(sequence) > 0 else 0

def calculate_mfe(sequence: str) -> float:
    """Calculates Minimum Free Energy (MFE) using ViennaRNA."""
    try:
        _, mfe = RNA.fold(sequence)
        return mfe
    except Exception:
        return 0.0

def protein_to_initial_dna(protein_seq: str) -> str:
    """Initialize with E. coli-preferred codons to create an initial DNA sequence."""
    ECOLI_PREFERRED = {
        'A': 'GCG', 'C': 'TGC', 'D': 'GAT', 'E': 'GAA', 
        'F': 'TTT', 'G': 'GGC', 'H': 'CAC', 'I': 'ATC', 'K': 'AAA',
        'L': 'CTG', 'M': 'ATG', 'N': 'AAC', 'P': 'CCG', 'Q': 'CAG',
        'R': 'CGC', 'S': 'AGC', 'T': 'ACC', 'V': 'GTG', 'W': 'TGG', 'Y': 'TAT'
    }
    return "".join(ECOLI_PREFERRED.get(aa, "NNN") for aa in protein_seq)

def main():
    parser = argparse.ArgumentParser(description="Optimize a protein sequence using the Beam Search Decoder.")
    parser.add_argument("--protein_sequence", type=str, required=True, help="The protein sequence to optimize.")
    parser.add_argument("--model_path", type=str, default="checkpoints/multitask_long.pt", help="Path to the trained model checkpoint.")
    parser.add_argument("--cai_csv", type=str, default="data/raw/ecoli_cai_weights.csv", help="Path to CAI weights CSV.")
    parser.add_argument("--beam_size", type=int, default=5, help="Beam size for beam search.")
    parser.add_argument("--lambda_cai", type=float, default=1.0, help="Weight for CAI in scoring.")
    parser.add_argument("--lambda_dg", type=float, default=1.0, help="Weight for dG penalty in scoring.")
    args = parser.parse_args()

    # Load CAI weights
    try:
        cai_df = pd.read_csv(args.cai_csv)
        cai_weights = cai_df.set_index('Codon')['Weight'].to_dict()
    except Exception as e:
        print(f"Error loading CAI weights from {args.cai_csv}: {e}")
        print("Please ensure the CSV has 'Codon' and 'Weight' columns.")
        return

    print("\n--- Optimizing Sequence with Beam Search Decoder ---")
    
    # 1. Get an initial DNA sequence from the protein sequence.
    initial_dna = protein_to_initial_dna(args.protein_sequence)
    
    # --- Evaluate Initial Sequence ---
    print("\n--- Initial DNA Sequence ---")
    initial_cai = calculate_cai(initial_dna, cai_weights)
    print(f"Sequence: {initial_dna}")
    print(f"CAI: {initial_cai:.4f}")
    
    # 2. Use the new `generate_optimized` function.
    optimized_results = generate_optimized(
        sequence=initial_dna,
        model_path=args.model_path,
        beam_size=args.beam_size,
        lambda_cai=args.lambda_cai,
        lambda_dg=args.lambda_dg
    )
    
    if not optimized_results:
        print("Optimization failed to produce any valid sequences.")
        return

    print("\n--- Top Optimized DNA Sequence ---")
    # Get the top result from the beam search
    optimized_sequence, best_score = optimized_results[0]
    
    optimized_gc = calculate_gc_content(optimized_sequence)
    optimized_mfe = calculate_mfe(optimized_sequence)
    optimized_cai = calculate_cai(optimized_sequence, cai_weights)
    
    print(f"Sequence: {optimized_sequence}")
    print(f"Score: {best_score:.4f}")
    print(f"CAI: {optimized_cai:.4f}")
    print(f"GC Content: {optimized_gc:.2f}%")
    print(f"MFE (ΔG): {optimized_mfe:.2f} kcal/mol")
    
    if len(optimized_results) > 1:
        print("\n--- Other Candidates ---")
        for i, (seq, score) in enumerate(optimized_results[1:]):
            print(f"Candidate {i+1} (Score: {score:.4f}): {seq}")

if __name__ == "__main__":
    main()
