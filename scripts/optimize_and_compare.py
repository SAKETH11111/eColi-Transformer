import argparse
import sys
from pathlib import Path
import numpy as np
import RNA

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from ecoli_transformer.mlm_decoder import iterative_optimization
from ecoli_transformer.model import CodonEncoder
from transformers import AutoTokenizer
import torch

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

def protein_to_optimal_codons(protein_seq: str, tokenizer) -> str:
    """Initialize with E. coli-preferred codons."""
    # This is a simplified version of what the CodonTransformer tokenizer does.
    # We'll use a hardcoded map for simplicity, but the real tokenizer is more sophisticated.
    ECOLI_PREFERRED = {
        'A': 'GCG', 'C': 'TGC', 'D': 'GAT', 'E': 'GAA', 
        'F': 'TTT', 'G': 'GGC', 'H': 'CAC', 'I': 'ATC', 'K': 'AAA',
        'L': 'CTG', 'M': 'ATG', 'N': 'AAC', 'P': 'CCG', 'Q': 'CAG',
        'R': 'CGC', 'S': 'AGC', 'T': 'ACC', 'V': 'GTG', 'W': 'TGG', 'Y': 'TAT'
    }
    return "".join(ECOLI_PREFERRED.get(aa, "NNN") for aa in protein_seq)

def main():
    parser = argparse.ArgumentParser(description="Optimize a protein sequence and compare its properties.")
    parser.add_argument("--protein_sequence", type=str, required=True, help="The protein sequence to optimize.")
    parser.add_argument("--model_path", type=str, default="checkpoints/multitask_long.pt", help="Path to the trained model checkpoint.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("adibvafa/CodonTransformer")
    model = CodonEncoder(vocab_size=tokenizer.vocab_size)
    
    # Note: We are not loading a checkpoint here, as we need to retrain with the new vocab
    # This will use the randomly initialized model for the test
    model.to(device)

    # --- Optimized Sequence ---
    print("\n--- Optimizing Sequence ---")
    initial_dna = protein_to_optimal_codons(args.protein_sequence, tokenizer)
    optimized_sequence = iterative_optimization(initial_dna, model, tokenizer, device)
    
    print("\n--- Optimized DNA Sequence ---")
    optimized_gc = calculate_gc_content(optimized_sequence)
    optimized_mfe = calculate_mfe(optimized_sequence)
    print(f"Sequence: {optimized_sequence}")
    print(f"GC Content: {optimized_gc:.2f}%")
    print(f"MFE (ΔG): {optimized_mfe:.2f} kcal/mol")

if __name__ == "__main__":
    main()
