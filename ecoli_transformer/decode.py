import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from ecoli_transformer.model import CodonEncoder
from ecoli_transformer.tokenizer import CodonTokenizer

# --- Biological Constraints ---
RESTRICTION_SITES = {
    "EcoRI": "GAATTC",
    "HindIII": "AAGCTT",
    "BamHI": "GGATCC",
    "BglII": "AGATCT",
    "XhoI": "CTCGAG",
}

def has_restriction_site(sequence: str, sites: dict = RESTRICTION_SITES) -> bool:
    """Checks if a sequence contains any of the given restriction sites."""
    for site in sites.values():
        if site in sequence:
            return True
    return False

# --- Beam Search Decoder ---
class BeamSearchDecoder:
    def __init__(self, model: CodonEncoder, tokenizer: CodonTokenizer, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def generate(self, start_sequence: str, beam_size: int = 5, max_len: int = 512, dg_penalty: float = 0.1):
        """
        Generates optimized sequences using beam search.
        """
        start_tokens = self.tokenizer.encode_cds(start_sequence)[0]
        start_node = (0.0, start_tokens, self.tokenizer.decode(start_tokens[1:-1])) # (score, tokens, sequence_str)
        
        live_beams = [start_node]
        finished_beams = []

        for step in range(max_len):
            if not live_beams:
                break

            new_beams = []
            for score, tokens, seq_str in live_beams:
                if tokens[-1] == self.tokenizer.sep_token_id:
                    finished_beams.append((score, tokens, seq_str))
                    continue

                input_ids = torch.LongTensor([tokens]).to(self.device)
                
                with torch.no_grad():
                    mlm_logits, _, _, dg_pred = self.model(input_ids)
                
                # --- Scoring ---
                log_probs = F.log_softmax(mlm_logits[:, -1, :], dim=-1)
                
                # Add dG penalty (we want to maximize dG, so penalize low values)
                # The model predicts dG, so a higher value is better (less negative)
                score_update = dg_penalty * torch.sigmoid(dg_pred.mean())
                
                top_k = torch.topk(log_probs, beam_size, dim=-1)
                
                for i in range(beam_size):
                    next_token = top_k.indices[0, i].item()
                    log_prob = top_k.values[0, i].item()
                    
                    new_tokens = tokens + [next_token]
                    new_seq_str = self.tokenizer.decode(new_tokens[1:-1])

                    # --- Constraint Checking ---
                    if has_restriction_site(new_seq_str):
                        continue
                    
                    new_score = score + log_prob + score_update.item()
                    new_beams.append((new_score, new_tokens, new_seq_str))

            live_beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_size]

        finished_beams.extend(live_beams)
        
        # Return top N beams, ensuring they end with a stop codon
        final_candidates = []
        for score, tokens, seq_str in sorted(finished_beams, key=lambda x: x[0], reverse=True):
            if self.tokenizer.is_stop_codon(self.tokenizer.id_to_token[tokens[-1]]):
                 final_candidates.append((seq_str, score))
        
        return final_candidates[:beam_size]

def generate_optimized(sequence: str, model_path: str, beam_size: int = 5) -> List[Tuple[str, float]]:
    """
    High-level function to load a model and generate optimized sequences.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = CodonEncoder()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Tokenizer
    tokenizer = CodonTokenizer()
    
    decoder = BeamSearchDecoder(model, tokenizer, device)
    optimized_sequences = decoder.generate(sequence, beam_size=beam_size)
    
    return optimized_sequences

# --- Smoke Test ---
if __name__ == '__main__':
    # This is a placeholder for a real sequence from your validation set
    # In a real scenario, you would load this from a FASTA file
    example_sequence = "ATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAGCTGACCCTGAAGTTCATCTGCACCACCGGCAAGCTGCCCGTGCCCTGGCCCACCCTCGTGACCACCCTGACCTACGGCGTGCAGTGCTTCAGCCGCTACCCCGACCACATGAAGCAGCACGACTTCTTCAAGTCCGCCATGCCCGAAGGCTACGTCCAGGAGCGCACCATCTTCTTCAAGGACGACGGCAACTACAAGACCCGCGCCGAGGTGAAGTTCGAGGGCGACACCCTGGTGAACCGCATCGAGCTGAAGGGCATCGACTTCAAGGAGGACGGCAACATCCTGGGGCACAAGCTGGAGTACAACTACAACAGCCACAACGTCTATATCATGGCCGACAAGCAGAAGAACGGCATCAAGGTGAACTTCAAGATCCGCCACAACATCGAGGACGGCAGCGTGCAGCTCGCCGACCACTACCAGCAGAACACCCCCATCGGCGACGGCCCCGTGCTGCTGCCCGACAACCACTACCTGAGCACCCAGTCCGCCCTGAGCAAAGACCCCAACGAGAAGCGCGATCACATGGTCCTGCTGGAGTTCGTGACCGCCGCCGGGATCACTCTCGGCATGGACGAGCTGTACAAGTAA"
    
    print("Running smoke test...")
    try:
        # You must have a trained model checkpoint to run this
        optimized_variants = generate_optimized(example_sequence, model_path="checkpoints/multitask_long.pt", beam_size=3)
        
        if optimized_variants:
            print("\n--- Generated Variants ---")
            for i, (seq, score) in enumerate(optimized_variants):
                print(f"Variant {i+1} (Score: {score:.4f}):")
                print(seq)
            print("\n✅ Smoke test complete.")
        else:
            print("\n❌ Smoke test failed: No variants were generated.")

    except FileNotFoundError:
        print("\n❌ Smoke test failed: Checkpoint 'checkpoints/multitask_long.pt' not found.")
    except Exception as e:
        print(f"\n❌ Smoke test failed with an error: {e}")
