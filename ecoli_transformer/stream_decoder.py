import torch
import torch.nn.functional as F
from typing import List
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from ecoli_transformer.model import CodonEncoder
from ecoli_transformer.tokenizer import CodonTokenizer, RESTRICTION_SITES

def has_restriction_site(sequence: str, sites: dict = RESTRICTION_SITES) -> bool:
    """Checks if a sequence contains any of the given restriction sites."""
    for site in sites.values():
        if site in sequence:
            return True
    return False

class STREAMDecoder:
    def __init__(self, model_path: str, device: str = 'cpu', deterministic: bool = True):
        self.device = torch.device(device)
        self.tokenizer = CodonTokenizer()
        self.model = self.load_model(model_path)
        self.model.eval()
        self.deterministic = deterministic

    def load_model(self, model_path: str) -> CodonEncoder:
        """Loads a trained CodonEncoder model."""
        model = CodonEncoder(vocab_size=self.tokenizer.vocab_size)
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
        except FileNotFoundError:
            print(f"Error: Model checkpoint not found at {model_path}")
            raise
        except Exception as e:
            print(f"Error loading model state_dict: {e}")
            raise
        return model.to(self.device)

    def get_synonymous_codon_ids(self, aa_unk_token: str) -> List[int]:
        """Gets the token IDs for all synonymous codons for a given AA_UNK token."""
        synonymous_tokens = self.tokenizer.get_synonymous_tokens(aa_unk_token)
        return self.tokenizer.encode(synonymous_tokens)

    def nucleus_sample(self, logits: torch.Tensor, valid_ids: List[int], p=0.9) -> int:
        """Performs nucleus sampling on a given set of logits, restricted to valid IDs."""
        # This is a simplified implementation. A robust one would handle edge cases better.
        filtered_logits = torch.full((self.tokenizer.vocab_size,), -float('Inf'), device=self.device)
        filtered_logits[valid_ids] = logits[valid_ids]
        
        probabilities = F.softmax(filtered_logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
        
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Find indices to remove
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = 0
        
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        probabilities[indices_to_remove] = 0
        probabilities /= probabilities.sum()
        
        return torch.multinomial(probabilities, 1).item()

    def apply_constraints(self, sequence: str) -> str:
        """Placeholder for applying biological constraints like avoiding restriction sites."""
        if has_restriction_site(sequence):
            # In a real implementation, we might try to repair this.
            # For now, we just flag it.
            print(f"Warning: Generated sequence contains restriction site(s).")
        return sequence

    def generate_optimized_sequence(self, protein_seq: str, organism_id: int = 0) -> str:
        """
        Generate optimized DNA using true bidirectional MLM prediction.
        """
        # 1. Convert protein to AA_UNK tokens
        aa_unk_tokens = self.tokenizer.protein_to_aa_unk_tokens(protein_seq)
        input_tokens = ['[CLS]'] + aa_unk_tokens + ['[SEP]']
        
        # 2. Encode and add organism context
        input_ids = torch.tensor([self.tokenizer.encode(input_tokens)], device=self.device)
        organism_ids = torch.tensor([organism_id], device=self.device)
        
        # 3. Single forward pass - bidirectional prediction
        with torch.no_grad():
            logits, _, _, _ = self.model(input_ids, organism_ids=organism_ids)
        
        # 4. Predict best codon for each position simultaneously
        predicted_tokens = []
        for i, token in enumerate(aa_unk_tokens):
            position_logits = logits[0, i + 1, :]  # +1 for CLS token
            
            valid_codon_ids = self.get_synonymous_codon_ids(token)
            if not valid_codon_ids:
                # Fallback to a generic codon if something goes wrong
                predicted_tokens.append(self.tokenizer.id_to_token[self.tokenizer.mask_id])
                continue

            if self.deterministic:
                # Create a tensor of -inf and fill in the valid logits
                filtered_logits = torch.full_like(position_logits, -float('Inf'))
                filtered_logits[valid_codon_ids] = position_logits[valid_codon_ids]
                best_id = torch.argmax(filtered_logits).item()
                predicted_tokens.append(self.tokenizer.id_to_token[best_id])
            else:
                predicted_id = self.nucleus_sample(position_logits, valid_codon_ids)
                predicted_tokens.append(self.tokenizer.id_to_token[predicted_id])
        
        # 5. Decode and apply biological constraints
        final_sequence = self.tokenizer.decode_to_str(self.tokenizer.encode(predicted_tokens))
        
        return self.apply_constraints(final_sequence)

if __name__ == '__main__':
    # This is a placeholder for a proper test, assuming a trained STREAM model exists.
    # You would run this after completing the training step.
    print("Running STREAMDecoder smoke test...")
    try:
        # Create a dummy model file for testing purposes
        # In a real scenario, this would be your trained model checkpoint
        dummy_model_path = "dummy_stream_model.pt"
        tokenizer = CodonTokenizer()
        model = CodonEncoder(vocab_size=tokenizer.vocab_size)
        torch.save({'model_state_dict': model.state_dict()}, dummy_model_path)

        decoder = STREAMDecoder(dummy_model_path, deterministic=True)
        protein = "MKT"
        result = decoder.generate_optimized_sequence(protein)
        print(f"Protein: {protein}")
        print(f"Optimized DNA: {result}")
        assert len(result) == len(protein) * 3
        print("\n✅ STREAMDecoder self-test complete.")
        
        # Clean up the dummy file
        os.remove(dummy_model_path)

    except Exception as e:
        print(f"\n❌ STREAMDecoder test failed: {e}")
        if 'dummy_model_path' in locals() and os.path.exists(dummy_model_path):
            os.remove(dummy_model_path)
