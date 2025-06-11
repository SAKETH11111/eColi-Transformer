import torch
import torch.nn.functional as F
from typing import List, Tuple
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

class BeamSearchDecoder:
    def __init__(self, model: CodonEncoder, tokenizer: CodonTokenizer, device: torch.device,
                 lambda_cai: float = 1.0, lambda_dg: float = 1.0):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.lambda_cai = lambda_cai
        self.lambda_dg = lambda_dg
        self.model.eval()  # set model to evaluation mode

    def generate(self, input_sequence: str, protein_sequence: str, beam_size: int = 5) -> List[Tuple[str, float]]:
        """
        Generate optimized sequences from a masked input coding sequence, constrained by a protein sequence.
        Returns a list of (sequence, score) tuples.
        """
        # Encode the input sequence. Unspecified codons (e.g. 'NNN') will become [MASK] tokens.
        token_ids, _ = self.tokenizer.encode_cds(input_sequence)
        
        # Create a mapping from codon index to amino acid
        aa_per_codon_idx = {i: aa for i, aa in enumerate(protein_sequence)}
        
        # Ensure the first codon is ATG (start codon)
        if len(token_ids) > 2: # Must have [CLS], codon, [SEP]
            token_ids[1] = self.tokenizer.codon_to_id['ATG']
        else:
            # If no codons present (just CLS and SEP), insert ATG as start codon
            token_ids = [self.tokenizer.cls_id, self.tokenizer.codon_to_id['ATG'], self.tokenizer.sep_id]

        # Initialize beam list: each beam is (score, token_id_list).
        beams = [(0.0, token_ids)]

        while True:
            # Check if all beams are complete (no [MASK] in any beam's token list).
            if all(self.tokenizer.mask_id not in tokens for _, tokens in beams):
                break

            # Find the index of the leftmost mask.
            try:
                mask_index = beams[0][1].index(self.tokenizer.mask_id)
            except ValueError:
                break # No mask found, should be caught by the check above but as a safeguard.

            all_expansions = []
            
            # Prepare batch input for all beams to speed up model inference
            # All beams should have the same length, so no padding is needed.
            input_batch = torch.tensor([tokens for _, tokens in beams], dtype=torch.long, device=self.device)

            # Run model to get logits and predictions for each sequence in the batch
            with torch.no_grad():
                mlm_logits, _, cai_pred_batch, dg_pred_batch = self.model(input_batch)

            # For each beam in the batch, get the distribution for the mask position
            for i in range(input_batch.size(0)):
                score, tokens = beams[i]
                
                # Get log-probabilities over vocab for this position
                logits = mlm_logits[i, mask_index, :]
                log_probs = F.log_softmax(logits, dim=-1)

                # Incorporate sequence-level predictions
                cai_pred = torch.sigmoid(cai_pred_batch[i]) if cai_pred_batch is not None else torch.tensor(0.0)
                dg_pred = torch.sigmoid(dg_pred_batch[i]) if dg_pred_batch is not None else torch.tensor(0.0)

                # Increase search space for candidates to find a valid one
                CANDIDATE_SEARCH_SIZE = 50
                top_candidates = torch.topk(log_probs, CANDIDATE_SEARCH_SIZE)

                for j in range(CANDIDATE_SEARCH_SIZE):
                    next_token_id = int(top_candidates.indices[j].item())
                    token_logp = float(top_candidates.values[j].item())

                    # Filter out unwanted special tokens
                    if next_token_id in (self.tokenizer.cls_id, self.tokenizer.pad_id, self.tokenizer.sep_id, self.tokenizer.mask_id):
                        continue
                    
                    # Enforce protein sequence constraint
                    codon_str = self.tokenizer.id_to_codon.get(next_token_id)
                    if codon_str:
                        target_aa = aa_per_codon_idx.get(mask_index - 1) # -1 for CLS token
                        if target_aa and codon_str not in self.tokenizer.get_codons_for_aa(target_aa):
                            continue

                    # Create the new sequence tokens: replace the mask with the chosen token
                    new_tokens = tokens[:]
                    new_tokens[mask_index] = next_token_id
                    
                    # Calculate new sequence score
                    new_score = score + token_logp
                    if self.lambda_cai > 0:
                        new_score += self.lambda_cai * float(cai_pred.item())
                    if self.lambda_dg > 0:
                        new_score += self.lambda_dg * float(dg_pred.item())

                    # Decode sequence to string for restriction site check
                    seq_str = self.tokenizer.decode(new_tokens)
                    if has_restriction_site(seq_str):
                        continue
                    
                    all_expansions.append((new_score, new_tokens))

            if not all_expansions:
                break # No valid expansions found

            # Prune to top beam_size beams
            all_expansions.sort(key=lambda x: x[0], reverse=True)
            beams = all_expansions[:beam_size]

        # Prepare final results
        results = []
        for score, tokens in beams:
            # Ensure the last codon is a stop codon
            if len(tokens) > 2: # at least CLS, CODON, SEP
                last_codon_id = tokens[-2]
                if not self.tokenizer.is_stop_codon(self.tokenizer.id_to_codon.get(last_codon_id, '')):
                    tokens[-2] = self.tokenizer.codon_to_id['TAA']
            
            seq_str = self.tokenizer.decode(tokens)
            results.append((seq_str, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:beam_size]


def get_protein_from_dna(dna_seq: str, tokenizer: CodonTokenizer) -> str:
    """Translates a DNA sequence into a protein sequence."""
    codons = [dna_seq[i:i+3] for i in range(0, len(dna_seq), 3)]
    protein_seq = []
    for codon in codons:
        aa = tokenizer.get_aa_for_codon(codon)
        if aa == '*': # Stop codon
            break
        protein_seq.append(aa)
    return "".join(protein_seq)

def generate_optimized(sequence: str, model_path: str, beam_size: int = 5, lambda_cai: float = 1.0, lambda_dg: float = 1.0) -> List[Tuple[str, float]]:
    """
    High-level function to load a model and generate optimized sequences.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Tokenizer
    tokenizer = CodonTokenizer()

    # Load model with old vocab size, load weights, then resize
    # This handles the mismatch between the saved model and the new tokenizer
    OLD_VOCAB_SIZE = 68 # 4 special + 64 codons
    model = CodonEncoder(vocab_size=OLD_VOCAB_SIZE)
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    except FileNotFoundError:
        print(f"Error: Model checkpoint not found at {model_path}")
        return []
    except RuntimeError as e:
        print(f"Error loading state_dict, likely a vocab size mismatch: {e}")
        print("Attempting to load with mismatched keys ignored...")
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    # Resize to new vocab size
    model.resize_token_embeddings(tokenizer.vocab_size)
    model.to(device)
    
    decoder = BeamSearchDecoder(model, tokenizer, device, lambda_cai=lambda_cai, lambda_dg=lambda_dg)
    
    # Translate original DNA to protein to enforce constraint
    protein_sequence = get_protein_from_dna(sequence, tokenizer)

    # Create a masked sequence if not already masked
    if "N" not in sequence.upper():
        codons = [sequence[i:i+3] for i in range(0, len(sequence), 3)]
        # Mask all codons except start and stop
        if len(codons) > 2:
            masked_codons = [codons[0]] + ['NNN'] * (len(codons) - 2) + [codons[-1]]
            sequence = "".join(masked_codons)

    optimized_sequences = decoder.generate(sequence, protein_sequence, beam_size=beam_size)
    
    return optimized_sequences

if __name__ == '__main__':
    example_sequence = "ATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAGCTGACCCTGAAGTTCATCTGCACCACCGGCAAGCTGCCCGTGCCCTGGCCCACCCTCGTGACCACCCTGACCTACGGCGTGCAGTGCTTCAGCCGCTACCCCGACCACATGAAGCAGCACGACTTCTTCAAGTCCGCCATGCCCGAAGGCTACGTCCAGGAGCGCACCATCTTCTTCAAGGACGACGGCAACTACAAGACCCGCGCCGAGGTGAAGTTCGAGGGCGACACCCTGGTGAACCGCATCGAGCTGAAGGGCATCGACTTCAAGGAGGACGGCAACATCCTGGGGCACAAGCTGGAGTACAACTACAACAGCCACAACGTCTATATCATGGCCGACAAGCAGAAGAACGGCATCAAGGTGAACTTCAAGATCCGCCACAACATCGAGGACGGCAGCGTGCAGCTCGCCGACCACTACCAGCAGAACACCCCCATCGGCGACGGCCCCGTGCTGCTGCCCGACAACCACTACCTGAGCACCCAGTCCGCCCTGAGCAAAGACCCCAACGAGAAGCGCGATCACATGGTCCTGCTGGAGTTCGTGACCGCCGCCGGGATCACTCTCGGCATGGACGAGCTGTACAAGTAA"
    model_checkpoint = "checkpoints/multitask_long.pt"

    print("Running smoke test with new BeamSearchDecoder...")
    try:
        # Mask the middle of the sequence for optimization
        codons = [example_sequence[i:i+3] for i in range(0, len(example_sequence), 3)]
        input_seq = codons[0] + ("NNN" * (len(codons) - 2)) + codons[-1]
        
        optimized_variants = generate_optimized(
            input_seq, 
            model_path=model_checkpoint, 
            beam_size=5,
            lambda_cai=1.0,
            lambda_dg=1.0
        )
        
        if optimized_variants:
            print("\n--- Generated Variants ---")
            for i, (seq, score) in enumerate(optimized_variants):
                print(f"Variant {i+1} (Score: {score:.4f}):")
                print(seq)
            print("\n✅ Smoke test complete.")
        else:
            print("\n❌ Smoke test failed: No variants were generated.")

    except FileNotFoundError:
        print(f"\n❌ Smoke test failed: Checkpoint '{model_checkpoint}' not found.")
    except Exception as e:
        print(f"\n❌ Smoke test failed with an error: {e}")
        import traceback
        traceback.print_exc()
