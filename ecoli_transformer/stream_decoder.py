import torch
import torch.nn.functional as F
from typing import List, Tuple

from ecoli_transformer.model import CodonEncoder
from ecoli_transformer.tokenizer import CodonTokenizer, RESTRICTION_SITES

def has_restriction_site(sequence: str, sites: dict = RESTRICTION_SITES) -> bool:
    """Checks if a sequence contains any of the given restriction sites."""
    for site in sites.values():
        if site in sequence:
            return True
    return False

def generate_optimized_sequence(protein_seq: str, model: CodonEncoder, tokenizer: CodonTokenizer, device: torch.device, deterministic: bool = True, top_p: float = 0.9) -> str:
    """
    Generates an optimized DNA sequence from a protein sequence using a bidirectional, masked language modeling approach.
    """
    model.eval()

    # 1. Convert protein sequence to masked amino acid tokens
    masked_input = [f"{aa}_UNK" for aa in protein_seq]
    
    # 2. Add special tokens and encode
    # This is a simplified encoding; a real implementation might have organism-specific tokens
    encoded_input = [tokenizer.cls_id] + [tokenizer.codon_to_id[t] for t in masked_input] + [tokenizer.sep_id]
    input_ids = torch.LongTensor([encoded_input]).to(device)

    # 3. Get model predictions for all positions simultaneously
    with torch.no_grad():
        logits, _, _, _ = model(input_ids)

    # 4. Decode the sequence
    if deterministic:
        # Greedy decoding
        predicted_ids = torch.argmax(logits, dim=-1).squeeze(0)
    else:
        # Nucleus sampling
        probs = F.softmax(logits, dim=-1).squeeze(0)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        probs[indices_to_remove] = 0
        
        predicted_ids = torch.multinomial(probs, 1).squeeze(-1)

    # 5. Convert IDs to sequence and apply constraints
    # We only care about the codon predictions, not the special tokens
    predicted_codons = [tokenizer.id_to_codon[id_.item()] for id_ in predicted_ids[1:-1]]
    optimized_sequence = "".join(predicted_codons)

    # Post-processing filter (simple version)
    if has_restriction_site(optimized_sequence):
        print("Warning: Generated sequence contains restriction sites. Returning original sequence.")
        return "".join(tokenizer.decode(input_ids.squeeze(0)[1:-1]))

    return optimized_sequence

# --- Smoke Test ---
if __name__ == '__main__':
    # This is a placeholder for a real protein sequence
    example_protein_sequence = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR"
    
    print("Running STREAM-style decoder smoke test...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = CodonTokenizer()
        model = CodonEncoder(vocab_size=tokenizer.vocab_size)
        checkpoint = torch.load("checkpoints/multitask_long.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        optimized_sequence = generate_optimized_sequence(example_protein_sequence, model, tokenizer, device)
        
        print("\n--- Original Protein Sequence ---")
        print(example_protein_sequence)
        print("\n--- Generated DNA Sequence ---")
        print(optimized_sequence)
        print("\n✅ STREAM-style decoder smoke test complete.")

    except FileNotFoundError:
        print("\n❌ Smoke test failed: Checkpoint 'checkpoints/multitask_long.pt' not found.")
    except Exception as e:
        print(f"\n❌ Smoke test failed with an error: {e}")
