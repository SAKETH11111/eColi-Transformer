#!/usr/bin/env python3
"""Debug script to test model outputs and identify the AGA repetition issue."""

import torch
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from ecoli_transformer.model import CodonEncoder
from ecoli_transformer.tokenizer_v2 import CodingTokenizerV2 as CodonTokenizer

def debug_model_outputs():
    """Test model with simple inputs to see if it generates AGA repetitions."""
    
    # Initialize tokenizer and model
    tokenizer = CodonTokenizer()
    device = torch.device("cpu")
    
    # Create a simple test sequence: ATG + 3 masked codons + TAA
    test_sequence = "ATGNNNNNNNNTAA"
    token_ids, _ = tokenizer.encode_cds(test_sequence)
    
    print("=== DEBUGGING MODEL OUTPUTS ===")
    print(f"Test sequence: {test_sequence}")
    print(f"Token IDs: {token_ids}")
    print(f"Tokens: {[tokenizer.id_to_token[id] for id in token_ids]}")
    print()
    
    # Check if we have a trained model
    model_path = "checkpoints/stream_full_finetuned.pt"
    if not Path(model_path).exists():
        print(f"Model checkpoint not found at {model_path}")
        print("Available checkpoints:")
        checkpoints_dir = Path("checkpoints")
        if checkpoints_dir.exists():
            for cp in checkpoints_dir.glob("*.pt"):
                print(f"  - {cp}")
        return
    
    # Load model
    try:
        checkpoint = torch.load(model_path, map_location=device)
        vocab_size = checkpoint['model_state_dict']['token_embedding.weight'].shape[0]
        model = CodonEncoder(vocab_size=vocab_size)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"Loaded model with vocab_size={vocab_size}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Test model inference on masked sequence
    input_tensor = torch.tensor([token_ids], dtype=torch.long)
    
    with torch.no_grad():
        mlm_logits, _, cai_pred, dg_pred = model(input_tensor)
    
    print("=== MODEL OUTPUTS ===")
    print(f"MLM logits shape: {mlm_logits.shape}")
    print(f"CAI prediction: {cai_pred.item() if cai_pred is not None else None}")
    print(f"dG prediction: {dg_pred.item() if dg_pred is not None else None}")
    
    # Check the first MASK position (should be index 2: [CLS, ATG, MASK, ...])
    mask_pos = 2
    mask_logits = mlm_logits[0, mask_pos, :]
    probs = torch.softmax(mask_logits, dim=-1)
    log_probs = torch.log_softmax(mask_logits, dim=-1)
    
    print(f"\n=== MASK POSITION {mask_pos} ANALYSIS ===")
    
    # Get top 10 predictions
    top_k = torch.topk(probs, 10)
    print("Top 10 token predictions:")
    for i in range(10):
        token_id = int(top_k.indices[i])
        prob = float(top_k.values[i])
        log_prob = float(log_probs[token_id])
        token = tokenizer.id_to_token.get(token_id, f"UNK_{token_id}")
        print(f"  {i+1:2d}. {token:>8s} (ID:{token_id:3d}) - Prob: {prob:.6f}, LogProb: {log_prob:.4f}")
    
    # Specifically check AGA
    aga_id = tokenizer.token_to_id.get('AGA')
    if aga_id is not None:
        aga_prob = float(probs[aga_id])
        aga_log_prob = float(log_probs[aga_id])
        print(f"\nAGA codon (ID:{aga_id}):")
        print(f"  Probability: {aga_prob:.6f}")
        print(f"  Log Probability: {aga_log_prob:.4f}")
        print(f"  Rank: {int((probs > aga_prob).sum()) + 1}")
    
    # Check if model is outputting uniform or near-uniform distributions
    entropy = -(probs * torch.log(probs + 1e-8)).sum()
    max_entropy = torch.log(torch.tensor(float(len(probs))))
    print(f"\nDistribution analysis:")
    print(f"  Entropy: {entropy:.4f}")
    print(f"  Max possible entropy: {max_entropy:.4f}")
    print(f"  Normalized entropy: {entropy/max_entropy:.4f}")
    
    if entropy/max_entropy > 0.9:
        print("  ⚠️  Distribution is nearly uniform - model may be undertrained!")
    
    # Test if the issue is in the beam search scoring
    print(f"\n=== BEAM SEARCH SCORING TEST ===")
    
    # Simulate what happens in beam search
    lambda_cai = 1.0
    lambda_dg = 1.0
    
    # Top candidate
    best_token_id = int(top_k.indices[0])
    best_log_prob = float(log_probs[best_token_id])
    
    # Calculate beam search score
    base_score = 0.0  # Initial beam score
    token_score = base_score + best_log_prob
    if cai_pred is not None:
        cai_contrib = lambda_cai * float(torch.sigmoid(cai_pred).item())
        token_score += cai_contrib
    if dg_pred is not None:
        dg_contrib = lambda_dg * float(torch.sigmoid(dg_pred).item())
        token_score += dg_contrib
    
    print(f"Best token: {tokenizer.id_to_token.get(best_token_id, 'UNK')}")
    print(f"Log prob contribution: {best_log_prob:.4f}")
    print(f"CAI contribution: {cai_contrib:.4f}")
    print(f"dG contribution: {dg_contrib:.4f}")
    print(f"Total score: {token_score:.4f}")

if __name__ == "__main__":
    debug_model_outputs()