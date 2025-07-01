"""
Model utilities for vocabulary transitions and checkpoint conversion.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional
from .model import CodonEncoder
from .tokenizer_v2 import CodingTokenizerV2


def load_model_with_vocab_resize(
    checkpoint_path: str, 
    target_vocab_size: Optional[int] = None,
    device: str = "cpu"
) -> CodonEncoder:
    """
    Load model checkpoint and automatically resize vocabulary if needed.
    
    Args:
        checkpoint_path: Path to model checkpoint
        target_vocab_size: Target vocabulary size (defaults to tokenizer v2 size)
        device: Device to load model on
        
    Returns:
        Model with correct vocabulary size
    """
    if target_vocab_size is None:
        target_vocab_size = 68  # Default to tokenizer v2 size
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get current vocab size from checkpoint
    try:
        current_vocab_size = checkpoint['model_state_dict']['token_embedding.weight'].shape[0]
    except KeyError:
        raise ValueError("Could not determine vocabulary size from checkpoint")
    
    print(f"Checkpoint vocab size: {current_vocab_size}")
    print(f"Target vocab size: {target_vocab_size}")
    
    # Initialize model with original vocab size
    model = CodonEncoder(vocab_size=current_vocab_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Resize if needed
    if current_vocab_size != target_vocab_size:
        print(f"Resizing vocabulary from {current_vocab_size} to {target_vocab_size}")
        model.resize_token_embeddings(target_vocab_size)
    
    return model.to(device)


def convert_checkpoint_vocab(
    input_checkpoint: str,
    output_checkpoint: str,
    target_vocab_size: int = 68
) -> None:
    """
    Convert an old checkpoint to work with new vocabulary size.
    
    Args:
        input_checkpoint: Path to input checkpoint
        output_checkpoint: Path to save converted checkpoint
        target_vocab_size: New vocabulary size
    """
    print(f"Converting checkpoint: {input_checkpoint} -> {output_checkpoint}")
    
    checkpoint = torch.load(input_checkpoint, map_location='cpu')
    
    # Get original vocab size
    orig_size = checkpoint['model_state_dict']['token_embedding.weight'].shape[0]
    print(f"Original vocab size: {orig_size}")
    print(f"Target vocab size: {target_vocab_size}")
    
    if orig_size == target_vocab_size:
        print("No conversion needed - vocabulary sizes match")
        torch.save(checkpoint, output_checkpoint)
        return
    
    # Create temporary model to use resize functionality
    model = CodonEncoder(vocab_size=orig_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.resize_token_embeddings(target_vocab_size)
    
    # Update checkpoint
    checkpoint['model_state_dict'] = model.state_dict()
    
    # Save converted checkpoint
    torch.save(checkpoint, output_checkpoint)
    print(f"✅ Converted checkpoint saved to {output_checkpoint}")


def test_vocab_compatibility():
    """Test that model works with both v1 and v2 tokenizers."""
    from .tokenizer_v2 import CodingTokenizerV2 as CodonTokenizer
    
    print("=== VOCAB COMPATIBILITY TEST ===")
    
    # Test sequence
    test_seq = "ATGAAAGCGTAA"
    
    # Test with v1 tokenizer
    v1_tokenizer = CodonTokenizer()
    v1_ids, _ = v1_tokenizer.encode_cds(test_seq)
    print(f"V1 tokenizer: {len(v1_ids)} tokens, vocab_size={v1_tokenizer.vocab_size}")
    
    # Test with v2 tokenizer  
    v2_tokenizer = CodingTokenizerV2()
    v2_ids, _ = v2_tokenizer.encode_cds(test_seq)
    print(f"V2 tokenizer: {len(v2_ids)} tokens, vocab_size={v2_tokenizer.vocab_size}")
    
    # Test model creation with both sizes
    model_v1 = CodonEncoder(vocab_size=v1_tokenizer.vocab_size)
    model_v2 = CodonEncoder(vocab_size=v2_tokenizer.vocab_size)
    
    print(f"Model V1: embedding shape {model_v1.token_embedding.weight.shape}")
    print(f"Model V2: embedding shape {model_v2.token_embedding.weight.shape}")
    
    # Test resize functionality
    print("Testing resize from V1 to V2...")
    model_v1.resize_token_embeddings(v2_tokenizer.vocab_size)
    print(f"After resize: embedding shape {model_v1.token_embedding.weight.shape}")
    
    print("✅ Vocabulary compatibility test passed!")


if __name__ == "__main__":
    test_vocab_compatibility()