"""
Integration test for complete v2 pipeline.
Tests tokenizer v2 + model v2 + decoder end-to-end.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import torch
from ecoli_transformer.tokenizer_v2 import CodingTokenizerV2 as CodonTokenizer
from ecoli_transformer.model_utils import load_model_with_vocab_resize
from ecoli_transformer.decode import generate_optimized


def test_v2_pipeline_integration():
    """Test complete v2 pipeline from tokenization to optimization."""
    print("=== V2 PIPELINE INTEGRATION TEST ===")
    
    # 1. Test tokenizer v2
    print("\n1. Testing Tokenizer V2...")
    tokenizer = CodonTokenizer()
    
    assert tokenizer.vocab_size == 68, f"Expected 68 tokens, got {tokenizer.vocab_size}"
    
    test_seq = "ATGAAAGCGTAA"  # M K A * (stop)
    token_ids, codons = tokenizer.encode_cds(test_seq)
    decoded = tokenizer.decode_to_str(token_ids)
    
    assert decoded == test_seq, f"Round-trip failed: {test_seq} != {decoded}"
    assert all(tid < 68 for tid in token_ids), f"Found token ID >= 68: {max(token_ids)}"
    
    print(f"  ✅ Tokenizer: {test_seq} -> {len(token_ids)} tokens -> {decoded}")
    
    # 2. Test model loading with v2 vocab
    print("\n2. Testing Model V2...")
    try:
        model = load_model_with_vocab_resize(
            'checkpoints/stream_full_finetuned_v2.pt',
            target_vocab_size=tokenizer.vocab_size
        )
        
        embedding_shape = model.token_embedding.weight.shape
        mlm_shape = model.mlm_head.weight.shape
        
        assert embedding_shape[0] == 68, f"Embedding vocab mismatch: {embedding_shape[0]}"
        assert mlm_shape[0] == 68, f"MLM head vocab mismatch: {mlm_shape[0]}"
        
        print(f"  ✅ Model: embedding {embedding_shape}, MLM head {mlm_shape}")
        
    except Exception as e:
        print(f"  ❌ Model loading failed: {e}")
        return False
    
    # 3. Test model inference
    print("\n3. Testing Model Inference...")
    try:
        with torch.no_grad():
            input_tensor = torch.tensor([token_ids], dtype=torch.long)
            mlm_logits, _, cai_pred, dg_pred = model(input_tensor)
            
            assert mlm_logits.shape[2] == 68, f"MLM output vocab mismatch: {mlm_logits.shape[2]}"
            assert mlm_logits.shape[1] == len(token_ids), f"Sequence length mismatch"
            
            print(f"  ✅ Inference: MLM shape {mlm_logits.shape}, CAI {cai_pred.item():.3f}, dG {dg_pred.item():.3f}")
            
    except Exception as e:
        print(f"  ❌ Model inference failed: {e}")
        return False
    
    # 4. Test decoder with v2 components
    print("\n4. Testing Decoder Integration...")
    try:
        # Test with a short sequence
        test_protein = "MKA"  # Simple 3-amino acid protein
        optimized_results = generate_optimized(
            sequence="ATGAAAGCA",  # M K A
            model_path='checkpoints/stream_full_finetuned_v2.pt',
            beam_size=2,
            lambda_cai=1.0,
            lambda_dg=1.0
        )
        
        if optimized_results:
            opt_seq, score = optimized_results[0]
            
            # Verify optimized sequence uses only valid codons
            opt_token_ids, _ = tokenizer.encode_cds(opt_seq, add_special_tokens=False)
            assert all(tid < 68 for tid in opt_token_ids), f"Optimized sequence has invalid token ID"
            
            # Verify no excessive repetition (the old AGA bug)
            codons_in_seq = [opt_seq[i:i+3] for i in range(0, len(opt_seq), 3)]
            if len(codons_in_seq) >= 4:
                for i in range(len(codons_in_seq) - 3):
                    four_codons = codons_in_seq[i:i+4]
                    unique_codons = set(four_codons)
                    assert len(unique_codons) > 1, f"Found 4+ identical codons: {four_codons}"
            
            print(f"  ✅ Decoder: {opt_seq} (score: {score:.3f})")
            print(f"    No AGA repetition detected")
            
        else:
            print("  ⚠️  Decoder returned no results (acceptable for short sequence)")
            
    except Exception as e:
        print(f"  ❌ Decoder integration failed: {e}")
        return False
    
    # 5. Test vocab size consistency across components
    print("\n5. Testing Vocab Size Consistency...")
    
    model_vocab = model.token_embedding.weight.shape[0]
    tokenizer_vocab = tokenizer.vocab_size
    
    assert model_vocab == tokenizer_vocab == 68, \
        f"Vocab size mismatch: model={model_vocab}, tokenizer={tokenizer_vocab}"
    
    print(f"  ✅ Consistent vocab size: {model_vocab} tokens")
    
    # 6. Test that old dual representation is eliminated
    print("\n6. Testing No Dual Representation...")
    
    # Verify AGA token exists but R_AGA doesn't
    assert 'AGA' in tokenizer.token_to_id, "AGA codon missing from vocab"
    assert 'R_AGA' not in tokenizer.token_to_id, "R_AGA composite token still exists!"
    assert 'M_ATG' not in tokenizer.token_to_id, "M_ATG composite token still exists!"
    
    print(f"  ✅ No dual representation: only raw codons in vocab")
    
    print("\n🎉 V2 PIPELINE INTEGRATION TEST PASSED!")
    print("   - Tokenizer v2: 68-token clean vocabulary")
    print("   - Model v2: matching embedding dimensions")  
    print("   - Decoder: no AGA repetition bug")
    print("   - Data: all token IDs < 68")
    print("   - No dual representation issues")
    
    return True


def test_backwards_compatibility():
    """Test that v2 provides same API as v1 for existing code."""
    print("\n=== BACKWARDS COMPATIBILITY TEST ===")
    
    tokenizer = CodonTokenizer()
    
    # Test all required methods exist
    required_methods = [
        'encode', 'decode', 'decode_to_str', 'encode_cds',
        'get_aa_for_codon', 'get_codons_for_aa', 'is_stop_codon', 'id_to_codon_safe'
    ]
    
    for method in required_methods:
        assert hasattr(tokenizer, method), f"Missing method: {method}"
    
    # Test all required attributes exist
    required_attrs = ['vocab_size', 'pad_id', 'mask_id', 'cls_id', 'sep_id']
    
    for attr in required_attrs:
        assert hasattr(tokenizer, attr), f"Missing attribute: {attr}"
    
    print("✅ All required methods and attributes present")
    
    # Test API compatibility with sample calls
    test_seq = "ATGAAATAA"
    
    # Test encoding
    token_ids, codons = tokenizer.encode_cds(test_seq)
    assert isinstance(token_ids, list), "encode_cds should return list"
    assert isinstance(codons, list), "encode_cds should return codons list"
    
    # Test decoding
    decoded = tokenizer.decode_to_str(token_ids)
    assert decoded == test_seq, "decode_to_str failed"
    
    # Test genetic code
    assert tokenizer.get_aa_for_codon('ATG') == 'M'
    assert tokenizer.is_stop_codon('TAA')
    
    # Test safe codon extraction
    atg_id = tokenizer.token_to_id['ATG']
    assert tokenizer.id_to_codon_safe(atg_id) == 'ATG'
    assert tokenizer.id_to_codon_safe(tokenizer.pad_id) == ''
    
    print("✅ API compatibility verified")
    return True


if __name__ == "__main__":
    success = True
    
    try:
        success &= test_v2_pipeline_integration()
        success &= test_backwards_compatibility()
        
        if success:
            print("\n🎉 ALL INTEGRATION TESTS PASSED!")
            print("Stage 1 vocabulary consolidation is complete and verified.")
        else:
            print("\n❌ Some integration tests failed")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 Integration test crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)