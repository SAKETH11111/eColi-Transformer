"""
Unit tests for tokenizer v2.
"""

# import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from ecoli_transformer.tokenizer_v2 import CodingTokenizerV2


class TestCodingTokenizerV2:
    """Test suite for the new 68-token tokenizer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tokenizer = CodingTokenizerV2()
    
    def test_vocab_size(self):
        """Test vocabulary size is exactly 68."""
        assert self.tokenizer.vocab_size == 68
        assert len(self.tokenizer.vocab) == 68
        
    def test_special_tokens(self):
        """Test special tokens are properly mapped."""
        assert self.tokenizer.pad_id == 0
        assert self.tokenizer.mask_id == 1
        assert self.tokenizer.cls_id == 2
        assert self.tokenizer.sep_id == 3
        
        assert self.tokenizer.token_to_id['[PAD]'] == 0
        assert self.tokenizer.token_to_id['[MASK]'] == 1
        assert self.tokenizer.token_to_id['[CLS]'] == 2
        assert self.tokenizer.token_to_id['[SEP]'] == 3
    
    def test_codon_tokens(self):
        """Test all 64 codons are in vocabulary."""
        # Check a few specific codons
        assert 'ATG' in self.tokenizer.token_to_id  # Start codon
        assert 'TAA' in self.tokenizer.token_to_id  # Stop codon
        assert 'GCA' in self.tokenizer.token_to_id  # Alanine
        assert 'AGA' in self.tokenizer.token_to_id  # Arginine (our problem child)
        
        # Check all codons are present
        bases = "ATGC"
        all_codons = [a+b+c for a in bases for b in bases for c in bases]
        
        for codon in all_codons:
            assert codon in self.tokenizer.token_to_id, f"Codon {codon} missing from vocabulary"
    
    def test_round_trip_encoding(self):
        """Test encode -> decode is identity."""
        test_sequences = [
            "ATG",  # Single codon
            "ATGTAA",  # Start + stop
            "ATGAAAGCGTAA",  # Short gene
            "ATGAAAGCGAAATTCGCGCGCTAA",  # Longer sequence
        ]
        
        for seq in test_sequences:
            token_ids, codons = self.tokenizer.encode_cds(seq, add_special_tokens=True)
            decoded = self.tokenizer.decode_to_str(token_ids)
            assert decoded == seq, f"Round-trip failed: {seq} -> {decoded}"
    
    def test_id_to_codon_safe(self):
        """Test safe codon extraction from token IDs."""
        # Test regular codon tokens
        atg_id = self.tokenizer.token_to_id['ATG']
        assert self.tokenizer.id_to_codon_safe(atg_id) == 'ATG'
        
        # Test special tokens return empty string
        assert self.tokenizer.id_to_codon_safe(self.tokenizer.pad_id) == ''
        assert self.tokenizer.id_to_codon_safe(self.tokenizer.mask_id) == ''
        assert self.tokenizer.id_to_codon_safe(self.tokenizer.cls_id) == ''
        assert self.tokenizer.id_to_codon_safe(self.tokenizer.sep_id) == ''
        
        # Test invalid ID
        assert self.tokenizer.id_to_codon_safe(999) == ''
    
    def test_genetic_code(self):
        """Test amino acid mappings are correct."""
        # Test some key mappings
        assert self.tokenizer.get_aa_for_codon('ATG') == 'M'  # Methionine
        assert self.tokenizer.get_aa_for_codon('TAA') == '*'  # Stop
        assert self.tokenizer.get_aa_for_codon('TAG') == '*'  # Stop
        assert self.tokenizer.get_aa_for_codon('TGA') == '*'  # Stop
        assert self.tokenizer.get_aa_for_codon('TTT') == 'F'  # Phenylalanine
        assert self.tokenizer.get_aa_for_codon('TTC') == 'F'  # Phenylalanine
        assert self.tokenizer.get_aa_for_codon('AGA') == 'R'  # Arginine (our former problem)
        
        # Test invalid codon
        assert self.tokenizer.get_aa_for_codon('XXX') == ''
    
    def test_stop_codon_detection(self):
        """Test stop codon identification."""
        stop_codons = ['TAA', 'TAG', 'TGA']
        non_stop_codons = ['ATG', 'AAA', 'GGG', 'TTT']
        
        for codon in stop_codons:
            assert self.tokenizer.is_stop_codon(codon), f"{codon} should be a stop codon"
        
        for codon in non_stop_codons:
            assert not self.tokenizer.is_stop_codon(codon), f"{codon} should not be a stop codon"
    
    def test_synonymous_codons(self):
        """Test getting synonymous codons for amino acids."""
        # Test methionine (only one codon)
        met_codons = self.tokenizer.get_codons_for_aa('M')
        assert met_codons == ['ATG']
        
        # Test stop codons
        stop_codons = self.tokenizer.get_codons_for_aa('*')
        assert set(stop_codons) == {'TAA', 'TAG', 'TGA'}
        
        # Test amino acid with multiple codons (e.g., Serine)
        ser_codons = self.tokenizer.get_codons_for_aa('S')
        assert len(ser_codons) == 6  # TCT, TCC, TCA, TCG, AGT, AGC
        assert 'TCT' in ser_codons
        assert 'AGC' in ser_codons
        
        # Test invalid amino acid
        assert self.tokenizer.get_codons_for_aa('X') == []
    
    def test_encode_cds_with_special_tokens(self):
        """Test CDS encoding with and without special tokens."""
        seq = "ATGAAATAA"
        
        # With special tokens
        token_ids_with, _ = self.tokenizer.encode_cds(seq, add_special_tokens=True)
        assert token_ids_with[0] == self.tokenizer.cls_id
        assert token_ids_with[-1] == self.tokenizer.sep_id
        assert len(token_ids_with) == 5  # [CLS] + 3 codons + [SEP]
        
        # Without special tokens
        token_ids_without, _ = self.tokenizer.encode_cds(seq, add_special_tokens=False)
        assert token_ids_without[0] != self.tokenizer.cls_id
        assert token_ids_without[-1] != self.tokenizer.sep_id
        assert len(token_ids_without) == 3  # Just 3 codons
    
    def test_invalid_sequences(self):
        """Test handling of invalid or incomplete sequences."""
        # Incomplete codon at end
        token_ids, codons = self.tokenizer.encode_cds("ATGAA", add_special_tokens=False)
        assert len(token_ids) == 1  # Only ATG
        assert len(codons) == 1
        
        # Empty sequence
        token_ids, codons = self.tokenizer.encode_cds("", add_special_tokens=False)
        assert len(token_ids) == 0
        assert len(codons) == 0
        
        # Sequence with unknown characters (replaced with mask)
        token_ids, codons = self.tokenizer.encode_cds("ATGNNN", add_special_tokens=False)
        assert len(token_ids) == 2
        assert token_ids[1] == self.tokenizer.mask_id  # NNN -> [MASK]
    
    def test_no_dual_representation(self):
        """Test that v2 tokenizer eliminates dual representation issue."""
        # In v1, both 'AGA' and 'R_AGA' could represent the same codon
        # In v2, only 'AGA' should exist
        
        assert 'AGA' in self.tokenizer.token_to_id
        assert 'R_AGA' not in self.tokenizer.token_to_id
        assert 'M_ATG' not in self.tokenizer.token_to_id
        assert 'A_GCA' not in self.tokenizer.token_to_id
        
        # Verify no composite tokens exist
        for token in self.tokenizer.vocab:
            if '_' in token:
                # Only special tokens should have underscores
                assert token.startswith('[') and token.endswith(']'), f"Found composite token: {token}"


def test_backward_compatibility():
    """Test that v2 tokenizer provides same API as v1."""
    tokenizer = CodingTokenizerV2()
    
    # Test that all required methods exist
    assert hasattr(tokenizer, 'encode')
    assert hasattr(tokenizer, 'decode')
    assert hasattr(tokenizer, 'decode_to_str')
    assert hasattr(tokenizer, 'encode_cds')
    assert hasattr(tokenizer, 'get_aa_for_codon')
    assert hasattr(tokenizer, 'get_codons_for_aa')
    assert hasattr(tokenizer, 'is_stop_codon')
    assert hasattr(tokenizer, 'id_to_codon_safe')
    
    # Test that all required attributes exist
    assert hasattr(tokenizer, 'vocab_size')
    assert hasattr(tokenizer, 'pad_id')
    assert hasattr(tokenizer, 'mask_id')
    assert hasattr(tokenizer, 'cls_id')
    assert hasattr(tokenizer, 'sep_id')


if __name__ == "__main__":
    # Run basic tests without pytest
    tokenizer = CodingTokenizerV2()
    
    print("=== RUNNING TOKENIZER V2 TESTS ===")
    
    # Test vocab size
    assert tokenizer.vocab_size == 68
    print("✅ Vocabulary size test passed")
    
    # Test round-trip
    test_seq = "ATGAAAGCGTAA"
    token_ids, _ = tokenizer.encode_cds(test_seq)
    decoded = tokenizer.decode_to_str(token_ids)
    assert decoded == test_seq
    print("✅ Round-trip encoding test passed")
    
    # Test special tokens
    assert tokenizer.id_to_codon_safe(tokenizer.pad_id) == ""
    assert tokenizer.id_to_codon_safe(tokenizer.token_to_id['ATG']) == "ATG"
    print("✅ Special token handling test passed")
    
    # Test genetic code
    assert tokenizer.get_aa_for_codon('ATG') == 'M'
    assert tokenizer.is_stop_codon('TAA')
    print("✅ Genetic code test passed")
    
    print("✅ All tokenizer v2 tests passed!")