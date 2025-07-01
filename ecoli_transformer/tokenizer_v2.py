"""
Tokenizer V2: Clean 68-token vocabulary for E. coli codon optimization.

Vocabulary: [PAD] [MASK] [CLS] [SEP] + 64 raw codons
- Eliminates dual representation (raw codons + AA-codon tokens)
- Direct DNA mapping for CAI, GC%, restriction site analysis
- Maintains sequence length = codon count for positional embeddings
"""

from typing import List, Tuple, Dict, Optional
import itertools

# --- Core Vocabulary Components ---
BASES = "ATGC"
SPECIAL_TOKENS = ['[PAD]', '[MASK]', '[CLS]', '[SEP]']
CODONS = [a+b+c for a in BASES for b in BASES for c in BASES]  # 64 codons
VOCAB = SPECIAL_TOKENS + CODONS

# --- Token Mappings ---
TOKEN_TO_ID: Dict[str, int] = {token: i for i, token in enumerate(VOCAB)}
ID_TO_TOKEN: Dict[int, str] = {i: token for i, token in enumerate(VOCAB)}

# --- Special Token IDs ---
PAD_ID = TOKEN_TO_ID['[PAD]']
MASK_ID = TOKEN_TO_ID['[MASK]']
CLS_ID = TOKEN_TO_ID['[CLS]']
SEP_ID = TOKEN_TO_ID['[SEP]']

# --- Genetic Code ---
GENETIC_CODE: Dict[str, str] = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
}

# --- Reverse mapping: amino acid to codons ---
AA_TO_CODONS: Dict[str, List[str]] = {}
for codon, aa in GENETIC_CODE.items():
    if aa not in AA_TO_CODONS:
        AA_TO_CODONS[aa] = []
    AA_TO_CODONS[aa].append(codon)

# --- Restriction sites for constraint checking ---
RESTRICTION_SITES = {
    "EcoRI": "GAATTC", "HindIII": "AAGCTT", "BamHI": "GGATCC",
    "BglII": "AGATCT", "XhoI": "CTCGAG",
}


class CodingTokenizerV2:
    """
    Clean codon tokenizer with 68-token vocabulary.
    Provides backward-compatible API for existing codebase.
    """
    
    def __init__(self):
        self.vocab = VOCAB
        self.vocab_size = len(VOCAB)
        self.token_to_id = TOKEN_TO_ID
        self.id_to_token = ID_TO_TOKEN
        
        # Special token IDs
        self.pad_id = PAD_ID
        self.mask_id = MASK_ID
        self.cls_id = CLS_ID
        self.sep_id = SEP_ID
        
        # Codon-specific mappings
        self.codon_to_id = {codon: TOKEN_TO_ID[codon] for codon in CODONS}
        self.id_to_codon = {TOKEN_TO_ID[codon]: codon for codon in CODONS}
        
    def encode(self, tokens: List[str]) -> List[int]:
        """Encode list of tokens to IDs."""
        return [self.token_to_id.get(token, self.mask_id) for token in tokens]
    
    def decode_to_str(self, token_ids: List[int]) -> str:
        """Decode token IDs back to DNA sequence string."""
        codons = []
        for token_id in token_ids:
            if token_id in self.id_to_codon:
                codons.append(self.id_to_codon[token_id])
            # Skip special tokens
        return "".join(codons)
    
    def id_to_codon_safe(self, token_id: int) -> str:
        """
        Convert token ID to codon string.
        Returns empty string for special tokens.
        """
        return self.id_to_codon.get(token_id, "")
    
    def encode_cds(self, cds: str, add_special_tokens: bool = True) -> Tuple[List[int], List[str]]:
        """
        Encode DNA coding sequence into token IDs.
        
        Args:
            cds: DNA sequence (will be upper-cased)
            add_special_tokens: Whether to add [CLS] and [SEP]
            
        Returns:
            (token_ids, codons): Token IDs and original codon strings
        """
        cds = cds.upper()
        # Split into codons, handle incomplete final codon
        codons = []
        for i in range(0, len(cds), 3):
            codon = cds[i:i+3]
            if len(codon) == 3:
                # Replace unknown codons with mask
                if codon in self.codon_to_id:
                    codons.append(codon)
                else:
                    # Handle 'NNN' or invalid codons
                    codons.append('[MASK]')
        
        tokens = codons[:]
        if add_special_tokens:
            tokens = ['[CLS]'] + tokens + ['[SEP]']
            
        token_ids = self.encode(tokens)
        return token_ids, codons
    
    def get_codons_for_aa(self, aa: str) -> List[str]:
        """Get all codons that encode a given amino acid."""
        return AA_TO_CODONS.get(aa, [])
    
    def get_aa_for_codon(self, codon: str) -> str:
        """Get amino acid encoded by a codon."""
        return GENETIC_CODE.get(codon, '')
    
    def is_stop_codon(self, codon: str) -> bool:
        """Check if codon is a stop codon."""
        return GENETIC_CODE.get(codon) == '*'
    
    # --- Backward compatibility aliases ---
    def decode(self, token_ids: List[int]) -> str:
        return self.decode_to_str(token_ids)

    # --- DEPRECATED STREAM METHODS ---
    # These methods were specific to the 152-token STREAM architecture
    
    def protein_to_aa_unk_tokens(self, protein_seq: str):
        raise NotImplementedError("protein_to_aa_unk_tokens is deprecated in tokenizer v2")
    
    def get_synonymous_tokens(self, aa_unk_token: str):
        raise NotImplementedError("get_synonymous_tokens is deprecated in tokenizer v2")
    
    def cds_to_aa_codon_tokens(self, cds: str):
        raise NotImplementedError("cds_to_aa_codon_tokens is deprecated in tokenizer v2")
        return self.decode_to_str(token_ids)


def validate_tokenizer():
    """Self-test for tokenizer v2."""
    tokenizer = CodingTokenizerV2()
    
    print("=== TOKENIZER V2 VALIDATION ===")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Expected: 68 (4 special + 64 codons)")
    assert tokenizer.vocab_size == 68, f"Expected 68 tokens, got {tokenizer.vocab_size}"
    
    # Test basic encoding/decoding
    test_sequence = "ATGAAATTT"  # M K F
    token_ids, codons = tokenizer.encode_cds(test_sequence)
    decoded = tokenizer.decode_to_str(token_ids)
    print(f"Test sequence: {test_sequence}")
    print(f"Token IDs: {token_ids}")
    print(f"Decoded: {decoded}")
    assert test_sequence == decoded, f"Round-trip failed: {test_sequence} != {decoded}"
    
    # Test special token handling
    special_id = tokenizer.pad_id
    special_codon = tokenizer.id_to_codon_safe(special_id)
    print(f"Special token [PAD] (ID {special_id}) -> codon: '{special_codon}'")
    assert special_codon == "", "Special tokens should return empty codon"
    
    # Test genetic code
    aa = tokenizer.get_aa_for_codon("ATG")
    print(f"ATG encodes: {aa}")
    assert aa == "M", "ATG should encode methionine"
    
    # Test stop codon
    is_stop = tokenizer.is_stop_codon("TAA")
    print(f"TAA is stop codon: {is_stop}")
    assert is_stop, "TAA should be recognized as stop codon"
    
    print("✅ All tokenizer v2 tests passed!")
    return True


if __name__ == "__main__":
    validate_tokenizer()
    # --- DEPRECATED STREAM METHODS ---
    # These methods were specific to the 152-token STREAM architecture
    # and are not compatible with the clean 68-token vocabulary
    
    def protein_to_aa_unk_tokens(self, protein_seq: str) -> List[str]:
        """DEPRECATED: STREAM-specific method not available in tokenizer v2."""
        raise NotImplementedError(
            "protein_to_aa_unk_tokens is deprecated in tokenizer v2. "
            "STREAM workflow used 152-token vocabulary with AA_UNK tokens. "
            "Use the clean 68-token approach or revert to tokenizer_legacy.py for STREAM experiments."
        )
    
    def get_synonymous_tokens(self, aa_unk_token: str) -> List[str]:
        """DEPRECATED: STREAM-specific method not available in tokenizer v2."""
        raise NotImplementedError(
            "get_synonymous_tokens is deprecated in tokenizer v2. "
            "STREAM workflow used AA_UNK -> AA_CODON token mapping. "
            "Use get_codons_for_aa() with raw amino acid letters instead."
        )
    
    def cds_to_aa_codon_tokens(self, cds: str) -> List[str]:
        """DEPRECATED: STREAM-specific method not available in tokenizer v2."""
        raise NotImplementedError(
            "cds_to_aa_codon_tokens is deprecated in tokenizer v2. "
            "STREAM workflow used AA_CODON composite tokens (e.g., 'M_ATG'). "
            "Use encode_cds() to get raw codon tokens instead."
        )
