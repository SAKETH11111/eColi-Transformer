import itertools
import random
from typing import List, Tuple, Dict

# --- Foundational Vocabulary Components ---
BASES = ['A', 'T', 'G', 'C']
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
STOP_CODON = '*'

# --- Genetic Code and Codon Mappings ---
AA_TO_CODONS: Dict[str, List[str]] = {
    'A': ['GCT', 'GCC', 'GCA', 'GCG'], 'C': ['TGT', 'TGC'], 'D': ['GAT', 'GAC'],
    'E': ['GAA', 'GAG'], 'F': ['TTT', 'TTC'], 'G': ['GGT', 'GGC', 'GGA', 'GGG'],
    'H': ['CAT', 'CAC'], 'I': ['ATT', 'ATC', 'ATA'], 'K': ['AAA', 'AAG'],
    'L': ['TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG'], 'M': ['ATG'],
    'N': ['AAT', 'AAC'], 'P': ['CCT', 'CCC', 'CCA', 'CCG'], 'Q': ['CAA', 'CAG'],
    'R': ['CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'], 'S': ['TCT', 'TCC', 'TCA', 'TCG', 'AGT', 'AGC'],
    'T': ['ACT', 'ACC', 'ACA', 'ACG'], 'V': ['GTT', 'GTC', 'GTA', 'GTG'],
    'W': ['TGG'], 'Y': ['TAT', 'TAC'], STOP_CODON: ['TAA', 'TAG', 'TGA']
}
CODON_TO_AA: Dict[str, str] = {codon: aa for aa, codons in AA_TO_CODONS.items() for codon in codons}
CODON_LIST: List[str] = list(CODON_TO_AA.keys())

# --- STREAM-Specific Vocabulary ---
SPECIALS = ['[PAD]', '[MASK]', '[CLS]', '[SEP]']
AA_UNK_TOKENS = [f"{aa}_UNK" for aa in AMINO_ACIDS]
AA_CODON_TOKENS = [f"{CODON_TO_AA[codon]}_{codon}" for codon in CODON_LIST]

# --- Final Vocabulary Assembly ---
VOCAB = SPECIALS + CODON_LIST + AA_UNK_TOKENS + AA_CODON_TOKENS
VOCAB_SIZE = len(VOCAB)

# --- ID Mappings ---
TOKEN_TO_ID: Dict[str, int] = {token: i for i, token in enumerate(VOCAB)}
ID_TO_TOKEN: Dict[int, str] = {i: token for i, token in enumerate(VOCAB)}

PAD_ID = TOKEN_TO_ID['[PAD]']
MASK_ID = TOKEN_TO_ID['[MASK]']
CLS_ID = TOKEN_TO_ID['[CLS]']
SEP_ID = TOKEN_TO_ID['[SEP]']

# --- Other Constants ---
RESTRICTION_SITES = {
    "EcoRI": "GAATTC", "HindIII": "AAGCTT", "BamHI": "GGATCC",
    "BglII": "AGATCT", "XhoI": "CTCGAG",
}

class CodonTokenizer:
    """
    Tokenizer for the STREAM (CodonTransformer) architecture.
    Handles a vocabulary of codons, special tokens, AA-UNK tokens, and AA-codon tokens.
    """
    def __init__(self):
        self.token_to_id = TOKEN_TO_ID
        self.id_to_token = ID_TO_TOKEN
        self.vocab = VOCAB
        self.vocab_size = VOCAB_SIZE
        self.pad_id = PAD_ID
        self.mask_id = MASK_ID
        self.cls_id = CLS_ID
        self.sep_id = SEP_ID

    def encode(self, tokens: List[str]) -> List[int]:
        """Encodes a list of tokens into their corresponding IDs."""
        return [self.token_to_id.get(t, self.mask_id) for t in tokens]

    def decode_to_str(self, token_ids: List[int]) -> str:
        """Decodes a list of token IDs back into a DNA sequence string."""
        codons = []
        for id_ in token_ids:
            if id_ in self.id_to_token:
                token = self.id_to_token[id_]
                if '_' in token:
                    # Handles 'A_GCA' -> 'GCA'
                    codon = token.split('_')[1]
                    if len(codon) == 3 and all(c in BASES for c in codon):
                        codons.append(codon)
                elif len(token) == 3 and all(c in BASES for c in token):
                     # Handles raw codon tokens like 'ATG'
                    codons.append(token)
        return "".join(codons)

    def protein_to_aa_unk_tokens(self, protein_seq: str) -> List[str]:
        """
        Convert protein sequence to amino acid UNK tokens.
        'MALW' -> ['M_UNK', 'A_UNK', 'L_UNK', 'W_UNK']
        """
        return [f"{aa}_UNK" for aa in protein_seq]

    def get_synonymous_tokens(self, aa_unk_token: str) -> List[str]:
        """
        Get all possible codon-specific tokens for an amino acid UNK token.
        'A_UNK' -> ['A_GCA', 'A_GCC', 'A_GCG', 'A_GCT']
        """
        aa = aa_unk_token.split('_')[0]
        if aa in AA_TO_CODONS:
            return [f"{aa}_{codon}" for codon in AA_TO_CODONS[aa]]
        return []

    def validate_protein_translation(self, token_sequence: List[str], protein_sequence: str) -> bool:
        """
        Ensure the predicted sequence of AA_CODON tokens translates to the correct protein.
        """
        if len(token_sequence) != len(protein_sequence):
            return False
        
        for i, token in enumerate(token_sequence):
            expected_aa = protein_sequence[i]
            if not token.startswith(f"{expected_aa}_"):
                return False
        return True

    def cds_to_aa_codon_tokens(self, cds: str) -> List[str]:
        """Converts a CDS string into a list of amino acid-codon tokens."""
        codons = [cds[i:i+3] for i in range(0, len(cds), 3) if len(cds[i:i+3]) == 3]
        aa_codon_tokens = []
        for codon in codons:
            aa = CODON_TO_AA.get(codon)
            if aa:
                if aa == STOP_CODON:
                    break
                aa_codon_tokens.append(f"{aa}_{codon}")
        return aa_codon_tokens

if __name__ == '__main__':
    tokenizer = CodonTokenizer()
    print(f"STREAM Vocab Size: {tokenizer.vocab_size}")
    assert tokenizer.vocab_size == 4 + 64 + 20 + 64

    protein = "MKT"
    unk_tokens = tokenizer.protein_to_aa_unk_tokens(protein)
    print(f"\nProtein '{protein}' -> UNK tokens: {unk_tokens}")
    assert unk_tokens == ['M_UNK', 'K_UNK', 'T_UNK']

    synonymous = tokenizer.get_synonymous_tokens('A_UNK')
    print(f"\nSynonymous tokens for 'A_UNK': {synonymous}")
    assert synonymous == ['A_GCT', 'A_GCC', 'A_GCA', 'A_GCG']

    valid_translation = tokenizer.validate_protein_translation(['M_ATG', 'K_AAA', 'T_ACC'], 'MKT')
    print(f"\nTranslation validation (correct): {valid_translation}")
    assert valid_translation

    invalid_translation = tokenizer.validate_protein_translation(['M_ATG', 'A_GCT', 'T_ACC'], 'MKT')
    print(f"Translation validation (incorrect): {invalid_translation}")
    assert not invalid_translation
    
    print("\n✅ Tokenizer self-tests passed.")
