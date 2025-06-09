import itertools
import random
from typing import List, Tuple, Dict

BASES = ['A', 'T', 'G', 'C']
CODON_LIST = [''.join(p) for p in itertools.product(BASES, repeat=3)]

SPECIALS = ['[PAD]', '[MASK]', '[CLS]', '[SEP]']

VOCAB = SPECIALS + CODON_LIST
VOCAB_SIZE = len(VOCAB)

CODON_TO_ID: Dict[str, int] = {token: i for i, token in enumerate(VOCAB)}
ID_TO_CODON: Dict[int, str] = {i: token for i, token in enumerate(VOCAB)}

CODON_PAIR_INDEX: Dict[Tuple[str, str], int] = {
    pair: i for i, pair in enumerate(itertools.product(CODON_LIST, repeat=2))
}
PAIR_VOCAB_SIZE = len(CODON_PAIR_INDEX)

PAD_ID = CODON_TO_ID['[PAD]']
MASK_ID = CODON_TO_ID['[MASK]']
CLS_ID = CODON_TO_ID['[CLS]']
SEP_ID = CODON_TO_ID['[SEP]']

class CodonTokenizer:
    """
    Tokenizes nucleotide sequences into codons and codon pairs.
    Includes special tokens [PAD], [MASK], [CLS], [SEP].
    Provides masking functionality based on Absci's contextual rule (BERT-like).
    """
    def __init__(self):
        self.codon_to_id = CODON_TO_ID
        self.id_to_codon = ID_TO_CODON
        self.codon_pair_index = CODON_PAIR_INDEX
        self.vocab = VOCAB
        self.vocab_size = VOCAB_SIZE
        self.pair_vocab_size = PAIR_VOCAB_SIZE
        self.pad_id = PAD_ID
        self.mask_id = MASK_ID
        self.cls_id = CLS_ID
        self.sep_id = SEP_ID
        self.codon_ids = [self.codon_to_id[c] for c in CODON_LIST]

    def _split_into_codons(self, cds: str) -> List[str]:
        """Splits a CDS string into a list of codons."""
        if len(cds) % 3 != 0:
            print(f"Warning: CDS length {len(cds)} is not divisible by 3. Truncating.")
            cds = cds[:len(cds) - (len(cds) % 3)]
        return [cds[i:i+3] for i in range(0, len(cds), 3)]

    def encode_cds(self, cds: str) -> Tuple[List[int], List[int]]:
        """
        Encodes a CDS string into token IDs and codon pair IDs.
        Adds [CLS] at the beginning and [SEP] at the end of token IDs.
        Pair IDs correspond to adjacent codon pairs in the original sequence.
        """
        codons = self._split_into_codons(cds)
        if not codons:
            return [self.cls_id, self.sep_id], []

        token_ids = [self.cls_id] + [self.codon_to_id.get(c, self.mask_id) for c in codons] + [self.sep_id]

        pair_ids = []
        for i in range(len(codons) - 1):
            pair = (codons[i], codons[i+1])
            pair_idx = self.codon_pair_index.get(pair)
            if pair_idx is not None:
                pair_ids.append(pair_idx)
            else:
                 print(f"Warning: Codon pair {pair} not found in index. Skipping.")


        return token_ids, pair_ids

    def decode(self, token_ids: List[int]) -> str:
        """Decodes a list of token IDs back into a string (codons joined)."""
        codons = [self.id_to_codon[id_] for id_ in token_ids if id_ not in [self.cls_id, self.sep_id, self.pad_id]]
        return "".join(codons)

    def mask(self, token_ids: List[int], mask_prob=0.15, random_prob=0.1, keep_prob=0.1) -> Tuple[List[int], List[int]]:
        """
        Applies BERT-style masking to token IDs.
        Based on "Absci contextual rule" (assumed BERT-like).
        Selects mask_prob fraction of tokens (excluding CLS, SEP, PAD).
        80% of selected -> [MASK]
        10% of selected -> random codon
        10% of selected -> original token

        Returns:
            Tuple[List[int], List[int]]: Masked token IDs and original labels
                                         (original token ID or PAD_ID if not masked).
        """
        masked_token_ids = list(token_ids)
        labels = [self.pad_id] * len(token_ids)

        eligible_indices = [
            i for i, token_id in enumerate(token_ids)
            if token_id not in [self.cls_id, self.sep_id, self.pad_id]
        ]

        num_to_mask = int(len(eligible_indices) * mask_prob)
        if num_to_mask == 0:
            return masked_token_ids, labels

        indices_to_mask = sorted(random.sample(eligible_indices, num_to_mask))

        for i in indices_to_mask:
            labels[i] = masked_token_ids[i]

            rand_val = random.random()
            if rand_val < (1.0 - random_prob - keep_prob):
                masked_token_ids[i] = self.mask_id
            elif rand_val < (1.0 - keep_prob):
                masked_token_ids[i] = random.choice(self.codon_ids)
            else:
                pass

        return masked_token_ids, labels

if __name__ == '__main__':
    tokenizer = CodonTokenizer()
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Codon pair vocab size: {tokenizer.pair_vocab_size}")
    print(f"CLS ID: {tokenizer.cls_id}, SEP ID: {tokenizer.sep_id}, PAD ID: {tokenizer.pad_id}, MASK ID: {tokenizer.mask_id}")

    cds_example = "ATGCGTTAACGTAAG"
    token_ids, pair_ids = tokenizer.encode_cds(cds_example)
    print(f"\nCDS: {cds_example}")
    print(f"Token IDs: {token_ids}")
    print(f"Pair IDs: {pair_ids}")
    print(f"Decoded: {tokenizer.decode(token_ids)}")

    masked_ids, labels = tokenizer.mask(token_ids)
    print(f"Masked IDs: {masked_ids}")
    print(f"Labels:     {labels}")

    cds_short = "ATG"
    token_ids_short, pair_ids_short = tokenizer.encode_cds(cds_short)
    print(f"\nCDS Short: {cds_short}")
    print(f"Token IDs: {token_ids_short}")
    print(f"Pair IDs: {pair_ids_short}") 