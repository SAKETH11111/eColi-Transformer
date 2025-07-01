"""
DEPRECATED: This tokenizer has been replaced by tokenizer_v2.py

The original 152-token tokenizer with dual representation (raw codons + AA-codon tokens)
caused the AGA repetition bug and has been superseded by the clean 68-token vocabulary.

Use: from ecoli_transformer.tokenizer_v2 import CodingTokenizerV2 as CodonTokenizer
"""

raise ImportError(
    "The original tokenizer.py is deprecated. "
    "Use: from ecoli_transformer.tokenizer_v2 import CodingTokenizerV2 as CodonTokenizer"
)