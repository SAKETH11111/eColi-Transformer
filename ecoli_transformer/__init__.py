"""
E. coli Transformer package for codon analysis and prediction.
"""

from .model import CodonEncoder
from .tokenizer import CodonTokenizer
from .dataio import *

__version__ = "0.1.0"
__all__ = ["CodonEncoder", "CodonTokenizer"]
