"""
DEPRECATED: build_encoded_tensors.py is specific to STREAM workflow

This script was designed for the 152-token STREAM architecture that used:
- AA_UNK tokens (e.g., 'M_UNK')  
- AA_CODON tokens (e.g., 'M_ATG')
- Complex masking for protein sequence prediction

The new v2 pipeline uses:
- Clean 68-token vocabulary with raw codons only
- Direct CDS tokenization via tools/retokenize_dataset.py
- Standard MLM training without STREAM complexity

For v2 dataset creation, use:
    python tools/retokenize_dataset.py --input data/raw --output data_v2

For STREAM experiments, use tokenizer_legacy.py or implement 
STREAM-specific tokenization separately.
"""

print(__doc__)
raise SystemExit("build_encoded_tensors.py is deprecated for v2 pipeline")
