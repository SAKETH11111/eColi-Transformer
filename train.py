#!/usr/bin/env python3
"""
Main entry point for training the E. coli Transformer model.
This script can be run from the project root directory.
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and run the training script
from ecoli_transformer.train import main

if __name__ == "__main__":
    main()
