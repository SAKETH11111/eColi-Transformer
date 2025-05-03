#!/usr/bin/env python3
"""
Profiles raw E. coli sequence data from .csv and .xlsx files.

Reads files from an input directory, detects sequence and CAI columns,
and calculates various statistics like counts, rule violations, length
distributions, and GC content.
"""

import argparse
from pathlib import Path
import sys
import os
import pandas as pd # Keep pandas for type hint if needed, or remove if not

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Import from the module now that the path is set
try:
    from ecoli_transformer.dataio import load_raw_dir, basic_stats
except ImportError as e:
    print(f"Error importing dataio module: {e}")
    print(f"Ensure ecoli_transformer/dataio.py exists and Python path is correct: {sys.path}")
    sys.exit(1)

# Removed unused imports like logging, Bio.Seq etc.

def main():
    parser = argparse.ArgumentParser(description="Profile raw E. coli sequence data.")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing raw .csv and .xlsx files.")
    args = parser.parse_args()

    input_path = Path(args.input_dir)

    if not input_path.is_dir():
        print(f"Error: Input directory not found: {input_path}")
        sys.exit(1)

    print(f"Loading data from: {input_path.resolve()}")
    df = load_raw_dir(input_path)

    if df.empty:
        print(f"No raw files found or loaded successfully in {input_path}")
        sys.exit(0)

    print("Calculating statistics...")
    stats = basic_stats(df)

    # Pretty-print the report
    print("\n--- Raw Data Profile ---")
    print(f"Files read: {stats.get('files_processed', 0)}")
    print(f"Rows loaded: {stats.get('rows', 0):,}   (missing CAI: {stats.get('missing_cai', 0):,})")
    print(f"Duplicate CDS (md5): {stats.get('dup_md5', 0):,}")
    # Combine start/stop violations for the report line
    start_stop_violations = stats.get('start_violation', 0) + stats.get('stop_violation', 0)
    # Ensure we don't double-count by only reporting unique violations
    # Note: A sequence could violate both start and stop. The current calculation sums them.
    # A more precise unique count would require checking per sequence.
    # For this report, summing them gives an idea of total violations observed.
    print(f"Start/stop violation: {start_stop_violations:,}") # Corrected key access
    print(f"Internal stop codons : {stats.get('internal_stop', 0):,}")
    print(f"Length (nt)  : min {stats.get('len_min', 0):,} | median {stats.get('len_median', 0):,} | max {stats.get('len_max', 0):,}")
    print(f"GC%           : {stats.get('gc_mean', 0.0):.1f} ± {stats.get('gc_std', 0.0):.1f}")
    print("------------------------\n")

if __name__ == "__main__":
    main() 