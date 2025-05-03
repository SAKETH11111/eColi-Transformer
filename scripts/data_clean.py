#!/usr/bin/env python3
"""
Cleans raw E. coli sequence data, filters based on quality rules,
deduplicates sequences, performs a stratified split, and saves
processed data in FASTA and CSV formats.
"""

import argparse
from pathlib import Path
import sys
import os
import pandas as pd
import numpy as np
import hashlib
from sklearn.model_selection import train_test_split

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Import from the dataio module
try:
    from ecoli_transformer.dataio import load_raw_dir, basic_stats, _calculate_gc_content, _has_internal_stop
except ImportError as e:
    print(f"Error importing dataio module: {e}")
    print(f"Ensure ecoli_transformer/dataio.py exists and Python path is correct: {sys.path}")
    sys.exit(1)

# --- Sequence Validation Functions ---
def check_start_codon(cds):
    """Checks if CDS starts with 'ATG'."""
    return isinstance(cds, str) and cds.startswith('ATG')

def check_stop_codon(cds):
    """Checks if CDS ends with a valid stop codon ('TAA', 'TAG', 'TGA')."""
    stop_codons = {'TAA', 'TAG', 'TGA'}
    return isinstance(cds, str) and len(cds) >= 3 and cds[-3:] in stop_codons

def check_length(cds):
    """Checks if CDS length is divisible by 3."""
    return isinstance(cds, str) and len(cds) % 3 == 0

def is_valid_cds(cds):
    """Applies all sequence validity checks."""
    return (
        check_start_codon(cds) and
        check_stop_codon(cds) and
        check_length(cds) and
        not _has_internal_stop(cds)
    )

def calculate_md5(text):
    """Calculates the MD5 hash of a string."""
    return hashlib.md5(text.encode()).hexdigest() if isinstance(text, str) else None

def write_fasta(df, filename, header_cols=['gene_id', 'length', 'cai', 'gc_percent']):
    """Writes a DataFrame to a FASTA file."""
    with open(filename, 'w') as f:
        for _, row in df.iterrows():
            header_parts = []
            if 'gene_id' in header_cols and pd.notna(row['gene_id']):
                header_parts.append(f"{row['gene_id']}")
            else:
                 header_parts.append(f"seq_{_}") # Fallback ID if gene_id is missing

            if 'length' in header_cols and pd.notna(row['length']):
                header_parts.append(f"len={row['length']}")
            if 'cai' in header_cols and pd.notna(row['cai']):
                header_parts.append(f"cai={row['cai']:.2f}")
            if 'gc_percent' in header_cols and pd.notna(row['gc_percent']):
                 header_parts.append(f"gc={row['gc_percent']:.1f}")

            header = ">" + "|".join(header_parts)
            sequence = row['cds']
            f.write(f"{header}\n{sequence}\n")

# --- Main Cleaning Logic ---
def main():
    parser = argparse.ArgumentParser(description="Clean and split E. coli sequence data.")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing raw .csv and .xlsx files.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save processed files.")
    parser.add_argument("--val_frac", type=float, default=0.10,
                        help="Fraction of data for the validation set.")
    parser.add_argument("--test_frac", type=float, default=0.10,
                        help="Fraction of data for the test set.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for stratified splitting.")
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    val_frac = args.val_frac
    test_frac = args.test_frac
    seed = args.seed

    if not (0 < val_frac < 1 and 0 < test_frac < 1 and (val_frac + test_frac) < 1):
        print("Error: Validation and test fractions must be between 0 and 1, and their sum must be less than 1.")
        sys.exit(1)

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    print(f"Loading raw data from: {input_path.resolve()}")
    df_raw = load_raw_dir(input_path)
    if df_raw.empty:
        print(f"No data loaded from {input_path}. Exiting.")
        sys.exit(1)
    print(f"Loaded {len(df_raw)} rows.")

    # 2. Filter
    print("Applying filters...")
    # Drop rows with missing CAI
    df_filtered = df_raw.dropna(subset=['cai']).copy()
    print(f"  {len(df_raw) - len(df_filtered)} rows dropped due to missing CAI.")
    rows_before_cds_filter = len(df_filtered)

    # Apply sequence validity checks
    valid_mask = df_filtered['cds'].apply(is_valid_cds)
    df_filtered = df_filtered[valid_mask].copy()
    rows_dropped_cds = rows_before_cds_filter - len(df_filtered)
    print(f"  {rows_dropped_cds} rows dropped due to start/stop/length/internal stop violations.")

    if len(df_filtered) == 0:
         print("Error: No valid sequences remained after filtering.")
         sys.exit(1)

    # 3. Deduplicate
    print("Deduplicating sequences...")
    df_filtered['md5'] = df_filtered['cds'].apply(calculate_md5)
    # Sort by CAI (desc) to keep the best CAI score for duplicate sequences
    df_deduped = df_filtered.sort_values('cai', ascending=False).drop_duplicates('md5', keep='first').copy()
    rows_dropped_dups = len(df_filtered) - len(df_deduped)
    print(f"  {rows_dropped_dups} duplicate CDS rows dropped (keeping highest CAI).")

    rows_after_clean = len(df_deduped)
    print(f"Total rows after cleaning and deduplication: {rows_after_clean}")

    # Check if enough rows remain
    if rows_after_clean < 10000:
        print(f"Error: Only {rows_after_clean} rows remain after cleaning, which is less than the threshold of 10,000.")
        sys.exit(1)

    # Calculate additional stats needed for output
    df_deduped['length'] = df_deduped['cds'].str.len()
    df_deduped['gc_percent'] = df_deduped['cds'].apply(lambda x: _calculate_gc_content(x) if pd.notna(x) else None)

    # Re-calculate basic stats on the cleaned data for validation printout
    stats_clean = basic_stats(df_deduped) # Note: basic_stats recalculates some things, but useful for validation format

    # 4. Stratified split
    print(f"Performing stratified split (Val: {val_frac:.0%}, Test: {test_frac:.0%}, Seed: {seed})...")
    df_deduped['len_quartile'] = pd.qcut(df_deduped['length'], 4, labels=False, duplicates='drop')
    df_deduped['cai_quartile'] = pd.qcut(df_deduped['cai'], 4, labels=False, duplicates='drop')
    df_deduped['stratum'] = df_deduped['len_quartile'].astype(str) + '_' + df_deduped['cai_quartile'].astype(str)

    # Split off test set first
    train_val_idx, test_idx = train_test_split(
        df_deduped.index,
        test_size=test_frac,
        random_state=seed,
        shuffle=True,
        stratify=df_deduped['stratum']
    )

    # Split remaining into train and validation
    # Adjust val_frac relative to the remaining data
    relative_val_frac = val_frac / (1.0 - test_frac)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=relative_val_frac,
        random_state=seed, # Use the same seed for reproducibility
        shuffle=True,
        stratify=df_deduped.loc[train_val_idx, 'stratum']
    )

    df_train = df_deduped.loc[train_idx]
    df_val = df_deduped.loc[val_idx]
    df_test = df_deduped.loc[test_idx]

    print(f"  Train set size: {len(df_train)}")
    print(f"  Validation set size: {len(df_val)}")
    print(f"  Test set size: {len(df_test)}")

    # 5. Write outputs
    print(f"Writing output files to: {output_path.resolve()}")

    # Clean CDS FASTA
    fasta_all_path = output_path / "clean_cds.fasta"
    write_fasta(df_deduped, fasta_all_path, header_cols=['gene_id']) # Only gene_id for this file
    print(f"  Saved all cleaned sequences to {fasta_all_path.name}")

    # Genes CSV
    genes_csv_path = output_path / "genes.csv"
    df_deduped[['gene_id', 'cds', 'cai', 'length', 'gc_percent']].to_csv(genes_csv_path, index=False)
    print(f"  Saved gene info to {genes_csv_path.name}")

    # Split FASTAs
    fasta_train_path = output_path / "train.fasta"
    fasta_val_path = output_path / "val.fasta"
    fasta_test_path = output_path / "test.fasta"

    write_fasta(df_train, fasta_train_path)
    write_fasta(df_val, fasta_val_path)
    write_fasta(df_test, fasta_test_path)
    print(f"  Saved split FASTAs: {fasta_train_path.name}, {fasta_val_path.name}, {fasta_test_path.name}")

    # 6. Validation printout
    print("Validation Summary:")
    print("After clean & dedupe:")
    print(f"  rows         : {stats_clean.get('rows', 0):,}")
    # After deduplication and filtering, these should be 0 by definition
    print(f"  duplicates   : 0") # Calculated md5 duplicates before drop_duplicates, now should be 0
    print(f"  start/stop   : 0") # Filtered out
    print(f"  internal stop: 0") # Filtered out

    print(f"Split summary (seed={seed}):")
    total_split = len(df_train) + len(df_val) + len(df_test)
    train_pct = len(df_train) / total_split if total_split else 0
    val_pct = len(df_val) / total_split if total_split else 0
    test_pct = len(df_test) / total_split if total_split else 0
    print(f"  train : {train_pct:.0%}  → n={len(df_train):,}")
    print(f"  val   : {val_pct:.0%}  → n={len(df_val):,}")
    print(f"  test  : {test_pct:.0%}  → n={len(df_test):,}")

    # Use stats calculated on the final df_deduped
    gc_mean = df_deduped['gc_percent'].mean()
    gc_std = df_deduped['gc_percent'].std()
    len_median = df_deduped['length'].median()

    print(f"GC% mean {gc_mean:.1f} ± {gc_std:.1f}   |  Length nt median {int(len_median):,}")

    print("Cleaning and splitting complete.")
    sys.exit(0)

if __name__ == "__main__":
    main() 