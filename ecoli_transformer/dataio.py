import pandas as pd
from pathlib import Path
import hashlib
import re
from typing import Dict, List, Optional, Tuple

def _clean_cds(seq: Optional[str]) -> Optional[str]:
    """Uppercase and remove non-ATGC characters."""
    if pd.isna(seq) or not isinstance(seq, str):
        return None
    cleaned = re.sub(r"[^ATGC]", "", seq.upper())
    return cleaned if cleaned else None

def _calculate_gc_content(seq: Optional[str]) -> Optional[float]:
    """Calculate GC content percentage."""
    if not seq:
        return None
    gc_count = seq.count('G') + seq.count('C')
    return (gc_count / len(seq)) * 100 if len(seq) > 0 else 0.0

def _has_internal_stop(seq: Optional[str]) -> bool:
    """Check for internal stop codons (TAA, TAG, TGA)."""
    if not seq or len(seq) < 6:
        return False
    stop_codons = {"TAA", "TAG", "TGA"}
    for i in range(3, len(seq) - 3, 3):
        codon = seq[i:i+3]
        if codon in stop_codons:
            return True
    return False

def _manual_single_col_csv(file_path: Path) -> pd.DataFrame:
    """
    Reads a single-column CSV file line by line, skipping the header.
    Assumes each line (after header) is a sequence.
    """
    sequences = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            next(f)
            for i, line in enumerate(f):
                seq = line.strip()
                if seq:
                    sequences.append({
                        "gene_id": f"{file_path.stem}_{i}",
                        "cds": seq,
                        "cai": pd.NA
                    })
    except Exception as e:
        print(f"Error reading {file_path} manually: {e}")
        return pd.DataFrame(columns=["gene_id", "cds", "cai"])

    if not sequences:
        return pd.DataFrame(columns=["gene_id", "cds", "cai"])

    return pd.DataFrame(sequences)

def load_raw_dir(dir_path: Path) -> pd.DataFrame:
    """
    Loads raw data files (.csv, .xlsx) from a directory, handling potential
    pandas TypeError by falling back to manual parsing for specific cases.
    Harmonizes column names and cleans CDS sequences.
    """
    all_dfs: List[pd.DataFrame] = []
    if not dir_path.is_dir():
        return pd.DataFrame()

    files_processed_count = 0
    for file_path in dir_path.iterdir():
        df = None
        if file_path.suffix == '.csv':
            try:
                df = pd.read_csv(file_path, dtype=str, na_filter=False, engine='c')
            except TypeError as te:
                if "Cannot convert numpy.ndarray to numpy.ndarray" in str(te):
                     print(f"Pandas TypeError reading {file_path}, attempting fallback to engine='python'.")
                     try:
                         df = pd.read_csv(file_path, dtype=str, na_filter=False, engine='python')
                     except TypeError as te_py:
                          if "Cannot convert numpy.ndarray to numpy.ndarray" in str(te_py):
                              print(f"Pandas TypeError persists with python engine for {file_path}, attempting manual read.")
                              df = _manual_single_col_csv(file_path)
                          else:
                              print(f"Unhandled TypeError with python engine for {file_path}: {te_py}")
                              continue
                     except Exception as e_py:
                        print(f"Error reading {file_path} with python engine: {e_py}")
                        continue
                else:
                     print(f"Unhandled TypeError reading {file_path} with C engine: {te}")
                     continue
            except Exception as e:
                print(f"Error reading {file_path} with C engine: {e}")
                continue

        elif file_path.suffix == '.xlsx':
            try:
                df = pd.read_excel(file_path, dtype=str, na_filter=False, engine='openpyxl')
            except Exception as e:
                print(f"Error reading Excel file {file_path}: {e}")
                continue

        else:
            continue

        if df is not None and not df.empty:
            df['source_file'] = file_path.name
            all_dfs.append(df)
            files_processed_count += 1

    if not all_dfs:
        print(f"No valid data files found or loaded from {dir_path}")
        return pd.DataFrame()

    combined_df = pd.concat(all_dfs, ignore_index=True)

    harmonized_cols = {}
    name_map = {
        'sequence': 'cds',
        'cds': 'cds',
        'cai': 'cai',
        'cai_score': 'cai',
        'gene id': 'gene_id',
        'gene_id': 'gene_id',
        'unnamed: 0': 'gene_id'
    }
    original_cols = combined_df.columns
    for col in original_cols:
        norm_col = col.lower().replace(' ', '_').replace('-', '_')
        if col.lower().startswith('unnamed:'):
             norm_col = 'unnamed:_0'

        if norm_col in name_map:
            target_name = name_map[norm_col]
            if target_name not in harmonized_cols:
                 harmonized_cols[target_name] = col
        elif norm_col == 'source_file':
             harmonized_cols['source_file'] = col


    final_df = combined_df[[harmonized_cols[target] for target in ['gene_id', 'cds', 'cai', 'source_file'] if target in harmonized_cols]].copy()
    final_df.rename(columns={harmonized_cols[target]: target for target in ['gene_id', 'cds', 'cai', 'source_file'] if target in harmonized_cols}, inplace=True)

    for col in ['gene_id', 'cds', 'cai']:
        if col not in final_df.columns:
            final_df[col] = pd.NA

    final_df['cds'] = final_df['cds'].apply(_clean_cds)
    final_df.dropna(subset=['cds'], inplace=True)

    if 'cai' in final_df.columns:
        final_df['cai'] = pd.to_numeric(final_df['cai'], errors='coerce')
    else:
         final_df['cai'] = pd.NA


    print(f"Processed {files_processed_count} files. Combined DataFrame shape: {final_df.shape}")
    return final_df[['gene_id', 'cds', 'cai', 'source_file']]


def basic_stats(df: pd.DataFrame) -> Dict[str, float | int | str]:
    """Calculates basic statistics for the loaded DataFrame."""
    if df.empty:
        return {
            "rows": 0, "files_processed": 0, "missing_cai": 0, "dup_md5": 0,
            "start_violation": 0, "stop_violation": 0, "internal_stop": 0,
            "len_min": 0, "len_median": 0, "len_max": 0,
            "gc_mean": 0.0, "gc_std": 0.0
        }

    stats: Dict[str, float | int | str] = {}
    stats["rows"] = len(df)
    stats["files_processed"] = df['source_file'].nunique()
    stats["missing_cai"] = df['cai'].isna().sum()

    df_cds_valid = df.dropna(subset=['cds'])
    if not df_cds_valid.empty:
        cds_hashes = df_cds_valid['cds'].apply(lambda x: hashlib.md5(x.encode()).hexdigest() if pd.notna(x) else None)
        stats["dup_md5"] = cds_hashes.duplicated().sum()
    else:
         stats["dup_md5"] = 0


    start_violations = 0
    stop_violations = 0
    internal_stops = 0
    lengths = []
    gc_contents = []

    stop_codons = {"TAA", "TAG", "TGA"}

    for seq in df_cds_valid['cds']:
        seq_len = len(seq)
        lengths.append(seq_len)

        if seq_len >= 3:
            if not seq.startswith("ATG"):
                start_violations += 1
            if seq[-3:] not in stop_codons:
                stop_violations += 1
            if _has_internal_stop(seq):
                internal_stops += 1
        else:
            start_violations += 1
            stop_violations += 1

        gc = _calculate_gc_content(seq)
        if gc is not None:
            gc_contents.append(gc)

    stats["start_violation"] = start_violations
    stats["stop_violation"] = stop_violations
    stats["internal_stop"] = internal_stops

    if lengths:
        length_series = pd.Series(lengths)
        stats["len_min"] = length_series.min()
        stats["len_median"] = int(length_series.median())
        stats["len_max"] = length_series.max()
    else:
        stats["len_min"] = 0
        stats["len_median"] = 0
        stats["len_max"] = 0

    if gc_contents:
        gc_series = pd.Series(gc_contents)
        stats["gc_mean"] = round(gc_series.mean(), 1)
        stats["gc_std"] = round(gc_series.std(), 1)
    else:
        stats["gc_mean"] = 0.0
        stats["gc_std"] = 0.0


    return stats 