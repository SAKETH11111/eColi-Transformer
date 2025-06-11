import torch
import numpy as np
from pathlib import Path
import argparse

def print_stats(name, tensor):
    """Prints detailed statistics for a given tensor."""
    if tensor.dtype == torch.float32:
        # Convert to numpy for easier stats calculation, ignoring NaNs
        data_np = tensor.cpu().numpy()
        nan_count = np.isnan(data_np).sum()
        data_clean = data_np[~np.isnan(data_np)]

        if data_clean.size == 0:
            print(f"\n--- Stats for {name} ---")
            print("  Tensor contains only NaNs.")
            return

        min_val, max_val, mean_val, std_val = (
            np.min(data_clean),
            np.max(data_clean),
            np.mean(data_clean),
            np.std(data_clean),
        )
        
        # Outlier detection
        lower_bound = mean_val - 3 * std_val
        upper_bound = mean_val + 3 * std_val
        outliers = data_clean[(data_clean < lower_bound) | (data_clean > upper_bound)]

        print(f"\n--- Stats for {name} ---")
        print(f"  - Min: {min_val:.4f}")
        print(f"  - Max: {max_val:.4f}")
        print(f"  - Mean: {mean_val:.4f}")
        print(f"  - Std Dev: {std_val:.4f}")
        print(f"  - NaN Count: {nan_count} / {data_np.size} ({nan_count/data_np.size:.2%})")
        print(f"  - Outliers (>3 std): {len(outliers)} found")

        # Histogram
        hist, bin_edges = np.histogram(data_clean, bins=10)
        print("  - Histogram:")
        for i in range(len(hist)):
            print(f"    {bin_edges[i]:.2f} to {bin_edges[i+1]:.2f}: {'#' * int(hist[i] / hist.max() * 20 if hist.max() > 0 else 0)}")

    elif tensor.dtype == torch.long:
        min_val, max_val = torch.min(tensor).item(), torch.max(tensor).item()
        mean_val = tensor.float().mean().item()
        print(f"\n--- Stats for {name} (Length) ---")
        print(f"  - Min: {min_val}")
        print(f"  - Max: {max_val}")
        print(f"  - Mean: {mean_val:.2f}")
        
        # Distribution
        hist = torch.histc(tensor.float(), bins=10, min=min_val, max=max_val)
        bin_edges = np.linspace(min_val, max_val, 11)
        print("  - Distribution:")
        for i in range(len(hist)):
            print(f"    {bin_edges[i]:.0f} to {bin_edges[i+1]:.0f}: {'#' * int(hist[i] / hist.max() * 20 if hist.max() > 0 else 0)}")


def main():
    parser = argparse.ArgumentParser(description="Debug and inspect processed tensor files.")
    parser.add_argument("files", nargs='+', help="List of .pt files to inspect (e.g., data/processed/train.pt)")
    args = parser.parse_args()

    for file_path_str in args.files:
        file_path = Path(file_path_str)
        if not file_path.exists():
            print(f"\nError: File not found at {file_path}")
            continue

        print(f"\n{'='*20} Inspecting: {file_path} {'='*20}")
        data = torch.load(file_path)

        # Validate fields exist
        required_fields = ['cai', 'mfe', 'length', 'token_ids', 'pair_ids']
        for field in required_fields:
            if field not in data:
                print(f"  - Field '{field}' not found in the data dictionary.")
                continue
            
            # Print stats for primary fields
            if field in ['cai', 'mfe', 'length']:
                print_stats(field.upper(), data[field])

        # Validate shapes and dtypes
        print("\n--- Tensor Shapes and Dtypes ---")
        for key, value in data.items():
            if isinstance(value, list):
                if len(value) > 0 and hasattr(value[0], 'shape'):
                    print(f"  - {key}: List of {len(value)} tensors, e.g., shape={value[0].shape}, dtype={value[0].dtype}")
                else:
                    print(f"  - {key}: List of {len(value)} items, e.g., type={type(value[0])}")
            else:
                print(f"  - {key}: shape={value.shape}, dtype={value.dtype}")

    print("\n✅ Tensor diagnostics complete")

if __name__ == "__main__":
    main()
