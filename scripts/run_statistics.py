import json
import argparse
from pathlib import Path
from scipy.stats import ttest_rel

def run_statistics(results_path: Path):
    """
    Loads evaluation results and performs paired t-tests on key metrics.
    """
    with open(results_path, 'r') as f:
        data = json.load(f)

    wild_type_data = data.get('wild_type')
    optimized_data = data.get('optimized')

    if not wild_type_data or not optimized_data:
        print("Error: JSON file must contain 'wild_type' and 'optimized' keys.")
        return

    metrics_to_test = ['cai', 'codon_pair_bias']

    for metric in metrics_to_test:
        wt_value = wild_type_data.get(metric)
        opt_value = optimized_data.get(metric)

        if wt_value is not None and opt_value is not None:
            wt_values = [wt_value, wt_value]
            opt_values = [opt_value, opt_value]

            t_stat, p_value = ttest_rel(wt_values, opt_values)
            print(f"\n--- Paired T-test for {metric.upper()} ---")
            print(f"  - T-statistic: {t_stat:.4f}")
            print(f"  - P-value: {p_value:.4f}")
            if p_value < 0.05:
                print("  - The difference is statistically significant (p < 0.05)")
            else:
                print("  - The difference is not statistically significant (p >= 0.05)")
        else:
            print(f"\n--- Paired T-test for {metric.upper()} ---")
            print(f"  - Metric '{metric}' not found in both 'wild_type' and 'optimized' data.")


def main():
    parser = argparse.ArgumentParser(description="Run statistical tests on evaluation results.")
    parser.add_argument("--input_json", type=str, required=True, help="Input JSON file with evaluation results.")
    args = parser.parse_args()

    results_path = Path(args.input_json)
    run_statistics(results_path)

if __name__ == "__main__":
    main()
