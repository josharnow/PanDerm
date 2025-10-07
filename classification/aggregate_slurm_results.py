import pandas as pd
import argparse
import os

def main(args):
    all_metrics = []
    
    print(f"Aggregating results from: {args.output_dir}")

    for fold in range(1, args.n_splits + 1):
        # Construct the expected filename for the results CSV of each fold
        fold_dir = os.path.join(args.output_dir, f"fold_{fold}")
        results_filename = f"fold_{fold}_{args.csv_filename}"
        results_path = os.path.join(fold_dir, results_filename)
        
        if os.path.exists(results_path):
            print(f"Reading results from: {results_path}")
            # Assuming the CSV from linear_eval.py has one row of metrics
            metrics_df = pd.read_csv(results_path)
            all_metrics.append(metrics_df.iloc[0].to_dict())
        else:
            print(f"Warning: Results file not found for fold {fold} at {results_path}")

    if not all_metrics:
        print("No metrics found to aggregate. Exiting.")
        return

    # Aggregate the metrics across folds
    aggregated_df = pd.DataFrame(all_metrics)
    
    # Calculate mean and standard deviation
    mean_metrics = aggregated_df.mean().to_frame('mean').T
    std_metrics = aggregated_df.std().to_frame('std').T
    
    final_summary = pd.concat([mean_metrics, std_metrics])

    print("\n--- Aggregated Cross-Validation Results ---")
    print(final_summary)

    # Save aggregated results
    aggregated_df.to_csv(os.path.join(args.output_dir, "all_folds_results.csv"), index=False)
    final_summary.to_csv(os.path.join(args.output_dir, "final_summary_results.csv"))
    print(f"\nSaved final results to {args.output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Aggregate results from k-fold CV')
    parser.add_argument('--output_dir', required=True, help='Path to the main output directory containing fold subdirectories')
    parser.add_argument('--n_splits', required=True, type=int, help='Number of folds that were run')
    parser.add_argument('--csv_filename', required=True, type=str, help='The base name of the results CSV file')
    args = parser.parse_args()
    main(args)