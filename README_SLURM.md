Running each fold of your cross-validation in parallel is a perfect use case for an HPC cluster with Slurm. This will dramatically speed up your experimentation.

The strategy is to change the Python script so it only runs a *single* fold, specified by a command-line argument. Then, we will use a Slurm job array to launch multiple instances of this script in parallel, with each instance handling a different fold. Finally, we'll create a separate script to gather the results.

Here is the complete solution, broken down into three parts.

-----

### Part 1: Modified Python Script for a Single Fold (`slurm_runner.py`)

This new script is a modification of the previous one. Instead of looping through the folds, it takes a `--fold` argument and only processes that specific fold. The Slurm job array will provide this fold number.

**Save this file as `classification/slurm_runner.py`:**

```python
import argparse
import os
import subprocess
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
import sys

def get_args_parser():
    parser = argparse.ArgumentParser('Slurm Runner for Single Fold of Nested CV', add_help=False)
    # This script needs all the same arguments as before
    parser.add_argument('--csv_path', required=True, type=str)
    parser.add_argument('--root_path', required=True, type=str)
    parser.add_argument('--model', required=True, type=str)
    parser.add_argument('--output_dir', default='./output_dir')
    parser.add_argument('--pretrained_checkpoint', default=None, type=str)
    parser.add_argument('--nb_classes', default=2, type=int)
    parser.add_argument('--batch_size', default=200, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--n_splits', default=5, type=int)
    parser.add_argument('--label_column', default='binary_label', type=str)
    parser.add_argument('--percent_data', default=1.0, type=float)
    parser.add_argument('--csv_filename', default='results.csv', type=str)
    parser.add_argument('--validation_size', default=0.15, type=float)
    # --- NEW ARGUMENT ---
    parser.add_argument('--fold', required=True, type=int, help='The fold number to process (1-based index)')
    return parser

def main(args):
    df = pd.read_csv(args.csv_path)
    if args.label_column not in df.columns:
        raise ValueError(f"Label column '{args.label_column}' not found in the CSV file.")

    # We still perform the k-fold split to ensure the data division is identical for each job
    outer_skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    
    # Get the specific train/test indices for the fold this job is responsible for
    all_splits = list(outer_skf.split(df, df[args.label_column]))
    try:
        # Adjust for 1-based fold index from Slurm
        train_val_index, test_index = all_splits[args.fold - 1]
    except IndexError:
        print(f"Error: Fold number {args.fold} is out of range for {args.n_splits} splits.")
        sys.exit(1)

    print(f"--- Processing Fold {args.fold}/{args.n_splits} ---")

    train_val_df = df.iloc[train_val_index].copy()
    test_df = df.iloc[test_index].copy()

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=args.validation_size,
        stratify=train_val_df[args.label_column],
        random_state=42
    )
    
    print(f"Fold {args.fold} sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    fold_output_dir = os.path.join(args.output_dir, f"fold_{args.fold}")
    if not os.path.exists(fold_output_dir):
        os.makedirs(fold_output_dir)

    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'
    
    fold_df = pd.concat([train_df, val_df, test_df])
    fold_csv_path = os.path.join(fold_output_dir, 'fold_data.csv')
    fold_df.to_csv(fold_csv_path, index=False)
    
    cmd = [
        'python', 'linear_eval.py',
        '--csv_path', fold_csv_path,
        '--root_path', args.root_path,
        '--model', args.model,
        '--nb_classes', str(args.nb_classes),
        '--batch_size', str(args.batch_size),
        '--num_workers', str(args.num_workers),
        '--output_dir', fold_output_dir,
        '--percent_data', str(args.percent_data),
        '--csv_filename', f"fold_{args.fold}_{args.csv_filename}"
    ]
    if args.pretrained_checkpoint:
        cmd.extend(['--pretrained_checkpoint', args.pretrained_checkpoint])

    print("Running command:", " ".join(cmd))
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()

    process.wait()
    if process.returncode == 0:
        print(f"--- Fold {args.fold} completed successfully ---")
    else:
        print(f"--- Subprocess for fold {args.fold} failed with return code {process.returncode} ---")
        sys.exit(1)

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
```

-----

### Part 2: Slurm Submission Script (`submit.sbatch`)

This is the batch script that you will submit to Slurm. It requests resources and launches the job array. Each job in the array will run the `slurm_runner.py` script for a different fold.

**Save this file in the same directory as your `Makefile`:**

```bash
#!/bin/bash
#SBATCH --job-name=nested_cv     # A name for your job
#SBATCH --output=slurm_logs/cv_job_%A_%a.out  # Log file for stdout and stderr, %A is job ID, %a is array task ID
#SBATCH --error=slurm_logs/cv_job_%A_%a.err
#SBATCH --nodes=1                # Request 1 node
#SBATCH --ntasks=1               # Request 1 task (process)
#SBATCH --cpus-per-task=16       # Number of CPUs per task (matches your --num_workers)
#SBATCH --mem=32G                # Memory per node
#SBATCH --time=02:00:00          # Maximum walltime (2 hours)
#SBATCH --gres=gpu:1             # Request 1 GPU per task

# --- Important: Set the number of splits for your array ---
#SBATCH --array=1-5              # Creates 5 jobs, with task IDs from 1 to 5

# --- Your Makefile Variables (replace with actual values or pass them) ---
# It's better to define these here or pass them as arguments for clarity
MAC_PYTHON="/path/to/your/python"
MAC_BATCH_SIZE=200
MAC_MODEL="YourModelName"
MAC_NB_CLASSES=2
MAC_PERCENT_DATA=1.0
MAC_CSV_FILENAME="PanDerm_Large_LP_result.csv"
MAC_OUTPUT_DIR="./nested_kfold_output"
MAC_CSV_PATH="/path/to/your/atlas-clinical-all.csv"
MAC_ROOT_PATH="/path/to/your/images/"
MAC_PRETRAINED_CHECKPOINT="/path/to/your/checkpoint.pth" # or None
MAC_NUM_WORKERS=16
MAC_N_SPLITS=5 # Should match the --array range above
MAC_LABEL_COLUMN="binary_label"


# --- Environment Setup ---
echo "Starting job array task ${SLURM_ARRAY_TASK_ID}"
# Create log directory if it doesn't exist
mkdir -p slurm_logs
# Activate your python environment (e.g., conda, virtualenv)
# source activate your_env_name

# --- Change to the correct directory ---
cd classification

# --- Run the Python script for the assigned fold ---
${MAC_PYTHON} slurm_runner.py \
    --fold ${SLURM_ARRAY_TASK_ID} \
    --batch_size ${MAC_BATCH_SIZE} \
    --model "${MAC_MODEL}" \
    --nb_classes ${MAC_NB_CLASSES} \
    --percent_data ${MAC_PERCENT_DATA} \
    --csv_filename "${MAC_CSV_FILENAME}" \
    --output_dir "${MAC_OUTPUT_DIR}" \
    --csv_path "${MAC_CSV_PATH}" \
    --root_path "${MAC_ROOT_PATH}" \
    --pretrained_checkpoint "${MAC_PRETRAINED_CHECKPOINT}" \
    --num_workers ${MAC_NUM_WORKERS} \
    --n_splits ${MAC_N_SPLITS} \
    --label_column "${MAC_LABEL_COLUMN}"

echo "Finished job array task ${SLURM_ARRAY_TASK_ID}"
```

-----

### Part 3: Aggregation Script (`aggregate_results.py`)

After all the Slurm jobs have finished, you will have results scattered across `fold_1`, `fold_2`, etc. This simple script will collect them, calculate the average and standard deviation, and save the final results.

**Save this file as `classification/aggregate_results.py`:**

```python
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
```

### How to Run

1. **Configure `submit.sbatch`**: Open `submit.sbatch` and **carefully** set the paths and variables at the top of the script to match your HPC environment and project structure.
2. **Submit the Job Array**: From your terminal, submit the job to Slurm.

    ```bash
    sbatch submit.sbatch
    ```

3. **Monitor the Jobs**: You can check the status of your jobs using `squeue -u your_username`. You will see 5 jobs queued or running.
4. **Aggregate Results**: Once all jobs are complete (you can check the `slurm_logs` directory), run the aggregation script.

    ```bash
    # Make sure you are in the root directory of your project
    python classification/aggregate_results.py \
        --output_dir "classification/nested_kfold_output" \
        --n_splits 5 \
        --csv_filename "PanDerm_Large_LP_result.csv"
    ```

This workflow is robust, scalable, and the standard way to handle such tasks on an HPC cluster.
