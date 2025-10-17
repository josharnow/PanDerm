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