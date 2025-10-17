import argparse
import os
import subprocess
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
import sys

def get_args_parser():
    parser = argparse.ArgumentParser('True Nested k-fold cross-validation', add_help=False)
    # Add all the original arguments from your Makefile
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
    parser.add_argument('--validation_size', default=0.15, type=float, help='Proportion of the training data to use for validation')

    return parser

def main(args):
    df = pd.read_csv(args.csv_path)
    if args.label_column not in df.columns:
        raise ValueError(f"Label column '{args.label_column}' not found in the CSV file.")

    outer_skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for fold, (train_val_index, test_index) in enumerate(outer_skf.split(df, df[args.label_column])):
        print(f"--- Outer Fold {fold+1}/{args.n_splits} ---")

        train_val_df = df.iloc[train_val_index].copy()
        test_df = df.iloc[test_index].copy()

        train_df, val_df = train_test_split(
            train_val_df,
            test_size=args.validation_size,
            stratify=train_val_df[args.label_column],
            random_state=42
        )
        
        print(f"Fold {fold+1} sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

        fold_output_dir = os.path.join(args.output_dir, f"fold_{fold+1}")
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
            '--csv_filename', f"fold_{fold+1}_{args.csv_filename}"
        ]
        if args.pretrained_checkpoint:
            cmd.extend(['--pretrained_checkpoint', args.pretrained_checkpoint])

        print("Running command:", " ".join(cmd))
        
        # --- MODIFICATION FOR REAL-TIME OUTPUT ---
        # Use Popen to start the process and capture stdout/stderr
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)

        # Read and print the output line by line in real-time
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()

        # Wait for the process to complete and get the return code
        process.wait()
        if process.returncode != 0:
            print(f"\n--- Subprocess for fold {fold+1} failed with return code {process.returncode} ---")
            # You might want to stop the script if a fold fails
            # sys.exit(1)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)