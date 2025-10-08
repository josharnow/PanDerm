#!/bin/bash

#SBATCH --job-name=panderm
#SBATCH --output=logs/phase_1/cv_job_%A_%a.out
#SBATCH --error=logs/phase_1/cv_job_%A_%a.err
#SBATCH --nodes=1 # Request 1 node
#SBATCH --ntasks=1 # Request 1 task (process)
#SBATCH --gres=gpu:1 # Request 1 GPU per task

# --- Important: Set the number of splits for your array ---
#SBATCH --array=1-2 # Creates 10 jobs, with task IDs from 1 to 10; matches n_splits in Makefile

# NOTE - Adding memory doesn't work (sbatch: error: Memory specification can not be satisfied sbatch: error: Batch job submission failed: Requested node configuration is not available)

# Project-specific variables
PYTHON="python3"
BATCH_SIZE=16
MODEL="PanDerm_Large_LP"
NB_CLASSES=2
PERCENT_DATA=1.0
CSV_FILENAME="PanDerm_Large_LP_result.csv"
OUTPUT_DIR="/home/PACE/ja50529n/MS Thesis/Model/PanDerm/output/phase_1" # TODO - Change name of directory on HPC before proceeding; old name is PanDerm_Large_LP_res
CSV_PATH="/home/PACE/ja50529n/MS Thesis/Thesis Data/Skin Cancer Project/PanDerm & SkinEHDLF/ISIC 2024 (SLICE-3D)/ISIC_2024_Training_GroundTruth.csv"
ROOT_PATH="/home/PACE/ja50529n/MS Thesis/Thesis Data/Skin Cancer Project/PanDerm & SkinEHDLF/ISIC 2024 (SLICE-3D)/ISIC_2024_Training_Input/"
PRETRAINED_CHECKPOINT="/home/PACE/ja50529n/MS Thesis/Model/PanDerm/pretrain_weight/panderm_ll_data6_checkpoint-499.pth"
NUM_WORKERS=0
N_SPLITS=2 # Should match the --array range above
LABEL_COLUMN="binary_label"



echo "Starting job on " `date`
echo "Running on node " `hostname`
echo "Visible CUDA device environment variables: $CUDA_VISIBLE_DEVICES"
echo "Starting job array task ${SLURM_ARRAY_TASK_ID}"

# pip3 install virtualenv

# create virtualenv with...
if [ ! -d venv ]; then
  python3 -m virtualenv -p python3 venv
  # install libraries with...
  venv/bin/pip install -r requirements.txt
  venv/bin/pip install -r classification/requirements.txt
  venv/bin/pip install -r segmentation/requirements.txt
fi

source venv/bin/activate
cd classification
mkdir -p "/home/PACE/ja50529n/MS Thesis/Model/PanDerm/output/PanDerm_Large_LP_res"
ulimit -n 4096

# --- Run the Python script for the assigned fold ---
# NOTE - This is part of the nested, stratified k-fold cross-validation setup
${PYTHON} slurm_runner.py \
    --fold ${SLURM_ARRAY_TASK_ID} \
    --batch_size ${BATCH_SIZE} \
    --model "${MODEL}" \
    --nb_classes ${NB_CLASSES} \
    --percent_data ${PERCENT_DATA} \
    --csv_filename "${CSV_FILENAME}" \
    --output_dir "${OUTPUT_DIR}" \
    --csv_path "${CSV_PATH}" \
    --root_path "${ROOT_PATH}" \
    --pretrained_checkpoint "${PRETRAINED_CHECKPOINT}" \
    --num_workers ${NUM_WORKERS} \
    --n_splits ${N_SPLITS} \
    --label_column "${LABEL_COLUMN}"

echo "Finished job array task ${SLURM_ARRAY_TASK_ID}"