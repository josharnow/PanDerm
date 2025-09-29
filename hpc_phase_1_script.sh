#!/bin/bash

#SBATCH --job-name=panderm_phase_1
#SBATCH --output=logs/panderm_phase_1_%j.out
#SBATCH --error=logs/panderm_phase_1_%j.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

# export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64

echo "Starting job on " `date`
echo "Running on node " `hostname`
echo $CUDA_VISIBLE_DEVICES

# pip3 install virtualenv

# create virtualenv with...
if [ ! -d venv ]; then
  python3 -m virtualenv -p python3 venv
fi
source venv/bin/activate

# install libraries with...
venv/bin/pip install -r requirements.txt
venv/bin/pip install -r classification/requirements.txt
venv/bin/pip install -r segmentation/requirements.txt
sleep 10
make linear_eval