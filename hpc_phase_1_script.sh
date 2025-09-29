#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

# export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64
#source venv/bin/activate

echo $CUDA_VISIBLE_DEVICES

# create virtualenv with...
python3 -m virtualenv -p python3 venv
source venv/bin/activate

# install libraries with...
venv/bin/pip install -r requirements.txt
sleep 10
venv/bin/python3 make linear_eval
# venv/bin/python3 example_script.py