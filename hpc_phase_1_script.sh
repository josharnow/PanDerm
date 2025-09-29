#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

# export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64

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
sleep 10
make linear_eval