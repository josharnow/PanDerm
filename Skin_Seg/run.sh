#!/bin/bash

seeds=(0 1 2 3 4)
percents=(100)
# Loop over seeds
for seed in "${seeds[@]}"
do
    # Loop over percents
    for percent in "${percents[@]}"
    do
        echo "Running cae experiment with seed $seed and percent $percent on isic18"
        CUDA_VISIBLE_DEVICES=0,1 python run.py \
            --worker 4 --gpu "0,1" --batch_size 1 --test_batch_size 1 \
            --epoch 100 --lr 1e-4 --weight_decay 0.05 \
            --log_name "cae_seg_lr_1e-4_decay_0.05_full_seed_$seed" --model "cae_seg" --size 224 \
            --dataset "ISIC2018" --dataset_path "/data2/wangzh/datasets/ISIC2018" \
            --save_name "/data3/wangzh/experiments/skinfm/finals/cae_seg_isic18/" \
            --seed $seed --smoke --percent $percent
    done
done
