tmp_my_name=PanDerm-Large_Finetune_HAM10000_cleaned
my_name=${tmp_my_name%.*}

seed=122

ADDRESS=ADDR_FOR_THIS_MACHINE
NNODES=4
RANK=RANK_FOR_THIS_MACHINE

MODEL_PATH=/home/syyan/XJ/PanDerm-open_source/pretrain_weight/panderm_ll_data6_checkpoint-499.pth

BATCH_SIZE=128
LR=5e-4

CUDA_VISIBLE_DEVICES=2 python3 run_class_finetuning.py \
    --model cae_large_patch16_224 \
    --finetune $MODEL_PATH \
    --nb_classes 7 \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --update_freq 1 \
    --warmup_epochs 10 \
    --epochs 50 --layer_decay 0.65 --drop_path 0.2 \
    --weight_decay 0.05 --mixup 0.8 --cutmix 1.0 \
    --weights \
    --sin_pos_emb \
    --percent_data 1.0 \
    --no_auto_resume \
    --exp_name $my_name \
    --imagenet_default_mean_and_std \
    --wandb_name "weight_sampler_max_recall_mask_B${BATCH_SIZE}_${LR}_${seed}" \
    --output_dir "/home/syyan/XJ/PanDerm-open_source/finetune/work_dir/HAM10000_cleaned_using_lp_setting" \
    --csv_path /home/syyan/XJ/PanDerm-open_source/data/linear_probing/HAM_clean.csv \
    --root_path /home/share/Uni_Eval/ISIC2018_reader/images/ \
    --seed ${seed}