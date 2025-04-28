seed=0


MODEL_PATH=/home/share/FM_Code/FM_Eval/Model_Weights/CAE/caev2_ll_data6_checkpoint-499.pth
RESUME_PATH=/home/syyan/XJ/PanDerm-open_source/finetune/work_dir/HAM10000_cleaned_using_lp_setting/checkpoint-best.pth

CUDA_VISIBLE_DEVICES=0 python run_class_finetuning.py \
    --model PanDerm_Large_FT \
    --finetune $MODEL_PATH \
    --nb_classes 7 \
    --batch_size 64 \
    --lr 5e-4 --update_freq 1 \
    --warmup_epochs 10 \
    --epochs 50 --layer_decay 0.65 --drop_path 0.2 \
    --weight_decay 0.05 \
    --mixup 0.8 --cutmix 1.0 \
    --sin_pos_emb \
    --percent_data 1.0 \
    --no_auto_resume \
    --exp_name $my_name \
    --imagenet_default_mean_and_std \
    --wandb_name "weight_sampler_max_recall_mask5e-4${seed}" \
    --output_dir /home/syyan/XJ/PanDerm-open_source/finetune/work_dir/HAM10000_cleaned_using_lp_setting \
    --csv_path /home/syyan/XJ/PanDerm-open_source/data/finetune/HAM10000/HAM_cleaned_test.csv \
    --test_csv_path /home/syyan/XJ/PanDerm-open_source/data/finetune/HAM10000/HAM_cleaned_test.csv \
    --root_path /home/share/Uni_Eval/ISIC2018_reader/images/ \
    --seed ${seed} \
    --resume ${RESUME_PATH} \
    --eval \
    --TTA