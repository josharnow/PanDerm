# PanDerm - Skin Classification Finetune

This implementation uses PanDerm for Skin Classification Finetune.

## Install Environment

```bash
cd finetune
pip install -r requirements.txt
```

## Data Preparation

The dataset used for the finetuning stage should be organized in a CSV file with the following structure:

**Required Columns**

- `image`: Path to the image file (e.g., ISIC_0034524.jpg)

- `label`: Numerical class label (e.g., 0, 1, 2)
- `split`: Dataset partition indicator (train, val, or test)

For Example:
```csv
image,label,split
ISIC_0034524.jpg,1,train
ISIC_0034525.jpg,1,train
ISIC_0034526.jpg,4,val
ISIC_0034527.jpg,1,val
ISIC_0034528.jpg,1,test
ISIC_0034529.jpg,0,test
```

**Dataset Preparation Notes**

* Ensure all image paths in the CSV are correctly referenced relative to your dataset directory. 
* Labels should be numerical and correspond to your class mapping (`class_mapping = {'BCC': 1, 'MEL': 0, ...}`)
* Each sample must be assigned to one of the splits: `train`, `val`, or `test`


## Fine-tuning with PanDerm Weight
1. **Download Pretrained Weights**
   - [PanDerm Large weights](https://drive.google.com/file/d/1XHKRk2p-dS1PFQE-xRbOM3yx47i3bXmi/view?usp=sharing)

2. **Set Pretrained Path**
TODO:
   - Modify `MODEL_PATH` parameter in finetune script, the path to the finetune script folder is [finetune/scripts](finetune/scripts). You can find more scripts in this folder.

3. **Start Training**

You could finetune our model on your dataset, here is a command line example:
```bash
MODEL_PATH=PATH_TO_YOUR_DOWNLOADED_PANDERM_PRETRAINED_WEIGHT

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
    --wandb_name Panderm-finetune \
    --output_dir OUTPUT_DIRECTORY \ # Your checkpoint and output result will be saved in this directory
    --csv_path /home/syyan/XJ/PanDerm-open_source/data/finetune/HAM10000/HAM_cleaned_training.csv \ # You could replace this with your finetune dataset csv
    --root_path /home/share/Uni_Eval/ISIC2018_reader/images/ \ # Optional: Dataset image root, you could set this with empty value.
    --seed 122 \
    --TTA # This is optional: You could comment this line by turn off Test Time Augmentation(TTA)
```

Here is an example to finetune our model on HAM10000:
```bash
cd finetune
bash scripts/HAM_cleaned/ft_HAM10000_train.sh # you can replace this with your_script.sh
```
Note: Remember to adjust the path config to your desired storage location.

4. **Evaluation**
```bash
cd finetune
bash scripts/HAM_cleaned/ft_HAM10000_test.sh # you can replace this with your_script.sh
```

5. **Test-Time Augmentation (TTA)**

We've implemented a Test-Time Augmentation pipeline to enhance the classification performance of our model. TTA works by applying multiple augmentations to each test image, running predictions on each augmented version, and then aggregating the results for a more robust prediction. You can modify the setting in the class `TTAHandler` [finetune/furnace/engine_for_finetuning.py](finetune/furnace/engine_for_finetuning.py) for better performance on your dataset.