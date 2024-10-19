# PanDerm - Skin Lesion Segmentation

This is an implementation of using PanDerm for skin lesion segmentation. 

## Install Environment

Main dependencies: Pytorch, Pytorch Lightning, MMSegmentation. 

You can directly install all the dependencies by running: 
```
pip install -r requirements.txt
```

## Data preparation

Please download [ISIC2018 task1](https://challenge.isic-archive.com/data/#2018) and [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000). After downloading the dataset, please change the function `_load_name()` in `./datasets/dataset_seg.py` to your data storage path. 

## Fine-tuning with PanDerm Weight

To finetune the PanDerm on skin lesion segmentation, please follow these steps: 

1. Download PanDerm pretrained weight [here](https://drive.google.com/file/d/1XHKRk2p-dS1PFQE-xRbOM3yx47i3bXmi/view?usp=sharing). 
2. Set pretrained path. This can be done by:
    * Change `cae_weight` in `./models/cae_seg.py`. 
    * Change parameter `pretrained` in `run.sh`. 
3. Start training. You can directly run `run.sh`. Note: you should change `save_name` config to your own storage path.
```bash
cd Skin_Seg
bash run.sh
```
4. For evaluation, you can add configuration `--evaluate` at the end of `run.sh`. It will directly load the checkpoint from your model storage path for evaluating.
```bash
cd Skin_Seg
bash run.sh --evaluate
```

## Starter Code for segmenting your own dermoscopic images

Check out our easy-to-follow Jupyter Notebook:

[**evaluate.ipynb**](Skin_Seg/evaluate.ipynb)

This notebook shows you how to:
- Load our pre-trained model
- Use it for skin lesion segmentation

