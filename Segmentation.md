# PanDerm - Skin Lesion Segmentation

This is an implementation of using PanDerm for skin lesion segmentation.

## Install Environment

Main dependencies: Pytorch, Pytorch Lightning, MMSegmentation.

Install all dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation

1. Download datasets:
   - [ISIC2018 task1](https://challenge.isic-archive.com/data/#2018)
   - [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
2. Update `_load_name()` in `./datasets/dataset_seg.py` with your data storage path.

## Fine-tuning with PanDerm Weight

### Steps:
1. Download PanDerm pretrained weights from [here](https://drive.google.com/file/d/1XHKRk2p-dS1PFQE-xRbOM3yx47i3bXmi/view?usp=sharing).
2. Set pretrained path:
   - Change `cae_weight` in `./models/cae_seg.py`.
   - Change parameter `pretrained` in `run.sh`.
3. Start training:
   ```bash
   cd Skin_Seg
   bash run.sh
   ```
   Note: Change `save_name` config to your own storage path.

4. For evaluation:
   Add `--evaluate` at the end of `run.sh`:
   ```bash
   cd Skin_Seg
   bash run.sh --evaluate
   ```
   This will load the checkpoint from your model storage path for evaluation.

## Starter Code for Segmenting Your Own Dermoscopic Images

Check out our Jupyter Notebook: [**evaluate.ipynb**](Skin_Seg/evaluate.ipynb)

This notebook demonstrates:
- Loading our pre-trained model
- Using it for skin lesion segmentation
