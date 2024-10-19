# PanDerm
A General-Purpose Multimodal Foundation Model for Dermatology

## Installation
```shell
git clone https://github.com/SiyuanYan1/PanDerm
cd PanDerm
conda create -n PanDerm python=3.10 -y
conda activate PanDerm
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## 1. Download PanDerm Pre-trained Weights

1. Download weights: [Google Drive link](https://drive.google.com/file/d/1XHKRk2p-dS1PFQE-xRbOM3yx47i3bXmi/view?usp=sharing)
2. Update model path in line 42 in `PanDerm/LP_Eval/models/builder.py`:

```python
root_path = '/path/to/your/PanDerm/Model_Weights/'
```

## 2. Data Organization for Classification

Use our pre-processed datasets to reproduce paper results. For custom datasets, follow the same format.

### Dataset Links

| Dataset | Processed Data | Original Data |
|---------|----------------|---------------|
| HAM10000 | [Download](https://drive.google.com/file/d/1D9Q4B50Z5tyj5fd5EE9QWmFrg66vGvfA/view?usp=sharing) | [Official](https://challenge.isic-archive.com/data/#2018) |
| BCN20000 | [Download](https://drive.google.com/file/d/1jn1h1jWjd4go7BQ5fFWMRBMtq7poSlfi/view?usp=sharing) | [Official](https://figshare.com/articles/journal_contribution/BCN20000_Dermoscopic_Lesions_in_the_Wild/24140028/1) |
| DDI | [Download](https://drive.google.com/file/d/1F5RVqBUIxYcub1OkBm6yHTyV2TkHc65B/view?usp=sharing) | [Official](https://ddi-dataset.github.io/index.html) |
| Derm7pt | [Download](https://drive.google.com/file/d/1OYAmqG93eWLdf7dIkulY_fr0ZScvRLRg/view?usp=sharing) | [Official](https://derm.cs.sfu.ca/Welcome.html) |
| Dermnet | [Download](https://drive.google.com/file/d/1WrvReon2gA3sF9rqQGqivglG7HLFJ8he/view?usp=sharing) | [Official](https://www.kaggle.com/datasets/shubhamgoel27/dermnet) |
| HIBA | [Download](https://drive.google.com/file/d/1Sg0gFhfBaNNoeunF7C0HZgDbp5EDV436/view?usp=sharing) | [Official](https://www.isic-archive.com) |
| MSKCC | [Download](https://drive.google.com/file/d/17ma4tREXHAq1ZcBT7lZBhwO-3UHSbDW2/view?usp=sharing) | [Official](https://www.isic-archive.com) |
| PAD-UFES | [Download](https://drive.google.com/file/d/1NLv0EH3QENuRxW-_-BSf4KMP9cPjBk9o/view?usp=sharing) | [Official](https://www.kaggle.com/datasets/mahdavi1202/skin-cancer) |
| PATCH16 | [Download](https://drive.google.com/file/d/1wDMIfYrQatkeADoneHgjXQrawVMK-TFL/view?usp=sharing) | [Official](https://heidata.uni-heidelberg.de/dataset.xhtml?persistentId=doi:10.11588/data/7QCR8S) |

## 3. Linear Evaluation on Downstream Tasks

Example using HAM10000:

### Key Parameters
- `nb_classes`: Number of classes in your dataset
- `batch_size`: Adjust based on GPU memory
- `percent_data`: Percentage of training data used (e.g., 0.1 for 10%)

### Evaluation Command

```bash
cd LP_Eval
CUDA_VISIBLE_DEVICES=0 python linear_eval.py \
  --batch_size 1000 \
  --model 'PanDerm' \
  --nb_classes 7 \
  --percent_data 1.0 \
  --csv_filename 'PanDerm_results.csv' \
  --output_dir "/path/to/output/" \
  --csv_path "/path/to/HAM_clean.csv" \
  --root_path "/path/to/ISIC2018/"
```

### Additional Evaluations
```bash
cd LP_Eval
bash script/lp.sh
```

### Starter Code
See [HAM_clean_evaluation.ipynb](LP_Eval/notebooks/HAM_clean_evaluation.ipynb) for:
- Loading pre-trained model
- Feature extraction
- Basic classification

## 4. Skin Lesion Segmentation

Refer to [Segmentation.md](Segmentation.md) for details.
