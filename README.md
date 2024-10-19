# PanDerm
A General-Purpose Multimodal Foundation Model for Dermatology


## Installation
First clone the repo and cd into the directory:
```shell
git clone https://github.com/SiyuanYan1/PanDerm
cd PanDerm
```
Then create a conda env and install the dependencies:
```shell
conda create -n PanDerm python=3.10 -y
conda activate PanDerm
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## 1. Download PanDerm Pre-trained Weights

### Obtaining the Model Weights
Download the pre-trained model weights from [this Google Drive link](https://drive.google.com/file/d/1XHKRk2p-dS1PFQE-xRbOM3yx47i3bXmi/view?usp=sharing).

### Configuring the Model Path
After downloading, you need to update the model weights path in the code:

1. Open the file `PanDerm/LP_Eval/models/builder.py`
2. Locate line 42
3. Replace the existing path with the directory where you saved the model weights:

```python
root_path = '/path/to/your/PanDerm/Model_Weights/'
```
## 2. Organise your data into this directory (Public datasets used in this study can be downloaded here)

Notice we clean these datasets when possible to avoid data leakage between splits, can be found in Download link, which is slightly different from the dataset provided in official website.

### Data split
| Dataset | Download Link  |   Official Website |
| ------------- | ------------------ |------------------ |
| HAM10000 | [Google Drive](https://drive.google.com/file/d/1D9Q4B50Z5tyj5fd5EE9QWmFrg66vGvfA/view?usp=sharing) | https://challenge.isic-archive.com/data/#2018
| BCN20000 | [Google Drive](https://drive.google.com/file/d/1jn1h1jWjd4go7BQ5fFWMRBMtq7poSlfi/view?usp=sharing)  | https://figshare.com/articles/journal_contribution/BCN20000_Dermoscopic_Lesions_in_the_Wild/24140028/1
| DDI| [Google Drive](https://drive.google.com/file/d/1F5RVqBUIxYcub1OkBm6yHTyV2TkHc65B/view?usp=sharing)       
| Derm7pt | [Google Drive](https://drive.google.com/file/d/1OYAmqG93eWLdf7dIkulY_fr0ZScvRLRg/view?usp=sharing)      
| Dermnet | [Google Drive](https://drive.google.com/file/d/1WrvReon2gA3sF9rqQGqivglG7HLFJ8he/view?usp=sharing)       
| HIBA| [Google Drive](https://drive.google.com/file/d/1Sg0gFhfBaNNoeunF7C0HZgDbp5EDV436/view?usp=sharing)       
| MSKCC | [Google Drive](https://drive.google.com/file/d/17ma4tREXHAq1ZcBT7lZBhwO-3UHSbDW2/view?usp=sharing)     
| PAD-UFES | [Google Drive](https://drive.google.com/file/d/1NLv0EH3QENuRxW-_-BSf4KMP9cPjBk9o/view?usp=sharing)       
| PATCH16 | [Google Drive](https://drive.google.com/file/d/1wDMIfYrQatkeADoneHgjXQrawVMK-TFL/view?usp=sharing)  




