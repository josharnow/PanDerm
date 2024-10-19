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

1. Download PanDerm pre-trained weights
The model weights can be found [here](https://drive.google.com/file/d/1XHKRk2p-dS1PFQE-xRbOM3yx47i3bXmi/view?usp=sharing)


