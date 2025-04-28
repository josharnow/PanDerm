# PanDerm

<div style="background-color: #e6f3ff; padding: 10px; border-radius: 5px; margin-bottom: 15px;">

## A Multimodal Vision Foundation Model for Clinical Dermatology

</div>

[![Arxiv Paper](https://img.shields.io/badge/Arxiv-Paper-red)](https://arxiv.org/pdf/2410.15038)
[![Cite](https://img.shields.io/badge/Cite-BibTeX-blue)](#citation)

> **Warning:** This repo is under construction!!!

**Abstract:** We introduce PanDerm, a multimodal dermatology foundation model addressing the challenge that current deep learning models excel only at specific tasks rather than meeting the complex, multimodal requirements of clinical dermatology practice. Pretrained through self-supervised learning on over 2 million skin disease images across four imaging modalities from multiple institutions, PanDerm demonstrates state-of-the-art performance across diverse tasks, including skin cancer screening, differential diagnosis, lesion segmentation, longitudinal monitoring, and prognosis prediction, often requiring less labeled data than existing approaches. Clinical reader studies show PanDerm outperforms clinicians in early melanoma detection, improves dermatologists' diagnostic skin cancer diagnosis accuracy, and enhances non-specialists' differential diagnosis capabilities across numerous skin conditions.

![alt text](overview.png)

<div style="background-color: #e6fff2; padding: 10px; border-radius: 5px; margin-bottom: 15px;">

## Updates

</div>

- **26/04/2025:** The ViT-base version of PanDerm (PanDerm_base) is now available, providing a smaller model for more widespread usage scenarios.
- **26/04/2025:** Released the finetuning script for image classification.

<div style="background-color: #fff2e6; padding: 10px; border-radius: 5px; margin-bottom: 15px;">

## About PanDerm

</div>

<details>
<summary><b>Click to expand details about PanDerm</b></summary>

### What is PanDerm?
PanDerm is a vision-centric multimodal foundation model pretrained on 2 million dermatological images. It provides specialized representations across four dermatological imaging modalities (dermoscopy, clinical images, TBP, and dermatopathology), delivering superior performance in skin cancer diagnosis, differential diagnosis of hundreds of skin conditions, disease progression monitoring, Total Body Photography-based applications, and image segmentation.

### Why use PanDerm?
PanDerm significantly outperforms clinically popular CNN models like ResNet, especially with limited labeled data. Its strong linear probing results offer a computationally efficient alternative with lower implementation barriers. PanDerm also demonstrates superior performance compared to existing foundation models while minimizing data leakage risk—a common concern with web-scale pretrained models like DINOv2, SwavDerm, and Derm Foundation. These combined advantages make PanDerm the ideal choice for replacing both traditional CNNs and other foundation models in clinical applications, including human-AI collaboration, multimodal image analysis, and various diagnostic and progression tasks.

> **Note**: PanDerm is a general-purpose dermatology foundation model and requires fine-tuning or linear probing before application to specific tasks.

</details>

<div style="background-color: #f2e6ff; padding: 10px; border-radius: 5px; margin-bottom: 15px;">

## Installation

</div>

<details>
<summary><b>Click to expand installation instructions</b></summary>

First, clone the repo and cd into the directory:

```shell
git clone https://github.com/SiyuanYan1/PanDerm
cd PanDerm/classification
