## SP&A-Net
[Project]() | [Arxiv](https://arxiv.org/pdf/1905.01164.pdf) | [CVF]() 
### Official tensorflow implementation of the paper: "A Novel Deep Learning Architecture for Defect Pattern Classification: Self-Proliferation-and-Attention Neural Network"
####  CVPR 2021


## SP&A-Net's Architecture.
* The first function of SP&A-Net is the self-proliferation, using a series of linear transformations to generate more feature maps at a cheaper cost. We can train image classifier in a more efficient way.
* The second function is self-attention, capturing the long-range dependencies of the feature map using the channel-wise and spatial attention mechanism.

## Code

### Install dependencies

```
python -m pip install -r requirements.txt
```

This code was tested with python 3.7  

###  Train
This script is based on CIFAR-10 as an example. For training, please run:

```
python TestRun.py
```

## Script Introduction

```Self_Proliferate.py``` is used to generate more feature maps (As paper section 3.1).

```Self_Attention.py``` is used to capturing the long-range dependencies of the feature map (As paper secton 3.2).

```Self_Proliferate_and_Attention.py``` follow the spirit of MobileNet,  "capture features in high dimensions and transfer information in low dimensions",  to make the network more efficient. (As paper secton 3.3).

```SPA_Net.py``` is the overall network architecture of SP&A-Net. Please refer section 3.4 of this paper.

```SP&A-Net-Test-Run.ipynb``` is in the form of a Jupyter Notebook as a simple display with CIFAR-10 as the training object.

## Contribution
In addition, this research won the TSMC 2020 machine learning competition champion. It has achieved substantial improvement in the industry.
