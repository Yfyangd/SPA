## SP&A-Net
### Official tensorflow implementation of the paper: "Semiconductor Defect Pattern Classification by Self-Proliferation-and-Attention Neural Network (SP&A-Net)"

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

```Self_Proliferate.py``` is used to generate more feature maps (As paper section 3.A).

```Self_Attention.py``` is used to capturing the long-range dependencies of the feature map (As paper secton 3.B).

```Self_Proliferate_and_Attention.py``` follow the spirit of MobileNet,  "capture features in high dimensions and transfer information in low dimensions",  to make the network more efficient. (As paper secton 3.C).

```SPA_Net.py``` is the overall network architecture of SP&A-Net. Please refer section 3.D of this paper.

```CircleLoss.py``` is used to estimate the loss rate during model training with two elemental deep feature learning approaches: class-level labels and pair-wise labels (as section 3-E).

```SP&A-Net-Test-Run.ipynb``` is in the form of a Jupyter Notebook as a simple display with CIFAR-10 as the training object.
