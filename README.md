# Fried Rice Lab

## Overview

We are updating this README, please be patient.

## News

**23.01.11** FRL code v2.0 has been released 🎉

**22.11.15** FRL code v1.0 has been released 😎

## Our Works

Oops. 🤪

## Quick Start

For your convenience, we have prepared a demo Set5 dataset located in `datasets/demo_data/Demo_Set5`, and a demo [ELAN](https://github.com/xindongzhang/ELAN) pre-training weight located in `modelzoo/ELAN/ELAN-demo.pth`. So you can try out the functions of the FRL code by simply using the following command:

- train:
    ```shell
    sh run.sh train.py options/task/LSR_x4.yml options/repr/ELAN/ELAN_LSR.yml
    ```
- test:
    ```shell
    sh run.sh test.py options/task/LSR_x4.yml options/repr/ELAN/ELAN_LSR.yml
    ```
- analyse:
    ```shell
    sh run.sh analyse.py options/task/LSR_x4.yml options/repr/ELAN/ELAN_LSR.yml
    ```
- interpret:
    ```shell
    sh run.sh interpret.py options/task/LSR_x4.yml options/repr/ELAN/ELAN_LSR.yml
    ```
- infer:
    ```shell
    sh run.sh infer.py options/task/LSR_x4.yml options/expe/repr/ELAN_LSR.yml
    ```

## New Features

If you are new to BasicSR, please refer to [BasicSR](https://github.com/XPixelGroup/BasicSR) and [BasicSR example](https://github.com/xinntao/BasicSR-examples) first.

Please replace `PairedImageDataset` with `IRDataset` and `SRModel` with `IRModel` to use these new features.

### 1. New data flow

By default, BasicSR converts the loaded image to an input tensor with values between [0, 1], and then converts the output tensor to the original data range to save it. However, perhaps you prefer to use the original data range, such as PNG images ([0, 255]) and TIF images ([0, 65535]).

The `IRDataset` and `IRModel`support different bits of data flows. You can try different data flows by simply changing the YML configuration as follows:

```yaml
# general settings
name: DemoArch
model_type: IRModel
num_gpu: 1
manual_seed: 42
bit: 8 # here!
```

The `bit` represents using the data flow with values between [0, 2 ** bit - 1]. Set it to 0 to use the default data flow of BasicSR.

> 🤠 The implementation of the new data flow refers to the code of [EDSR](https://github.com/sanghyun-son/EDSR-PyTorch) and [RCAN-it](https://github.com/zudi-lin/rcan-it). Since many codes of existing image super-resolution works are based on these codes, you can reproduce these using the FRL code. We hope this will make your work easier.

> ⚠️ Using new data flows may result in inaccurate metric calculations (error less than 0.0001). For accurate metric results, use `evaluate.m` located in `scripts`. PR is welcomed.

### 2. New configuration loading and arch customizing

A complete image restoration experiment consists of three parts: data, arch, and training strategy. We made some minor changes to the YML configuration loading of BasicSR to decouple them. The FRL code loads two YML configurations instead of one:

- **expe.yml**: contains arch-related and strategy-related configurations
- **task.yml**: contains data-related configurations

Therefore, please use the following command to run FRL code:

```shell
python func.py -expe_opt expe.yml -task_opt task.yml
```

Different from BasicSR, all arches in the FRL code **must** have the following four args:

- `upscale`: upscale factor, such as: 2, 3, 4, and 8 for "lsr" or "csr", 1 for "denoising"
- `num_in_ch`: input channel number
- `num_out_ch`: output channel number
- `task`: image restoration task, such as: "lsr", "csr" or "denoising"

Therefore, an arch implementation should be as follows:

```python
import torch


class DemoArch(torch.nn.Module):
    def __init__(self, upscale: int, num_in_ch: int, num_out_ch: int, task: str,
                 num_groups: int, num_blocks: int, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
```

> ⚠️ These four args are **task-related** and should be defined in `task.yml`. Thus, your `expe.yml` should look like:
>
> ```yaml
> network_g:
>     type: DemoArch
>     num_groups: 20
>     num_blocks: 10
> ```

### 3. New functions

#### 3.1 Analyze

This method will analyse the complexity of an arch, including:

- **#Params**: total learnable parameter number

- **#FLOPs**: floating point operations (measure using an image of size 3 * 256 * 256 by default)

- **#Acts**: number of elements of all outputs of convolutional layers (measure using an image of size 3 * 256 * 256 by default)

- **#Conv**: number of convolutional layers

- **#Memory**: maximum GPU memory consumption when inferring a dataset

- **#Ave. Time**: average inference time per image on a dataset
  > ⚠️ The **#Ave. Time** result of the first dataset is **incorrect** (higher than the real value). We are working on it.

Run the following command to get the results:

```shell
sh run.sh analyse.py options/task/LSR_x4.yml options/repr/ELAN/ELAN_LSR.yml
```

![](figs/analyse.png)

#### 3.2 Interpret

This method comes from the paper "Interpreting Super-Resolution Networks with Local Attribution Maps". When reconstructing the patches marked with red boxes, a higher DI indicates involving a larger range of contextual information, and a darker color indicates a higher degree of contribution.

Run the following command to get the results:

```shell
sh run.sh interpret.py options/task/LSR_x4.yml options/repr/ELAN/ELAN_LSR.yml
```

![](figs/interpret.png)

![](figs/lam_result.png)For more details, please refer to official [Colab](https://colab.research.google.com/drive/1ZodQ8CRCfHw0y6BweG9zB3YrK_lYWcDk?usp=sharing#scrollTo=oUaR2N96819-) of LAM.

#### 3.3 Infer

You can use this method to restore your own image.

Run the following command to get the results:

![](figs/infer.png)

## How to Use

### 1. Preparation

#### 1.1 Environment

```shell
conda create -n frl python
conda activate frl
pip install torch torchvision basicsr einops timm matplotlib
```

> ⚠️ If you are from mainland China, please run the following command before `pip`:
>
> ```shell
> pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
> ```

#### 1.2 Data

Updating. 🤪

#### 1.3 Pretraining weight

We have provided some pre-training weights. Please download them under `modelzoo/${arch name}`.

| Models | Links                                                           |
|--------|-----------------------------------------------------------------|
| ELAN   | [Link](https://github.com/Fried-Rice-Lab/FriedRiceLab/releases) |

### 2. ...

Updating. 🤪

## Acknowledgments

This code is mainly built on [BasicSR](https://github.com/XPixelGroup/BasicSR). We thank its developers for creating such a useful toolbox. The code of the function `analyse.py` is based on [NTIRE2022 ESR](https://github.com/ofsoundof/NTIRE2022_ESR), and the code of the function `interpret.py` is based on [LAM](https://github.com/X-Lowlevel-Vision/LAM_Demo). All other image restoration network codes are from their official GitHub. More details can be found in their implementations.

## Contact

This repository is maintained by [Jinpeng Shi](https://github.com/jinpeng-s) (jinpeeeng.s@gmail.com). Special thanks to [Tianle Liu](https://github.com/TIANLE233) (tianle.l@outlook.com) for his excellent code testing work. Due to our limited capacity, we welcome any PR.