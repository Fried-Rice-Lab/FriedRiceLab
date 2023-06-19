# (EMT) Efficient Mixed Transformer for Single Image Super-Resolution [[arXiv](https://arxiv.org/abs/2305.11403)]
Ling Zheng*, [Jinchen Zhu*](https://github.com/Jinchen2028), [Jinpeng Shi*](https://github.com/jinpeng-s), Shizhuang Weng^

> *: (Co-)first author(s)
> 
> ^: corresponding author(s)

## Table of contents

<!--ts-->

- [Abstract](#abstract)
- [Environment](#environment)
- [Demo test](#demo-test)
- [Retraining](#retraining)
- [Citation](#citation)
- [Contact](#contact)

<!--te-->
  
## Abstract
*Recently, Transformer-based methods have achieved impressive results in single image super-resolution (SISR). However, the lack of locality mechanism and high complexity limit their application in the field of super-resolution (SR). To solve these problems, we propose a new method, Efficient Mixed Transformer (EMT) in this study. Specifically, we propose the Mixed Transformer Block (MTB), consisting of multiple consecutive transformer layers, in some of which the Pixel Mixer (PM) is used to replace the Self-Attention (SA). PM can enhance the local knowledge aggregation with pixel shifting operations. At the same time, no additional complexity is introduced as PM has no parameters and floating-point operations. Moreover, we employ striped window for SA (SWSA) to gain an efficient global dependency modelling by utilizing image anisotropy. Experimental results show that EMT outperforms the existing methods on benchmark dataset and achieved state-of-the-art performance.*

## Environment
```shell
conda create -n frl python
conda activate frl
pip install torch torchvision basicsr einops timm matplotlib
```

## Demo test
```shell
python test.py -expe_opt options/repr/EMT/EMT.yml -task_opt options/task/LSR_x4.yml
```
## Retraining
```shell
python train.py -expe_opt options/repr/EMT/EMT.yml -task_opt options/task/LSR_x4.yml
```

## Citation

If EMT helps your research or work, please consider citing the following works:

----------
```BibTex
@article{zheng2023efficient,
  title={Efficient Mixed Transformer for Single Image Super-Resolution},
  author={Zheng, Ling and Zhu, Jinchen and Shi, Jinpeng and Weng, Shizhuang},
  journal={arXiv preprint arXiv:2305.11403},
  year={2023}
}
```

## Contact

If you have any questions, please contact [Jinchen Zhu](https://github.com/Jinchen2028) (jinchen.z@outlook.com), Shizhuang Weng (weng_1989@126.com).
