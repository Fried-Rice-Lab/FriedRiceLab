# Fried Rice Lab

## Overview

We are updating this README, please be patient.

## How to Use

### Easy Start

```shell
# function(.py) + task(.yml) + experiment(.yml)

# train + x4 lightweight SR + using IMDN
sh run.sh train.py options/task/LSR_x4.yml options/expe/IMDN/IMDN_LSR.yml 

# test + x4 classical SR + using EDSR
sh run.sh train.py options/task/CSR_x4.yml options/expe/EDSR/EDSR_CSR.yml

# analyse.py + x4 lightweight SR + using RFDN
sh run.sh train.py options/task/LSR_x4.yml options/expe/RFDN/RFDN_LSR.yml

# infer.py + x4 lightweight SR + using BSRN
sh run.sh train.py options/task/LSR_x4.yml options/expe/BSRN/BSRN_LSR.yml  

# interpret.py + x4 lightweight SR + using ELAN
sh run.sh train.py options/task/LSR_x4.yml options/expe/ELAN/ELAN_LSR.yml  
```

### Custom Arch

Different from BasicSR, all Arch **must** have the following four args:

- `upscale`
- `num_in_ch`
- `num_out_ch`
- `task`

```python
class YourArch(torch.nn.Module):
    def __init__(self, upscale: int, num_in_ch: int, num_out_ch: int, task: str,
                 *args, **kwargs) -> None:
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
```

## Acknowledgments

This code is built on [BasicSR](https://github.com/XPixelGroup/BasicSR). We thank its developers for creating such a
useful toolbox.

## Contact

If you have any questions, please e-mail jinpeeeng.s@gmail.com.