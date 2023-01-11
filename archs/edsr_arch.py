# ---------------------------------------------------------------------------
# Enhanced Deep Residual Networks for Single Image Super-Resolution
# Official GitHub: https://github.com/sanghyun-son/EDSR-PyTorch
#
# Modified by Tianle Liu (tianle.l@outlook.com)
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn
from basicsr.utils.registry import ARCH_REGISTRY

from archs.utils import Conv2d3x3, Upsampler


class ResBlock(nn.Module):
    r"""Res Block.

    Args:
        planes: Number of input channels
        res_scale:
        act_layer:

    """

    def __init__(self, planes: int, res_scale: float = 1.0, act_layer: nn.Module = nn.ReLU) -> None:
        super(ResBlock, self).__init__()

        self.res_scale = res_scale

        self.body = nn.Sequential(Conv2d3x3(planes, planes, bias=True),
                                  act_layer(),
                                  Conv2d3x3(planes, planes, bias=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x) * self.res_scale


@ARCH_REGISTRY.register()
class _EDSR(nn.Module):
    r"""Enhanced Deep Residual Network.
    """

    def __init__(self, upscale: int, num_in_ch: int, num_out_ch: int, task: str,
                 planes: int = 256, n_blocks: int = 32, act_layer: nn.Module = nn.ReLU):
        super(_EDSR, self).__init__()

        modules_head = [Conv2d3x3(num_in_ch, planes)]
        self.head = nn.Sequential(*modules_head)

        modules_body = [ResBlock(planes, act_layer=act_layer) for _ in range(n_blocks)]
        self.body = nn.Sequential(*modules_body)

        self.tail = nn.Sequential(Upsampler(upscale=upscale, in_channels=planes,
                                            out_channels=planes, upsample_mode=task),
                                  Conv2d3x3(planes, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # head
        head_x = self.head(x)

        # body
        body_x = self.body(head_x)
        body_x = body_x + head_x

        # tail
        tail_x = self.tail(body_x)

        return tail_x


if __name__ == '__main__':
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # net = EDSR(upscale=4, planes=64, n_blocks=16)
    # print(count_parameters(net))
