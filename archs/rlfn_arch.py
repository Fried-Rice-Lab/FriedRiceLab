# -----------------------------------------------------------------
# Residual Local Feature Network for Efficient Super-Resolution
# Official GitHub: https://github.com/bytedance/RLFN
#
# Modified by Jinchen Zhu (jinchen.z@outlook.com)
# -----------------------------------------------------------------
import torch
import torch.nn as nn
from basicsr.utils.registry import ARCH_REGISTRY

from archs.utils import Conv2d1x1, Conv2d3x3, ESA, Upsampler


class RLFB(nn.Module):
    r"""Residual Local Feature Block.

    Args:
        planes:

    """

    def __init__(self, planes: int):
        super(RLFB, self).__init__()

        modules_body = list()
        for i in range(3):
            modules_body.append(Conv2d3x3(planes, planes))
            modules_body.append(nn.ReLU())
        self.body = nn.Sequential(*modules_body)

        self.tail = nn.Sequential(Conv2d1x1(planes, planes),
                                  ESA(planes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tail(self.body(x) + x)


@ARCH_REGISTRY.register()
class RLFN(nn.Module):
    r"""Residual Local Feature Network.

    Args:
        upscale:
        planes:
        num_blocks: Number of RLFB

    """

    def __init__(self, upscale: int, num_in_ch: int, num_out_ch: int, task: str,
                 planes: int, num_blocks: int = 6):
        super(RLFN, self).__init__()

        self.head = Conv2d3x3(num_in_ch, planes)

        self.body = nn.Sequential(*[RLFB(planes) for _ in range(num_blocks)],
                                  Conv2d3x3(planes, planes))

        self.tail = Upsampler(upscale=upscale, in_channels=planes,
                              out_channels=num_out_ch, upsample_mode=task)

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

    # net = RLFN(upscale=4, planes=52)
    # print(count_parameters(net))

    # data = torch.randn((1, 3, 64, 64))
    # print(net(data).size())
