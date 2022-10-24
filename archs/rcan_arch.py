# ---------------------------------------------------------------------------
# Image Super-Resolution Using Very Deep Residual Channel Attention Networks
# Official GitHub: https://github.com/yulunzhang/RCAN
#
# Modified by Tianle Liu (tianle.l@outlook.com)
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn
from basicsr.utils.registry import ARCH_REGISTRY

from archs.utils import Conv2d1x1, Conv2d3x3, Upsampler


class CA(nn.Module):
    r"""Channel Attention.

    Args:
        planes: Number of input channels
        reduction:
        act_layer:

    """

    def __init__(self, planes: int, reduction: int = 16,
                 act_layer: nn.Module = nn.ReLU) -> None:
        super(CA, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            Conv2d1x1(planes, planes // reduction, bias=False),
            act_layer(inplace=True),
            Conv2d1x1(planes // reduction, planes, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        return x * self.sigmoid(avg_out)


class RCAB(nn.Module):
    r"""Residual Channel Attention Block.

    Args:
        planes: Number of input channels
        res_scale:
        act_layer:

    """

    def __init__(self, planes: int, res_scale: float = 1.0,
                 act_layer: nn.Module = nn.ReLU) -> None:
        super(RCAB, self).__init__()

        self.res_scale = res_scale

        self.rcab = nn.Sequential(Conv2d3x3(planes, planes),
                                  act_layer(inplace=True),
                                  Conv2d3x3(planes, planes),
                                  CA(planes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.rcab(x) * self.res_scale


class RG(nn.Module):
    r"""Residual Groups.

    Args:
        planes: Number of input channels
        num_blocks: Number of RCAB

    """

    def __init__(self, planes: int, num_blocks: int) -> None:
        super(RG, self).__init__()

        self.body = nn.Sequential(*[RCAB(planes) for _ in range(num_blocks)])
        self.conv = Conv2d3x3(planes, planes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(self.body(x))


class RIR(nn.Module):
    r"""Residual in Residual.

    Args:
        planes: Number of input channels
        num_groups: Number of RG
        num_blocks: Number of RCAB

    """

    def __init__(self, planes: int, num_groups: int, num_blocks: int) -> None:
        super(RIR, self).__init__()

        self.rir = nn.Sequential(*[RG(planes, num_blocks) for _ in range(num_groups)])
        self.conv = Conv2d3x3(planes, planes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(self.rir(x))


@ARCH_REGISTRY.register()
class _RCAN(nn.Module):
    r"""Residual Channel Attention Network.

        Args:
            upscale:
            planes: Number of input channels
            num_groups: Number of RG
            num_blocks: Number of RCAB

        """

    def __init__(self, upscale: int, num_in_ch: int, num_out_ch: int, task: str,
                 planes: int = 64, num_groups: int = 10, num_blocks: int = 20) -> None:
        super(_RCAN, self).__init__()

        self.head = nn.Sequential(Conv2d3x3(num_in_ch, planes))

        self.body = nn.Sequential(RIR(planes, num_groups, num_blocks))

        self.tail = nn.Sequential(Upsampler(upscale=upscale, in_channels=planes,
                                            out_channels=planes, upsample_mode=task),
                                  Conv2d3x3(planes, num_out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # head
        head_x = self.head(x)

        # body
        body_x = self.body(head_x)

        # tail
        tail_x = self.tail(body_x)
        return tail_x


if __name__ == '__main__':
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # net = _RCAN(upscale=4, planes=64, num_groups=10, num_blocks=20)
    # print(count_parameters(net))

    # data = torch.randn(1, 3, 120, 80)
    # print(net(data).size())
