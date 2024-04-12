# ----------------------------------------------------------------------
# Multi-scale Attention Network for Single Image Super-Resolution
# Official GitHub: https://github.com/icandle/MAN
#
# Modified by Tianle Liu (tianle.l@outlook.com)
# ----------------------------------------------------------------------
import torch
import torch.nn as nn
from basicsr.utils.registry import ARCH_REGISTRY

from archs.utils import Conv2d1x1
from archs.utils import Conv2d3x3
from archs.utils import LayerNorm4D
from archs.utils import MeanShift
from archs.utils import Upsampler


class GSAU(nn.Module):
    r"""Gated Spatial Attention Unit.

    Args:
        n_feats: Number of input channels

    """

    def __init__(self, n_feats: int) -> None:
        super().__init__()
        i_feats = n_feats * 2

        self.Conv1 = Conv2d1x1(n_feats, i_feats)
        self.DWConv1 = nn.Conv2d(
            n_feats, n_feats, 7, 1, 7 // 2, groups=n_feats)
        self.Conv2 = Conv2d1x1(n_feats, n_feats)

        self.norm = LayerNorm4D(n_feats)
        self.scale = nn.Parameter(torch.zeros(
            (1, n_feats, 1, 1)), requires_grad=True)

    def forward(self, x) -> torch.Tensor:
        shortcut = x.clone()

        x = self.Conv1(self.norm(x))
        a, x = torch.chunk(x, 2, dim=1)
        x = x * self.DWConv1(a)
        x = self.Conv2(x)

        return x * self.scale + shortcut


class MLKA(nn.Module):
    r"""Multi-scale Large Kernel Attention.

    Args:
        n_feats: Number of input channels

    """

    def __init__(self, n_feats: int) -> None:
        super().__init__()
        i_feats = 2 * n_feats

        self.n_feats = n_feats
        self.i_feats = i_feats
        self.norm = LayerNorm4D(n_feats)
        self.scale = nn.Parameter(torch.zeros(
            (1, n_feats, 1, 1)), requires_grad=True)

        self.LKA7 = nn.Sequential(
            nn.Conv2d(n_feats // 3, n_feats // 3, 7,
                      1, 7 // 2, groups=n_feats // 3),
            nn.Conv2d(n_feats // 3, n_feats // 3, 9, 1, (9 // 2)
                      * 4, groups=n_feats // 3, dilation=4),
            nn.Conv2d(n_feats // 3, n_feats // 3, 1, 1, 0))
        self.LKA5 = nn.Sequential(
            nn.Conv2d(n_feats // 3, n_feats // 3, 5,
                      1, 5 // 2, groups=n_feats // 3),
            nn.Conv2d(n_feats // 3, n_feats // 3, 7, 1, (7 // 2)
                      * 3, groups=n_feats // 3, dilation=3),
            nn.Conv2d(n_feats // 3, n_feats // 3, 1, 1, 0))
        self.LKA3 = nn.Sequential(
            nn.Conv2d(n_feats // 3, n_feats // 3,
                      3, 1, 1, groups=n_feats // 3),
            nn.Conv2d(n_feats // 3, n_feats // 3, 5, 1, (5 // 2)
                      * 2, groups=n_feats // 3, dilation=2),
            nn.Conv2d(n_feats // 3, n_feats // 3, 1, 1, 0))

        self.X3 = nn.Conv2d(n_feats // 3, n_feats // 3,
                            3, 1, 1, groups=n_feats // 3)
        self.X5 = nn.Conv2d(n_feats // 3, n_feats // 3, 5,
                            1, 5 // 2, groups=n_feats // 3)
        self.X7 = nn.Conv2d(n_feats // 3, n_feats // 3, 7,
                            1, 7 // 2, groups=n_feats // 3)

        self.proj_first = Conv2d1x1(n_feats, i_feats)
        self.proj_last = Conv2d1x1(n_feats, n_feats)

    def forward(self, x) -> torch.Tensor:
        shortcut = x.clone()

        x = self.norm(x)
        x = self.proj_first(x)
        a, x = torch.chunk(x, 2, dim=1)
        a_1, a_2, a_3 = torch.chunk(a, 3, dim=1)
        a = torch.cat([self.LKA3(a_1) * self.X3(a_1), self.LKA5(a_2)
                      * self.X5(a_2), self.LKA7(a_3) * self.X7(a_3)], dim=1)
        x = self.proj_last(x * a)

        return x * self.scale + shortcut


class MAB(nn.Module):
    r"""Multi-scale Attention Block.

    Args:
        n_feats: Number of input channels

    """

    def __init__(self, n_feats):
        super().__init__()

        self.LKA = MLKA(n_feats)
        self.LFE = GSAU(n_feats)

    def forward(self, x):
        x = self.LKA(x)
        x = self.LFE(x)

        return x


class LKAT(nn.Module):
    r"""Large Kernel Attention Tail.

    Args:
        n_feats: Number of input channels

    """

    def __init__(self, n_feats):
        super().__init__()

        self.conv0 = nn.Sequential(
            Conv2d1x1(n_feats, n_feats),
            nn.GELU())

        self.att = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 7, 1, 7 // 2, groups=n_feats),
            nn.Conv2d(n_feats, n_feats, 9, 1, (9 // 2)
                      * 3, groups=n_feats, dilation=3),
            nn.Conv2d(n_feats, n_feats, 1, 1, 0))

        self.conv1 = Conv2d1x1(n_feats, n_feats)

    def forward(self, x):
        x = self.conv0(x)
        x = x * self.att(x)
        x = self.conv1(x)
        return x


class ResGroup(nn.Module):
    def __init__(self, n_resblocks, n_feats):
        super(ResGroup, self).__init__()
        self.body = nn.ModuleList([MAB(n_feats)
                                   for _ in range(n_resblocks)])

        self.body_t = LKAT(n_feats)

    def forward(self, x):
        res = x.clone()

        for i, block in enumerate(self.body):
            res = block(res)
        x = self.body_t(res) + x

        return x


@ARCH_REGISTRY.register()
class MAN(nn.Module):
    r"""Multi-scale Attention Network.

    Args:
        n_feats: Number of input channels
        n_resblocks:
        upscale:
        num_in_ch:
        num_out_ch:
        task:

    """

    def __init__(self, upscale: int, num_in_ch: int, num_out_ch: int,
                 task: str, n_resblocks=36, n_feats=180):
        super(MAN, self).__init__()

        self.sub_mean = MeanShift(255)
        self.add_mean = MeanShift(255, sign=1)

        self.head = Conv2d3x3(num_in_ch, n_feats)

        self.body = ResGroup(n_resblocks, n_feats)

        self.tail = Upsampler(upscale=upscale, in_channels=n_feats,
                              out_channels=num_out_ch, upsample_mode=task)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # reduce the mean of pixels
        sub_x = self.sub_mean(x)

        # head
        head_x = self.head(sub_x)

        # body
        body_x = self.body(head_x)

        # tail
        tail_x = self.tail(body_x)

        # add the mean of pixels
        x = self.add_mean(tail_x)

        return x


if __name__ == '__main__':
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # MAN-light
    net = MAN(upscale=4,  n_resblocks=24, n_feats=60,
              num_in_ch=3, num_out_ch=3, task='lsr')
    print(count_parameters(net))

    data = torch.randn(1, 3, 120, 80)
    print(net(data).size())

