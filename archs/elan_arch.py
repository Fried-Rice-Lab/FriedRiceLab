# ---------------------------------------------------------------------------
# Efficient Long-Range Attention Network for Image Super-resolution
# Official GitHub: https://github.com/xindongzhang/ELAN
#
# Modified by Tianle Liu (tianle.l@outlook.com)
# ---------------------------------------------------------------------------
import math

import torch
import torch.nn as nn
import torch.nn.functional as f
from basicsr.utils.registry import ARCH_REGISTRY
from einops import rearrange

from archs.utils import Conv2d1x1, Conv2d3x3, ShiftConv2d1x1, MeanShift, Upsampler


class LFE(nn.Module):
    r"""Local Feature Extraction.

       Args:
           planes: Number of input channels
           r_expand: Channel expansion ratio
           act_layer:

       """

    def __init__(self, planes: int, r_expand: int = 2,
                 act_layer: nn.Module = nn.ReLU) -> None:
        super(LFE, self).__init__()

        self.lfe = nn.Sequential(ShiftConv2d1x1(planes, planes * r_expand),
                                 act_layer(inplace=True),
                                 ShiftConv2d1x1(planes * r_expand, planes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lfe(x)


class GMSA(nn.Module):
    r"""Residual Feature Distillation Network.

       Args:
           planes: Number of input channels
           shifts:
           window_sizes: Window size
           pass_attn:

       """

    def __init__(self, planes: int = 60, shifts: int = 0,
                 window_sizes: tuple = (4, 8, 12), pass_attn: int = 0) -> None:

        super(GMSA, self).__init__()
        self.shifts = shifts
        self.window_sizes = window_sizes

        if pass_attn == 0:
            self.split_chns = [planes * 2 // 3, planes * 2 // 3, planes * 2 // 3]
            self.project_inp = nn.Sequential(
                Conv2d1x1(planes, planes * 2),
                nn.BatchNorm2d(planes * 2)
            )
            self.project_out = Conv2d1x1(planes, planes)
        else:
            self.split_chns = [planes // 3, planes // 3, planes // 3]
            self.project_inp = nn.Sequential(
                Conv2d1x1(planes, planes),
                nn.BatchNorm2d(planes)
            )
            self.project_out = Conv2d1x1(planes, planes)

    def forward(self, x: torch.Tensor, prev_atns: list = None):
        b, c, h, w = x.shape
        x = self.project_inp(x)
        xs = torch.split(x, self.split_chns, dim=1)
        ys = []
        atns = []
        if prev_atns is None:
            for idx, x_ in enumerate(xs):
                wsize = self.window_sizes[idx]
                if self.shifts > 0:
                    x_ = torch.roll(x_, shifts=(-wsize // 2, -wsize // 2), dims=(2, 3))
                q, v = rearrange(
                    x_, 'b (qv c) (h dh) (w dw) -> qv (b h w) (dh dw) c',
                    qv=2, dh=wsize, dw=wsize
                )
                atn = (q @ q.transpose(-2, -1))
                atn = atn.softmax(dim=-1)
                y_ = (atn @ v)
                y_ = rearrange(
                    y_, '(b h w) (dh dw) c-> b (c) (h dh) (w dw)',
                    h=h // wsize, w=w // wsize, dh=wsize, dw=wsize
                )
                if self.shifts > 0:
                    y_ = torch.roll(y_, shifts=(wsize // 2, wsize // 2), dims=(2, 3))
                ys.append(y_)
                atns.append(atn)
            y = torch.cat(ys, dim=1)
            y = self.project_out(y)
            return y, atns
        else:
            for idx, x_ in enumerate(xs):
                wsize = self.window_sizes[idx]
                if self.shifts > 0:
                    x_ = torch.roll(x_, shifts=(-wsize // 2, -wsize // 2), dims=(2, 3))
                atn = prev_atns[idx]
                v = rearrange(
                    x_, 'b (c) (h dh) (w dw) -> (b h w) (dh dw) c',
                    dh=wsize, dw=wsize
                )
                y_ = (atn @ v)
                y_ = rearrange(
                    y_, '(b h w) (dh dw) c-> b (c) (h dh) (w dw)',
                    h=h // wsize, w=w // wsize, dh=wsize, dw=wsize
                )
                if self.shifts > 0:
                    y_ = torch.roll(y_, shifts=(wsize // 2, wsize // 2), dims=(2, 3))
                ys.append(y_)
            y = torch.cat(ys, dim=1)
            y = self.project_out(y)
            return y, prev_atns


class ELAB(nn.Module):
    r"""Residual Feature Distillation Network.

       Args:
           planes: Number of input channels
           r_expand: Channel expansion ratio
           shifts:
           window_sizes: Window size
           n_share: Depth of shared attention.

       """

    def __init__(self, planes: int = 60, r_expand: int = 2, shifts: int = 0,
                 window_sizes: tuple = (4, 8, 12), n_share: int = 1) -> None:
        super(ELAB, self).__init__()

        self.modules_lfe = nn.ModuleList([LFE(planes=planes, r_expand=r_expand)
                                          for _ in range(n_share + 1)])
        self.modules_gmsa = nn.ModuleList([GMSA(planes=planes, shifts=shifts,
                                                window_sizes=window_sizes, pass_attn=i)
                                           for i in range(n_share + 1)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        atn = None
        for module1, module2 in zip(self.modules_lfe, self.modules_gmsa):
            x = module1(x) + x
            y, atn = module2(x, atn)
            x = y + x
        return x


@ARCH_REGISTRY.register()
class ELAN(nn.Module):
    r"""Residual Feature Distillation Network.

       Args:
           upscale:
           planes: Number of input channels
           num_blocks: Number of RFDB
           window_sizes: Window size
           n_share: Depth of shared attention
           r_expand: Channel expansion ratio

       """

    def __init__(self, upscale: int, num_in_ch: int, num_out_ch: int, task: str,
                 planes: int = 60, num_blocks: int = 24,
                 window_sizes: tuple = (4, 8, 12), n_share: int = 1, r_expand: int = 2) -> None:
        super(ELAN, self).__init__()

        self.window_sizes = window_sizes
        self.upscale = upscale
        self.sub_mean = MeanShift(255)
        self.add_mean = MeanShift(255, sign=1)

        self.head = Conv2d3x3(num_in_ch, planes)

        m_body = [ELAB(planes, r_expand, i % 2, window_sizes, n_share)
                  for i in range(num_blocks // (1 + n_share))]
        self.body = nn.Sequential(*m_body)

        self.tail = Upsampler(upscale=upscale, in_channels=planes,
                              out_channels=num_out_ch, upsample_mode=task)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        wsize = self.window_sizes[0]
        for i in range(1, len(self.window_sizes)):
            wsize = wsize * self.window_sizes[i] // math.gcd(wsize, self.window_sizes[i])
        mod_pad_h = (wsize - h % wsize) % wsize
        mod_pad_w = (wsize - w % wsize) % wsize
        x = f.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.size()
        x = self.check_image_size(x)

        # reduce the mean of pixels
        sub_x = self.sub_mean(x)

        # head
        head_x = self.head(sub_x)

        # body
        body_x = self.body(head_x)
        body_x = body_x + head_x

        # tail
        tail_x = self.tail(body_x)

        # add the mean of pixels
        add_x = self.add_mean(tail_x)

        return add_x[:, :, 0:h * self.upscale, 0:w * self.upscale]


if __name__ == '__main__':
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    # ELAN
    net = ELAN(upscale=4, planes=180, window_sizes=(4, 8, 16), num_blocks=36, n_share=0)
    print(count_parameters(net))

    # ELAN-light
    net = ELAN(upscale=4, planes=60, window_sizes=(4, 8, 16), num_blocks=24, n_share=1)
    print(count_parameters(net))

    data = torch.randn(1, 3, 120, 80)
    print(net(data).size())
