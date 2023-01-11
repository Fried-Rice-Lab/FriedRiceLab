# -------------------------------------------------------
# Efficient Image Super-Resolution Using Pixel Attention
# Official GitHub: https://github.com/zhaohengyuan1/PAN
# -------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as f
from basicsr.utils.registry import ARCH_REGISTRY

from archs.utils import Conv2d1x1, Conv2d3x3, PixelAttention


class SCPA(nn.Module):
    def __init__(self, planes):
        super(SCPA, self).__init__()
        group_width = planes // 2
        self.conv11_up = Conv2d1x1(planes, group_width, bias=False)
        self.conv11_down = Conv2d1x1(planes, group_width, bias=False)
        self.conv33 = Conv2d3x3(group_width, group_width, bias=False)
        self.PAConv = PAConv(group_width)
        self.conv11_tail = Conv2d1x1(planes, planes, bias=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        # the upper branch
        up_head = self.conv11_up(x)
        up_head = self.lrelu(up_head)
        down_head = self.conv11_down(x)
        down_head = self.lrelu(down_head)
        up_body = self.PAConv(up_head)
        up_body = self.lrelu(up_body)
        # the lower branch
        down_body = self.conv33(down_head)
        down_body = self.lrelu(down_body)
        tail = torch.cat([up_body, down_body], dim=1)
        tail = self.conv11_tail(tail)
        out = x + tail
        return out


class PAConv(nn.Module):
    def __init__(self, planes):
        super(PAConv, self).__init__()
        self.conv11_up = Conv2d1x1(planes, planes)
        self.sig = nn.Sigmoid()
        self.conv33_down1 = Conv2d3x3(planes, planes, bias=False)
        self.conv33_down2 = Conv2d3x3(planes, planes, bias=False)

    def forward(self, x):
        up = self.conv11_up(x)
        up = self.sig(up)
        down = self.conv33_down1(x)
        tail = torch.mul(up, down)
        out = self.conv33_down2(tail)
        return out


class PanUpsamper(nn.Module):
    def __init__(self, planes: int, tail_planes: int, out_channels: int, scale: int) -> None:
        super(PanUpsamper, self).__init__()
        self.scale = scale
        self.conv33_first = Conv2d3x3(planes, tail_planes)
        self.pa_first = PixelAttention(tail_planes)
        self.conv33_second = Conv2d3x3(tail_planes, tail_planes)
        if self.scale == 4:
            self.conv33_third = Conv2d3x3(tail_planes, tail_planes)
            self.pa_second = PixelAttention(tail_planes)
            self.conv33_fourth = Conv2d3x3(tail_planes, tail_planes)
        self.conv33_fifth = Conv2d3x3(tail_planes, out_channels)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if self.scale == 2 or self.scale == 3:
            out = self.conv33_first(f.interpolate(x, scale_factor=self.scale, mode='nearest'))
            out = self.lrelu(self.pa_first(out))
            out = self.lrelu(self.conv33_second(out))
        elif self.scale == 4:
            out = self.conv33_first(f.interpolate(x, scale_factor=2, mode='nearest'))
            out = self.lrelu(self.pa_first(out))
            out = self.lrelu(self.conv33_second(out))
            out = self.conv33_third(f.interpolate(out, scale_factor=2, mode='nearest'))
            out = self.lrelu(self.pa_second(out))
            out = self.lrelu(self.conv33_fourth(out))
        return self.conv33_fifth(out)


@ARCH_REGISTRY.register()
class PAN(nn.Module):
    r"""Efficient Image Super-Resolution Using Pixel Attention.
    """

    def __init__(self, upscale: int, num_in_ch: int, num_out_ch: int, task: str,
                 planes: int, n_blocks: int, tail_planes: int):
        super(PAN, self).__init__()

        self.scale = upscale

        self.conv33_first = Conv2d3x3(num_in_ch, planes)

        self.body = nn.Sequential(*[SCPA(planes=planes) for _ in range(n_blocks)])

        self.conv33_middle = Conv2d3x3(planes, planes)
        self.upsamper = PanUpsamper(planes, tail_planes, num_out_ch, upscale)

    def forward(self, x):
        # head
        head = self.conv33_first(x)
        # body
        body = self.conv33_middle(self.body(head))
        body = head + body
        # tail
        tail = self.upsamper(body)
        input_lr = f.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        tail = tail + input_lr
        return tail


if __name__ == '__main__':
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # net = PAN(in_channels=3, out_channels=3, planes=40, n_blocks=16, tail_planes=24, scale=2)
    # print(count_parameters(net))

    # data = torch.randn(1, 3, 120, 80)
    # print(net(data).size())
