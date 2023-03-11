# ---------------------------------------------------------------------------
# Blueprint Separable Residual Network for Efficient Image Super-Resolution
# Official GitHub: https://github.com/xiaom233/BSRN
#
# Modified by Yulong Liu (yl.liu88@outlook.com)
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as f
from basicsr.utils.registry import ARCH_REGISTRY

from archs.utils import Conv2d1x1, Conv2d3x3, CCA, Upsampler, DWConv2d


class DWConv2d33(DWConv2d):
    r"""

    Args:
        stride(tuple). Default: 1

    """

    def __init__(self, in_channels: int, out_channels: int, stride: tuple = 1,
                 dilation: tuple = (1, 1), groups: int = None, bias: bool = True,
                 **kwargs) -> None:
        super(DWConv2d33, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=(3, 3), stride=stride, padding=(1, 1),
                                         dilation=dilation, groups=groups, bias=bias, **kwargs)


class BlueprintSeparableConv(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple = (3, 3),
                 stride: tuple = 1, padding: tuple = 1, dilation: tuple = (1, 1), bias: bool = True,
                 mid_channels: int = None, **kwargs) -> None:
        super(BlueprintSeparableConv, self).__init__()

        # pointwise
        if mid_channels is not None:  # BSConvS
            self.pw = nn.Sequential(Conv2d1x1(in_channels, mid_channels, bias=False),
                                    Conv2d1x1(mid_channels, out_channels, bias=False))

        else:  # BSConvU
            self.pw = Conv2d1x1(in_channels, out_channels, bias=False)

        # depthwise
        self.dw = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                                  stride=stride, padding=padding, dilation=dilation, groups=out_channels,
                                  bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dw(self.pw(x))


class BSConv2d33(BlueprintSeparableConv):
    pass


class ESA(nn.Module):
    r"""Enhanced Spatial Attention.

    Args:
        in_channels:
        planes:
        num_conv: Number of conv layers in the conv group

    """

    def __init__(self, in_channels, planes: int = None, num_conv: int = 3, conv_layer=Conv2d3x3,
                 **kwargs) -> None:
        super(ESA, self).__init__()

        planes = planes or in_channels // 4
        self.head_conv = Conv2d1x1(in_channels, planes)

        self.stride_conv = conv_layer(planes, planes, stride=(2, 2), **kwargs)
        conv_group = list()
        for i in range(num_conv):
            if i != 0:
                conv_group.append(nn.ReLU(inplace=True))
            conv_group.append(conv_layer(planes, planes, **kwargs))
        self.group_conv = nn.Sequential(*conv_group)
        self.useless_conv = Conv2d1x1(planes, planes)  # maybe nn.Identity()?

        self.tail_conv = Conv2d1x1(planes, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv-1
        head_output = self.head_conv(x)

        # Stride Conv
        stride_output = self.stride_conv(head_output)
        # Pooling
        pool_output = f.max_pool2d(stride_output, kernel_size=7, stride=3)
        # Conv Group
        group_output = self.group_conv(pool_output)
        # Upsampling
        upsample_output = f.interpolate(group_output, (x.size(2), x.size(3)),
                                        mode='bilinear', align_corners=False)

        # Conv-1
        tail_output = self.tail_conv(upsample_output + self.useless_conv(head_output))
        # Sigmoid
        sig_output = torch.sigmoid(tail_output)

        return x * sig_output


class ESDB(nn.Module):
    r"""Efficient Separable Distillation Block
    """

    def __init__(self, planes: int, distillation_rate: float = 0.5, conv_layer=Conv2d3x3,
                 **kwargs) -> None:
        super(ESDB, self).__init__()

        distilled_channels = int(planes * distillation_rate)

        self.c1_d = Conv2d1x1(planes, distilled_channels)
        self.c1_r = conv_layer(planes, planes, **kwargs)

        self.c2_d = Conv2d1x1(planes, distilled_channels)
        self.c2_r = conv_layer(planes, planes, **kwargs)

        self.c3_d = Conv2d1x1(planes, distilled_channels)
        self.c3_r = conv_layer(planes, planes, **kwargs)

        self.c4_r = conv_layer(planes, distilled_channels, **kwargs)

        self.c5 = Conv2d1x1(distilled_channels * 4, planes)

        self.cca = CCA(planes)
        self.esa = ESA(planes, conv_layer=conv_layer, **kwargs)

        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d_c1 = self.act(self.c1_d(x))
        r_c1 = self.c1_r(x)
        r_c1 = self.act(x + r_c1)

        d_c2 = self.act(self.c2_d(r_c1))
        r_c2 = self.c2_r(r_c1)
        r_c2 = self.act(r_c1 + r_c2)

        d_c3 = self.act(self.c3_d(r_c2))
        r_c3 = self.c3_r(r_c2)
        r_c3 = self.act(r_c2 + r_c3)

        r_c4 = self.c4_r(r_c3)
        r_c4 = self.act(r_c4)

        out = torch.cat([d_c1, d_c2, d_c3, r_c4], dim=1)
        out = self.c5(out)

        out_fused = self.cca(self.esa(out))

        return out_fused + x


@ARCH_REGISTRY.register()
class BSRN(nn.Module):
    r"""Blueprint Separable Residual Network.
    """

    def __init__(self, upscale: int, num_in_ch: int, num_out_ch: int, task: str,
                 planes: int, num_modules: int, num_times: int,
                 conv_type: str, mid_channels: int = None) -> None:
        super(BSRN, self).__init__()

        kwargs = dict()
        if conv_type == 'bsconv_u':
            conv_layer = BlueprintSeparableConv
        elif conv_type == 'bsconv_s':
            kwargs = {'mid_channel': mid_channels}
            conv_layer = BlueprintSeparableConv
        elif conv_type == 'dwconv':
            conv_layer = DWConv2d33
        elif conv_type == 'conv':
            conv_layer = Conv2d3x3
        else:
            raise NotImplementedError

        self.num_times = num_times

        self.head = conv_layer(num_in_ch * num_times, planes, **kwargs)

        self.body = nn.ModuleList([ESDB(planes, conv_layer=conv_layer, **kwargs)
                                   for _ in range(num_modules)])
        self.body_tail = nn.Sequential(Conv2d1x1(planes * num_modules, planes),
                                       nn.GELU(),
                                       conv_layer(planes, planes, **kwargs))

        self.tail = Upsampler(upscale=upscale, in_channels=planes,
                              out_channels=num_out_ch, upsample_mode=task)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # head
        x = torch.cat([x] * self.num_times, dim=1)
        head_x = self.head(x)

        # body
        body_x = head_x
        output_list = list()
        for module in self.body:
            body_x = module(body_x)
            output_list.append(body_x)
        body_x = self.body_tail(torch.cat(output_list, dim=1))
        body_x = body_x + head_x

        # tail
        tail_x = self.tail(body_x)
        return tail_x


if __name__ == '__main__':
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    net = BSRN(upscale=4, planes=64, num_modules=8, num_times=4, conv_type='bsconv_s')
    print(count_parameters(net))

    data = torch.randn(1, 3, 64, 64)
    print(net(data).size())
