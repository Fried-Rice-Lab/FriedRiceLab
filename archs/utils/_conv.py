# -------------------------------------------------------------------------------------
# Basic Modules for Image Restoration Networks
#
# Implemented/Modified by Fried Rice Lab (https://github.com/Fried-Rice-Lab)
# -------------------------------------------------------------------------------------
import torch
from torch import nn as nn
from torch.nn import functional as f

from ._linear import GroupLinear

__all__ = ['Conv2d1x1', 'Conv2d3x3', 'ContextGatedConv2d',
           'ShiftConv2d1x1', 'MeanShift', 'AffineConv2d1x1',
           'DepthwiseSeparableConv2d']


class Conv2d1x1(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, stride: tuple = (1, 1),
                 dilation: tuple = (1, 1), groups: int = 1, bias: bool = True,
                 **kwargs) -> None:
        super(Conv2d1x1, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=(1, 1), stride=stride, padding=(0, 0),
                                        dilation=dilation, groups=groups, bias=bias, **kwargs)


class Conv2d3x3(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, stride: tuple = (1, 1),
                 dilation: tuple = (1, 1), groups: int = 1, bias: bool = True,
                 **kwargs) -> None:
        super(Conv2d3x3, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=(3, 3), stride=stride, padding=(1, 1),
                                        dilation=dilation, groups=groups, bias=bias, **kwargs)


class ContextGatedConv2d(nn.Conv2d):
    r"""Context-Gated Convolution.

    This version of Context-Gated Convolution supports any input size. Modified
    from https://github.com/XudongLinthu/context-gated-convolution.

    Args:
        in_channels (int):
        out_channels (int):
        kernel_size (tuple):
        stride (tuple):
        padding (tuple):
        dilation (tuple):
        linear_groups (int):
        codec_bias (bool):
        use_bn (bool):
        act_layer (nn.Module):

    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple,
                 padding: tuple, dilation: tuple = (1, 1), linear_groups: int = 12,
                 codec_bias: bool = False, use_bn: bool = True, act_layer: nn.Module = nn.ReLU,
                 **kwargs) -> None:
        super(ContextGatedConv2d, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                                 dilation=dilation, groups=1, bias=False, **kwargs)  # TODO add bias

        self.latent_size = self.kernel_size[0] * self.kernel_size[1] // 2 + 1  # e
        self.act = act_layer(inplace=True)
        self._act = act_layer(inplace=False)  # TODO: fix it

        """Context Encoding"""
        self.avg_pool = nn.AdaptiveAvgPool2d(self.kernel_size)
        self.global_info_encoder = nn.Linear(self.kernel_size[0] * self.kernel_size[1], self.latent_size,
                                             bias=codec_bias)

        """Sub-kernel Generation for wc"""
        self.in_channels_context_bn = nn.BatchNorm1d(in_channels) if use_bn else nn.Identity()
        self.in_channels_context_decoder = nn.Linear(self.latent_size, self.kernel_size[0] * self.kernel_size[1],
                                                     bias=codec_bias)

        """Sub-kernel Generation for wo"""
        self.pre_out_channels_context_bn = nn.BatchNorm1d(in_channels) if use_bn else nn.Identity()
        if in_channels % linear_groups == 0:
            self.out_channels_context_encoder = GroupLinear(in_channels, out_channels,
                                                            groups=linear_groups, bias=codec_bias)
        else:
            self.out_channels_context_encoder = nn.Linear(in_channels, out_channels,
                                                          bias=codec_bias)
        self.beh_out_channels_context_bn = nn.BatchNorm1d(out_channels) if use_bn else nn.Identity()
        self.out_channels_context_decoder = nn.Linear(self.latent_size, self.kernel_size[0] * self.kernel_size[1],
                                                      bias=codec_bias)

        """Gate Generation"""
        self.sig = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        h_ = int(((h + 2 * self.padding[0] - self.dilation[0] *
                   (self.kernel_size[0] - 1) - 1) / self.stride[0]) + 1)
        w_ = int(((w + 2 * self.padding[1] - self.dilation[1] *
                   (self.kernel_size[0] - 1) - 1) / self.stride[1]) + 1)
        weight = self.weight

        """Context Encoding"""
        global_info = self.avg_pool(x).view(b, c, -1)  # b c k*k
        global_info = self.global_info_encoder(global_info)  # b c e

        """Sub-kernel Generation for wc"""
        in_channels_context = self.act(self.in_channels_context_bn(global_info))  # b c e
        wc = self.in_channels_context_decoder(in_channels_context)  # b c k*k
        wc = wc.view(b, 1, c, self.kernel_size[0], self.kernel_size[1])  # b 1 c k k

        """Sub-kernel Generation for wo"""
        out_channels_context = self._act(self.pre_out_channels_context_bn(global_info)). \
            permute(0, 2, 1)  # b c e -> b e c
        out_channels_context = self.out_channels_context_encoder(out_channels_context). \
            permute(0, 2, 1)  # b e c -> b e o -> b o e
        out_channels_context = self.act(self.beh_out_channels_context_bn(out_channels_context))  # b o e
        wo = self.out_channels_context_decoder(out_channels_context)  # b o k*k
        wo = wo.view(b, self.out_channels, 1, self.kernel_size[0], self.kernel_size[1])  # b o 1 k k

        """Gate Generation"""
        wg = self.sig(wc + wo)  # b o c k k
        wg = wg * weight.unsqueeze(0)  # b o c k k
        wg = wg.view(b, self.out_channels, -1)  # b o c*k*k

        """Conv"""
        x_patch = f.unfold(x, self.kernel_size, self.dilation, self.padding, self.stride)  # b i*k*k h_*w_
        output = torch.matmul(wg, x_patch)  # b o c*k*k @ b c*k*k h_*w_ -> b o h_*w_
        output = output.view(b, self.out_channels, h_, w_)  # b o h_ w_
        return output


class ShiftConv2d1x1(nn.Conv2d):
    r"""

    Args:
        in_channels (int):
        out_channels (int):
        stride (tuple):
        dilation (tuple):
        bias (bool):
        shift_mode (str):
        val (float):

    """

    def __init__(self, in_channels: int, out_channels: int, stride: tuple = (1, 1),
                 dilation: tuple = (1, 1), bias: bool = True, shift_mode: str = '+', val: float = 1.,
                 **kwargs) -> None:
        super(ShiftConv2d1x1, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                             kernel_size=(1, 1), stride=stride, padding=(0, 0),
                                             dilation=dilation, groups=1, bias=bias, **kwargs)

        assert in_channels % 5 == 0, f'{in_channels} % 5 != 0.'

        channel_per_group = in_channels // 5
        self.mask = nn.Parameter(torch.zeros((in_channels, 1, 3, 3)), requires_grad=False)
        if shift_mode == '+':
            self.mask[0 * channel_per_group:1 * channel_per_group, 0, 1, 2] = val
            self.mask[1 * channel_per_group:2 * channel_per_group, 0, 1, 0] = val
            self.mask[2 * channel_per_group:3 * channel_per_group, 0, 2, 1] = val
            self.mask[3 * channel_per_group:4 * channel_per_group, 0, 0, 1] = val
            self.mask[4 * channel_per_group:, 0, 1, 1] = val
        elif shift_mode == 'x':
            self.mask[0 * channel_per_group:1 * channel_per_group, 0, 0, 0] = val
            self.mask[1 * channel_per_group:2 * channel_per_group, 0, 0, 2] = val
            self.mask[2 * channel_per_group:3 * channel_per_group, 0, 2, 0] = val
            self.mask[3 * channel_per_group:4 * channel_per_group, 0, 2, 2] = val
            self.mask[4 * channel_per_group:, 0, 1, 1] = val
        else:
            raise NotImplementedError(f'Unknown shift mode for ShiftConv2d1x1: {shift_mode}.')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = f.conv2d(input=x, weight=self.mask, bias=None,
                     stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=self.in_channels)
        x = f.conv2d(input=x, weight=self.weight, bias=self.bias,
                     stride=(1, 1), padding=(0, 0), dilation=(1, 1))
        return x


class MeanShift(nn.Conv2d):
    r"""

    Args:
        rgb_range (int):
        sign (int):
        data_type (str):

    """

    def __init__(self, rgb_range: int, sign: int = -1, data_type: str = 'DIV2K') -> None:
        super(MeanShift, self).__init__(3, 3, kernel_size=(1, 1))

        rgb_std = (1.0, 1.0, 1.0)
        if data_type == 'DIV2K':
            # RGB mean for DIV2K 1-800
            rgb_mean = (0.4488, 0.4371, 0.4040)
        elif data_type == 'DF2K':
            # RGB mean for DF2K 1-3450
            rgb_mean = (0.4690, 0.4490, 0.4036)
        else:
            raise NotImplementedError(f'Unknown data type for MeanShift: {data_type}.')

        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class AffineConv2d1x1(nn.Conv2d):
    r"""

    Args:
        in_channels (int):
        val (float):

    """

    def __init__(self, in_channels: int, val: float = 0.1) -> None:
        super().__init__(in_channels, in_channels, kernel_size=(1, 1))

        self.weight = nn.Parameter(torch.ones(1, in_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, in_channels, 1, 1))

        nn.init.constant_(self.weight, val)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight + self.bias


class DepthwiseSeparableConv2d(nn.Module):
    r"""

    Args:
        in_channels (int):
        out_channels (int):
        kernel_size (tuple):
        stride (tuple):
        padding (tuple):
        dilation (tuple):
        groups (int):
        bias (bool):

    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple,
                 padding: tuple, dilation: tuple = (1, 1), groups: int = None, bias: bool = True,
                 **kwargs) -> None:  # noqa
        super(DepthwiseSeparableConv2d, self).__init__()

        groups = groups or in_channels

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.pointwise = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                   kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
                                   dilation=(1, 1), groups=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.conv(x))


if __name__ == '__main__':
    a = torch.arange(1, 21).reshape(2, 10, 1, 1).float()
    a = f.pad(a, (1, 1, 1, 1))
