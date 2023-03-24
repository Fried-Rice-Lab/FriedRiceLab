# -------------------------------------------------------------------------------------
# Basic Modules for Image Restoration Networks
#
# Implemented/Modified by Fried Rice Lab (https://github.com/Fried-Rice-Lab)
# -------------------------------------------------------------------------------------
import math
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as f

from archs.utils._conv import Conv2d1x1, Conv2d3x3

__all__ = ['DistBackbone', 'TransformerGroup', 'UBackbone', 'Upsampler']


class DistBackbone(nn.Module):
    def __init__(self, planes: int, dist_num: int, dist_rate: float,
                 rema_layer: Union[nn.Module, any] = Conv2d3x3, rema_layer_kwargs: dict = {},
                 dist_layer: Union[nn.Module, any] = Conv2d1x1, dist_layer_kwargs: dict = {},
                 act_layer: Union[nn.Module, any] = nn.ReLU, act_layer_kwargs: dict = {}) -> None:
        super().__init__()

        dist_planes = int(planes * dist_rate)

        self.rema_layers = nn.ModuleList([
            rema_layer(planes, planes, **rema_layer_kwargs)
            for _ in range(dist_num + 1)
        ])
        self.rema_tail = rema_layer(planes, planes, **rema_layer_kwargs)  # useless ?
        # self.rema_tail = nn.Identity(planes, dist_planes, **rema_layer_kwargs)

        self.dist_layers = nn.ModuleList([
            dist_layer(planes, dist_planes, **dist_layer_kwargs)
            for _ in range(dist_num + 1)
        ])

        self.tail = Conv2d1x1(dist_planes * dist_num + planes, planes)

        self.act = act_layer(**act_layer_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        rema_outputs = list()
        for rema_layer in self.rema_layers[:-1]:
            x = self.act(x + rema_layer(x))
            rema_outputs.append(x)
        x = self.act(self.rema_tail(x))

        dist_outputs = list()
        for dist_layer, rema_output in zip(self.dist_layers, rema_outputs):
            dist_outputs.append(self.act(dist_layer(rema_output)))

        return identity + self.tail(torch.cat([*dist_outputs, x], dim=1))


class TransformerGroup(nn.Module):
    r"""

    Args:
        sa_list:
        mlp_list:
        conv_list:

    """

    def __init__(self, sa_list: list, mlp_list: list, conv_list: list = None) -> None:
        super(TransformerGroup, self).__init__()

        assert len(sa_list) == len(mlp_list)

        self.sa_list = nn.ModuleList(sa_list)
        self.mlp_list = nn.ModuleList(mlp_list)
        self.conv = nn.Sequential(*conv_list if conv_list is not None else [nn.Identity()])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            x: b c h w
        """
        for sa, mlp in zip(self.sa_list, self.mlp_list):
            x = x + sa(x)
            x = x + mlp(x)
        return self.conv(x)


class _EncoderTail(nn.Module):
    def __init__(self, planes: int) -> None:
        super(_EncoderTail, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(in_channels=planes, out_channels=2 * planes,
                                            kernel_size=(2, 2), stride=(2, 2), bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class _DecoderHead(nn.Module):
    def __init__(self, planes: int) -> None:
        super(_DecoderHead, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(in_channels=planes, out_channels=2 * planes,
                                            kernel_size=(1, 1), bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class UBackbone(nn.Module):
    def __init__(self, planes: int,
                 encoder: nn.Module, encoder_kwargs: dict, encoder_nums: list,
                 middler: nn.Module, middler_kwargs: dict, middler_num: int,
                 decoder: nn.Module, decoder_kwargs: dict, decoder_nums: list,
                 **kwargs) -> None:  # noqa
        super().__init__()

        if len(encoder_nums) == len(decoder_nums):
            self.stage_num = len(encoder_nums)
        else:
            raise NotImplementedError

        self.encoders = nn.ModuleList(
            [nn.Sequential(
                *[encoder(planes * (2 ** i), **encoder_kwargs) for _ in range(num)]
            ) for i, num in enumerate(encoder_nums)]
        )
        self.encoder_tails = nn.ModuleList(
            [_EncoderTail(planes * (2 ** i)) for i in range(self.stage_num)]
        )

        self.middler = nn.Sequential(
            *[middler(planes * (2 ** self.stage_num), **middler_kwargs) for _ in range(middler_num)]
        )

        self.decoder_heads = nn.ModuleList(
            [_DecoderHead(planes * (2 ** (i + 1))) for i in range(self.stage_num)][::-1]
        )
        self.decoders = nn.ModuleList(
            [nn.Sequential(
                *[decoder(planes * (2 ** i), **decoder_kwargs) for _ in range(num)]
            ) for i, num in enumerate(decoder_nums)][::-1]
        )

    def check_image_size(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.size()
        mod_pad_h = ((2 ** self.stage_num) - h % (2 ** self.stage_num)) % (2 ** self.stage_num)
        mod_pad_w = ((2 ** self.stage_num) - w % (2 ** self.stage_num)) % (2 ** self.stage_num)
        x = f.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        x = self.check_image_size(x)

        encoder_outputs = list()
        for encoder, encoder_tail in zip(self.encoders, self.encoder_tails):
            x = encoder(x)
            encoder_outputs.append(x)
            x = encoder_tail(x)

        x = self.middler(x)

        for decoder_head, decoder, encoder_output in zip(self.decoder_heads, self.decoders, encoder_outputs[::-1]):
            x = decoder_head(x)
            x = x + encoder_output
            x = decoder(x)

        return x[:, :, :h, :w]


class Upsampler(nn.Sequential):
    r"""Tail of the image restoration network.

    Args:
        upscale (int):
        in_channels (int):
        out_channels (int):
        upsample_mode (str):

    """

    def __init__(self, upscale: int, in_channels: int,
                 out_channels: int, upsample_mode: str = 'csr') -> None:

        layer_list = list()
        if upsample_mode == 'csr':  # classical
            if (upscale & (upscale - 1)) == 0:  # 2^n?
                for _ in range(int(math.log(upscale, 2))):
                    layer_list.append(Conv2d3x3(in_channels, 4 * in_channels))
                    layer_list.append(nn.PixelShuffle(2))
            elif upscale == 3:
                layer_list.append(Conv2d3x3(in_channels, 9 * in_channels))
                layer_list.append(nn.PixelShuffle(3))
            else:
                raise ValueError(f'Upscale {upscale} is not supported.')
            layer_list.append(Conv2d3x3(in_channels, out_channels))
        elif upsample_mode == 'lsr':  # lightweight
            layer_list.append(Conv2d3x3(in_channels, out_channels * (upscale ** 2)))
            layer_list.append(nn.PixelShuffle(upscale))
        elif upsample_mode == 'denoising' or upsample_mode == 'deblurring' or upsample_mode == 'deraining':
            layer_list.append(Conv2d3x3(in_channels, out_channels))
        else:
            raise ValueError(f'Upscale mode {upscale} is not supported.')

        super(Upsampler, self).__init__(*layer_list)


if __name__ == '__main__':
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    module = DistBackbone(planes=20, dist_num=8, dist_rate=0.123,
                          rema_layer=Conv2d1x1, rema_layer_kwargs={'bias': True},
                          dist_layer=Conv2d1x1, dist_layer_kwargs={'bias': False},
                          act_layer=nn.LeakyReLU, act_layer_kwargs={'negative_slope': 0.05, 'inplace': True})
    print(count_parameters(module))

    data = torch.randn(1, 20, 10, 10)
    print(module(data).size())


    class Conv(nn.Conv2d):
        def __init__(self, in_channels: int, kernel_size: tuple, padding: tuple = (0, 0),
                     dilation: tuple = (1, 1), groups: int = 1, bias: bool = True,
                     **kwargs) -> None:
            super(Conv, self).__init__(in_channels=in_channels, out_channels=in_channels,
                                       kernel_size=kernel_size, stride=(1, 1), padding=padding,
                                       dilation=dilation, groups=groups, bias=bias, **kwargs)


    u = UBackbone(planes=4,
                  encoder=Conv, encoder_kwargs={'kernel_size': 1, 'padding': 0}, encoder_nums=[2, 2, 2],
                  middler=Conv, middler_kwargs={'kernel_size': 3, 'padding': 1}, middler_num=2,
                  decoder=Conv, decoder_kwargs={'kernel_size': 5, 'padding': 2}, decoder_nums=[2, 2, 2])
    print(count_parameters(u), u)

    data = torch.randn(1, 4, 49, 49)
    print(u(data).size())
