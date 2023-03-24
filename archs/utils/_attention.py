# -------------------------------------------------------------------------------------
# Basic Modules for Image Restoration Networks
#
# Implemented/Modified by Fried Rice Lab (https://github.com/Fried-Rice-Lab)
# -------------------------------------------------------------------------------------
import math
import torch
import torch.nn as nn
import torch.nn.functional as f
from einops import rearrange

from ._conv import Conv2d1x1, Conv2d3x3

__all__ = ['ChannelAttention', 'SpatialAttention', 'PixelAttention',
           'SABase4D', 'CrissCrossAttention',
           'CBAM', 'CCA', 'ESA']


class ChannelAttention(nn.Module):
    r"""

    Args:
        planes (int):
        reduction (int):
        act_layer (nn.Module):

    """

    def __init__(self, planes: int, reduction: int = 8, act_layer: nn.Module = nn.ReLU) -> None:
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP = nn.Sequential(
            Conv2d3x3(planes, planes // reduction, bias=False),
            act_layer(inplace=True),
            Conv2d3x3(planes // reduction, planes, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_MLP(self.avg_pool(x))
        max_out = self.shared_MLP(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class SpatialAttention(nn.Module):
    r"""
    """

    def __init__(self) -> None:
        super(SpatialAttention, self).__init__()

        self.conv = Conv2d3x3(2, 1, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = self.conv(torch.cat([avg_out, max_out], dim=1))
        return x * self.sig(x)


class PixelAttention(nn.Module):
    r"""Pixel Attention from "Efficient Image Super-Resolution Using Pixel Attention".

    Args:
        planes (int):
        bias (bool):

    """

    def __init__(self, planes: int, bias: bool = False) -> None:
        super(PixelAttention, self).__init__()

        self.conv = Conv2d1x1(planes, planes, bias=bias)  # 同时建立C * H * W的注意力

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.conv(x))


class SABase4D(nn.Module):
    r"""Self attention for 4D input.

    Args:
        dim (int): Number of input channels
        num_heads (int): Number of attention heads
        attn_layer (list): Layers used to calculate attn
        proj_layer (list): Layers used to proj output
        window_list (tuple): List of window sizes. Input will be equally divided
            by channel to use different windows sizes
        shift_list (tuple): list of shift sizes
        return_attns (bool): Returns attns or not

    Returns:
        b c h w -> b c h w
    """

    def __init__(self, dim: int,
                 num_heads: int,
                 attn_layer: list = None,
                 proj_layer: list = None,
                 window_list: tuple = ((8, 8),),
                 shift_list: tuple = None,
                 return_attns: bool = False,
                 ) -> None:
        super(SABase4D, self).__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.return_attns = return_attns

        self.window_list = window_list
        if shift_list is not None:
            assert len(shift_list) == len(window_list)
            self.shift_list = shift_list
        else:
            self.shift_list = ((0, 0),) * len(window_list)

        self.attn = nn.Sequential(*attn_layer if attn_layer is not None else [nn.Identity()])
        self.proj = nn.Sequential(*proj_layer if proj_layer is not None else [nn.Identity()])

    @staticmethod
    def check_image_size(x: torch.Tensor, window_size: tuple) -> torch.Tensor:
        _, _, h, w = x.size()
        windows_num_h = math.ceil(h / window_size[0])
        windows_num_w = math.ceil(w / window_size[1])
        mod_pad_h = windows_num_h * window_size[0] - h
        mod_pad_w = windows_num_w * window_size[1] - w
        return f.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

    def forward(self, x: torch.Tensor) -> torch.Tensor or tuple:
        r"""
        Args:
            x: b c h w

        Returns:
            b c h w -> b c h w
        """
        # calculate qkv
        qkv = self.attn(x)
        _, C, _, _ = qkv.size()

        # split channels
        qkv_list = torch.split(qkv, [C // len(self.window_list)] * len(self.window_list), dim=1)

        output_list = list()
        if self.return_attns:
            attn_list = list()

        for attn_slice, window_size, shift_size in zip(qkv_list, self.window_list, self.shift_list):
            _, _, h, w = attn_slice.size()
            attn_slice = self.check_image_size(attn_slice, window_size)

            # roooll!
            if shift_size != (0, 0):
                attn_slice = torch.roll(attn_slice, shifts=shift_size, dims=(2, 3))

            # cal attn
            _, _, H, W = attn_slice.size()
            q, v = rearrange(attn_slice, 'b (qv head c) (nh ws1) (nw ws2) -> qv (b head nh nw) (ws1 ws2) c',
                             qv=2, head=self.num_heads,
                             ws1=window_size[0], ws2=window_size[1])
            attn = (q @ q.transpose(-2, -1))
            attn = f.softmax(attn, dim=-1)
            if self.return_attns:
                attn_list.append(attn.reshape(self.num_heads, -1,
                                              window_size[0] * window_size[1],
                                              window_size[0] * window_size[1]))  # noqa
            output = rearrange(attn @ v, '(b head nh nw) (ws1 ws2) c -> b (head c) (nh ws1) (nw ws2)',
                               head=self.num_heads,
                               nh=H // window_size[0], nw=W // window_size[1],
                               ws1=window_size[0], ws2=window_size[1])

            # roooll back!
            if shift_size != (0, 0):
                output = torch.roll(output, shifts=(-shift_size[0], -shift_size[1]), dims=(2, 3))

            output_list.append(output[:, :, :h, :w])

        # proj output
        output = self.proj(torch.cat(output_list, dim=1))

        if self.return_attns:
            return output, attn_list
        else:
            return output


class CrissCrossAttention(nn.Module):
    r"""Criss-Cross Attention from "CCNet: Criss-Cross Attention for Semantic Segmentation".

    Args:
        planes (int):
        reduction (int):

    """

    def __init__(self, planes: int, reduction: int = 8) -> None:
        super(CrissCrossAttention, self).__init__()

        self.q = Conv2d1x1(in_channels=planes, out_channels=planes // reduction)
        self.k = Conv2d1x1(in_channels=planes, out_channels=planes // reduction)
        self.v = Conv2d1x1(in_channels=planes, out_channels=planes)

        self.gamma = nn.Parameter(torch.zeros(1))

    @staticmethod
    def inf(b: int, h: int, w: int, device) -> torch.Tensor:
        return -torch.diag(torch.tensor(float("inf")).to(device).repeat(h), 0).unsqueeze(0).repeat(b * w, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.size()

        proj_query = self.q(x)  # b c h w -> b c/8 h w
        proj_query_h = proj_query.permute(0, 3, 1, 2). \
            contiguous().view(b * w, -1, h).permute(0, 2, 1)  # b c/8 h w -> b w c/8 h -> b*w c/8 h -> b*w h c/8
        proj_query_w = proj_query.permute(0, 2, 1, 3). \
            contiguous().view(b * h, -1, w).permute(0, 2, 1)  # b c/8 h w -> b h c/8 w -> b*h c/8 w -> b*h w c/8

        proj_key = self.k(x)  # b c h w -> b c/8 h w
        proj_key_h = proj_key.permute(0, 3, 1, 2). \
            contiguous().view(b * w, -1, h)  # b c/8 h w -> b w c/8 h -> b*w c/8 h
        proj_key_w = proj_key.permute(0, 2, 1, 3). \
            contiguous().view(b * h, -1, w)  # b c/8 h w -> b h c/8 w -> b*h c/8 w

        proj_value = self.v(x)  # b c h w -> b c h w
        proj_value_h = proj_value.permute(0, 3, 1, 2). \
            contiguous().view(b * w, -1, h)  # b c h w -> b w c h -> b*w c h
        proj_value_w = proj_value.permute(0, 2, 1, 3). \
            contiguous().view(b * h, -1, w)  # b c h w -> b h c w -> b*h c w

        energy_h = torch.bmm(proj_query_h, proj_key_h)  # b*w h c/8 @ b*w c/8 h -> b*w h h
        energy_h = energy_h + self.inf(b, h, w, x.device)  # b*w h h + b*w h h -> b*w h h
        energy_h = energy_h.view(b, w, h, h).permute(0, 2, 1, 3)  # b*w h h -> b w h h -> b h w h

        energy_w = torch.bmm(proj_query_w, proj_key_w).view(b, h, w, w)  # b*h w c/8 @ b*h c/8 w -> b*h w w -> b h w w

        concate = f.softmax(torch.cat([energy_h, energy_w], 3), dim=3)  # b h w h + b h w w -> b h w (h + w)

        att_h = concate[:, :, :, 0:h].permute(0, 2, 1, 3).contiguous().view(b * w, h, h)  # b*w h h
        att_w = concate[:, :, :, h:h + w].contiguous().view(b * h, w, w)  # b*h w w

        out_h = torch.bmm(proj_value_h, att_h.permute(0, 2, 1)).view(b, w, -1, h).permute(0, 2, 3, 1)
        out_w = torch.bmm(proj_value_w, att_w.permute(0, 2, 1)).view(b, h, -1, w).permute(0, 2, 1, 3)

        return self.gamma * (out_h + out_w) + x


class CBAM(nn.Module):
    r"""CBAM from "CBAM: Convolutional Block Attention Module".

    Args:
        planes (int):
        reduction (int):
        act_layer (nn.Module):

    """

    def __init__(self, planes: int, reduction: int = 8, act_layer: nn.Module = nn.ReLU) -> None:
        super(CBAM, self).__init__()

        self.ca = ChannelAttention(planes, reduction, act_layer)
        self.sa = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sa(self.ca(x))


class CCA(nn.Module):
    r"""Contrast-aware Channel Attention.
    """

    def __init__(self, planes: int, reduction: int = 16) -> None:
        super(CCA, self).__init__()

        self.conv = nn.Sequential(Conv2d1x1(planes, planes // reduction),
                                  nn.ReLU(inplace=True),
                                  Conv2d1x1(planes // reduction, planes))
        self.sig = nn.Sigmoid()

    @staticmethod
    def contrast(x: torch.Tensor) -> torch.Tensor:
        # cal stdv
        mean = x.sum(3, keepdim=True).sum(2, keepdim=True) / (x.size(2) * x.size(3))
        var = (x - mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (x.size(2) * x.size(3))
        stdv = var.pow(0.5)
        # cal pool
        pool = f.adaptive_avg_pool2d(x, 1)
        return pool + stdv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.sig(self.conv(self.contrast(x)))


class ESA(nn.Module):
    r"""Enhanced Spatial Attention.

    Args:
        in_channels:
        planes:
        num_conv: Number of conv layers in the conv group

    """

    def __init__(self, in_channels: int, planes: int = 16, num_conv: int = 1):
        super(ESA, self).__init__()

        self.head_conv = Conv2d1x1(in_channels, planes)

        self.stride_conv = nn.Conv2d(planes, planes, kernel_size=3, stride=2)
        conv_group = list()
        for i in range(num_conv):
            if i != 0:
                conv_group.append(nn.ReLU(inplace=True))
            conv_group.append(Conv2d3x3(planes, planes))
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


if __name__ == '__main__':
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    net = CCA(32)

    data = torch.randn((1, 32, 64, 64))
    print(net(data).size())
