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

__all__ = ['SABase4D']


class SABase4D(nn.Module):
    r"""Self attention for 4D input.

    Args:
        dim (int): Number of input channels
        num_heads (int): Number of attention heads
        attn_layer (list): layers used to calculate attn
        proj_layer (list): layers used to proj output
        window_list (tuple): list of window sizes. Input will be equally divided
            by channel to use different windows sizes
        shift_list (tuple): list of shift sizes

    Returns:
        b c h w -> b c h w
    """

    def __init__(self, dim: int,
                 num_heads: int,
                 attn_layer: list = None,
                 proj_layer: list = None,
                 window_list: tuple = ((8, 8),),
                 shift_list: tuple = None,
                 ) -> None:
        super(SABase4D, self).__init__()

        self.dim = dim
        self.num_heads = num_heads

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        return output


if __name__ == '__main__':
    dim = 11
    data = torch.randn(1, dim, 16, 16)
    layer = SABase4D(dim=dim,
                     num_heads=1,
                     attn_layer=[nn.Conv2d(dim, dim * 2, 1), nn.BatchNorm2d(dim * 2)],
                     proj_layer=[nn.Conv2d(dim, dim, 1)],
                     window_list=((8, 8),))
    print(layer(data) == data)
