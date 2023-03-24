# -------------------------------------------------------------------------------------
# Basic Modules for Image Restoration Networks
#
# Implemented/Modified by Fried Rice Lab (https://github.com/Fried-Rice-Lab)
# -------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as f
from einops import rearrange

__all__ = ['ChannelMixer', 'PixelMixer', 'Scale']


class Scale(nn.Module):
    r"""Learnable scale.

    Args:
        shape (tuple):
        init_value (float): the value to fill the scale with

    """

    def __init__(self, shape: tuple, init_value: float = 1e-5) -> None:
        super(Scale, self).__init__()

        self.scale = nn.Parameter(torch.zeros(shape))
        nn.init.constant_(self.scale, init_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class PixelMixer(nn.Module):
    r"""Pixel Mixer.

    Args:
        planes (int):
        mix_margin (int):

    """

    def __init__(self, planes: int, mix_margin: int = 1) -> None:
        super(PixelMixer, self).__init__()

        assert planes % 5 == 0

        self.planes = planes
        self.mix_margin = mix_margin

        self.mask = torch.zeros((self.planes, 1, mix_margin * 2 + 1, mix_margin * 2 + 1),
                                requires_grad=False)
        self.mask[3::5, 0, 0, mix_margin] = 1.
        self.mask[2::5, 0, -1, mix_margin] = 1.
        self.mask[1::5, 0, mix_margin, 0] = 1.
        self.mask[0::5, 0, mix_margin, -1] = 1.
        self.mask[4::5, 0, mix_margin, mix_margin] = 1.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.mix_margin
        return f.conv2d(input=f.pad(x, pad=(m, m, m, m), mode='circular'),
                        weight=self.mask, bias=None, stride=(1, 1), padding=(0, 0),
                        dilation=(1, 1), groups=self.planes)


class ChannelMixer(nn.Module):
    r"""Channel Mixer.

    Args:
        num_slices (int):

    """

    def __init__(self, num_slices: int = 2) -> None:
        super(ChannelMixer, self).__init__()

        self.num_slices = num_slices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.num_slices

        x = rearrange(x, 'b (s c) h w -> b s c h w', s=s)
        x = x.permute(0, 2, 1, 3, 4)
        x = rearrange(x, 'b c s h w -> b (c s) h w')

        return x


if __name__ == '__main__':
    pm = PixelMixer(10)
    data = torch.randn(2, 10, 10, 10)
    print(pm(data).size())
