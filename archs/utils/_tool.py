# -------------------------------------------------------------------------------------
# Basic Modules for Image Restoration Networks
#
# Implemented/Modified by Fried Rice Lab (https://github.com/Fried-Rice-Lab)
# -------------------------------------------------------------------------------------
import torch
import torch.nn as nn
from einops import rearrange

__all__ = ['Scale', 'ChannelMixer']


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
    from torch.nn import functional as f

    a = torch.arange(0, 10).reshape(1, 10, 1, 1)
    a = f.pad(a, (1, 1, 1, 1))
    print(a)

    l = ChannelMixer()
    print(l(a))
