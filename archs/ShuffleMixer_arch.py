# ----------------------------------------------------------------------
# ShuffleMixer: An Efficient ConvNet for Image Super-Resolution
# Official GitHub: https://github.com/sunny2109/ShuffleMixer
#
# Modified by Tianle Liu (tianle.l@outlook.com)
# ----------------------------------------------------------------------
import numbers

import torch
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY
from einops import rearrange
from torch import nn

from archs.utils import Conv2d1x1, Conv2d3x3


class PointMlp(nn.Module):
    def __init__(self, dim, mlp_ratio=4) -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)

        self.fc = nn.Sequential(
            Conv2d1x1(dim, hidden_dim),
            nn.SiLU(inplace=True),
            Conv2d1x1(hidden_dim, dim),
        )

    def forward(self, x) -> torch.Tensor:
        return self.fc(x)


class SplitPointMlp(nn.Module):
    def __init__(self, dim, mlp_ratio=2) -> None:
        super().__init__()
        hidden_dim = int(dim // mlp_ratio * 2)

        self.fc = nn.Sequential(
            Conv2d1x1(dim // 2, hidden_dim),
            nn.SiLU(inplace=True),
            Conv2d1x1(hidden_dim, dim // 2),
        )

    def forward(self, x) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.fc(x1)
        x = torch.cat([x1, x2], dim=1)
        return rearrange(x, 'b (g d) h w -> b (d g) h w', g=8)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape) -> None:
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x) -> torch.Tensor:
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape) -> None:
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x) -> torch.Tensor:
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='BiasFree') -> None:
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x) -> torch.Tensor:
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class SMLayer(nn.Module):
    r"""Shuffle Mixer Layer.

    Args:
        dim: Number of input channels
        kernel_size:
        mlp_ratio:

    """

    def __init__(self, dim, kernel_size, mlp_ratio=2) -> None:
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)

        self.spatial = nn.Conv2d(dim, dim, kernel_size, 1, kernel_size // 2, groups=dim)

        self.mlp1 = SplitPointMlp(dim, mlp_ratio)
        self.mlp2 = SplitPointMlp(dim, mlp_ratio)

    def forward(self, x) -> torch.Tensor:
        x = self.mlp1(self.norm1(x)) + x
        x = self.spatial(x)
        x = self.mlp2(self.norm2(x)) + x
        return x


class FMBlock(nn.Module):
    r"""Feature Mixing Block.

    Args:
        dim: Number of input channels
        kernel_size:
        mlp_ratio:

    """

    def __init__(self, dim, kernel_size, mlp_ratio=2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            SMLayer(dim, kernel_size, mlp_ratio),
            SMLayer(dim, kernel_size, mlp_ratio),
        )

        self.conv = nn.Sequential(
            Conv2d3x3(dim, dim + 16),
            nn.SiLU(inplace=True),
            Conv2d1x1(dim + 16, dim)
        )

    def forward(self, x) -> torch.Tensor:
        x = self.net(x) + x
        x = self.conv(x) + x
        return x


@ARCH_REGISTRY.register()
class ShuffleMixer(nn.Module):
    """
    Args:
        dim (int): Number of channels. Default: 64 (32 for the tiny model).
        kerenl_size (int): kernel size of Depthwise convolution. Default:7 (3 for the tiny model).
        n_blocks (int): Number of feature mixing blocks. Default: 5.
        mlp_ratio (int): The expanding factor of point-wise MLP. Default: 2.
        upscaling_factor: The upscaling factor. [2, 3, 4]
    """

    def __init__(self, upscale=4, num_in_ch=3, num_out_ch=3, task='lsr', dim=64, kernel_size=7, n_blocks=5,
                 mlp_ratio=2):
        super().__init__()
        self.scale = upscale

        self.head = Conv2d3x3(num_in_ch, dim, bias=False)

        self.body = nn.Sequential(*[FMBlock(dim, kernel_size, mlp_ratio)
                                    for _ in range(n_blocks)])

        if self.scale == 4:
            self.upsapling = nn.Sequential(
                Conv2d1x1(dim, dim * 4),
                nn.PixelShuffle(2),
                nn.SiLU(inplace=True),
                Conv2d1x1(dim, dim * 4),
                nn.PixelShuffle(2),
                nn.SiLU(inplace=True)
            )
        else:
            self.upsapling = nn.Sequential(
                Conv2d1x1(dim, dim * self.scale ** 2),
                nn.PixelShuffle(self.scale),
                nn.SiLU(inplace=True)
            )

        self.tail = Conv2d3x3(dim, num_out_ch, bias=True)

    def forward(self, x):
        base = x

        x = self.head(x)
        x = self.body(x)
        x = self.upsapling(x)
        x = self.tail(x)

        base = F.interpolate(base, scale_factor=self.scale, mode='bilinear', align_corners=False)
        return x + base


if __name__ == '__main__':
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    net = ShuffleMixer(dim=64, kernel_size=7, n_blocks=5, mlp_ratio=2, upscale=4)
    print(count_parameters(net))

    data = torch.randn(1, 3, 120, 80)
    print(net(data).size())
