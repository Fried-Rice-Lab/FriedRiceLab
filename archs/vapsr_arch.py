# ----------------------------------------------------------------------
# Efficient Image Super-Resolution using Vast-Receptive-Field Attention
# Official GitHub: https://github.com/zhoumumu/VapSR
#
# Modified by Tianle Liu (tianle.l@outlook.com)
# ----------------------------------------------------------------------
import torch
import torch.nn as nn
from basicsr.archs.arch_util import default_init_weights
from basicsr.utils.registry import ARCH_REGISTRY


class Attention(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        self.pw = nn.Conv2d(dim, dim, 1)
        self.dw = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.dw_d = nn.Conv2d(dim, dim, 5, stride=1, padding=6, groups=dim, dilation=3)

    def forward(self, x) -> torch.Tensor:
        u = x.clone()

        x = self.pw(x)
        x = self.dw(x)
        x = self.dw_d(x)

        return u * x


class VAB(nn.Module):
    r"""Vast-Receptive-Field Attention Block.

    Args:
        dim: Number of input channels
        d_atten:

    """

    def __init__(self, dim, d_atten) -> None:
        super().__init__()

        self.proj_1 = nn.Conv2d(dim, d_atten, 1)
        self.atten = Attention(d_atten)
        self.proj_2 = nn.Conv2d(d_atten, dim, 1)
        self.pixel_norm = nn.LayerNorm(dim)
        self.act = nn.GELU()
        default_init_weights([self.pixel_norm], 0.1)

    def forward(self, x) -> torch.Tensor:
        shorcut = x.clone()

        x = self.proj_1(x)
        x = self.act(x)
        x = self.atten(x)
        x = self.proj_2(x)
        x = x + shorcut

        x = x.permute(0, 2, 3, 1)
        x = self.pixel_norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()

        return x


# scale X4 use this version
def pixelshuffle(in_channels, out_channels) -> nn.Sequential:
    upconv1 = nn.Conv2d(in_channels, 64, 3, 1, 1)
    pixel_shuffle = nn.PixelShuffle(2)
    upconv2 = nn.Conv2d(16, out_channels * 4, 3, 1, 1)
    lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    return nn.Sequential(*[upconv1, pixel_shuffle, lrelu, upconv2, pixel_shuffle])


# both scale X2 and X3 use this version
def pixelshuffle_single(in_channels, out_channels, upscale_factor=2) -> nn.Sequential:
    upconv1 = nn.Conv2d(in_channels, 56, 3, 1, 1)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    upconv2 = nn.Conv2d(56, out_channels * upscale_factor * upscale_factor, 3, 1, 1)
    lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    return nn.Sequential(*[upconv1, lrelu, upconv2, pixel_shuffle])


def make_layer(block, n_layers, *kwargs):
    layers = []
    for _ in range(n_layers):
        layers.append(block(*kwargs))
    return nn.Sequential(*layers)


@ARCH_REGISTRY.register()
class vapsr(nn.Module):
    r"""VAst-receptive-field Pixel Attention Network.

     Args:
         dim: Number of input channels
         num_block:
         d_atten:
         conv_groups:
         upscale:
         num_in_ch:
         num_out_ch:
         task:

     """

    def __init__(self, upscale: int, num_in_ch: int, num_out_ch: int, task: str,
                 dim=48, num_block=21, d_atten=64, conv_groups=1) -> None:
        super(vapsr, self).__init__()

        self.conv_first = nn.Conv2d(num_in_ch, dim, 3, 1, 1)
        self.body = make_layer(VAB, num_block, dim, d_atten)

        # conv_groups=2 for VapSR-S, 1 for VapSR
        self.conv_body = nn.Conv2d(dim, dim, 3, 1, 1, groups=conv_groups)

        if upscale == 4:
            self.upsampler = pixelshuffle(dim, num_out_ch)
        else:
            self.upsampler = pixelshuffle_single(dim, num_out_ch, upscale_factor=upscale)

    def forward(self, x) -> torch.Tensor:
        x = self.conv_first(x)

        body_feat = self.body(x)

        body_out = self.conv_body(body_feat)
        x = x + body_out

        out = self.upsampler(x)

        return out


if __name__ == '__main__':
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    # vapSR
    net = vapsr(upscale=4, num_in_ch=3, num_out_ch=3, task='lsr')
    print(count_parameters(net))

    data = torch.randn(1, 3, 120, 80)
    print(net(data).size())
