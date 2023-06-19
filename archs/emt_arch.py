import torch
import torch.nn as nn
from basicsr.utils.registry import ARCH_REGISTRY
import torch.nn.functional as f
from archs.utils import MeanShift, Conv2d1x1, Conv2d3x3, ShiftConv2d1x1,\
     SABase4D, TransformerGroup, Upsampler, Swish

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
        self.mask = nn.Parameter(torch.zeros((self.planes, 1, mix_margin * 2 + 1, mix_margin * 2 + 1)),
                                 requires_grad=False)

        self.mask[3::5, 0, 0, mix_margin] = 1.
        self.mask[2::5, 0, -1, mix_margin] = 1.
        self.mask[1::5, 0, mix_margin, 0] = 1.
        self.mask[0::5, 0, mix_margin, -1] = 1.
        self.mask[4::5, 0, mix_margin, mix_margin] = 1.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.mix_margin
        x = f.conv2d(input=f.pad(x, pad=(m, m, m, m), mode='circular'),
                     weight=self.mask, bias=None, stride=(1, 1), padding=(0, 0),
                     dilation=(1, 1), groups=self.planes)
        return x


class Mlp(nn.Module):
    r"""Multi-layer perceptron.

    Args:
        in_features: Number of input channels
        hidden_features:
        out_features: Number of output channels
        act_layer:

    """

    def __init__(self, in_features: int, hidden_features: int = None,
                 out_features: int = None, act_layer: nn.Module = nn.GELU) -> None:
        super().__init__()

        hidden_features = hidden_features or in_features
        out_features = out_features or in_features

        self.fc1 = ShiftConv2d1x1(in_features, hidden_features, bias=True)
        self.act = act_layer()
        self.fc2 = ShiftConv2d1x1(hidden_features, out_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            x: b c h w
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class TokenMixer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()

        self.token_mixer = PixelMixer(planes=dim)
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.token_mixer(x) - x)


class MixedTransformerBlock(TransformerGroup):
    def __init__(self, dim: int, num_layer: int, num_heads: int, num_GTLs: int,
                 window_list: tuple = None, shift_list: tuple = None,
                 mlp_ratio: int = None, act_layer: nn.Module = nn.GELU) -> None:
        token_mixer_list = [TokenMixer(dim) if _ > (num_GTLs - 1) else SABase4D(dim=dim, num_heads=num_heads,
                                                                                attn_layer=[Conv2d1x1(dim, dim * 2),
                                                                                            nn.BatchNorm2d(dim * 2)],
                                                                                proj_layer=[Conv2d1x1(dim, dim)],
                                                                                window_list=window_list,
                                                                                shift_list=shift_list if (_ + 1) % 2 == 0 else None)
                            for _ in range(num_layer)]

        mlp_list = [Mlp(dim, dim * mlp_ratio, act_layer=act_layer)
                    for _ in range(num_layer)]

        super(MixedTransformerBlock, self). \
            __init__(sa_list=token_mixer_list, mlp_list=mlp_list, conv_list=None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            x: b c h w
        """
        for sa, mlp in zip(self.sa_list, self.mlp_list):
            x = x + sa(x)
            x = x + mlp(x)

        return self.conv(x)


@ARCH_REGISTRY.register()
class EMT(nn.Module):
    r"""Efficient Mixed Transformer Super-Resolution Network!
    """

    def __init__(self, upscale: int, num_in_ch: int, num_out_ch: int, task: str, dim: int, n_blocks: int, n_layers: int,
                 num_heads: int, mlp_ratio: int, n_GTLs: int, window_list: tuple, shift_list: tuple):
        super(EMT, self).__init__()

        self.sub_mean = MeanShift(255, sign=-1, data_type='DF2K')
        self.add_mean = MeanShift(255, sign=1, data_type='DF2K')

        self.head = Conv2d3x3(num_in_ch, dim)

        self.body = nn.Sequential(
            *[MixedTransformerBlock(dim=dim, num_layer=n_layers, num_heads=num_heads, num_GTLs=n_GTLs,
                                    window_list=window_list, shift_list=shift_list,
                                    mlp_ratio=mlp_ratio, act_layer=Swish)
              for _ in range(n_blocks)])

        self.tail = Upsampler(upscale=upscale, in_channels=dim, out_channels=num_out_ch, upsample_mode=task)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # reduce the mean of pixels
        sub_x = self.sub_mean(x)

        # head
        head_x = self.head(sub_x)

        # body
        body_x = self.body(head_x)
        body_x = body_x + head_x

        # tail
        tail_x = self.tail(body_x)

        # add the mean of pixels
        add_x = self.add_mean(tail_x)

        return add_x

if __name__ == '__main__':
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    net = EMT(upscale=4, dim=60, n_blocks=6, n_layers=6, num_in_ch=3, num_out_ch=3, num_heads=3, mlp_ratio=2,
              n_GTLs=2, window_list=[ [32, 8],[8, 32] ], shift_list=[ [16, 4],[4, 16]], task='lsr')
    print(count_parameters(net))
