# ----------------------------------------------------------------------
# Image Super-Resolution using Efficient Striped Window Transformer
#
# Implemented by Jinpeng Shi (jinpeeeng.s@gmail.com)
# ----------------------------------------------------------------------
import torch
import torch.nn as nn
from basicsr.utils.registry import ARCH_REGISTRY

from archs.utils import MeanShift, Conv2d1x1, Conv2d3x3, ShiftConv2d1x1, \
    SABase4D, TransformerGroup as _TransformerGroup, Upsampler, Swish


class MLP4D(nn.Module):
    r"""Multi-layer perceptron for 4D input.

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


class TransformerGroup(_TransformerGroup):
    def __init__(self, n_t: int, dim: int, num_heads: int,
                 window_list: tuple = ((24, 6), (6, 24)), shift_list: tuple = ((12, 3), (3, 12)),
                 mlp_ratio: int = 2, act_layer: nn.Module = nn.GELU) -> None:
        sa_list = [SABase4D(dim=dim,
                            num_heads=num_heads,
                            attn_layer=[Conv2d1x1(dim, dim * 2),
                                        nn.BatchNorm2d(dim * 2)],
                            proj_layer=[Conv2d1x1(dim, dim)],
                            window_list=window_list,
                            shift_list=shift_list if (i + 1) % 2 == 0 else None)
                   for i in range(n_t)]

        mlp_list = [MLP4D(dim, dim * mlp_ratio, act_layer=act_layer)
                    for _ in range(n_t)]

        conv_list = [Conv2d3x3(in_channels=dim, out_channels=dim)]

        super(TransformerGroup, self). \
            __init__(sa_list=sa_list, mlp_list=mlp_list, conv_list=conv_list)


@ARCH_REGISTRY.register()
class ESWT(nn.Module):
    r"""Image Super-Resolution using Efficient Striped Window Transformer
    """

    def __init__(self, upscale: int, num_in_ch: int, num_out_ch: int, task: str,
                 n_t: int, n_g: int, dim: int, num_heads: int = 1,
                 window_list: tuple = ((24, 6), (6, 24)),
                 shift_list: tuple = ((12, 3), (3, 12))) -> None:
        super(ESWT, self).__init__()

        self.sub_mean = MeanShift(255, sign=-1, data_type='DF2K')
        self.add_mean = MeanShift(255, sign=1, data_type='DF2K')

        self.head = nn.Sequential(Conv2d3x3(num_in_ch, dim))

        self.body = nn.Sequential(*[TransformerGroup(n_t=n_t, dim=dim, num_heads=num_heads,
                                                     window_list=window_list, shift_list=shift_list,
                                                     act_layer=Swish)  # noqa
                                    for _ in range(n_g)])

        self.tail = Upsampler(upscale=upscale, in_channels=dim,
                              out_channels=num_out_ch, upsample_mode=task)

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


    net = ESWT(upscale=4, n_t=6, n_g=3, dim=60)
    print(count_parameters(net))

    data = torch.randn(1, 3, 120, 80)
    print(net(data).size())
