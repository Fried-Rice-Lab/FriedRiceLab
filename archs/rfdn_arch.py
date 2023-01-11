# --------------------------------------------------------------------------------
# Residual Feature Distillation Network for Lightweight Image Super-Resolution
# Official GitHub: https://github.com/njulj/RFDN
#
# Modified by Tianle Liu (tianle.l@outlook.com)
# --------------------------------------------------------------------------------
import torch
import torch.nn as nn
from basicsr.utils.registry import ARCH_REGISTRY

from archs.utils import Conv2d1x1, Conv2d3x3, CCA, ESA, Upsampler


class RFDB(nn.Module):
    r"""Residual Feature Distillation Block.

       Args:
           planes: Number of input channels
           distillation_rate:
           act_layer:

       """

    def __init__(self, planes: int, distillation_rate: float = 0.5,
                 act_layer: nn.Module = nn.LeakyReLU) -> None:
        super(RFDB, self).__init__()

        self.act = act_layer(negative_slope=0.05, inplace=True)
        self.distilled_channels = int(planes * distillation_rate)

        self.c1_d = Conv2d1x1(planes, self.distilled_channels, bias=True)
        self.c1_r = Conv2d3x3(planes, planes, bias=True)
        self.c2_d = Conv2d1x1(planes, self.distilled_channels, bias=True)
        self.c2_r = Conv2d3x3(planes, planes, bias=True)
        self.c3_d = Conv2d1x1(planes, self.distilled_channels, bias=True)
        self.c3_r = Conv2d3x3(planes, planes, bias=True)
        self.c4 = Conv2d3x3(planes, self.distilled_channels, bias=True)

        self.c5 = Conv2d1x1(self.distilled_channels * 4, planes, bias=True)

        self.cca = CCA(planes, reduction=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d_c1 = self.act(self.c1_d(x))
        r_c1 = self.c1_r(x)
        r_c1 = self.act(r_c1 + x)

        d_c2 = self.act(self.c2_d(r_c1))
        r_c2 = self.c2_r(r_c1)
        r_c2 = self.act(r_c2 + r_c1)

        d_c3 = self.act(self.c3_d(r_c2))
        r_c3 = self.c3_r(r_c2)
        r_c3 = self.act(r_c3 + r_c2)

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([d_c1, d_c2, d_c3, r_c4], dim=1)
        out_fused = self.cca(self.c5(out))

        return x + out_fused


class ERFDB(nn.Module):
    r"""Enhanced Residual Feature Distillation Block.

       Args:
           planes: Number of input channels
           distillation_rate:
           act_layer:

       """

    def __init__(self, planes: int, distillation_rate: float = 0.5,
                 act_layer: nn.Module = nn.LeakyReLU) -> None:
        super(ERFDB, self).__init__()

        self.act = act_layer(negative_slope=0.05, inplace=True)
        self.distilled_channels = int(planes * distillation_rate)

        self.c1_d = Conv2d1x1(planes, self.distilled_channels, bias=True)
        self.c1_r = Conv2d3x3(planes, planes, bias=True)
        self.c2_d = Conv2d1x1(planes, self.distilled_channels, bias=True)
        self.c2_r = Conv2d3x3(planes, planes, bias=True)
        self.c3_d = Conv2d1x1(planes, self.distilled_channels, bias=True)
        self.c3_r = Conv2d3x3(planes, planes, bias=True)
        self.c4 = Conv2d3x3(planes, self.distilled_channels, bias=True)

        self.c5 = Conv2d1x1(self.distilled_channels * 4, planes, bias=True)

        self.cca = ESA(planes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d_c1 = self.act(self.c1_d(x))
        r_c1 = self.c1_r(x)
        r_c1 = self.act(r_c1 + x)

        d_c2 = self.act(self.c2_d(r_c1))
        r_c2 = self.c2_r(r_c1)
        r_c2 = self.act(r_c2 + r_c1)

        d_c3 = self.act(self.c3_d(r_c2))
        r_c3 = self.c3_r(r_c2)
        r_c3 = self.act(r_c3 + r_c2)

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([d_c1, d_c2, d_c3, r_c4], dim=1)
        out_fused = self.cca(self.c5(out))

        return x + out_fused


@ARCH_REGISTRY.register()
class RFDN(nn.Module):
    r"""Residual Feature Distillation Network.

       Args:
           upscale:
           planes: Number of input channels
           num_blocks: Number of RFDB
           distillation_rate:

       """

    def __init__(self, upscale: int, num_in_ch: int, num_out_ch: int, task: str,
                 planes: int = 48, num_blocks: int = 6, distillation_rate: float = 0.5) -> None:
        super(RFDN, self).__init__()

        self.head = Conv2d3x3(num_in_ch, planes, bias=True)

        self.body = nn.ModuleList([RFDB(planes, distillation_rate)
                                   for _ in range(num_blocks)])

        self.body_tail = nn.Sequential(Conv2d1x1(planes * num_blocks, planes, bias=True),
                                       nn.LeakyReLU(negative_slope=0.05, inplace=True))

        self.tail = Upsampler(upscale=upscale, in_channels=planes,
                              out_channels=num_out_ch, upsample_mode=task)

        self.lr_conv = Conv2d3x3(planes, planes, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # head
        head_x = self.head(x)

        # body
        body_x = head_x
        output_list = list()
        for module in self.body:
            body_x = module(body_x)
            output_list.append(body_x)
        body_x = self.body_tail(torch.cat(output_list, dim=1))
        body_x = self.lr_conv(body_x) + head_x

        # tail
        tail_x = self.tail(body_x)

        return tail_x


@ARCH_REGISTRY.register()
class ERFDN(nn.Module):
    r"""Enhanced Residual Feature Distillation Network.

       Args:
           upscale:
           planes: Number of input channels
           num_blocks: Number of ERFDB
           distillation_rate:

       """

    def __init__(self, upscale: int, num_in_ch: int, num_out_ch: int, task: str,
                 planes: int = 48, num_blocks: int = 6, distillation_rate: float = 0.5) -> None:
        super(ERFDN, self).__init__()

        self.head = Conv2d3x3(num_in_ch, planes, bias=True)

        self.body = nn.ModuleList([ERFDB(planes, distillation_rate)
                                   for _ in range(num_blocks)])

        self.body_tail = nn.Sequential(Conv2d1x1(planes * num_blocks, planes, bias=True),
                                       nn.LeakyReLU(negative_slope=0.05, inplace=True))

        self.tail = Upsampler(upscale=upscale, in_channels=planes,
                              out_channels=num_out_ch, upsample_mode=task)

        self.lr_conv = Conv2d3x3(planes, planes, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # head
        head_x = self.head(x)

        # body
        body_x = head_x
        output_list = list()
        for module in self.body:
            body_x = module(body_x)
            output_list.append(body_x)
        body_x = self.body_tail(torch.cat(output_list, dim=1))
        body_x = self.lr_conv(body_x) + head_x

        # tail
        tail_x = self.tail(body_x)

        return tail_x


if __name__ == '__main__':
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # net = RFDN(upscale=4, planes=48, num_blocks=6)
    # print(count_parameters(net))

    # net = ERFDN(upscale=4, planes=50, num_blocks=4, distillation_rate=0.5)
    # print(count_parameters(net))

    # data = torch.randn(1, 3, 120, 80)
    # print(net(data).size())
