# --------------------------------------------------------------------------------
# Lightweight Image Super-Resolution with Information Multi-distillation Network
# Official GitHub: https://github.com/Zheng222/IMDN
#
# Modified by Yulong Liu (yl.liu88@outlook.com)
# --------------------------------------------------------------------------------
import torch
import torch.nn as nn
from basicsr.utils.registry import ARCH_REGISTRY

from archs.utils import Conv2d1x1, Conv2d3x3, CCA, Upsampler


class IMDB(nn.Module):
    r"""Information Multi-Distillation Block.
    """

    def __init__(self, in_channels: int, distillation_rate: float = 0.25) -> None:
        super(IMDB, self).__init__()

        self.distilled_channels = int(in_channels * distillation_rate)
        self.remaining_channels = int(in_channels - self.distilled_channels)
        self.c1 = Conv2d3x3(in_channels, in_channels)
        self.c2 = Conv2d3x3(self.remaining_channels, in_channels)
        self.c3 = Conv2d3x3(self.remaining_channels, in_channels)
        self.c4 = Conv2d3x3(self.remaining_channels, self.distilled_channels)
        self.act = nn.LeakyReLU(negative_slope=0.05, inplace=True)
        self.c5 = Conv2d1x1(in_channels, in_channels)
        self.cca = CCA(self.distilled_channels * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_c1 = self.act(self.c1(x))
        distilled_c1, remaining_c1 = torch.split(out_c1, [self.distilled_channels, self.remaining_channels], dim=1)

        out_c2 = self.act(self.c2(remaining_c1))
        distilled_c2, remaining_c2 = torch.split(out_c2, [self.distilled_channels, self.remaining_channels], dim=1)

        out_c3 = self.act(self.c3(remaining_c2))
        distilled_c3, remaining_c3 = torch.split(out_c3, [self.distilled_channels, self.remaining_channels], dim=1)

        out_c4 = self.c4(remaining_c3)

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, out_c4], dim=1)

        return self.c5(self.cca(out)) + x


@ARCH_REGISTRY.register()
class IMDN(nn.Module):
    r"""Information Multi-Distillation Network.
    """

    def __init__(self, upscale: int, num_in_ch: int, num_out_ch: int, task: str,
                 planes: int, num_modules: int) -> None:
        super(IMDN, self).__init__()

        self.head = Conv2d3x3(num_in_ch, planes)

        self.body = nn.ModuleList([IMDB(in_channels=planes) for _ in range(num_modules)])

        self.body_tail = nn.Sequential(Conv2d1x1(planes * num_modules, planes),
                                       nn.LeakyReLU(negative_slope=0.05, inplace=True),
                                       Conv2d3x3(planes, planes))

        self.tail = Upsampler(upscale=upscale, in_channels=planes,
                              out_channels=num_out_ch, upsample_mode=task)

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
        body_x = body_x + head_x

        # tail
        tail_x = self.tail(body_x)
        return tail_x


if __name__ == '__main__':
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    net = IMDN(upscale=4, planes=64, num_modules=6)
    print(count_parameters(net))

    data = torch.randn(1, 3, 120, 80)
    print(net(data).size())
