import torch
import torch.nn as nn
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class DebugArch(torch.nn.Module):
    def __init__(self, upscale: int, num_in_ch: int, num_out_ch: int, task: str) -> None:
        super().__init__()

        self.upscale = upscale
        self.parameter = nn.Parameter(torch.ones(1), requires_grad=True)
        self.body = nn.PixelShuffle(upscale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x * self.parameter] * (2 ** self.upscale), dim=1)
        return self.body(x)
