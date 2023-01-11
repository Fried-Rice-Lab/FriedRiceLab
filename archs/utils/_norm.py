# -------------------------------------------------------------------------------------
# Basic Modules for Image Restoration Networks
#
# Implemented/Modified by Fried Rice Lab (https://github.com/Fried-Rice-Lab)
# -------------------------------------------------------------------------------------
import torch
import torch.nn as nn

__all__ = ['LayerNorm4D']


class LayerNorm4D(nn.Module):
    r"""LayerNorm for 4D input.

    Modified from https://github.com/sail-sg/poolformer.

    Args:
        num_channels (int): Number of channels expected in input
        eps (float): A value added to the denominator for numerical stability. Default: 1e-5

    """

    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            x: b c h w

        Returns:
            b c h w -> b c h w
        """
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1).unsqueeze(-1) * x \
            + self.bias.unsqueeze(-1).unsqueeze(-1)
        return x
