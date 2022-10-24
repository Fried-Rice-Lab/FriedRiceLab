# -------------------------------------------------------------------------------------
# Basic Modules for Image Restoration Networks
#
# Implemented/Modified by Fried Rice Lab (https://github.com/Fried-Rice-Lab)
# -------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as f

__all__ = ['GroupLinear']


class GroupLinear(nn.Linear):
    r"""

    Args:
        in_features:
        out_features:
        groups:
        bias:

    """

    def __init__(self, in_features: int, out_features: int,
                 groups: int, bias: bool = True, **kwargs) -> None:
        super(GroupLinear, self).__init__(in_features=in_features // groups,
                                          out_features=out_features // groups,
                                          bias=bias, **kwargs)

        assert in_features % groups == 0, f'{in_features} % {groups} != 0.'

        self.in_features = in_features
        self.out_features = out_features
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x.size()) == 2 or len(x.size()) == 3

        x_len = len(x.size()) == 2
        if x_len:
            x = x.unsqueeze(dim=1)  # b c -> b l c
        b, l, c = x.size()
        o, c, g = self.out_features, self.in_features, self.groups
        x = x.reshape(b, l, g, c // g).permute(0, 2, 1, 3)  # b l c -> b l g c/g -> b g l c/g
        x = f.linear(x, self.weight, self.bias)  # b g l c/g -> b g l o/g
        x = x.permute(0, 2, 1, 3).reshape(b, l, o)  # b g l o/g -> b l g o/g -> b l o
        if x_len:
            return x.squeeze(dim=1)
        else:
            return x
