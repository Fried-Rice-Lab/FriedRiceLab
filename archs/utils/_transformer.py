# -------------------------------------------------------------------------------------
# Basic Modules for Image Restoration Networks
#
# Implemented/Modified by Fried Rice Lab (https://github.com/Fried-Rice-Lab)
# -------------------------------------------------------------------------------------
import torch
import torch.nn as nn

__all__ = ['TransformerGroup']


class TransformerGroup(nn.Module):
    r"""

    Args:
        sa_list:
        mlp_list:
        conv_list:

    """

    def __init__(self, sa_list: list, mlp_list: list, conv_list: list = None) -> None:
        super(TransformerGroup, self).__init__()

        assert len(sa_list) == len(mlp_list)

        self.sa_list = nn.ModuleList(sa_list)
        self.mlp_list = nn.ModuleList(mlp_list)
        self.conv = nn.Sequential(*conv_list if conv_list is not None else [nn.Identity()])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            x: b c h w
        """
        for sa, mlp in zip(self.sa_list, self.mlp_list):
            x = x + sa(x)
            x = x + mlp(x)
        return self.conv(x)


if __name__ == '__main__':
    pass
