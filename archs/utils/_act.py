# -------------------------------------------------------------------------------------
# Basic Modules for Image Restoration Networks
#
# Implemented/Modified by Fried Rice Lab (https://github.com/Fried-Rice-Lab)
# -------------------------------------------------------------------------------------
import torch
import torch.nn as nn

__all__ = ['Swish']


class _Swish(torch.autograd.Function):  # noqa
    @staticmethod
    def forward(ctx, i):  # noqa
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):  # noqa
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish(nn.Module):
    r"""A memory-efficient implementation of Swish. The original code is from
        https://github.com/zudi-lin/rcan-it/blob/main/ptsr/model/_utils.py.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa
        return _Swish.apply(x)
