# ----------------------------------------------------------------------
# Fast and Memory-Efficient Network Towards Efficient Image Super-Resolution
# Official GitHub: https://github.com/NJU-Jet/FMEN
#
# Modified by Tianle Liu (tianle.l@outlook.com)
# ----------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY

from archs.utils import Conv2d1x1, Conv2d3x3, Upsampler


def pad_tensor(t, pattern):
    pattern = pattern.view(1, -1, 1, 1)
    t = F.pad(t, (1, 1, 1, 1), 'constant', 0)

    t[:, :, 0:1, :] = pattern
    t[:, :, -1:, :] = pattern
    t[:, :, :, 0:1] = pattern
    t[:, :, :, -1:] = pattern

    return t


def get_bn_bias(bn_layer):
    gamma, beta, mean, var, eps = bn_layer.weight, bn_layer.bias, bn_layer.running_mean, bn_layer.running_var, bn_layer.eps
    std = (var + eps).sqrt()
    bn_bias = beta - mean * gamma / std

    return bn_bias


class RRRB(nn.Module):
    """ Residual in residual reparameterizable block.
    Using reparameterizable block to replace single 3x3 convolution.

    Diagram:
        ---Conv1x1--Conv3x3-+-Conv1x1--+--
                   |________|
         |_____________________________|


    Args:
        n_feats (int): The number of feature maps.
        ratio (int): Expand ratio.
    """

    def __init__(self, n_feats, ratio=2):
        super(RRRB, self).__init__()
        self.expand_conv = Conv2d1x1(n_feats, ratio * n_feats)
        self.fea_conv = nn.Conv2d(ratio * n_feats, ratio * n_feats, 3, 1, 0)
        self.reduce_conv = Conv2d1x1(ratio * n_feats, n_feats)

    def forward(self, x):
        out = self.expand_conv(x)
        out_identity = out

        # explicitly padding with bias for reparameterizing in the test phase
        b0 = self.expand_conv.bias
        out = pad_tensor(out, b0)

        out = self.fea_conv(out) + out_identity
        out = self.reduce_conv(out)
        out = out + x

        return out


class RRRB_i(nn.Module):
    def __init__(self, n_feats):
        super(RRRB_i, self).__init__()
        self.rep_conv = Conv2d3x3(n_feats, n_feats)

    def forward(self, x):
        out = self.rep_conv(x)

        return out


class ERB(nn.Module):
    """ Enhanced residual block for building FEMN.

    Diagram:
        --RRRB--LeakyReLU--RRRB--

    Args:
        n_feats (int): Number of feature maps.
        ratio (int): Expand ratio in RRRB.
    """

    def __init__(self, n_feats, ratio=2, act=nn.LeakyReLU(0.1)):
        super(ERB, self).__init__()
        self.conv1 = RRRB(n_feats, ratio)
        self.conv2 = RRRB(n_feats, ratio)
        self.act = act

    def forward(self, x):
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)

        return out


class ERB_i(nn.Module):
    def __init__(self, n_feats, act=nn.LeakyReLU(0.1)):
        super(ERB_i, self).__init__()
        self.conv1 = RRRB_i(n_feats)
        self.conv2 = RRRB_i(n_feats)
        self.act = act

    def forward(self, x):
        res = self.conv1(x)
        res = self.act(res)
        res = self.conv2(res)

        return res


class HFAB(nn.Module):
    """ High-Frequency Attention Block.

    Diagram:
        ---BN--Conv--[ERB]*up_blocks--BN--Conv--BN--Sigmoid--*--
         |___________________________________________________|

    Args:
        n_feats (int): Number of HFAB input feature maps.
        up_blocks (int): Number of ERBs for feature extraction in this HFAB.
        mid_feats (int): Number of feature maps in ERB.

    Note:
        Batch Normalization (BN) is adopted to introduce global contexts and achieve sigmoid unsaturated area.

    """

    def __init__(self, n_feats, up_blocks, mid_feats, ratio, act=nn.LeakyReLU(0.1)):
        super(HFAB, self).__init__()
        self.bn1 = nn.BatchNorm2d(n_feats)
        self.bn2 = nn.BatchNorm2d(mid_feats)
        self.bn3 = nn.BatchNorm2d(n_feats)
        self.act = act

        self.squeeze = nn.Conv2d(n_feats, mid_feats, 3, 1, 0)

        convs = [ERB(mid_feats, ratio) for _ in range(up_blocks)]
        self.convs = nn.Sequential(*convs)

        self.excitate = nn.Conv2d(mid_feats, n_feats, 3, 1, 0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # explicitly padding with bn bias
        out = self.bn1(x)

        bn1_bias = get_bn_bias(self.bn1)
        out = pad_tensor(out, bn1_bias)

        out = self.act(self.squeeze(out))
        out = self.act(self.convs(out))

        # explicitly padding with bn bias
        out = self.bn2(out)

        bn2_bias = get_bn_bias(self.bn2)
        out = pad_tensor(out, bn2_bias)

        out = self.excitate(out)
        out = self.sigmoid(self.bn3(out))

        return out * x


class HFAB_i(nn.Module):
    def __init__(self, n_feats, up_blocks, mid_feats, act=nn.LeakyReLU(0.1)):
        super(HFAB_i, self).__init__()
        self.act = act
        self.squeeze = nn.Conv2d(n_feats, mid_feats, 3, 1, 1)
        convs = [ERB_i(mid_feats) for _ in range(up_blocks)]
        self.convs = nn.Sequential(*convs)
        self.excitate = nn.Conv2d(mid_feats, n_feats, 3, 1, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.act(self.squeeze(x))
        out = self.act(self.convs(out))
        out = self.excitate(out)
        out = self.sigmoid(out)

        return out * x


@ARCH_REGISTRY.register()
class FMEN(nn.Module):
    """ Fast and Memory-Efficient Network

    Diagram:
        --Conv--Conv-HFAB-[ERB-HFAB]*down_blocks-Conv-+-Upsample--
               |______________________________________|

    Args:
        down_blocks (int): Number of [ERB-HFAB] pairs.
        up_blocks (list): Number of ERBs in each HFAB.
        mid_feats (int): Number of feature maps in branch ERB.
        n_feats (int): Number of feature maps in trunk ERB.
        backbone_expand_ratio (int): Expand ratio of RRRB in trunk ERB.
        attention_expand_ratio (int): Expand ratio of RRRB in branch ERB.
    """

    def __init__(self, upscale: int, num_in_ch: int, num_out_ch: int, task: str, up_blocks=(2, 1, 1, 1, 1),
                 down_blocks=4, n_feats=50, mid_feats=16, backbone_expand_ratio=2, attention_expand_ratio=2,
                 mode='train'):
        super(FMEN, self).__init__()

        self.down_blocks = down_blocks

        # define head module
        self.head = Conv2d3x3(num_in_ch, n_feats)
        if mode == 'train':
            # warm up
            self.warmup = nn.Sequential(
                Conv2d3x3(n_feats, n_feats),
                HFAB(n_feats, up_blocks[0], mid_feats - 4, attention_expand_ratio)
            )

            # define body module
            ERBs = [ERB(n_feats, backbone_expand_ratio) for _ in range(self.down_blocks)]
            HFABs = [HFAB(n_feats, up_blocks[i + 1], mid_feats, attention_expand_ratio)
                     for i in range(self.down_blocks)]

            self.ERBs = nn.ModuleList(ERBs)
            self.HFABs = nn.ModuleList(HFABs)
        elif mode == 'infer':
            # warm up
            self.warmup = nn.Sequential(
                Conv2d3x3(n_feats, n_feats),
                HFAB_i(n_feats, up_blocks[0], mid_feats - 4)
            )

            # define body module
            ERBs = [ERB_i(n_feats) for _ in range(self.down_blocks)]
            HFABs = [HFAB_i(n_feats, up_blocks[i + 1], mid_feats)
                     for i in range(self.down_blocks)]

            self.ERBs = nn.ModuleList(ERBs)
            self.HFABs = nn.ModuleList(HFABs)
        else:
            raise ValueError(f'mode {mode} is not supported.')

        self.lr_conv = Conv2d3x3(n_feats, n_feats)

        # define tail module
        self.tail = Upsampler(upscale=upscale, in_channels=n_feats,
                              out_channels=num_out_ch, upsample_mode=task)

    def forward(self, x):
        x = self.head(x)

        h = self.warmup(x)

        for i in range(self.down_blocks):
            h = self.ERBs[i](h)
            h = self.HFABs[i](h)

        h = self.lr_conv(h)

        h = h + x
        x = self.tail(h)

        return x


def make_model():
    return FMEN(upscale=4, num_in_ch=3, num_out_ch=3, task='lsr', mode='infer')


if __name__ == '__main__':
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    net = FMEN(upscale=4, num_in_ch=3, num_out_ch=3, task='lsr', mode='train')
    print(count_parameters(net))
    net = FMEN(upscale=4, num_in_ch=3, num_out_ch=3, task='lsr', mode='infer')
    print(count_parameters(net))

    data = torch.randn(1, 3, 120, 80)
    print(net(data).size())
