# ----------------------------------------------------------------------
# Edge-enhanced Feature Distillation Network for Efficient Super-Resolution
# Official GitHub: https://github.com/icandle/EFDN
#
# Modified by Tianle Liu (tianle.l@outlook.com)
# ----------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY

from archs.utils import Upsampler, ESA, Conv2d1x1, Conv2d3x3


def multiscale(kernel, target_kernel_size):
    H_pixels_to_pad = (target_kernel_size - kernel.size(2)) // 2
    W_pixels_to_pad = (target_kernel_size - kernel.size(3)) // 2
    return F.pad(kernel, [H_pixels_to_pad, H_pixels_to_pad, W_pixels_to_pad, W_pixels_to_pad])


class SeqConv3x3(nn.Module):
    def __init__(self, seq_type, inp_planes, out_planes, depth_multiplier):
        super(SeqConv3x3, self).__init__()

        self.type = seq_type
        self.inp_planes = inp_planes
        self.out_planes = out_planes

        if self.type == 'conv1x1-conv3x3':
            self.mid_planes = int(out_planes * depth_multiplier)
            conv0 = Conv2d1x1(self.inp_planes, self.mid_planes)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            conv1 = Conv2d3x3(self.mid_planes, self.out_planes)
            self.k1 = conv1.weight
            self.b1 = conv1.bias

        elif self.type == 'conv1x1-sobelx':
            conv0 = Conv2d1x1(self.inp_planes, self.out_planes)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(scale)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(bias)

            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0] = 1.0
                self.mask[i, 0, 1, 0] = 2.0
                self.mask[i, 0, 2, 0] = 1.0
                self.mask[i, 0, 0, 2] = -1.0
                self.mask[i, 0, 1, 2] = -2.0
                self.mask[i, 0, 2, 2] = -1.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)

        elif self.type == 'conv1x1-sobely':
            conv0 = Conv2d1x1(self.inp_planes, self.out_planes)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(torch.FloatTensor(scale))
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(torch.FloatTensor(bias))

            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0] = 1.0
                self.mask[i, 0, 0, 1] = 2.0
                self.mask[i, 0, 0, 2] = 1.0
                self.mask[i, 0, 2, 0] = -1.0
                self.mask[i, 0, 2, 1] = -2.0
                self.mask[i, 0, 2, 2] = -1.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)

        elif self.type == 'conv1x1-laplacian':
            conv0 = Conv2d1x1(self.inp_planes, self.out_planes)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(torch.FloatTensor(scale))
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(torch.FloatTensor(bias))

            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 1] = 1.0
                self.mask[i, 0, 1, 0] = 1.0
                self.mask[i, 0, 1, 2] = 1.0
                self.mask[i, 0, 2, 1] = 1.0
                self.mask[i, 0, 1, 1] = -4.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)
        else:
            raise ValueError('the type of seqconv is not supported!')

    def forward(self, x):
        if self.type == 'conv1x1-conv3x3':
            # conv-1x1
            y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1)
            # explicitly padding with bias
            y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0)
            b0_pad = self.b0.view(1, -1, 1, 1)
            y0[:, :, 0:1, :] = b0_pad
            y0[:, :, -1:, :] = b0_pad
            y0[:, :, :, 0:1] = b0_pad
            y0[:, :, :, -1:] = b0_pad
            # conv-3x3
            y1 = F.conv2d(input=y0, weight=self.k1, bias=self.b1, stride=1)
        else:
            # conv-1x1
            y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1)
            # explicitly padding with bias
            y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0)
            b0_pad = self.b0.view(1, -1, 1, 1)
            y0[:, :, 0:1, :] = b0_pad
            y0[:, :, -1:, :] = b0_pad
            y0[:, :, :, 0:1] = b0_pad
            y0[:, :, :, -1:] = b0_pad
            # conv-3x3
            y1 = F.conv2d(input=y0, weight=self.scale * self.mask, bias=self.bias,
                          stride=1, groups=self.out_planes)
        return y1

    def rep_params(self):
        device = self.k0.get_device()
        if device < 0:
            device = None

        if self.type == 'conv1x1-conv3x3':
            # re-param conv kernel
            RK = F.conv2d(input=self.k1, weight=self.k0.permute(1, 0, 2, 3))
            # re-param conv bias
            RB = torch.ones(1, self.mid_planes, 3, 3, device=device) * self.b0.view(1, -1, 1, 1)
            RB = F.conv2d(input=RB, weight=self.k1).view(-1, ) + self.b1
        else:
            tmp = self.scale * self.mask
            k1 = torch.zeros((self.out_planes, self.out_planes, 3, 3), device=device)
            for i in range(self.out_planes):
                k1[i, i, :, :] = tmp[i, 0, :, :]
            b1 = self.bias

            # re-param conv kernel
            RK = F.conv2d(input=k1, weight=self.k0.permute(1, 0, 2, 3))
            # re-param conv bias
            RB = torch.ones(1, self.out_planes, 3, 3, device=device) * self.b0.view(1, -1, 1, 1)
            RB = F.conv2d(input=RB, weight=k1).view(-1, ) + b1
        return RK, RB


class EDBB(nn.Module):
    r"""Edge-enhanced Diverse Branch Block.

    Args:
        n_feats: Number of input channels
        depth_multiplier:
        deploy:

    """

    def __init__(self, n_feats, depth_multiplier=1, deploy=False):
        super(EDBB, self).__init__()
        self.deploy = deploy
        self.depth_multiplier = depth_multiplier
        self.n_feats = n_feats

        if deploy:
            self.rep_conv = Conv2d3x3(n_feats, n_feats)
        else:
            self.rep_conv = Conv2d3x3(n_feats, n_feats)
            self.conv1x1 = Conv2d1x1(n_feats, n_feats)
            self.conv1x1_3x3 = SeqConv3x3('conv1x1-conv3x3', n_feats, n_feats, depth_multiplier)
            self.conv1x1_sbx = SeqConv3x3('conv1x1-sobelx', n_feats, n_feats, -1)
            self.conv1x1_sby = SeqConv3x3('conv1x1-sobely', n_feats, n_feats, -1)
            self.conv1x1_lpl = SeqConv3x3('conv1x1-laplacian', n_feats, n_feats, -1)

        self.act = nn.PReLU(n_feats)

    def forward(self, x):
        if self.deploy:
            y = self.rep_conv(x)
        else:
            y = self.rep_conv(x) + self.conv1x1(x) + self.conv1x1_3x3(x) + \
                self.conv1x1_sbx(x) + self.conv1x1_sby(x) + self.conv1x1_lpl(x) + x

        return self.act(y)

    def switch_to_deploy(self):

        if self.deploy:
            return
        self.deploy = True

        K0, B0 = self.rep_conv.weight, self.rep_conv.bias
        K1, B1 = self.conv1x1_3x3.rep_params()
        K2, B2 = self.conv1x1_sbx.rep_params()
        K3, B3 = self.conv1x1_sby.rep_params()
        K4, B4 = self.conv1x1_lpl.rep_params()
        K5, B5 = multiscale(self.conv1x1.weight, 3), self.conv1x1.bias
        RK, RB = (K0 + K1 + K2 + K3 + K4 + K5), (B0 + B1 + B2 + B3 + B4 + B5)

        device = RK.get_device()
        if device < 0:
            device = None
        K_idt = torch.zeros(self.n_feats, self.n_feats, 3, 3, device=device)
        for i in range(self.n_feats):
            K_idt[i, i, 1, 1] = 1.0
        B_idt = 0.0
        RK, RB = RK + K_idt, RB + B_idt

        self.rep_conv = Conv2d3x3(self.n_feats, self.n_feats)
        self.rep_conv.weight.data = RK
        self.rep_conv.bias.data = RB

        for para in self.parameters():
            para.detach_()

        self.__delattr__('conv1x1')
        self.__delattr__('conv1x1_3x3')
        self.__delattr__('conv1x1_sbx')
        self.__delattr__('conv1x1_sby')
        self.__delattr__('conv1x1_lpl')


class EFDB(nn.Module):
    r"""Edge-enhanced Feature Distillation Block.

    Args:
        n_feats: Number of input channels

    """

    def __init__(self, n_feats=48, deploy=False):
        super(EFDB, self).__init__()

        self.conv1 = Conv2d1x1(n_feats, n_feats)

        self.conv2 = EDBB(n_feats, deploy=deploy)
        self.conv3 = EDBB(n_feats, deploy=deploy)

        self.act = nn.PReLU(n_feats)

        self.fuse = Conv2d1x1(n_feats * 2, n_feats)
        self.att = ESA(n_feats, n_feats // 4, 3)
        self.branch = nn.ModuleList([Conv2d1x1(n_feats, n_feats // 2) for _ in range(4)])

    def forward(self, x):
        out1 = self.act(self.conv1(x))
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)

        out = self.fuse(torch.cat([self.branch[0](x), self.branch[1](out1),
                                   self.branch[2](out2), self.branch[3](out3)], dim=1))

        out = self.att(out)
        out = out + x

        return out


@ARCH_REGISTRY.register()
class EFDN(nn.Module):
    r"""Edge-enhanced Feature Distillation Network.

    Args:
        n_feats: Number of input channels

    """

    def __init__(self, upscale: int, num_in_ch: int, num_out_ch: int, task: str,
                 n_feats=48, deploy=False):
        super(EFDN, self).__init__()
        self.head = Conv2d3x3(num_in_ch, n_feats)

        self.cells = nn.ModuleList([EFDB(n_feats, deploy=deploy) for _ in range(4)])

        self.local_fuse = nn.ModuleList([Conv2d1x1(n_feats * 2, n_feats) for _ in range(3)])

        self.tail = Upsampler(upscale=upscale, in_channels=n_feats,
                              out_channels=num_out_ch, upsample_mode=task)

    def forward(self, x):
        out0 = self.head(x)

        out1 = self.cells[0](out0)
        out2 = self.cells[1](out1)
        out2_fuse = self.local_fuse[0](torch.cat([out1, out2], dim=1))
        out3 = self.cells[2](out2_fuse)
        out3_fuse = self.local_fuse[1](torch.cat([out2, out3], dim=1))
        out4 = self.cells[3](out3_fuse)
        out4_fuse = self.local_fuse[2](torch.cat([out2, out4], dim=1))

        out = out4_fuse + out0
        out = self.tail(out)

        return out


if __name__ == '__main__':
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    # EFDN
    net = EFDN(upscale=4, n_feats=48, num_in_ch=3, num_out_ch=3, task='lsr', deploy=False)
    print(count_parameters(net))
    net = EFDN(upscale=4, n_feats=48, num_in_ch=3, num_out_ch=3, task='lsr', deploy=True)
    print(count_parameters(net))

    data = torch.randn(1, 3, 120, 80)
    print(net(data).size())
