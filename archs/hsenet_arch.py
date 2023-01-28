# -------------------------------------------------------------------------------------
# Hybrid-Scale Self-Similarity Exploitation for Remote Sensing Image Super-Resolution
# Official GitHub: https://github.com/Shaosifan/HSENet
# -------------------------------------------------------------------------------------
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False


class BasicBlock(nn.Sequential):
    def __init__(
            self, conv, in_channels, out_channels, kernel_size, stride=1, bias=True,
            bn=False, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)
        super(BasicBlock, self).__init__(*m)


class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class DownBlock(nn.Module):
    def __init__(self, scale):
        super().__init__()

        self.scale = scale

    def forward(self, x):
        n, c, h, w = x.size()
        x = x.view(n, c, h // self.scale, self.scale, w // self.scale, self.scale)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        x = x.view(n, c * (self.scale ** 2), h // self.scale, w // self.scale)
        return x


# NONLocalBlock2D
# ref: https://github.com/AlexHex7/Non-local_pytorch/blob/master/Non-Local_pytorch_0.4.1_to_1.1.0/lib/non_local_dot_product.py
# ref: https://github.com/yulunzhang/RNAN/blob/master/SR/code/model/common.py
class NonLocalBlock2D(nn.Module):
    def __init__(self, in_channels, inter_channels):
        super(NonLocalBlock2D, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        self.g = nn.Conv2d(in_channels=in_channels, out_channels=inter_channels,
                           kernel_size=1, stride=1, padding=0)
        self.W = nn.Conv2d(in_channels=inter_channels, out_channels=in_channels,
                           kernel_size=1, stride=1, padding=0)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)

        # use dot production
        # N = f.size(-1)
        # f_div_C = f / N

        # use embedding gaussian
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


class AdjustedNonLocalBlock(nn.Module):
    def __init__(self, in_channels, inter_channels):
        super(AdjustedNonLocalBlock, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        self.g = nn.Conv2d(in_channels=in_channels, out_channels=inter_channels,
                           kernel_size=1, stride=1, padding=0)
        self.W = nn.Conv2d(in_channels=inter_channels, out_channels=in_channels,
                           kernel_size=1, stride=1, padding=0)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

    def forward(self, x0, x1):
        batch_size = x0.size(0)

        g_x = self.g(x0).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x1).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x0).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)

        # use embedding gaussian
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x0.size()[2:])
        W_y = self.W(y)
        z = W_y + x0

        return z


# hybrid-scale self-similarity exploitation module
class HSEM(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True,
                 bn=False, act=nn.ReLU(True)):
        super(HSEM, self).__init__()

        base_scale = []
        base_scale.append(SSEM(conv, n_feats, kernel_size, bias=bias, bn=bn, act=act))

        down_scale = []
        down_scale.append(SSEM(conv, n_feats, kernel_size, bias=bias, bn=bn, act=act))

        tail = []
        tail.append(BasicBlock(conv, n_feats, n_feats, kernel_size, bias=bias, bn=bn, act=act))

        self.NonLocal_base = AdjustedNonLocalBlock(n_feats, n_feats // 2)

        self.base_scale = nn.Sequential(*base_scale)
        self.down_scale = nn.Sequential(*down_scale)
        self.tail = nn.Sequential(*tail)

    def forward(self, x):
        add_out = x

        # base scale
        x_base = self.base_scale(x)

        # 1/2 scale
        x_down = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        x_down = self.down_scale(x_down)

        # fusion x_down and x_down2
        x_down = F.interpolate(x_down, size=(x_base.shape[2], x_base.shape[3]),
                               mode='bilinear')
        ms = self.NonLocal_base(x_base, x_down)
        ms = self.tail(ms)

        add_out = add_out + ms

        return add_out


# single-scale self-similarity exploitation module
class SSEM(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True,
                 bn=False, act=nn.ReLU(True)):
        super(SSEM, self).__init__()

        head = []
        head.append(BasicBlock(conv, n_feats, n_feats, kernel_size, bias=bias, bn=bn))

        MB = []  # main branch
        MB.append(BasicBlock(conv, n_feats, n_feats, kernel_size, bias=bias, bn=bn))
        MB.append(BasicBlock(conv, n_feats, n_feats, kernel_size, bias=bias, bn=bn))

        AB = []  # attention branch
        AB.append(NonLocalBlock2D(n_feats, n_feats // 2))
        AB.append(nn.Conv2d(n_feats, n_feats, 1, padding=0, bias=True))

        sigmoid = []
        sigmoid.append(nn.Sigmoid())

        tail = []
        tail.append(BasicBlock(conv, n_feats, n_feats, kernel_size, bias=bias, bn=bn))

        self.head = nn.Sequential(*head)
        self.MB = nn.Sequential(*MB)
        self.AB = nn.Sequential(*AB)
        self.sigmoid = nn.Sequential(*sigmoid)
        self.tail = nn.Sequential(*tail)

    def forward(self, x):
        add_out = x
        x_head = self.head(x)
        x_MB = self.MB(x_head)
        x_AB = self.AB(x_head)
        x_AB = self.sigmoid(x_AB)
        x_MB_AB = x_MB * x_AB
        x_tail = self.tail(x_MB_AB)

        add_out = add_out + x_tail
        return add_out


# multi-scale self-similarity block
class BasicModule(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True,
                 bn=False, act=nn.ReLU(True)):
        super(BasicModule, self).__init__()

        head = [
            BasicBlock(conv, n_feats, n_feats, kernel_size, bias=bias, bn=bn, act=act)
            for _ in range(2)
        ]

        body = []
        body.append(HSEM(conv, n_feats, kernel_size, bias=bias, bn=bn, act=act))

        tail = [
            BasicBlock(conv, n_feats, n_feats, kernel_size, bias=bias, bn=bn, act=act)
            for _ in range(2)
        ]

        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)

    def forward(self, x):
        add_out = x

        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)

        add_out = add_out + x

        return add_out


@ARCH_REGISTRY.register()
class HSENet(nn.Module):
    def __init__(self, upscale: int, num_in_ch: int = 3, num_out_ch: int = 3, task: str = 'csr',
                 conv=default_conv):
        super(HSENet, self).__init__()

        n_feats = 64
        kernel_size = 3
        scale = upscale
        act = nn.ReLU(True)

        self.n_BMs = 10

        rgb_mean = (0.4916, 0.4991, 0.4565)  # UCMerced data
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(255, rgb_mean, rgb_std)

        # define head body
        m_head = [conv(num_in_ch, n_feats, kernel_size)]

        # define main body
        self.body_modulist = nn.ModuleList([
            BasicModule(conv, n_feats, kernel_size, act=act)
            for _ in range(self.n_BMs)
        ])

        # define tail body
        m_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, num_out_ch, kernel_size)
        ]

        self.add_mean = MeanShift(255, rgb_mean, rgb_std, 1)
        self.head = nn.Sequential(*m_head)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        # main body
        add_out = x

        for i in range(self.n_BMs):
            x = self.body_modulist[i](x)
        add_out = add_out + x

        x = self.tail(add_out)
        x = self.add_mean(x)

        return x


if __name__ == '__main__':
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    net = HSENet(upscale=4)
    print(count_parameters(net))

    data = torch.randn(1, 3, 48, 48)
    print(net(data).size())
