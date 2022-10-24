# ---------------------------------------------------------------------------------------------------------
# Lightweight Bimodal Network for Single-Image Super-Resolution via Symmetric CNN and Recursive Transformer
# Official GitHub: https://github.com/wzx0826/LBNet
# ---------------------------------------------------------------------------------------------------------
import math

import torch
import torch.nn as nn
from basicsr.utils.registry import ARCH_REGISTRY

from archs.utils import Upsampler


def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows - 1) * strides[0] + effective_k_row - rows)
    padding_cols = max(0, (out_cols - 1) * strides[1] + effective_k_col - cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ZeroPad2d(paddings)(images)
    return images


def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']

    if padding == 'same':
        images = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError('Unsupported padding type: {}.\
                Only "same" or "valid" are supported.'.format(padding))

    unfold = torch.nn.Unfold(kernel_size=ksizes,
                             dilation=rates,
                             padding=0,
                             stride=strides)
    patches = unfold(images)
    return patches  # [N, C*k*k, L], L is the total number of such blocks


def reverse_patches(images, out_size, ksizes, strides, padding):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    unfold = torch.nn.Fold(output_size=out_size,
                           kernel_size=ksizes,
                           dilation=1,
                           padding=padding,
                           stride=strides)
    patches = unfold(images)
    return patches


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features // 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class EffAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., reduction=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.reduce = nn.Linear(dim, dim // reduction, bias=qkv_bias)  # dim // 1 for LBNet-T
        self.qkv = nn.Linear(dim // reduction, dim // reduction * 3, bias=qkv_bias)  # dim // 1 for LBNet-T
        self.proj = nn.Linear(dim // reduction, dim)  # dim // 1 for LBNet-T
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        x = self.reduce(x)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q_all = torch.split(q, math.ceil(N // 16), dim=-2)
        k_all = torch.split(k, math.ceil(N // 16), dim=-2)
        v_all = torch.split(v, math.ceil(N // 16), dim=-2)

        output = list()
        for q, k, v in zip(q_all, k_all, v_all):
            attn = (q @ k.transpose(-2, -1)) * self.scale  # 16*8*37*37
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            trans_x = (attn @ v).transpose(1, 2)  # .reshape(B, N, C)
            output.append(trans_x)
        x = torch.cat(output, dim=1)
        x = x.reshape(B, N, C)
        x = self.proj(x)
        return x


class TransBlock(nn.Module):
    def __init__(
            self, n_feat=64, dim=64, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
            drop_path=0., act_layer=nn.ReLU, norm_layer=nn.LayerNorm, reduction=1):
        super(TransBlock, self).__init__()
        self.dim = dim
        self.atten = EffAttention(self.dim, num_heads=num_heads, qkv_bias=False, qk_scale=None, attn_drop=0.,
                                  proj_drop=0., reduction=reduction)
        self.norm1 = nn.LayerNorm(self.dim)
        self.mlp = Mlp(in_features=dim, hidden_features=dim // 4, act_layer=act_layer, drop=drop)
        self.norm2 = nn.LayerNorm(self.dim)

    def forward(self, x):
        x = x + self.atten(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        res = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        y = self.sigmoid(x)
        return y * res


class FRDAB(nn.Module):
    def __init__(self, n_feats=32):
        super(FRDAB, self).__init__()

        self.c1 = default_conv(n_feats, n_feats, 1)
        self.c2 = default_conv(n_feats, n_feats // 2, 3)
        self.c3 = default_conv(n_feats, n_feats // 2, 3)
        self.c4 = default_conv(n_feats * 2, n_feats, 3)
        self.c5 = default_conv(n_feats // 2, n_feats // 2, 3)
        self.c6 = default_conv(n_feats * 2, n_feats, 1)

        self.se = CALayer(channel=2 * n_feats, reduction=16)
        self.sa = SpatialAttention()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        res = x

        y1 = self.act(self.c1(x))

        y2 = self.act(self.c2(y1))
        y3 = self.act(self.c3(y1))
        cat1 = torch.cat([y1, y2, y3], 1)

        y4 = self.act(self.c4(cat1))
        y5 = self.c5(y3)  # 16
        cat2 = torch.cat([y2, y5, y4], 1)

        ca_out = self.se(cat2)
        sa_out = self.sa(cat2)
        y6 = ca_out + sa_out

        y7 = self.c6(y6)
        output = res + y7

        return output


class LFFM(nn.Module):
    def __init__(self, n_feats=32):
        super(LFFM, self).__init__()

        self.b1 = FRDAB(n_feats=n_feats)
        self.b2 = FRDAB(n_feats=n_feats)
        self.b3 = FRDAB(n_feats=n_feats)

        self.c1 = nn.Conv2d(2 * n_feats, n_feats, 1, stride=1, padding=0, groups=2)
        self.c2 = nn.Conv2d(3 * n_feats, n_feats, 1, stride=1, padding=0, groups=2)
        self.c3 = nn.Conv2d(4 * n_feats, n_feats, 1, stride=1, padding=0, groups=1)

    def forward(self, x):
        res = x

        out1 = self.b1(x)
        dense1 = torch.cat([x, out1], 1)

        out2 = self.b2(self.c1(dense1))

        dense2 = torch.cat([x, out1, out2], 1)
        out3 = self.b3(self.c2(dense2))

        dense3 = torch.cat([x, out1, out2, out3], 1)
        out4 = self.c3(dense3)

        output = res + out4

        return output


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


def default_conv_stride2(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, stride=2,
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


@ARCH_REGISTRY.register()
class LBNet(nn.Module):
    def __init__(self, upscale: int, num_in_ch: int, num_out_ch: int, task: str,
                 n_feat=32, num_head=8, reduction=1, conv=default_conv):
        super(LBNet, self).__init__()

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(255, rgb_mean, rgb_std)

        self.head = conv(num_in_ch, n_feat, 3)

        self.r1 = LFFM(n_feats=n_feat)
        self.r2 = LFFM(n_feats=n_feat)
        self.r3 = LFFM(n_feats=n_feat)

        self.se1 = CALayer(channel=n_feat, reduction=16)
        self.se2 = CALayer(channel=n_feat, reduction=16)
        self.se3 = CALayer(channel=n_feat, reduction=16)

        self.attention = TransBlock(n_feat=n_feat, dim=n_feat * 9, num_heads=num_head, reduction=reduction)
        self.attention2 = TransBlock(n_feat=n_feat, dim=n_feat * 9, num_heads=num_head, reduction=reduction)

        self.c1 = default_conv(6 * n_feat, n_feat, 1)
        self.c2 = default_conv(n_feat, n_feat, 3)
        self.c3 = default_conv(n_feat, n_feat, 3)

        modules_tail = [
            Upsampler(upscale=upscale, in_channels=n_feat,
                      out_channels=num_out_ch, upsample_mode=task)
        ]
        self.tail = nn.Sequential(*modules_tail)

        self.add_mean = MeanShift(255, rgb_mean, rgb_std, 1)

    def forward(self, x):
        y_input1 = self.sub_mean(x)
        y_input = self.head(y_input1)
        res = y_input

        y1 = self.r1(y_input)
        y2 = self.r2(y1)
        y3 = self.r3(y2)

        y5 = self.r1(y3 + self.se1(y1))
        y6 = self.r2(y5 + self.se2(y2))
        y6_1 = self.r3(y6 + self.se3(y3))

        y7 = torch.cat([y1, y2, y3, y5, y6, y6_1], dim=1)
        y8 = self.c1(y7)

        b, c, h, w = y8.shape
        y8 = extract_image_patches(y8, ksizes=[3, 3],
                                   strides=[1, 1],
                                   rates=[1, 1],
                                   padding='same')  # 16*2304*576
        y8 = y8.permute(0, 2, 1)
        out_transf1 = self.attention(y8)
        out_transf1 = self.attention(out_transf1)
        out_transf1 = self.attention(out_transf1)
        out1 = out_transf1.permute(0, 2, 1)
        out1 = reverse_patches(out1, (h, w), (3, 3), 1, 1)
        y9 = self.c2(out1)

        y9 = extract_image_patches(y9, ksizes=[3, 3],
                                   strides=[1, 1],
                                   rates=[1, 1],
                                   padding='same')  # 16*2304*576
        y9 = y9.permute(0, 2, 1)
        out_transf2 = self.attention2(y9)
        out_transf2 = self.attention2(out_transf2)
        out_transf2 = self.attention2(out_transf2)
        out2 = out_transf2.permute(0, 2, 1)
        out2 = reverse_patches(out2, (h, w), (3, 3), 1, 1)

        y10 = self.c3(out2)

        output = y10 + res
        output = self.tail(output)

        y = self.add_mean(output)

        return y


if __name__ == '__main__':
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    # LBNet_x4
    net = LBNet(upscale=4, n_feat=32, num_head=8, reduction=2)
    print(count_parameters(net))

    # LBNet-T_x4
    net = LBNet(upscale=4, n_feat=18, num_head=6, reduction=1)
    print(count_parameters(net))

    data = torch.randn(1, 3, 120, 80)
    print(net(data).size())
