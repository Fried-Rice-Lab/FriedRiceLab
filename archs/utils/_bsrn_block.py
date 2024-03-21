'''
This repository is used to implement all blocks and tools for Efficient SR
@author
    LI Zehyuan from SIAT
    LIU yingqi from SIAT
'''
from functools import partial

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from distutils.version import LooseVersion



# 1*1卷积使用nn.Linear实现
class DepthWiseConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=True, padding_mode="zeros", with_norm=True, bn_kwargs=None):
        super(DepthWiseConv, self).__init__()
        # 也相当于分组为1的分组卷积
        self.depth_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=in_ch,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=in_ch)
        self.point_conv = nn.Linear(in_ch, out_ch)
        # self.point_conv = nn.Conv2d(in_channels=in_ch,
        #                             out_channels=out_ch,
        #                             kernel_size=1,
        #                             stride=1,
        #                             padding=0,
        #                             groups=1)

    def forward(self, input):
        out = self.depth_conv(input)
        out = out.permute(0, 2, 3, 1)
        out = self.point_conv(out)
        out = out.permute(0, 3, 1, 2)
        return out


class BSConvU(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=True, padding_mode="zeros", with_ln=False, bn_kwargs=None):
        super().__init__()
        self.with_ln = with_ln
        # check arguments
        if bn_kwargs is None:
            bn_kwargs = {}

        # pointwise
        self.pw = nn.Linear(in_channels, out_channels)
        # self.pw=torch.nn.Conv2d(
        #         in_channels=in_channels,
        #         out_channels=out_channels,
        #         kernel_size=(1, 1),
        #         stride=1,
        #         padding=0,
        #         dilation=1,
        #         groups=1,
        #         bias=False,
        # )
        # batchnorm
        if with_ln:
            self.ln = torch.nn.LayerNorm(out_channels, **bn_kwargs)

        # depthwise
        self.dw = torch.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=out_channels,
                bias=bias,
                padding_mode=padding_mode,
        )

    def forward(self, fea):
        fea = fea.permute(0, 2, 3, 1)
        fea = self.pw(fea)
        if self.with_ln:
            fea = self.ln(fea)
        fea = self.dw(fea.permute(0, 3, 1, 2))
        return fea


class BSConvS(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True,
                 padding_mode="zeros", p=0.25, min_mid_channels=4, with_ln=True, bn_kwargs=None):
        super().__init__()
        self.with_ln = with_ln
        # check arguments
        assert 0.0 <= p <= 1.0
        mid_channels = min(in_channels, max(min_mid_channels, math.ceil(p * in_channels)))
        if bn_kwargs is None:
            bn_kwargs = {}

        # pointwise 1
        self.pw1 = nn.Linear(in_channels, mid_channels)
        # self.pw1 = torch.nn.Conv2d(
        #     in_channels=in_channels,
        #     out_channels=mid_channels,
        #     kernel_size=(1, 1),
        #     stride=1,
        #     padding=0,
        #     dilation=1,
        #     groups=1,
        #     bias=False,
        # )

        # batchnorm
        if with_ln:
            self.ln1 = torch.nn.LayerNorm(mid_channels, **bn_kwargs)

        # pointwise 2
        self.pw2 = nn.Linear(mid_channels, out_channels)
        # self.add_module("pw2", torch.nn.Conv2d(
        #     in_channels=mid_channels,
        #     out_channels=out_channels,
        #     kernel_size=(1, 1),
        #     stride=1,
        #     padding=0,
        #     dilation=1,
        #     groups=1,
        #     bias=False,
        # ))

        # batchnorm
        if with_ln:
            self.ln2 = torch.nn.LayerNorm(out_channels, **bn_kwargs)

        # depthwise
        self.dw = torch.nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=out_channels,
            bias=bias,
            padding_mode=padding_mode,
        )

    def forward(self, x):
        fea = x.permute(0, 2, 3, 1)
        fea = self.pw1(fea)
        if self.with_ln:
            fea = self.ln1(fea)
        fea = self.pw2(fea)
        if self.with_ln:
            fea = self.ln2(fea)
        fea = self.dw(fea.permute(0, 3, 1, 2))
        return fea

    def _reg_loss(self):
        W = self[0].weight[:, :, 0, 0]
        WWt = torch.mm(W, torch.transpose(W, 0, 1))
        I = torch.eye(WWt.shape[0], device=WWt.device)
        return torch.norm(WWt - I, p="fro")


# replicate the input RGB channels at the head of the net
class ChannelReplicate(nn.Module):
    def __init__(self, factor=3):
        super(ChannelReplicate, self).__init__()
        self.factor = factor

    def forward(self, input):
        template = input
        for i in range(0, self.factor-1):
            input = torch.cat((template, input), 1)
        return input


# shuffleNetv1
# shufflblock combine group_conv with channle shuffle
class ShuffleV1Block(nn.Module):
    def __init__(self, inp, oup, *, group, first_group, mid_channels, ksize, stride):
        super(ShuffleV1Block, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp
        self.group = group

        if stride == 2:
            outputs = oup - inp
        else:
            outputs = oup

        branch_main_1 = [
            # pw
            nn.Conv2d(inp, mid_channels, 1, 1, 0, groups=1 if first_group else group, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(mid_channels, mid_channels, ksize, stride, pad, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
        ]
        branch_main_2 = [
            # pw-linear
            nn.Conv2d(mid_channels, outputs, 1, 1, 0, groups=group, bias=False),
            nn.BatchNorm2d(outputs),
        ]
        self.branch_main_1 = nn.Sequential(*branch_main_1)
        self.branch_main_2 = nn.Sequential(*branch_main_2)

        if stride == 2:
            self.branch_proj = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, old_x):
        x = old_x
        x_proj = old_x
        x = self.branch_main_1(x)
        if self.group > 1:
            x = self.channel_shuffle_v1(x)
        # 此处我觉得代码有问题，channle shuffle应该在branch_main_1中的pw之后
        x = self.branch_main_2(x)
        if self.stride == 1:
            return F.relu(x + x_proj)
        elif self.stride == 2:
            return torch.cat((self.branch_proj(x_proj), F.relu(x)), 1)

        def channel_shuffle_v1(self, x):
            batchsize, num_channels, height, width = x.data.size()
            assert num_channels % self.group == 0
            group_channels = num_channels // self.group

            x = x.reshape(batchsize, group_channels, self.group, height, width)
            x = x.permute(0, 2, 1, 3, 4)
            #将x换位，相当于x = (batchsize, self.group, group_channles, height, width)
            x = x.reshape(batchsize, num_channels, height, width)

            return x


# shufflev2block
# Gconv-->conv , remove channle shuffle in the branch
class ShuffleV2Block(nn.Module):
    def __init__(self, inp, oup, mid_channels, *, ksize, stride):
        super(ShuffleV2Block, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp

        outputs = oup - inp

        branch_main = [
            # pw
            nn.Conv2d(inp, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(mid_channels, mid_channels, ksize, stride, pad, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            # pw-linear
            nn.Conv2d(mid_channels, outputs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=True),
        ]
        self.branch_main = nn.Sequential(*branch_main)

        if stride == 2:
            branch_proj = [
                # dw
                nn.Conv2d(inp, inp, ksize, stride, pad, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
            ]
            self.branch_proj = nn.Sequential(*branch_proj)
        else:
            self.branch_proj = None

    def forward(self, old_x):
        if self.stride==1:
            x_proj, x = self.channel_shuffle_v2(old_x)
            return torch.cat((x_proj, self.branch_main(x)), 1)
        # 将上一个block得到的old_x经过两个不同分支，注意这里的恒等体现了channle spilt. 然后横着concat
        elif self.stride==2:
            x_proj = old_x
            x = old_x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)
        # 对应paper中第二幅图

    def channel_shuffle_v2(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert (num_channels % 4 == 0)
        x = x.reshape(batchsize * num_channels // 2, 2, height * width)
        x = x.permute(1, 0, 2)
        x = x.reshape(2, -1, num_channels // 2, height, width)
        return x[0], x[1]
# Performer
# for simpler you can just import and use like below
# from performer_pytorch import FastAttention
# https://github.com/lucidrains/performer-pytorch for more details




# Linear Transformer
# https://github.com/idiap/fast-transformers
# Linear Transformer
class FeatureMap(nn.Module):
    """Define the FeatureMap interface."""
    def __init__(self, query_dims):
        super().__init__()
        self.query_dims = query_dims

    def new_feature_map(self, device):
        """Create a new instance of this feature map. In particular, if it is a
        random feature map sample new parameters."""
        raise NotImplementedError()

    def forward_queries(self, x):
        """Encode the queries `x` using this feature map."""
        return self(x)

    def forward_keys(self, x):
        """Encode the keys `x` using this feature map."""
        return self(x)

    def forward(self, x):
        """Encode x using this feature map. For symmetric feature maps it
        suffices to define this function, but for asymmetric feature maps one
        needs to define the `forward_queries` and `forward_keys` functions."""
        raise NotImplementedError()

    @classmethod
    def factory(cls, *args, **kwargs):
        """Return a function that when called with the query dimensions returns
        an instance of this feature map.
        It is inherited by the subclasses so it is available in all feature
        maps.
        """
        def inner(query_dims):
            return cls(query_dims, *args, **kwargs)
        return inner


class ActivationFunctionFeatureMap(FeatureMap):
    """Define a feature map that is simply an element-wise activation
    function."""
    def __init__(self, query_dims, activation_function):
        super().__init__(query_dims)
        self.activation_function = activation_function

    def new_feature_map(self, device):
        return

    def forward(self, x):
        return self.activation_function(x)


class LinearAttention(nn.Module):
    """Implement unmasked attention using dot product of feature maps in
    O(N D^2) complexity.
    Given the queries, keys and values as Q, K, V instead of computing
        V' = softmax(Q.mm(K.t()), dim=-1).mm(V),
    we make use of a feature map function Φ(.) and perform the following
    computation
        V' = normalize(Φ(Q).mm(Φ(K).t())).mm(V).
    The above can be computed in O(N D^2) complexity where D is the
    dimensionality of Q, K and V and N is the sequence length. Depending on the
    feature map, however, the complexity of the attention might be limited.
    Arguments
    ---------
        feature_map: callable, a callable that applies the feature map to the
                     last dimension of a tensor (default: elu(x)+1)
        eps: float, a small number to ensure the numerical stability of the
             denominator (default: 1e-6)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, query_dimensions, feature_map=None, eps=1e-6,
                 event_dispatcher=""):
        super(LinearAttention, self).__init__()
        elu_feature_map = ActivationFunctionFeatureMap.factory(
            lambda x: torch.nn.functional.elu(x) + 1
        )
        self.feature_map = (
            feature_map(query_dimensions) if feature_map else
            elu_feature_map(query_dimensions)
        )
        self.eps = eps
        # self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, queries, keys, values, attn_mask, query_lengths,
                key_lengths):
        # Apply the feature map to the queries and keys
        self.feature_map.new_feature_map(queries.device)
        Q = self.feature_map.forward_queries(queries)
        K = self.feature_map.forward_keys(keys)

        # Apply the key padding mask and make sure that the attn_mask is
        # all_ones
        if not attn_mask.all_ones:
            raise RuntimeError(("LinearAttention does not support arbitrary "
                                "attention masks"))
        K = K * key_lengths.float_matrix[:, :, None, None]

        # Compute the KV matrix, namely the dot product of keys and values so
        # that we never explicitly compute the attention matrix and thus
        # decrease the complexity
        KV = torch.einsum("nshd,nshm->nhmd", K, values)

        # Compute the normalizer
        Z = 1/(torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1))+self.eps)

        # Finally compute and return the new values
        V = torch.einsum("nlhd,nhmd,nlh->nlhm", Q, KV, Z)

        return V.contiguous()
