# ===========================================================================
# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===========================================================================
# ----------------------------------------------------------------------------------------------------
# Image Super-Resolution with Cross-Scale Non-Local Attention and Exhaustive Self-Exemplars Mining
# Official GitHub: https://github.com/Lornatang/CSNLN-PyTorch
# ----------------------------------------------------------------------------------------------------
import torch
from basicsr.utils.registry import ARCH_REGISTRY
from torch import nn
from torch.nn import functional as F


# Reference from `https://github.com/SHI-Labs/Cross-Scale-Non-Local-Attention/blob/master/src/model/utils/tools.py`
def same_padding(images, kernel_size: list, stride: list, rates: list):
    assert len(images.size()) == 4

    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + stride[0] - 1) // stride[0]
    out_cols = (cols + stride[1] - 1) // stride[1]
    effective_k_row = (kernel_size[0] - 1) * rates[0] + 1
    effective_k_col = (kernel_size[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows - 1) * stride[0] + effective_k_row - rows)
    padding_cols = max(0, (out_cols - 1) * stride[1] + effective_k_col - cols)

    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ZeroPad2d(paddings)(images)

    return images


# Reference from `https://github.com/SHI-Labs/Cross-Scale-Non-Local-Attention/blob/master/src/model/utils/tools.py`
def reduce_sum(x, axis=None, keepdim=False):
    if axis is None:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.sum(x, dim=i, keepdim=keepdim)
    return x


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, stride: int, padding: int) -> None:
        super(ResidualConvBlock, self).__init__()
        self.rcb = nn.Sequential(
            nn.Conv2d(channels, channels, (kernel_size, kernel_size), (stride, stride), (padding, padding)),
            nn.PReLU(),
            nn.Conv2d(channels, channels, (kernel_size, kernel_size), (stride, stride), (padding, padding)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.rcb(x)

        out = torch.add(out, identity)

        return out


# Reference from `https://github.com/SHI-Labs/Cross-Scale-Non-Local-Attention/blob/master/src/model/utils/tools.py`
def extract_image_patches(images, kernel_size: list, stride: list, rates: list):
    assert len(images.size()) == 4

    images = same_padding(images, kernel_size, stride, rates)
    # [N, C*k*k, L], L is the total number of such blocks
    patches = torch.nn.Unfold(kernel_size=kernel_size, dilation=rates, padding=(0, 0), stride=stride)(images)

    return patches


# Reference from `https://github.com/SHI-Labs/Cross-Scale-Non-Local-Attention/blob/master/src/model/utils/tools.py`
class CrossScaleAttention(nn.Module):
    def __init__(self, channels: int, reduction: int, kernel_size: int, stride: int, scale: int,
                 softmax_scale: int = 10) -> None:
        super(CrossScaleAttention, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

        self.scale = scale
        self.softmax_scale = softmax_scale

        self.register_buffer("escape", torch.FloatTensor([1e-4]))

        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, (1, 1), (1, 1), (0, 0)),
            nn.PReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, (1, 1), (1, 1), (0, 0)),
            nn.PReLU(),
        )
        self.conv_assembly = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 1), (1, 1), (0, 0)),
            nn.PReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract embedded features
        assembly = self.conv_assembly(x)

        conv1 = self.conv1(x)
        conv2 = self.conv2(F.interpolate(x, scale_factor=1. / self.scale, mode="bilinear"))

        assembly_shape = list(assembly.size())

        conv1_weight_groups = torch.split(conv1, 1, dim=0)
        # kernel size on input for matching
        kernel = self.kernel_size * self.scale

        # raw_w is extracted for reconstruction. [N, C*k*k, L]
        assembly_weight = extract_image_patches(assembly,
                                                kernel_size=[kernel, kernel],
                                                stride=[self.stride * self.scale, self.stride * self.scale],
                                                rates=[1, 1])
        # assembly_weight_shape: [N, C, k, k, L]
        assembly_weight = assembly_weight.view(assembly_shape[0], assembly_shape[1], kernel, kernel, -1)
        # assembly_weight_shape: [N, L, C, k, k]
        assembly_weight = assembly_weight.permute(0, 4, 1, 2, 3)
        assembly_weight_groups = torch.split(assembly_weight, 1, dim=0)

        # downscaling X to form Y for cross-scale matching
        conv2_weight = extract_image_patches(conv2,
                                             kernel_size=[self.kernel_size, self.kernel_size],
                                             stride=[self.stride, self.stride],
                                             rates=[1, 1])
        conv2_shape = list(conv2.size())
        # conv2_weight shape: [N, C, k, k, L]
        conv2_weight = conv2_weight.view(conv2_shape[0], conv2_shape[1], self.kernel_size, self.kernel_size, -1)
        # conv2_weight shape: [N, L, C, k, k]
        conv2_weight = conv2_weight.permute(0, 4, 1, 2, 3)
        conv2_weight_groups = torch.split(conv2_weight, 1, dim=0)

        outs = []
        softmax_scale = self.softmax_scale

        for conv1_weight_group, conv2_weight_group, assembly_weight_group in zip(conv1_weight_groups,
                                                                                 conv2_weight_groups,
                                                                                 assembly_weight_groups):
            # normalize
            conv2_weight_group = conv2_weight_group[0]  # [L, C, k, k]
            max_conv2_weight_group = torch.max(
                torch.sqrt(reduce_sum(torch.pow(conv2_weight_group, 2), axis=[1, 2, 3], keepdim=True)), self.escape)
            norm_conv2_weight_group = conv2_weight_group / max_conv2_weight_group

            # Compute correlation map [1*c*H*W]
            conv1_weight_group = same_padding(conv1_weight_group, [self.kernel_size, self.kernel_size], [1, 1],
                                              [1, 1])  # xi:
            # [1, L, H, W] L = conv2_shape[2] * conv2_shape[3]
            out = F.conv2d(conv1_weight_group, norm_conv2_weight_group, stride=(1, 1))

            # (B=1, C=32 * 32, H=32, W=32)
            out = out.view(1, conv2_shape[2] * conv2_shape[3], assembly_shape[2], assembly_shape[3])
            # rescale matching score
            out = F.softmax(out * softmax_scale, dim=1)

            # deconv for reconsturction
            assembly_weight_group = assembly_weight_group[0]
            out = F.conv_transpose2d(out, assembly_weight_group, stride=self.stride * self.scale, padding=self.scale)

            out = out / 6.
            outs.append(out)

        outs = torch.cat(outs, dim=0)

        return outs


# Reference from `https://github.com/SHI-Labs/Cross-Scale-Non-Local-Attention/blob/master/src/model/utils/tools.py`
class NonLocalAttention(nn.Module):
    def __init__(self, channels: int, reduction: int) -> None:
        super(NonLocalAttention, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, (1, 1), (1, 1), (0, 0)),
            nn.PReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, (1, 1), (1, 1), (0, 0)),
            nn.PReLU(),
        )
        self.conv_assembly = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 1), (1, 1), (0, 0)),
            nn.PReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        assembly = self.conv_assembly(x)

        batch_size, channel, height, width = conv1.shape
        conv1 = conv1.permute(0, 2, 3, 1).view((batch_size, height * width, channel))
        conv2 = conv2.view(batch_size, channel, height * width)

        score = torch.matmul(conv1, conv2)
        score = F.softmax(score, dim=2)

        assembly = assembly.view(batch_size, -1, height * width).permute(0, 2, 1)
        out = torch.matmul(score, assembly)
        out = out.permute(0, 2, 1).view(batch_size, -1, height, width)

        return out


# Reference from `https://github.com/SHI-Labs/Cross-Scale-Non-Local-Attention/blob/master/src/model/utils/tools.py`
class MultiSourceProjection(nn.Module):
    def __init__(self, channels: int, kernel_size: list, scale: list) -> None:
        super(MultiSourceProjection, self).__init__()
        if scale == 2 or 4:
            de_kernel_size = 6
            stride = 2
            padding = 2
            upscale_factor = 2
        elif scale == 3:
            de_kernel_size = 9
            stride = 3
            padding = 3
            upscale_factor = 3

        self.cross_scale_attention = CrossScaleAttention(channels=128, reduction=2, kernel_size=3, stride=1,
                                                         scale=upscale_factor)
        self.non_local_attention = NonLocalAttention(channels=128, reduction=2)
        self.upsampling = nn.Sequential(
            nn.ConvTranspose2d(channels, channels, de_kernel_size, (stride, stride), (padding, padding)),
            nn.PReLU(),
        )
        self.encoder = ResidualConvBlock(channels, kernel_size, 1, kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cross_scale_attention = self.cross_scale_attention(x)
        non_local_attention = self.non_local_attention(x)
        upsampling = self.upsampling(non_local_attention)

        diff = self.encoder(cross_scale_attention - upsampling)
        out = upsampling + diff

        return out


class SelfExemplarMining(nn.Module):
    def __init__(self, channels: int, kernel_size: list, scale: list):
        super(SelfExemplarMining, self).__init__()
        self.scale = scale

        if scale == 2 or 4:
            stride_kernel_size = 6
            stride = 2
            padding = 2
        elif scale == 3:
            stride_kernel_size = 9
            stride = 3
            padding = 3

        self.multi_source_projection1 = MultiSourceProjection(channels, kernel_size=kernel_size, scale=scale)
        self.multi_source_projection2 = MultiSourceProjection(channels, kernel_size=kernel_size, scale=scale)  # !
        self.down_conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, (stride_kernel_size, stride_kernel_size), (stride, stride),
                      (padding, padding)),
            nn.PReLU(),
        )
        self.down_conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, (stride_kernel_size, stride_kernel_size), (stride, stride),
                      (padding, padding)),
            nn.PReLU(),
        )
        # X4
        self.down_conv3 = nn.Sequential(
            nn.Conv2d(channels, channels, (8, 8), (4, 4), (2, 2)),
            nn.PReLU(),
        )
        self.down_conv4 = nn.Sequential(
            nn.Conv2d(channels, channels, (8, 8), (4, 4), (2, 2)),
            nn.PReLU(),
        )

        self.diff_encode1 = nn.Sequential(
            nn.ConvTranspose2d(channels, channels, (stride_kernel_size, stride_kernel_size), (stride, stride),
                               (padding, padding)),
            nn.PReLU(),
        )
        # X4
        self.diff_encode2 = nn.Sequential(
            nn.ConvTranspose2d(channels, channels, (8, 8), (4, 4), (2, 2)),
            nn.PReLU(),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, (kernel_size, kernel_size), (1, 1), (kernel_size // 2, kernel_size // 2)),
            nn.PReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        multi_source_projection1 = self.multi_source_projection1(x)
        down_conv1 = self.down_conv1(multi_source_projection1)
        diff1 = torch.sub(x, down_conv1)
        diff_encode1 = self.diff_encode1(diff1)
        estimate = torch.add(multi_source_projection1, diff_encode1)

        if self.scale == 4:
            multi_source_projection2 = self.multi_source_projection2(estimate)
            down_conv3 = self.down_conv3(multi_source_projection2)
            diff2 = torch.sub(x, down_conv3)
            diff_encode2 = self.diff_encode2(diff2)
            estimate = torch.add(multi_source_projection2, diff_encode2)
            down_conv4 = self.down_conv4(estimate)
            out = self.conv(down_conv4)
        else:
            down_conv2 = self.down_conv2(estimate)
            out = self.conv(down_conv2)

        return out, estimate


@ARCH_REGISTRY.register()
class CSNLN(nn.Module):
    def __init__(self, upscale: int, num_in_ch: int = 3, num_out_ch: int = 3, task: str = 'csr', ):
        super(CSNLN, self).__init__()
        # First layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 128, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(),
            nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(),
        )

        # Self-Exemplars Mining (SEM) Cell
        self.self_exemplar_mining = SelfExemplarMining(channels=128, kernel_size=3, scale=upscale)

        # Final output layer
        self.conv2 = nn.Conv2d(1536, 3, (3, 3), (1, 1), (1, 1))

        self.register_buffer("mean", torch.Tensor([0.4488, 0.4371, 0.4040]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The images by subtracting the mean RGB value of the DIV2K dataset.
        out = x.sub_(self.mean).mul_(255.)

        out = self.conv1(out)
        sems = []
        for _ in range(12):
            out, estimate = self.self_exemplar_mining(out)
            sems.append(estimate)
        out = torch.cat(sems, dim=1)
        out = self.conv2(out)

        out = out.div_(255.).add_(self.mean)

        return out


if __name__ == '__main__':
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    net = CSNLN(upscale=4)
    print(count_parameters(net))

    data = torch.randn(1, 3, 32, 32)
    print(net(data).size())
