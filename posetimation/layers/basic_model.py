#!/usr/bin/python
# -*- coding:utf8 -*-
"""
    Author: Haoming Chen
    E-mail: chenhaomingbob@163.com
    Time: 2020/06/09
    Description:
"""
import torch
import torch.nn as nn
import torchvision.ops as ops

from torch.nn import functional as F
# from thirdparty.deform_conv import ModulatedDeformConv
from torchvision.ops import DeformConv2d

BN_MOMENTUM = 0.1


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False, groups=groups)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, skip_norm=False, act='ReLU'):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, groups=groups)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        assert act in ['ReLU', 'LeakyReLU'], "Not Expectation act function {}".format(act)
        if act == 'ReLU':
            self.act_fun = nn.ReLU(inplace=True)
        elif act == 'LeakyReLU':
            self.act_fun = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.conv2 = conv3x3(planes, planes, stride, groups=groups)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride
        self.skip_norm = skip_norm

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if not self.skip_norm:
            out = self.bn1(out)

        out = self.act_fun(out)

        out = self.conv2(out)
        if not self.skip_norm:
            out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.act_fun(out)

        return out


class Bottleneck(nn.Module):
    """
    From HRNet
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        def _func_factory(conv, bn, relu, has_bn, has_relu):
            def func(x):
                x = conv(x)
                if has_bn:
                    x = bn(x)
                if has_relu:
                    x = relu(x)
                return x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode='nearest'):
        super(Interpolate, self).__init__()
        self.interpolate = F.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = self.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class ChainOfBasicBlocks(nn.Module):
    def __init__(self, input_channel, ouput_channel, kernel_height=None, kernel_width=None, dilation=None, num_blocks=1, groups=1, skip_norm=False, act='ReLU'):
        # def __init__(self, input_channel, ouput_channel, kernel_height, kernel_width, dilation, num_blocks, groups=1):
        super(ChainOfBasicBlocks, self).__init__()
        stride = 1
        if skip_norm:
            downsample = nn.Sequential(nn.Conv2d(input_channel, ouput_channel, kernel_size=1, stride=stride, bias=False, groups=groups))
        else:
            downsample = nn.Sequential(nn.Conv2d(input_channel, ouput_channel, kernel_size=1, stride=stride, bias=False, groups=groups),
                                       nn.BatchNorm2d(ouput_channel, momentum=BN_MOMENTUM))
        layers = []
        layers.append(BasicBlock(input_channel, ouput_channel, stride, downsample, groups, skip_norm=skip_norm, act=act))

        for i in range(1, num_blocks):
            layers.append(BasicBlock(ouput_channel, ouput_channel, stride, downsample=None, groups=groups, skip_norm=skip_norm, act=act))

        # return nn.Sequential(*layers)
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        return self.layers(input)


class DeformableCONV(nn.Module):
    def __init__(self, num_joints, k, dilation):
        super(DeformableCONV, self).__init__()

        self.deform_conv = modulated_deform_conv(num_joints, k, k, dilation, num_joints).cuda()

    def forward(self, x, offsets, mask):
        return self.deform_conv(x, offsets, mask)


def modulated_deform_conv(n_channels, kernel_height, kernel_width, deformable_dilation, deformable_groups):
    assert kernel_height == kernel_width
    conv_offset2d = DeformConv2d(
        n_channels,
        n_channels,
        kernel_height,
        # (kernel_height, kernel_width),
        stride=1,
        padding=int(kernel_height / 2) * deformable_dilation,
        dilation=deformable_dilation,
        groups=deformable_groups,
        # deformable_groups=deformable_groups
    )

    # conv_offset2d = ModulatedDeformConv(
    #     n_channels,
    #     n_channels,
    #     (kernel_height, kernel_width),
    #     stride=1,
    #     padding=int(kernel_height / 2) * deformable_dilation,
    #     dilation=deformable_dilation,
    #     deformable_groups=deformable_groups
    # )
    return conv_offset2d


class AdaptBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, outplanes, stride=1,
                 downsample=None, dilation=1, deformable_groups=1, act='ReLU'):
        super(AdaptBlock, self).__init__()
        regular_matrix = torch.tensor([[-1, -1, -1, 0, 0, 0, 1, 1, 1], \
                                       [-1, 0, 1, -1, 0, 1, -1, 0, 1]])
        self.register_buffer('regular_matrix', regular_matrix.float())
        self.downsample = downsample

        self.transform_matrix_conv = nn.Conv2d(inplanes, 4, 3, 1, 1, bias=True)
        self.translation_conv = nn.Conv2d(inplanes, 2, 3, 1, 1, bias=True)
        self.adapt_conv = ops.DeformConv2d(inplanes, outplanes, kernel_size=3, stride=stride, \
                                           padding=dilation, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(outplanes, momentum=BN_MOMENTUM)
        assert act in ['ReLU', 'LeakyReLU'], "Not Expectation act function {}".format(act)
        if act == 'ReLU':
            self.relu = nn.ReLU(inplace=True)
        elif act == 'LeakyReLU':
            self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        N, _, H, W = x.shape
        transform_matrix = self.transform_matrix_conv(x)  # C: 36 -> 4
        transform_matrix = transform_matrix.permute(0, 2, 3, 1).reshape((N * H * W, 2, 2))  #
        offset = torch.matmul(transform_matrix, self.regular_matrix)  # N*H*W, 2, 2  x  2, 9
        offset = offset - self.regular_matrix
        offset = offset.transpose(1, 2).reshape((N, H, W, 2 * 3 * 3)).permute(0, 3, 1, 2)

        translation = self.translation_conv(x)
        offset[:, 0::2, :, :] += translation[:, 0:1, :, :]
        offset[:, 1::2, :, :] += translation[:, 1:2, :, :]

        out = self.adapt_conv(x, offset)
        out = self.bn(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class AdaptBlockV2(nn.Module):
    expansion = 1

    def __init__(self, inplanes, outplanes, stride=1,
                 downsample=None, dilation=1, deformable_groups=1, act='ReLU'):
        super(AdaptBlockV2, self).__init__()
        regular_matrix = torch.tensor([[-1, -1, -1, 0, 0, 0, 1, 1, 1], \
                                       [-1, 0, 1, -1, 0, 1, -1, 0, 1]])
        self.register_buffer('regular_matrix', regular_matrix.float())
        self.downsample = downsample
        self.deformable_groups = deformable_groups
        transform_matrix_conv_layers = []
        translation_conv_layers = []
        mask_conv_layers = []
        for i in range(deformable_groups):
            transform_matrix_conv = nn.Conv2d(inplanes, 4, (3, 3), (1, 1), (1, 1), bias=True)
            translation_conv = nn.Conv2d(inplanes, 2, (3, 3), (1, 1), (1, 1), bias=True)
            mask_conv = nn.Conv2d(inplanes,  3 * 3, (3, 3), (1, 1), (1, 1))
            transform_matrix_conv_layers.append(transform_matrix_conv)
            translation_conv_layers.append(translation_conv)
            mask_conv_layers.append(mask_conv)

        self.transform_matrix_conv_layers = nn.ModuleList(transform_matrix_conv_layers)
        self.translation_conv_layers = nn.ModuleList(translation_conv_layers)
        self.mask_conv_layers = nn.ModuleList(mask_conv_layers)

        self.adapt_conv = ops.DeformConv2d(inplanes, outplanes, kernel_size=3, stride=stride, \
                                           padding=dilation, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(outplanes, momentum=BN_MOMENTUM)
        assert act in ['ReLU', 'LeakyReLU'], "Not Expectation act function {}".format(act)
        if act == 'ReLU':
            self.relu = nn.ReLU(inplace=True)
        elif act == 'LeakyReLU':
            self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        N, _, H, W = x.shape
        offsets = []
        masks = []

        for i in range(self.deformable_groups):
            transform_matrix = self.transform_matrix_conv_layers[i](x)
            transform_matrix = transform_matrix.permute(0, 2, 3, 1).reshape((N * H * W, 2, 2))
            offset = torch.matmul(transform_matrix, self.regular_matrix)
            offset = offset - self.regular_matrix
            offset = offset.transpose(1, 2).reshape((N, H, W, 18)).permute(0, 3, 1, 2)
            translation = self.translation_conv_layers[i](x)
            offset[:, 0::2, :, :] += translation[:, 0:1, :, :]
            offset[:, 1::2, :, :] += translation[:, 1:2, :, :]
            mask = self.mask_conv_layers[i](x)
            offsets.append(offset)
            masks.append(mask)
        offsets = torch.cat(offsets, dim=1)
        masks = torch.cat(masks, dim=1)

        out = self.adapt_conv(x, offsets, masks)
        out = self.bn(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class DeformBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, outplanes, stride=1,
                 downsample=None, dilation=1, deformable_groups=1):
        super(DeformBlock, self).__init__()
        # regular_matrix = torch.tensor([[-1, -1, -1, 0, 0, 0, 1, 1, 1], \
        #                                [-1, 0, 1, -1, 0, 1, -1, 0, 1]])
        # self.register_buffer('regular_matrix', regular_matrix.float())
        self.downsample = downsample
        # self.transform_matrix_conv = nn.Conv2d(inplanes, 4, 3, 1, 1, bias=True)
        # self.translation_conv = nn.Conv2d(inplanes, 2, 3, 1, 1, bias=True)

        self.offset_conv = nn.Conv2d(inplanes, 2 * 3 * 3, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.adapt_conv = ops.DeformConv2d(inplanes, outplanes, kernel_size=3, stride=stride, \
                                           padding=dilation, dilation=dilation, bias=False, groups=deformable_groups)

        self.bn = nn.BatchNorm2d(outplanes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        residual = x

        # N, _, H, W = x.shape
        # transform_matrix = self.transform_matrix_conv(x)
        # transform_matrix = transform_matrix.permute(0, 2, 3, 1).reshape((N * H * W, 2, 2))
        # offset = torch.matmul(transform_matrix, self.regular_matrix)
        # offset = offset - self.regular_matrix
        # offset = offset.transpose(1, 2).reshape((N, H, W, 18)).permute(0, 3, 1, 2)
        #
        # translation = self.translation_conv(x)
        # offset[:, 0::2, :, :] += translation[:, 0:1, :, :]
        # offset[:, 1::2, :, :] += translation[:, 1:2, :, :]
        offset = self.offset_conv(x)
        out = self.adapt_conv(x, offset)
        out = self.bn(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
