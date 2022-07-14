#!/usr/bin/python
# -*- coding:utf8 -*-
"""
    Author: Haoming Chen
    E-mail: chenhaomingbob@163.com
    Time: 2020/12/29
    Description:
"""
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
BN_MOMENTUM = 0.1
class conv_bn_relu(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding, dilation,
                 has_bias=True, has_bn=True, has_relu=True, efficient=False, groups=1, act='ReLU'):
        super(conv_bn_relu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        self.has_relu = has_relu
        self.efficient = efficient

        self.bn = None
        self.relu = None
        if self.has_bn:
            self.bn = nn.BatchNorm2d(out_planes, momentum=BN_MOMENTUM)
        assert act in ['ReLU', 'LeakyReLU', 'SiLU'], "Not Expectation act function {}".format(act)
        if self.has_relu:
            if act == 'ReLU':
                self.relu = nn.ReLU(inplace=True)
            elif act == 'LeakyReLU':
                self.relu = nn.LeakyReLU(inplace=True)
            elif act == 'SiLU':
                # SiLU (Swish) activation function
                if hasattr(nn, 'SiLU'):
                    self.relu = nn.SiLU(inplace=True)
                else:
                    class SiLU(nn.Module):
                        def forward(self, x):
                            return x * torch.sigmoid(x)

                    self.relu = SiLU()

        # self.funs = []
        # self.funs.append(self.conv)
        # if self.has_bn:
        #     self.funs.append(self.bn)
        # if self.has_relu:
        #     self.funs.append(self.relu)
        # self.funs = nn.Sequential(self.funs)

    def forward(self, x):
        def _func_factory(conv, bn, relu, has_bn, has_relu):
            def func(x):
                x = conv(x)
                if has_bn:
                    x = bn(x)
                if has_relu:
                    x = relu(x)
                return x

            return func

        func = _func_factory(self.conv, self.bn, self.relu, self.has_bn, self.has_relu)

        if self.efficient:
            x = checkpoint(func, x)
        else:
            x = func(x)
        # x = self.funs(x)

        return x


class ChainOfBasicDilationBlocks(nn.Module):
    def __init__(self, input_channel, ouput_channel, padding=1, dilation=1, num_blocks=1, groups=1):
        super(ChainOfBasicDilationBlocks, self).__init__()
        stride = 1
        downsample = nn.Sequential(nn.Conv2d(input_channel, ouput_channel, kernel_size=1, stride=stride, bias=False, groups=groups),
                                   nn.BatchNorm2d(ouput_channel, momentum=BN_MOMENTUM))
        layers = []
        layers.append(BasicDilationBlock(input_channel, ouput_channel, stride, padding, dilation, downsample, groups))

        for i in range(1, num_blocks):
            layers.append(BasicDilationBlock(ouput_channel, ouput_channel, stride, padding, dilation, downsample=None, groups=groups))

        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        return self.layers(input)


class BasicDilationBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, padding=1, dilation=1, downsample=None, groups=1):
        super(BasicDilationBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, (3, 3), stride, padding, dilation, groups=groups)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.act_fun = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, (3, 3), stride, padding, dilation, groups=groups)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act_fun(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.act_fun(out)

        return out
