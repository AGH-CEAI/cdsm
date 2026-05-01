# CDSM © 2026 by AGH University of Krakow is licensed under CC BY-NC-ND 4.0. 
# To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-nd/4.0/ 

import sys

sys.setrecursionlimit(5000)

import torch
from torch import nn


def make_norm_layer(norm, planes):
    if norm == 'batch':
        return nn.BatchNorm2d(planes, momentum=0.9997, eps=4e-5)
    elif norm == 'layer':
        return nn.GroupNorm(1, planes, eps=4e-5)
    elif norm == 'instance':
        return nn.GroupNorm(planes, planes, eps=4e-5)
    elif 'group' in norm:
        div = int(norm.split('_')[1]) if len(norm.split('_')) > 1 else 2
        groups = max(1, planes // div)
        return nn.GroupNorm(groups, planes, eps=4e-5)
    else:
        raise ValueError('normalization should be one of \'batch\', \'instance\', \'layer\' or \'group_X\'')


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1, norm='batch'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            bias=False,
            dilation=dilation,
        )
        self.norm1 = make_norm_layer(norm, planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=dilation,
            bias=False,
            dilation=dilation,
        )
        self.norm2 = make_norm_layer(norm, planes)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out += residual
        out = self.relu(out)

        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual, norm='batch'):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            stride=1,
            bias=False,
            padding=(kernel_size - 1) // 2,
        )
        self.norm = make_norm_layer(norm, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.norm(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):
    def __init__(
        self,
        levels,
        block,
        in_channels,
        out_channels,
        stride=1,
        level_root=False,
        root_dim=0,
        root_kernel_size=1,
        dilation=1,
        root_residual=False,
        norm='batch'
    ):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride, dilation=dilation, norm=norm)
            self.tree2 = block(out_channels, out_channels, 1, dilation=dilation, norm=norm)
        else:
            self.tree1 = Tree(
                levels - 1,
                block,
                in_channels,
                out_channels,
                stride,
                root_dim=0,
                root_kernel_size=root_kernel_size,
                dilation=dilation,
                root_residual=root_residual,
                norm=norm
            )
            self.tree2 = Tree(
                levels - 1,
                block,
                out_channels,
                out_channels,
                root_dim=root_dim + out_channels,
                root_kernel_size=root_kernel_size,
                dilation=dilation,
                root_residual=root_residual,
                norm=norm
            )
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size, root_residual, norm=norm)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, bias=False
                ),
                make_norm_layer(norm, out_channels),
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(nn.Module):
    def __init__(
        self,
        levels,
        output_levels,
        strides,
        channels,
        norm,
        block=BasicBlock,
        residual_root=False,
    ):
        super(DLA, self).__init__()
        self.levels = levels
        self.output_levels = output_levels
        self.channels = channels
        # self.base_layer = nn.Sequential(
        #     nn.Conv2d(channels[0], channels[0], kernel_size=7, stride=1, padding=3, bias=False),
        #     make_norm_layer(norm, channels[0]),
        #     nn.ReLU(inplace=True),
        # )
        self.level0 = self._make_conv_level(channels[0], channels[0], levels[0], stride=strides[0], norm=norm)
        self.level1 = self._make_conv_level(channels[0], channels[1], levels[1], stride=strides[1], norm=norm)
        if levels[2]:
            self.level2 = Tree(
                levels[2],
                block,
                channels[1],
                channels[2],
                strides[2],
                level_root=False,
                root_residual=residual_root,
            )
        if levels[3]:
            self.level3 = Tree(
                levels[3],
                block,
                channels[2],
                channels[3],
                strides[3],
                level_root=True,
                root_residual=residual_root,
            )
        if levels[4]:
            self.level4 = Tree(
                levels[4],
                block,
                channels[3],
                channels[4],
                strides[4],
                level_root=True,
                root_residual=residual_root,
            )
        if levels[5]:
            self.level5 = Tree(
                levels[5],
                block,
                channels[4],
                channels[5],
                strides[5],
                level_root=True,
                root_residual=residual_root,
            )

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_level(self, block, inplanes, planes, blocks, stride=1, norm='batch'):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.MaxPool2d(stride, stride=stride),
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False),
                make_norm_layer(norm, planes),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1, norm='batch'):
        modules = []
        for i in range(convs):
            modules.extend(
                [
                    nn.Conv2d(
                        inplanes,
                        planes,
                        kernel_size=3,
                        stride=stride if i == 0 else 1,
                        padding=dilation,
                        bias=False,
                        dilation=dilation,
                    ),
                    make_norm_layer(norm, planes),
                    nn.ReLU(inplace=True),
                ]
            )
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        y = []
        # x = self.base_layer(x)
        for i in range(len(self.levels)):
            if self.levels[i] is None:
                break
            x = getattr(self, "level{}".format(i))(x)
            if i in self.output_levels:
                y.append(x)
        return y


def dla34(levels=None, outputs=None, strides=None, channels=None, norm=None, **kwargs):
    return DLA(levels=levels, output_levels=outputs, strides=strides,
               channels=channels, norm=norm, block=BasicBlock, **kwargs)
