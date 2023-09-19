"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the OLTR project which
notice below and in LICENSE in the root directory of
this source tree.

Copyright (c) 2019, Zhongqi Miao
All rights reserved.
"""

import math
import torch.nn as nn
import torch.nn.functional as F
from layers.ModulatedAttLayer import ModulatedAttLayer
from models.Aux_Layers3 import *


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

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

class My_Layers(nn.Module):
    def __init__(self, block, planes, blocks, stride=1, inplanes=128):
        super(My_Layers, self).__init__()
        downsample = None
        self.inplanes = inplanes
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        self.layer1 = nn.Sequential(block(self.inplanes, planes, stride, downsample))
        # layers.append()
        self.inplanes = planes * block.expansion
        for i in range(1, blocks // 2):
            self.layer1.add_module('block_{0}'.format(i), block(self.inplanes, planes))

        self.layer2 = nn.Sequential(block(self.inplanes, planes))

        for i in range(blocks // 2 + 1, blocks):
            self.layer2.add_module('block_{0}'.format(i), block(self.inplanes, planes))

    def get_inplanes(self, ):
        return self.inplanes

    def forward(self, x):
        x = self.layer1(x)
        out = self.layer2(x)

        return out, x

class ResNet(nn.Module):

    def __init__(self, block, layers, use_modulatedatt=False, groups=32, width_per_group=4, use_fc=False, dropout=None,
                 num_classes=1000, normalized=False, scale=30):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = My_Layers(block, 64, layers[0], inplanes=self.inplanes)
        self.inplanes = self.layer1.get_inplanes()
        self.layer2 = My_Layers(block, 128, layers[1], stride=2, inplanes=self.inplanes)
        self.inplanes = self.layer2.get_inplanes()
        self.layer3 = My_Layers(block, 256, layers[2], stride=2, inplanes=self.inplanes)
        self.inplanes = self.layer3.get_inplanes()
        self.layer4 = My_Layers(block, 512, layers[3], stride=2, inplanes=self.inplanes)
        self.inplanes = self.layer4.get_inplanes()
        self.avgpool = nn.AvgPool2d(7, stride=1)

        self.aux_layer2 = Aux_Layer2(block.expansion * 128, num_classes, base_width=width_per_group, groups=groups,
                                     normalized=normalized, scale=scale,
                                     dropout=dropout)
        '''
        self.aux_layer3 = Aux_Layer2(block.expansion * 256, num_classes, base_width=width_per_group, groups=groups,
                                     normalized=normalized, scale=scale,
                                     dropout=dropout)
        '''
        self.aux_layer4 = Aux_Layer2(block.expansion * 256, num_classes, base_width=width_per_group, groups=groups,
                                     normalized=normalized, scale=scale,
                                     dropout=dropout)
        '''
        self.aux_layer5 = Aux_Layer2(block.expansion * 512, num_classes, base_width=width_per_group, groups=groups,
                                     normalized=normalized, scale=scale,
                                     dropout=dropout)
        '''
        # self.aux_layer3 = Aux_Layer2(block.expansion * 256, num_classes, base_width=width_per_group, groups=groups, normalized=normalized, scale=scale,
        #                             dropout=dropout)

        self.use_fc = use_fc
        self.use_dropout = True if dropout else False

        if self.use_fc:
            print('Using fc.')
            self.fc_add = nn.Linear(512 * block.expansion, 512)

        if self.use_dropout:
            print('Using dropout.')
            self.dropout = nn.Dropout(p=dropout)

        self.use_modulatedatt = use_modulatedatt
        if self.use_modulatedatt:
            print('Using self attention.')
            self.modulatedatt = ModulatedAttLayer(in_channels=512 * block.expansion)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, *args):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x, _ = self.layer1(x)
        x, logits1 = self.layer2(x)
        logits2 = self.aux_layer2(x)

        x, logits3 = self.layer3(x)
        #logits3 = self.aux_layer3(logits3)
        logits4 = self.aux_layer4(x)

        x, logits5 = self.layer4(x)

        if self.use_modulatedatt:
            x, feature_maps = self.modulatedatt(x)
        else:
            feature_maps = None

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)

        if self.use_fc:
            x = F.relu(self.fc_add(x))

        if self.use_dropout:
            x = self.dropout(x)

        outputs = [x] + [logits2] + [logits4]
        return outputs