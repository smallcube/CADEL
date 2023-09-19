import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
from torch.autograd import Variable
import sys, os
import numpy as np
import random
from models.CosNormClassifier import CosNorm_Classifier
from models.NormedLinear import NormedLinear


class Aux_Layer1(nn.Module):
    def __init__(self, inplanes, num_classes=1000, reduction=8, normalized=False, scale=30, dropout=None):
        super(Aux_Layer1, self).__init__()
        self.use_dropout = True if dropout else False
        if self.use_dropout:
            print('Using dropout.')
            self.dropout = nn.Dropout(p=dropout)

        self.preBlock = nn.Sequential(
            nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, groups=16),
            nn.BatchNorm2d(2 * inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * inplanes, 2 * inplanes, kernel_size=3, groups=16),
            nn.BatchNorm2d(2 * inplanes),
            nn.ReLU(inplace=True)
        )

        channel = 2 * inplanes
        self.seBlock = nn.Sequential(
            nn.Linear(channel, (int)(channel / reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear((int)(channel / reduction), channel, bias=False),
            nn.Sigmoid()
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        if normalized:
            self.FC = CosNorm_Classifier(channel, num_classes, scale=scale)
            #self.FC = NormedLinear(channel, num_classes, scale=scale)
        else:
            self.FC = nn.Linear(channel, num_classes)

    def forward(self, x):
        # x1 = self.conv(x)
        x2 = self.preBlock(x)

        b, c, _, _ = x2.size()
        y = self.avg(x2).view(b, c)
        y = self.seBlock(y).view(b, c, 1, 1)
        out = x2 * y.expand_as(x2)

        out = self.avg(out)
        out = out.view(out.size(0), -1)
        if self.training:
            out = F.dropout(out)
        out = self.FC(out)
        if self.use_dropout:
            out = self.dropout(out)

        return out


class Aux_Layer2(nn.Module):
    def __init__(self, inplanes, num_classes=1000, reduction=8, normalized=False, scale=30, dropout=None):
        super(Aux_Layer2, self).__init__()
        self.use_dropout = True if dropout else False
        if self.use_dropout:
            print('Using dropout.')
            self.dropout = nn.Dropout(p=dropout)

        self.preBlock = nn.Sequential(
            nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, groups=16),
            nn.BatchNorm2d(2 * inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * inplanes, 2 * inplanes, kernel_size=3, groups=16),
            nn.BatchNorm2d(2 * inplanes),
            nn.ReLU(inplace=True)
        )

        channel = 2 * inplanes
        self.seBlock = nn.Sequential(
            nn.Linear(channel, (int)(channel / reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear((int)(channel / reduction), channel, bias=False),
            nn.Sigmoid()
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        if normalized:
            self.FC = CosNorm_Classifier(channel, num_classes, scale=scale)
            #self.FC = NormedLinear(channel, num_classes, scale=scale)
        else:
            self.FC = nn.Linear(channel, num_classes)

    def forward(self, x):
        x2 = self.preBlock(x)

        b, c, _, _ = x2.size()
        y = self.avg(x2).view(b, c)
        y = self.seBlock(y).view(b, c, 1, 1)
        out = x2 * y.expand_as(x2)

        out = self.avg(out)
        out = out.view(out.size(0), -1)
        if self.training:
            out = F.dropout(out)

        out = self.FC(out)
        if self.use_dropout:
            out = self.dropout(out)

        return out


class Aux_Layer3(nn.Module):
    def __init__(self, inplanes, num_classes=1000, reduction=8, normalized=False, dropout=None):
        super(Aux_Layer3, self).__init__()
        self.use_dropout = True if dropout else False
        if self.use_dropout:
            print('Using dropout.')
            self.dropout = nn.Dropout(p=dropout)

        self.preBlock = nn.Sequential(
            nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, padding=0),
            nn.BatchNorm2d(2 * inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * inplanes, 2 * inplanes, kernel_size=1, groups=16),
            nn.BatchNorm2d(2 * inplanes),
            nn.ReLU(inplace=True)
        )

        channel = 2 * inplanes
        self.seBlock = nn.Sequential(
            nn.Linear(channel, (int)(channel / reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear((int)(channel / reduction), channel, bias=False),
            nn.Sigmoid()
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        if normalized:
            self.FC = CosNorm_Classifier(channel, num_classes)
        else:
            self.FC = nn.Linear(channel, num_classes)

    def forward(self, x):
        x2 = self.preBlock(x)

        b, c, _, _ = x2.size()
        y = self.avg(x2).view(b, c)
        y = self.seBlock(y).view(b, c, 1, 1)
        out = x2 * y.expand_as(x2)

        out = self.avg(out)
        out = out.view(out.size(0), -1)
        if self.training:
            out = F.dropout(out)
        out = self.FC(out)
        if self.use_dropout:
            out = self.dropout(out)

        return out


class Aux_Layer4(nn.Module):
    def __init__(self, inplanes, num_classes=1000, reduction=8, normalized=False, dropout=None):
        super(Aux_Layer4, self).__init__()
        self.use_dropout = True if dropout else False
        if self.use_dropout:
            print('Using dropout.')
            self.dropout = nn.Dropout(p=dropout)

        self.preBlock = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            nn.Conv2d(inplanes, 2 * inplanes, kernel_size=1, groups=16),
            nn.BatchNorm2d(2 * inplanes),
            nn.ReLU(inplace=True)
        )

        channel = 2 * inplanes
        self.seBlock = nn.Sequential(
            nn.Linear(channel, (int)(channel / reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear((int)(channel / reduction), channel, bias=False),
            nn.Sigmoid()
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        if normalized:
            self.FC = CosNorm_Classifier(channel, num_classes)
        else:
            self.FC = nn.Linear(channel, num_classes)

    def forward(self, x):
        x2 = self.preBlock(x)

        b, c, _, _ = x2.size()
        y = self.avg(x2).view(b, c)
        y = self.seBlock(y).view(b, c, 1, 1)
        out = x2 * y.expand_as(x2)

        out = self.avg(out)
        out = out.view(out.size(0), -1)
        if self.training:
            out = F.dropout(out)
        out = self.FC(out)

        if self.use_dropout:
            out = self.dropout(out)

        return out
