import torch
import math
import torch.nn as nn
from torch.nn.parameter import Parameter

from models.CosNormClassifier import CosNorm_Classifier
import torch.nn.functional as F

import pdb

class Aux_Classifier(nn.Module):
    def __init__(self, in_dims, out_dims=1000, groups=32, reduction=8, normalized=False, scale=30, dropout=None):
        super(Aux_Classifier, self).__init__()
        self.use_dropout = True if dropout else False
        if self.use_dropout:
            print('Using dropout.')
            self.dropout = nn.Dropout(p=dropout)

        #width = int(num_classes * (base_width / 64.)) * groups
        width = 2048

        #mid_channel = inplanes * 4
        mid_channel = 4096
        #print("width=", width, "    inplane=", mid_channel)
        self.preBlock = nn.Sequential(
            nn.Conv2d(in_dims, width, kernel_size=3, padding=1, groups=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, kernel_size=3, padding=0, groups=groups),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, kernel_size=3, padding=0, groups=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, mid_channel, kernel_size=3, padding=0),
            nn.BatchNorm2d(mid_channel),
            # nn.ReLU(inplace=True)
        )

        channel = mid_channel
        self.seBlock = nn.Sequential(
            nn.Linear(channel, (int)(channel / reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear((int)(channel / reduction), channel, bias=False),
            nn.Sigmoid()
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        if normalized:
            self.FC = CosNorm_Classifier(channel, out_dims, scale=scale)
        else:
            self.FC = nn.Linear(channel, out_dims)

    def forward(self, x):
        # x1 = self.conv(x)
        x2 = self.preBlock(x)
        x2 = F.relu(x2)

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


def create_model(in_dims, out_dims=1000, groups=32, reduction=8, normalized=False, scale=30, dropout=None):
    print('Creating Aux_Classifier.')
    return Aux_Classifier(in_dims=in_dims, out_dims=out_dims, groups=groups, 
                            reduction=reduction, normalized=normalized, scale=scale, dropout=dropout)