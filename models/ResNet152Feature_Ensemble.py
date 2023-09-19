"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from models.ResNetFeature_EnsembleV2 import *
from utils import *
from os import path
from torchvision import models
from models.utils import weight_name_alignment
        
def create_model(use_selfatt=False, use_fc=False, dropout=None, checkpoint_dir=None, *args):
    print('Loading Scratch ResNet 152 Feature Model.')
    resnet = ResNet(Bottleneck, [3, 8, 36, 3], use_modulatedatt=use_selfatt, use_fc=use_fc, dropout=dropout)

    if checkpoint_dir is not None:
        resnet152 = torch.load(checkpoint_dir)
        resnet = weight_name_alignment(resnet152, resnet)
    else:
        print('No Pretrained Weights For Feature Model.')
    
    return resnet
