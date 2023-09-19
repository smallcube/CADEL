"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from models.ResNetFeature_Ensemble import *
from utils import *
from os import path
        
def create_model(use_selfatt=False, use_fc=False, dropout=None, stage1_weights=False, dataset=None, log_dir=None, test=False, num_classes=1000, normalized=False, scale=30, *args):
    
    print('Loading Scratch ResNet 50 Feature Model.')
    resnet = ResNet(Bottleneck, [3, 4, 6, 3], use_modulatedatt=use_selfatt, use_fc=use_fc, dropout=dropout, groups=32, width_per_group=4, 
                num_classes=num_classes, normalized=normalized, scale=scale)

    if not test:
        if stage1_weights:
            print('Loading %s Stage 1 ResNet 50 Weights.' % dataset)
            if log_dir is not None:
                weight_dir = log_dir
            else:
                weight_dir = './logs/%s/stage1' % dataset
            print('==> Loading weights from %s' % weight_dir)
            resnet = load_weights(model=resnet,
                                    weights_path=weight_dir)
        else:
            print('No Pretrained Weights For Feature Model.')

    return resnet
