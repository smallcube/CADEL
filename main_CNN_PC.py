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

import os
import argparse
import pprint
from data import dataloader
from run_networks_CNN_DDP import model
import warnings
import yaml
from utils import source_import, get_value

if __name__ == '__main__':

    data_root = {'ImageNet': './data/ImageNet_LT/',
                 'Places': './data/Places_LT/',
                 'iNaturalist18': './data/iNaturalist18'}

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default="./config/iNaturalist18/resnet50_stage2.yaml", type=str)

    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--test_open', default=False, action='store_true')
    parser.add_argument('--output_logits', default=False)
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--save_feat', type=str, default='')

    # KNN testing parameters
    parser.add_argument('--knn', default=False, action='store_true')
    parser.add_argument('--feat_type', type=str, default='cl2n')
    parser.add_argument('--dist_type', type=str, default='l2')

    # Learnable tau
    parser.add_argument('--val_as_train', default=False, action='store_true')
    parser.add_argument('--local_rank', default=os.getenv('LOCAL_RANK', -1), type=int)

    args = parser.parse_args()


    def update(config, args):
        # Change parameters
        config['model_dir'] = get_value(config['model_dir'], args.model_dir)
        config['training_opt']['batch_size'] = \
            get_value(config['training_opt']['batch_size'], args.batch_size)
        
        config['local_rank'] = args.local_rank
        # Testing with KNN
        if args.knn and args.test:
            training_opt = config['training_opt']
            classifier_param = {
                'feat_dim': training_opt['feature_dim'],
                'num_classes': training_opt['num_classes'],
                'feat_type': args.feat_type,
                'dist_type': args.dist_type,
                'log_dir': training_opt['log_dir']}
            classifier = {
                'def_file': './models/KNNClassifier.py',
                'params': classifier_param,
                'optim_params': config['networks']['classifier']['optim_params']}
            config['networks']['classifier'] = classifier

        return config


    # ============================================================================
    # LOAD CONFIGURATIONS
    with open(args.cfg) as f:
        config = yaml.safe_load(f)
    config = update(config, args)

    test_mode = args.test
    test_open = args.test_open
    if test_open:
        test_mode = True
    output_logits = args.output_logits
    training_opt = config['training_opt']
    #relatin_opt = config['memory']
    dataset = training_opt['dataset']

    if not os.path.isdir(training_opt['log_dir']):
        os.makedirs(training_opt['log_dir'])

    print('Loading dataset from: %s' % data_root[dataset.rstrip('_LT')])
    pprint.pprint(config)


    def split2phase(split):
        if split == 'train' and args.val_as_train:
            return 'train_val'
        else:
            return split


    sampler_defs = training_opt['sampler']
    if sampler_defs:
        if sampler_defs['type'] == 'ClassAwareSampler':
            sampler_dic = {
                'sampler': source_import(sampler_defs['def_file']).get_sampler(),
                'params': {'num_samples_cls': sampler_defs['num_samples_cls']}
            }
        elif sampler_defs['type'] in ['MixedPrioritizedSampler',
                                        'ClassPrioritySampler']:
            sampler_dic = {
                'sampler': source_import(sampler_defs['def_file']).get_sampler(),
                'params': {k: v for k, v in sampler_defs.items() \
                            if k not in ['type', 'def_file']}
            }
    else:
        sampler_dic = None

    splits = ['train', 'train_plain', 'val']
    if dataset not in ['iNaturalist18', 'ImageNet']:
        splits.append('test')
    data = {x: dataloader.load_data(data_root=data_root[dataset.rstrip('_LT')],
                                    dataset=dataset, phase=split2phase(x),
                                    batch_size=training_opt['batch_size'],
                                    sampler_dic=sampler_dic,
                                    num_workers=training_opt['num_workers'])
            for x in splits}

    #taos = [0.05, 0.1, 0.2,0.3]
    #taos = [1.6, 1.7, 1.8, 1.9, 2, 2.5, 3]
    taos = [0.9, 1.0,1.1,1.2,1.3, 1.4, 1.5]

    best_tao = 0
    best_acc = 0
    training_model = model(config, data, test=False)

    for tao in taos:
        print('tao=', tao)
        rsl = training_model.eval(phase='val', tao=tao, post_hoc=True)
        if rsl['val_all'] > best_acc:
            best_acc = rsl['val_all']
            best_tao = tao
    print("bset_tao=", best_tao, "   best_acc=", best_acc)
    training_model.eval(phase='test', tao=best_tao, post_hoc=True)


    print('ALL COMPLETED.')
