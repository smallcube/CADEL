import os
import argparse
import pprint
from data import dataloader
from run_networks_vit_ddp import model
import warnings
import yaml
from utils import source_import, get_value
import torch
import torch.distributed as dist



if __name__=='__main__':

    data_root = {'ImageNet': './data/ImageNet_LT/',
                'Places': './data/Places_LT/',
                'iNaturalist18': './data/iNaturalist18/'}

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='./config/Places_LT/ViT_B_16_224_stage1.yaml', type=str)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--test', default=False, action='store_true')
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
        #config['local_rank'] = int(os.environ['LOCAL_RANK'])

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
    #pprint.pprint(config)

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
    
    local_rank=args.local_rank
    if local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        #device = torch.device("cuda", args.local_rank)
        dist.init_process_group(backend='gloo', rank=local_rank)

    if local_rank != -1:
        data = {x: dataloader.load_data_distributed(data_root=data_root[dataset.rstrip('_LT')],
                                                    dataset=dataset, phase=split2phase(x),
                                                    batch_size=training_opt['batch_size'],
                                                    num_workers=training_opt['num_workers'],
                                                    image_size=training_opt['image_size'])
                for x in splits}
        
    else:
        data = {x: dataloader.load_data(data_root=data_root[dataset.rstrip('_LT')],
                                        dataset=dataset, phase=split2phase(x),
                                        batch_size=training_opt['batch_size'],
                                        sampler_dic=sampler_dic,
                                        num_workers=training_opt['num_workers'],
                                        image_size=training_opt['image_size'])
                for x in splits}


    training_model = model(config, data, test=False)

    training_model.train()
            
    print('ALL COMPLETED.')
