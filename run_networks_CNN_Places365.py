import os
import copy
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from utils import *
from logger import Logger
import time
import numpy as np
import warnings
import pdb
from models.util.loss import *
from torch.distributions import normal
from models.utils import mixup_data


class model():
    def __init__(self, config, data, test=False):
        if config['local_rank'] == -1:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cuda', config['local_rank'])
        self.config = config
        self.criterions = config['criterions']
        self.training_opt = self.config['training_opt']
        self.memory = self.config['memory']
        self.data = data
        self.test_mode = test
        self.num_gpus = torch.cuda.device_count()
        self.do_shuffle = config['shuffle'] if 'shuffle' in config else False

        # Setup logger
        self.logger = Logger(self.training_opt['log_dir'])

        # Initialize model
        self.init_models()
        self.init_weight()

        # Load pre-trained model parameters
        if 'model_dir' in self.config and self.config['model_dir'] is not None:
            self.load_model(self.config['model_dir'])
        
        if 'pretrain_dir' in self.config and self.config['pretrain_dir'] is not None:
            self.load_pretrain(self.config['pretrain_dir'])

        # Under training mode, initialize training steps, optimizers, schedulers, criterions, and centroids
        if not self.test_mode:

            # If using steps for training, we need to calculate training steps
            # for each epoch based on actual number of training data instead of
            # oversampled data number
            print('Using steps for training.')
            self.training_data_num = len(self.data['train'].dataset)
            self.epoch_steps = int(self.training_data_num \
                                   / self.training_opt['batch_size'])

            # Initialize model optimizer and scheduler
            print('Initializing model optimizer.')
            self.scheduler_params = self.training_opt['scheduler_params']
            self.model_optimizer, \
            self.model_optimizer_scheduler = self.init_optimizers(self.model_optim_params_list)
            #self.init_criterions()
            #if self.memory['init_centroids']:
            #    self.criterions['FeatureLoss'].centroids.data = \
            #        self.centroids_cal(self.data['train_plain'])

            # Set up log file
            self.log_file = os.path.join(self.training_opt['log_dir'], 'log.txt')
            '''
            if os.path.isfile(self.log_file):
                os.remove(self.log_file)
            '''
            self.logger.log_cfg(self.config)
        else:
            if 'KNNClassifier' in self.config['networks']['classifier']['def_file']:
                self.load_model()
                if not self.networks['classifier'].initialized:
                    cfeats = self.get_knncentroids()
                    print('===> Saving features to %s' %
                          os.path.join(self.training_opt['log_dir'], 'cfeats.pkl'))
                    with open(os.path.join(self.training_opt['log_dir'], 'cfeats.pkl'), 'wb') as f:
                        pickle.dump(cfeats, f)
                    self.networks['classifier'].update(cfeats)
            self.log_file = None

    def init_models(self, optimizer=True):
        networks_defs = self.config['networks']
        self.networks = {}
        self.model_optim_params_list = []

        print("Using", torch.cuda.device_count(), "GPUs.")

        for key, val in networks_defs.items():

            # Networks
            def_file = val['def_file']
            # model_args = list(val['params'].values())
            # model_args.append(self.test_mode)
            model_args = val['params']
            # model_args.update({'test': self.test_mode})
            #print("key=", key)

            self.networks[key] = source_import(def_file).create_model(**model_args)

            
            
            if self.config['local_rank'] != -1:
                #print('local_ranklocal_ranklocal_ranklocal_rank=', self.config['local_rank'])
                self.networks[key] = self.networks[key].to(self.device)
                self.networks[key] = nn.parallel.DistributedDataParallel(self.networks[key], 
                                                                         device_ids=[self.config['local_rank']],
                                                                         output_device=self.config['local_rank'])
                
            else:
                self.networks[key] = nn.DataParallel(self.networks[key]).cuda()
            
            
            if 'fix' in val and val['fix']:
                print('Freezing feature weights except for self attention weights (if exist).')
                for param_name, param in self.networks[key].named_parameters():
                    #print(param_name)
                    # Freeze all parameters except self attention parameters
                    if 'fc' not in param_name:
                        param.requires_grad = False
                    # print('  | ', param_name, param.requires_grad)
            # Optimizer list
            optim_params = val['optim_params']
            self.model_optim_params_list.append({'params': self.networks[key].parameters(),
                                                 'lr': optim_params['lr'],
                                                 'momentum': optim_params['momentum'],
                                                 'weight_decay': optim_params['weight_decay']})
        

    def init_criterions(self):
        criterion_defs = self.config['criterions']
        self.criterions = {}
        self.criterion_weights = {}

        for key, val in criterion_defs.items():
            def_file = val['def_file']
            loss_args = list(val['loss_params'].values())

            self.criterions[key] = source_import(def_file).create_loss(*loss_args).cuda()
            self.criterion_weights[key] = val['weight']

            if val['optim_params']:
                print('Initializing criterion optimizer.')
                optim_params = val['optim_params']
                optim_params = [{'params': self.criterions[key].parameters(),
                                 'lr': optim_params['lr'],
                                 'momentum': optim_params['momentum'],
                                 'weight_decay': optim_params['weight_decay']}]
                # Initialize criterion optimizer and scheduler
                self.criterion_optimizer, \
                self.criterion_optimizer_scheduler = self.init_optimizers(optim_params)
            else:
                self.criterion_optimizer = None

    def init_optimizers(self, optim_params):
        optimizer = optim.SGD(optim_params)
        if self.config['coslr']:
            print("===> Using coslr eta_min={}".format(self.config['endlr']))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, self.training_opt['num_epochs'], eta_min=self.config['endlr'])
        else:
            '''
            scheduler = optim.lr_scheduler.StepLR(optimizer,
                                                  step_size=self.scheduler_params['step_size'],
                                                  gamma=self.scheduler_params['gamma'])
            '''

            def lr_lambda(epoch):
                if epoch >= self.training_opt['step2']:
                    lr = self.training_opt['g'] * self.training_opt['g']
                elif epoch >= self.training_opt['step1']:
                    lr = self.training_opt['g']
                else:
                    lr = 1

                warmup_epoch = self.training_opt['warmup_epoch']
                if epoch < warmup_epoch:
                    lr = lr * float(1 + epoch) / warmup_epoch

                return lr

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        # print(optimizer.)
        return optimizer, scheduler

    def init_weight(self):
        #prepare for logits adjustment
        cls_num = np.load(self.training_opt['num_dir'])['num_per_class']
        cls_num = torch.tensor(cls_num).view(1, -1).cuda()
        self.distribution_source = torch.log(cls_num / torch.sum(cls_num)).view(1, -1)
        self.distribution_target = np.log(1.0 / self.training_opt['num_classes'])

    def batch_forward(self, inputs, labels=None, centroids=False, feature_ext=False, phase='train'):
        '''
        This is a general single batch running function.
        '''

        # Calculate Features
        outputs = self.networks['feat_model'](inputs)
        self.outputs = []
        self.outputs += [self.networks['classifier1'](outputs[0])]
        self.outputs += [self.networks['classifier2'](outputs[1])]
        self.outputs += [self.networks['classifier3'](outputs[2])]

    def batch_backward(self):
        # Zero out optimizer gradients
        self.model_optimizer.zero_grad()
        #if self.criterion_optimizer:
        #    self.criterion_optimizer.zero_grad()
        # Back-propagation from loss outputs
        self.loss.backward()
        # Step optimizers
        self.model_optimizer.step()
        #if self.criterion_optimizer:
        #    self.criterion_optimizer.step()


    def shuffle_batch(self, x, y):
        index = torch.randperm(x.size(0))
        x = x[index]
        y = y[index]
        return x, y

    def train(self):
        # When training the network
        print_str = ['Phase: train']
        print_write(print_str, self.log_file)
        time.sleep(0.25)

        print_write(['Do shuffle??? --- ', self.do_shuffle], self.log_file)

        # Initialize best model
        best_model_weights = {}
        best_model_weights['feat_model'] = copy.deepcopy(self.networks['feat_model'].state_dict())
        best_model_weights['classifier1'] = copy.deepcopy(self.networks['classifier1'].state_dict())
        best_model_weights['classifier2'] = copy.deepcopy(self.networks['classifier2'].state_dict())
        best_model_weights['classifier3'] = copy.deepcopy(self.networks['classifier3'].state_dict())
        best_acc = 0.0
        best_epoch = 0
        # best_centroids = self.centroids

        end_epoch = self.training_opt['num_epochs']

        # Loop over epochs
        for epoch in range(1, end_epoch + 1):
            self.epoch = epoch
            for model in self.networks.values():
                model.train()

            if self.config['local_rank'] != -1:
                self.data['train'].sampler.set_epoch(epoch)

            torch.cuda.empty_cache()

            # Iterate over dataset
            total_preds = []
            total_labels = []

            for step, (inputs, labels, indexes) in enumerate(self.data['train']):
                # print(labels)
                # Break when step equal to epoch step
                if step == self.epoch_steps:
                    break
                if self.do_shuffle:
                    inputs, labels = self.shuffle_batch(inputs, labels)
                inputs, labels = inputs.cuda(), labels.cuda()

                y_b = None
                lam = 1.0
                if 'mixup' in self.criterions:
                    mixup_para = self.criterions['mixup']
                    inputs, labels, y_b, lam = mixup_data(inputs, labels, alpha=mixup_para['alpha'])

                # If on training phase, enable gradients
                with torch.set_grad_enabled(True):

                    # If training, forward with loss, and no top 5 accuracy calculation
                    self.batch_forward(inputs, labels,
                                       centroids=self.memory['centroids'],
                                       phase='train')
                    
                    weight1, weight2 = None, None
                    self.loss = 0
                    logits_backup = 0
                    for i in range(0, len(self.outputs)):
                        loss_i, weight1, weight2 = ensemble_loss_v2(pred=self.outputs[i], target=labels, target2=y_b, lam=lam,
                                                                    weight1=weight1, weight2=weight2,
                                                                    bins=self.training_opt['bins'], 
                                                                    gamma=self.training_opt['gamma'],
                                                                    base_weight=self.training_opt['base_weight'],
                                                                    tempture=1.0-1.0*self.epoch/end_epoch)
                        self.loss = self.loss + loss_i
                        logits_backup = logits_backup + self.outputs[i]
                    del weight1, weight2, self.outputs

                    #logits_backup /= len(self.outputs) - 1
                    self.logits_ensemble = logits_backup


                    #self.batch_loss(labels, y_b, lam)
                    self.batch_backward()

                    # Tracking predictions
                    _, preds = torch.max(self.logits_ensemble, 1)
                    total_preds.append(torch2numpy(preds))
                    total_labels.append(torch2numpy(labels))

                    # update the accumulated prob
                    # Output minibatch training results
                    if step % self.training_opt['display_step'] == 0:
                        minibatch_loss_feat = self.loss_feat.item() \
                            if 'FeatureLoss' in self.criterions.keys() else None
                        # minibatch_loss_perf = self.loss_perf.item() \
                        #    if 'PerformanceLoss' in self.criterions else None
                        minibatch_loss_total = self.loss.item()
                        minibatch_acc = mic_acc_cal(preds, labels)

                        lr_current = max([param_group['lr'] for param_group in self.model_optimizer.param_groups])

                        print_str = ['Epoch: [%d/%d]'
                                     % (epoch, self.training_opt['num_epochs']),
                                     'Step: %5d'
                                     % (step),
                                     'Loss: %.3f'
                                     % (self.loss.item()),
                                     # 'Minibatch_loss_performance: %.3f'
                                     # % (minibatch_loss_perf) if minibatch_loss_perf else '',
                                     'current_learning_rate: %0.5f'
                                     % (lr_current),
                                     'Minibatch_accuracy_micro: %.3f'
                                     % (minibatch_acc)]
                        print_write(print_str, self.log_file)

                        loss_info = {
                            'Epoch': epoch,
                            'Step': step,
                            'Total': minibatch_loss_total,
                            # 'CE': minibatch_loss_perf,
                            'feat': minibatch_loss_feat
                        }

                        self.logger.log_loss(loss_info)
                    del self.logits_ensemble
                    del self.loss
                    # del pt_batch
                    del inputs

                # Update priority weights if using PrioritizedSampler
                # if self.training_opt['sampler'] and \
                #    self.training_opt['sampler']['type'] == 'PrioritizedSampler':
                if hasattr(self.data['train'].sampler, 'update_weights'):
                    if hasattr(self.data['train'].sampler, 'ptype'):
                        ptype = self.data['train'].sampler.ptype
                    else:
                        ptype = 'score'
                    ws = get_priority(ptype, self.logits.detach(), labels)
                    # ws = logits2score(self.logits.detach(), labels)
                    inlist = [indexes.cpu().numpy(), ws]
                    if self.training_opt['sampler']['type'] == 'ClassPrioritySampler':
                        inlist.append(labels.cpu().numpy())
                    self.data['train'].sampler.update_weights(*inlist)
                    # self.data['train'].sampler.update_weights(indexes.cpu().numpy(), ws)

            if hasattr(self.data['train'].sampler, 'get_weights'):
                self.logger.log_ws(epoch, self.data['train'].sampler.get_weights())
            if hasattr(self.data['train'].sampler, 'reset_weights'):
                self.data['train'].sampler.reset_weights(epoch)

            # After every epoch, validation
            rsls = {'epoch': epoch}
            rsls_train = self.eval_with_preds(total_preds, total_labels)
            rsls_eval = self.eval(phase='val')
            rsls.update(rsls_train)
            rsls.update(rsls_eval)

            # Reset class weights for sampling if pri_mode is valid
            if hasattr(self.data['train'].sampler, 'reset_priority'):
                ws = get_priority(self.data['train'].sampler.ptype,
                                  self.total_logits.detach(),
                                  self.total_labels)
                self.data['train'].sampler.reset_priority(ws, self.total_labels.cpu().numpy())

            # Log results
            self.logger.log_acc(rsls)

            # Under validation, the best model need to be updated
            if self.eval_acc_mic_top1 > best_acc:
                best_epoch = epoch
                best_acc = self.eval_acc_mic_top1
                # best_centroids = self.centroids
                best_model_weights['feat_model'] = copy.deepcopy(self.networks['feat_model'].state_dict())
                best_model_weights['classifier1'] = copy.deepcopy(self.networks['classifier1'].state_dict())
                best_model_weights['classifier2'] = copy.deepcopy(self.networks['classifier2'].state_dict())
                best_model_weights['classifier3'] = copy.deepcopy(self.networks['classifier3'].state_dict())

            print('===> Saving checkpoint')
            self.save_latest(epoch)

            # Set model modes and set scheduler
            # In training, step optimizer scheduler and set model to train()
            self.model_optimizer_scheduler.step()
            #if self.criterion_optimizer:
            #    self.criterion_optimizer_scheduler.step()

            del self.logits
            torch.cuda.empty_cache()

        print()
        print('Training Complete.')

        print_str = ['Best validation accuracy is %.3f at epoch %d' % (best_acc, best_epoch)]
        print_write(print_str, self.log_file)
        # Save the best model and best centroids if calculated
        self.save_model(epoch, best_epoch, best_model_weights, best_acc)

        # Test on the test set
        self.reset_model(best_model_weights)
        self.eval('test' if 'test' in self.data else 'val')
        print('Done')

    def eval_with_preds(self, preds, labels):
        # Count the number of examples
        n_total = sum([len(p) for p in preds])

        # Split the examples into normal and mixup
        normal_preds, normal_labels = [], []
        mixup_preds, mixup_labels1, mixup_labels2, mixup_ws = [], [], [], []
        for p, l in zip(preds, labels):
            if isinstance(l, tuple):
                mixup_preds.append(p)
                mixup_labels1.append(l[0])
                mixup_labels2.append(l[1])
                mixup_ws.append(l[2] * np.ones_like(l[0]))
            else:
                normal_preds.append(p)
                normal_labels.append(l)

        # Calculate normal prediction accuracy
        rsl = {'train_all': 0., 'train_many': 0., 'train_median': 0., 'train_low': 0.}
        if len(normal_preds) > 0:
            normal_preds, normal_labels = list(map(np.concatenate, [normal_preds, normal_labels]))
            n_top1 = mic_acc_cal(normal_preds, normal_labels)
            n_top1_many, \
            n_top1_median, \
            n_top1_low, = shot_acc(normal_preds, normal_labels, self.data['train'])
            rsl['train_all'] += len(normal_preds) / n_total * n_top1
            rsl['train_many'] += len(normal_preds) / n_total * n_top1_many
            rsl['train_median'] += len(normal_preds) / n_total * n_top1_median
            rsl['train_low'] += len(normal_preds) / n_total * n_top1_low

        # Calculate mixup prediction accuracy
        if len(mixup_preds) > 0:
            mixup_preds, mixup_labels, mixup_ws = \
                list(map(np.concatenate, [mixup_preds * 2, mixup_labels1 + mixup_labels2, mixup_ws]))
            mixup_ws = np.concatenate([mixup_ws, 1 - mixup_ws])
            n_top1 = weighted_mic_acc_cal(mixup_preds, mixup_labels, mixup_ws)
            n_top1_many, \
            n_top1_median, \
            n_top1_low, = weighted_shot_acc(mixup_preds, mixup_labels, mixup_ws, self.data['train'])
            rsl['train_all'] += len(mixup_preds) / 2 / n_total * n_top1
            rsl['train_many'] += len(mixup_preds) / 2 / n_total * n_top1_many
            rsl['train_median'] += len(mixup_preds) / 2 / n_total * n_top1_median
            rsl['train_low'] += len(mixup_preds) / 2 / n_total * n_top1_low

        # Top-1 accuracy and additional string
        print_str = ['\n Training acc Top1: %.3f \n' % (rsl['train_all']),
                     'Many_top1: %.3f' % (rsl['train_many']),
                     'Median_top1: %.3f' % (rsl['train_median']),
                     'Low_top1: %.3f' % (rsl['train_low']),
                     '\n']
        print_write(print_str, self.log_file)

        return rsl

    def eval(self, phase='val', openset=False, save_feat=False, tao=1.0, post_hoc=False):

        print_str = ['Phase: %s' % (phase)]
        print_write(print_str, self.log_file)
        time.sleep(0.25)

        if openset:
            print('Under openset test mode. Open threshold is %.1f'
                  % self.training_opt['open_threshold'])

        torch.cuda.empty_cache()

        # In validation or testing mode, set model to eval() and initialize running loss/correct
        for model in self.networks.values():
            model.eval()

        self.total_logits = torch.empty((0, self.training_opt['num_classes'])).cuda()
        self.total_labels = torch.empty(0, dtype=torch.long).cuda()
        self.total_paths = np.empty(0)

        get_feat_only = save_feat
        feats_all, labels_all, idxs_all, logits_all = [], [], [], []
        featmaps_all = []
        # Iterate over dataset
        for inputs, labels, paths in tqdm(self.data[phase]):
            inputs, labels = inputs.cuda(), labels.cuda()

            # If on training phase, enable gradients
            with torch.set_grad_enabled(False):

                # In validation or testing
                self.batch_forward(inputs, labels,
                                   centroids=self.memory['centroids'],
                                   phase=phase)
                logits_backup = 0
                for i in range(2, len(self.outputs)):
                    if post_hoc:
                        logits_backup = logits_backup + self.outputs[i] + tao * (self.distribution_target - self.distribution_source)
                    else:
                        logits_backup = logits_backup + self.outputs[i]
                #logits_backup /= len(self.outputs) - 1
                self.logits = logits_backup

                if not get_feat_only:
                    self.total_logits = torch.cat((self.total_logits, self.logits))
                    self.total_labels = torch.cat((self.total_labels, labels))
                    self.total_paths = np.concatenate((self.total_paths, paths))

                if get_feat_only:
                    logits_all.append(self.logits.cpu().numpy())
                    feats_all.append(self.features.cpu().numpy())
                    labels_all.append(labels.cpu().numpy())
                    idxs_all.append(paths.numpy())

        if get_feat_only:
            typ = 'feat'
            if phase == 'train_plain':
                name = 'train{}_all.pkl'.format(typ)
            elif phase == 'test':
                name = 'test{}_all.pkl'.format(typ)
            elif phase == 'val':
                name = 'val{}_all.pkl'.format(typ)

            fname = os.path.join(self.training_opt['log_dir'], name)
            print('===> Saving feats to ' + fname)
            with open(fname, 'wb') as f:
                pickle.dump({
                    'feats': np.concatenate(feats_all),
                    'labels': np.concatenate(labels_all),
                    'idxs': np.concatenate(idxs_all),
                },
                    f, protocol=4)
            return
        probs, preds = F.softmax(self.total_logits.detach(), dim=1).max(dim=1)

        if openset:
            preds[probs < self.training_opt['open_threshold']] = -1
            self.openset_acc = mic_acc_cal(preds[self.total_labels == -1],
                                           self.total_labels[self.total_labels == -1])
            print('\n\nOpenset Accuracy: %.3f' % self.openset_acc)

        # Calculate the overall accuracy and F measurement
        self.eval_acc_mic_top1 = mic_acc_cal(preds[self.total_labels != -1],
                                             self.total_labels[self.total_labels != -1])
        self.eval_f_measure = F_measure(preds, self.total_labels, openset=openset,
                                        theta=self.training_opt['open_threshold'])
        self.many_acc_top1, \
        self.median_acc_top1, \
        self.low_acc_top1, \
        self.cls_accs = shot_acc(preds[self.total_labels != -1],
                                 self.total_labels[self.total_labels != -1],
                                 self.data['train'],
                                 acc_per_cls=True)
        # Top-1 accuracy and additional string
        print_str = ['\n\n',
                     'Phase: %s'
                     % (phase),
                     '\n\n',
                     'Evaluation_accuracy_micro_top1: %.3f'
                     % (self.eval_acc_mic_top1),
                     '\n',
                     'Averaged F-measure: %.3f'
                     % (self.eval_f_measure),
                     '\n',
                     'Many_shot_accuracy_top1: %.3f'
                     % (self.many_acc_top1),
                     'Median_shot_accuracy_top1: %.3f'
                     % (self.median_acc_top1),
                     'Low_shot_accuracy_top1: %.3f'
                     % (self.low_acc_top1),
                     '\n']

        rsl = {phase + '_all': self.eval_acc_mic_top1,
               phase + '_many': self.many_acc_top1,
               phase + '_median': self.median_acc_top1,
               phase + '_low': self.low_acc_top1,
               phase + '_fscore': self.eval_f_measure}

        if phase == 'val':
            print_write(print_str, self.log_file)
        else:
            acc_str = ["{:.1f} \t {:.1f} \t {:.1f} \t {:.1f}".format(
                self.many_acc_top1 * 100,
                self.median_acc_top1 * 100,
                self.low_acc_top1 * 100,
                self.eval_acc_mic_top1 * 100)]
            if self.log_file is not None and os.path.exists(self.log_file):
                print_write(print_str, self.log_file)
                print_write(acc_str, self.log_file)
            else:
                print(*print_str)
                print(*acc_str)

        if phase == 'test':
            with open(os.path.join(self.training_opt['log_dir'], 'cls_accs.pkl'), 'wb') as f:
                pickle.dump(self.cls_accs, f)
        return rsl

    def centroids_cal(self, data, save_all=False):

        centroids = torch.zeros(self.training_opt['num_classes'],
                                self.training_opt['feature_dim']).cuda()

        print('Calculating centroids.')

        torch.cuda.empty_cache()
        for model in self.networks.values():
            model.eval()

        feats_all, labels_all, idxs_all = [], [], []

        # Calculate initial centroids only on training data.
        with torch.set_grad_enabled(False):
            for inputs, labels, idxs in tqdm(data):
                inputs, labels = inputs.cuda(), labels.cuda()

                # Calculate Features of each training data
                self.batch_forward(inputs, feature_ext=True)
                # Add all calculated features to center tensor
                for i in range(len(labels)):
                    label = labels[i]
                    centroids[label] += self.features[i]
                # Save features if requried
                if save_all:
                    feats_all.append(self.features.cpu().numpy())
                    labels_all.append(labels.cpu().numpy())
                    idxs_all.append(idxs.numpy())

        if save_all:
            fname = os.path.join(self.training_opt['log_dir'], 'feats_all.pkl')
            with open(fname, 'wb') as f:
                pickle.dump({'feats': np.concatenate(feats_all),
                             'labels': np.concatenate(labels_all),
                             'idxs': np.concatenate(idxs_all)},
                            f)
        # Average summed features with class count
        centroids /= torch.tensor(class_count(data)).float().unsqueeze(1).cuda()

        return centroids

    def get_knncentroids(self):
        datakey = 'train_plain'
        assert datakey in self.data

        print('===> Calculating KNN centroids.')

        torch.cuda.empty_cache()
        for model in self.networks.values():
            model.eval()

        feats_all, labels_all = [], []

        # Calculate initial centroids only on training data.
        with torch.set_grad_enabled(False):
            for inputs, labels, idxs in tqdm(self.data[datakey]):
                inputs, labels = inputs.cuda(), labels.cuda()

                # Calculate Features of each training data
                self.batch_forward(inputs, feature_ext=True)

                feats_all.append(self.features.cpu().numpy())
                labels_all.append(labels.cpu().numpy())

        feats = np.concatenate(feats_all)
        labels = np.concatenate(labels_all)

        featmean = feats.mean(axis=0)

        def get_centroids(feats_, labels_):
            centroids = []
            for i in np.unique(labels_):
                centroids.append(np.mean(feats_[labels_ == i], axis=0))
            return np.stack(centroids)

        # Get unnormalized centorids
        un_centers = get_centroids(feats, labels)

        # Get l2n centorids
        l2n_feats = torch.Tensor(feats.copy())
        norm_l2n = torch.norm(l2n_feats, 2, 1, keepdim=True)
        l2n_feats = l2n_feats / norm_l2n
        l2n_centers = get_centroids(l2n_feats.numpy(), labels)

        # Get cl2n centorids
        cl2n_feats = torch.Tensor(feats.copy())
        cl2n_feats = cl2n_feats - torch.Tensor(featmean)
        norm_cl2n = torch.norm(cl2n_feats, 2, 1, keepdim=True)
        cl2n_feats = cl2n_feats / norm_cl2n
        cl2n_centers = get_centroids(cl2n_feats.numpy(), labels)

        return {'mean': featmean,
                'uncs': un_centers,
                'l2ncs': l2n_centers,
                'cl2ncs': cl2n_centers}

    def reset_model(self, model_state):
        for key, model in self.networks.items():
            weights = model_state[key]
            weights = {k: weights[k] for k in weights if k in model.state_dict()}
            model.load_state_dict(weights)

    def load_model(self, model_dir=None):
        model_dir = self.training_opt['log_dir'] if model_dir is None else model_dir
        if not model_dir.endswith('.pth'):
            model_dir = os.path.join(model_dir, 'final_model_checkpoint.pth')

        print('Validation on the best model.')
        print('Loading model from %s' % (model_dir))

        checkpoint = torch.load(model_dir)
        model_state = checkpoint['state_dict_best']

        self.centroids = checkpoint['centroids'] if 'centroids' in checkpoint else None

        for key, model in self.networks.items():
            weights = model_state[key]
            weights = {k: weights[k] for k in weights if k in model.state_dict()}
            x = model.state_dict()
            x.update(weights)
            model.load_state_dict(x)
    
    def load_pretrain(self, model_dir=None):
        model_dir = self.training_opt['log_dir'] if model_dir is None else model_dir
        print('Loading model from %s' % (model_dir))

        checkpoint = torch.load(model_dir, map_location='cpu')
        model_state = checkpoint['state_dict_best']

        for key, model in self.networks.items():
            weights = model_state[key]
            weights = {k: weights[k] for k in weights if k in model.state_dict() and model.state_dict()[k].shape==weights[k].shape}
            #for k in weights:
            #    print("model=", weights[k].shape)
            x = model.state_dict()
            x.update(weights)
            model.load_state_dict(x)

    def save_latest(self, epoch):
        model_weights = {}
        model_weights['feat_model'] = copy.deepcopy(self.networks['feat_model'].state_dict())
        #model_weights['classifier'] = copy.deepcopy(self.networks['classifier'].state_dict())
        model_weights['classifier1'] = copy.deepcopy(self.networks['classifier1'].state_dict())
        model_weights['classifier2'] = copy.deepcopy(self.networks['classifier2'].state_dict())
        model_weights['classifier3'] = copy.deepcopy(self.networks['classifier3'].state_dict())

        model_states = {
            'epoch': epoch,
            'state_dict': model_weights
        }

        if self.config['local_rank'] > 0:
            model_dir = os.path.join(self.training_opt['log_dir'],
                                 'latest_model_checkpoint_{}_.pth'.format(self.config['local_rank']))
        else:
            model_dir = os.path.join(self.training_opt['log_dir'],
                                 'latest_model_checkpoint.pth')
        torch.save(model_states, model_dir)

    def save_model(self, epoch, best_epoch, best_model_weights, best_acc, centroids=None):

        model_states = {'epoch': epoch,
                        'best_epoch': best_epoch,
                        'state_dict_best': best_model_weights,
                        'best_acc': best_acc,
                        'centroids': centroids}

        if self.config['local_rank'] > 0:
            model_dir = os.path.join(self.training_opt['log_dir'],
                                 'final_model_checkpoint_{}_.pth'.format(self.config['local_rank']))
        else:
            model_dir = os.path.join(self.training_opt['log_dir'],
                                 'final_model_checkpoint.pth')

        torch.save(model_states, model_dir)

    def output_logits(self, openset=False):
        filename = os.path.join(self.training_opt['log_dir'],
                                'logits_%s' % ('open' if openset else 'close'))
        print("Saving total logits to: %s.npz" % filename)
        np.savez(filename,
                 logits=self.total_logits.detach().cpu().numpy(),
                 labels=self.total_labels.detach().cpu().numpy(),
                 paths=self.total_paths)
