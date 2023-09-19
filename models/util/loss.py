import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.loss import SoftTargetCrossEntropy
from typing import Optional


class ST_CE_loss(nn.Module):
    """
        CE loss, timm implementation for mixup
    """
    def __init__(self):
        super(ST_CE_loss, self).__init__()

    def forward(self, x, target):
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()


class Bal_CE_loss(nn.Module):
    '''
        Paper: https://arxiv.org/abs/2007.07314
        Code: https://github.com/google-research/google-research/tree/master/logit_adjustment
    '''
    def __init__(self, args):
        super(Bal_CE_loss, self).__init__()
        prior = np.array(args['cls_num'])
        prior = np.log(prior / np.sum(prior))
        prior = torch.from_numpy(prior).type(torch.FloatTensor)
        self.prior = args['bal_tau'] * prior

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prior = self.prior.to(x.device)
        x = x + prior
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()


class BCE_loss(nn.Module):

    def __init__(self, args,
                target_threshold=None, 
                type=None,
                reduction='mean', 
                pos_weight=None):
        super(BCE_loss, self).__init__()
        self.lam = 1.
        self.K = 1.
        self.smoothing = args['smoothing']
        self.target_threshold = target_threshold
        self.weight = None
        self.pi = None
        self.reduction = reduction
        self.register_buffer('pos_weight', pos_weight)

        if type == 'Bal':
            self._cal_bal_pi(args)
        if type == 'CB':
            self._cal_cb_weight(args)

    def _cal_bal_pi(self, args):
        cls_num = torch.Tensor(args['cls_num'])
        self.pi = cls_num / torch.sum(cls_num)

    def _cal_cb_weight(self, args):
        eff_beta = 0.9999
        effective_num = 1.0 - np.power(eff_beta, args['cls_num'])
        per_cls_weights = (1.0 - eff_beta) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(args['cls_num'])
        self.weight = torch.FloatTensor(per_cls_weights).to(args.device)

    def _bal_sigmod_bias(self, x):
        pi = self.pi.to(x.device)
        bias = torch.log(pi) - torch.log(1-pi)
        x = x + self.K * bias
        return x

    def _neg_reg(self, labels, logits, weight=None):
        if weight == None:
            weight = torch.ones_like(labels).to(logits.device)
        pi = self.pi.to(logits.device)
        bias = torch.log(pi) - torch.log(1-pi)
        logits = logits * (1 - labels) * self.lam + logits * labels # neg + pos
        logits = logits + self.K * bias
        weight = weight / self.lam * (1 - labels) + weight * labels # neg + pos
        return logits, weight

    def _one_hot(self, x, target):
        num_classes = x.shape[-1]
        off_value = self.smoothing / num_classes
        on_value = 1. - self.smoothing + off_value
        target = target.long().view(-1, 1)
        target = torch.full((target.size()[0], num_classes),
            off_value, device=x.device, 
            dtype=x.dtype).scatter_(1, target, on_value)
        return target

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert x.shape[0] == target.shape[0]
        if target.shape != x.shape:
            target = self._one_hot(x, target)
        if self.target_threshold is not None:
            target = target.gt(self.target_threshold).to(dtype=target.dtype)
        weight = self.weight
        if self.pi != None: x = self._bal_sigmod_bias(x)
        # if self.lam != None:
        #     x, weight = self._neg_reg(target, x)
        C = x.shape[-1] # + log C
        return C * F.binary_cross_entropy_with_logits(
                    x, target, weight, self.pos_weight,
                    reduction=self.reduction)


class LS_CE_loss(nn.Module):
    """
        label smoothing without mixup
    """
    def __init__(self, smoothing=0.1):
        super(LS_CE_loss, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class MiSLAS_loss(nn.Module):
    ''' 
        Paper: Improving Calibration for Long-Tailed Recognition
        Code: https://github.com/Jia-Research-Lab/MiSLAS
    '''
    def __init__(self, args, shape='concave', power=None):
        super(MiSLAS_loss, self).__init__()

        cls_num_list = args.cls_num
        n_1 = max(cls_num_list)
        n_K = min(cls_num_list)
        smooth_head = 0.3
        smooth_tail = 0.0

        if shape == 'concave':
            self.smooth = smooth_tail + (smooth_head - smooth_tail) * np.sin((np.array(cls_num_list) - n_K) * np.pi / (2 * (n_1 - n_K)))
        elif shape == 'linear':
            self.smooth = smooth_tail + (smooth_head - smooth_tail) * (np.array(cls_num_list) - n_K) / (n_1 - n_K)
        elif shape == 'convex':
            self.smooth = smooth_head + (smooth_head - smooth_tail) * np.sin(1.5 * np.pi + (np.array(cls_num_list) - n_K) * np.pi / (2 * (n_1 - n_K)))
        elif shape == 'exp' and power is not None:
            self.smooth = smooth_tail + (smooth_head - smooth_tail) * np.power((np.array(cls_num_list) - n_K) / (n_1 - n_K), power)

        self.smooth = torch.from_numpy(self.smooth)
        self.smooth = self.smooth.float()

    def forward_oneway(self, x, target):
        smooth = self.smooth.to(x.device)
        smoothing = smooth[target]
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss

    def forward(self, x, target):
        loss = 0
        if target.shape == x.shape: # to match mixup
            '''
                x.shape: batch * nClass
                target: one hot [0, 0, 0, 0.4, 0, 0, 0.6, 0, 0, 0]
            '''
            _, idx_ = torch.topk(target, k=2, dim=1, largest=True)
            i1, i2 = idx_[:,0], idx_[:,1]
            v1 = target[torch.tensor([i for i in range(x.shape[0])]), i1]
            v2 = target[torch.tensor([i for i in range(x.shape[0])]), i2]
            loss_y1 = self.forward_oneway(x, i1)
            loss_y2 = self.forward_oneway(x, i2)
            loss = v1.mul(loss_y1) + v2.mul(loss_y2)
        else:
            loss = self.forward_oneway(x, target)
        return loss.mean()


class LADE_loss(nn.Module):
    '''NOTE can not work with mixup, plz set mixup=0 and cutmix=0
        Paper: Disentangling Label Distribution for Long-tailed Visual Recognition
        Code: https://github.com/hyperconnect/LADE
    '''
    def __init__(self, args, remine_lambda=0.1):
        super().__init__()
        cls_num = torch.tensor(args.cls_num)
        self.prior = cls_num / torch.sum(cls_num)
        self.num_classes = args.nb_classes
        self.balanced_prior = torch.tensor(1. / self.num_classes).float()
        self.remine_lambda = remine_lambda
        self.cls_weight = (cls_num.float() / torch.sum(cls_num.float()))

    def mine_lower_bound(self, x_p, x_q, num_samples_per_cls):
        N = x_p.size(-1)
        first_term = torch.sum(x_p, -1) / (num_samples_per_cls + 1e-8)
        second_term = torch.logsumexp(x_q, -1) - np.log(N)
        return first_term - second_term, first_term, second_term

    def remine_lower_bound(self, x_p, x_q, num_samples_per_cls):
        loss, first_term, second_term = self.mine_lower_bound(x_p, x_q, num_samples_per_cls)
        reg = (second_term ** 2) * self.remine_lambda
        return loss - reg, first_term, second_term

    def forward(self, x, target, q_pred=None):
        """
            x: N x C
            target: N
        """
        prior = self.prior.to(x.device)
        balanced_prior = self.balanced_prior.to(x.device)
        cls_weight = self.cls_weight.to(x.device)
        per_cls_pred_spread = x.T * (target == torch.arange(0, self.num_classes).view(-1, 1).type_as(target))  # C x N
        pred_spread = (x - torch.log(prior + 1e-9) + torch.log(balanced_prior + 1e-9)).T  # C x N
        num_samples_per_cls = torch.sum(target == torch.arange(0, self.num_classes).view(-1, 1).type_as(target), -1).float()  # C
        estim_loss, _, _ = self.remine_lower_bound(per_cls_pred_spread, pred_spread, num_samples_per_cls)
        return - torch.sum(estim_loss * cls_weight)


class LDAM_loss(nn.Module):
    '''
        Paper: Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss
        Code: https://github.com/kaidic/LDAM-DRW
    '''
    def __init__(self, args):
        super(LDAM_loss, self).__init__()
        cls_num_list = args.cls_num
        self.drw = False
        self.epoch = 0
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (0.5 / np.max(m_list))
        self.m_list = torch.FloatTensor(m_list)
        self.s = 30

    def forward_oneway(self, x, target):
        m_list = self.m_list.to(x.device)
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        index_float = index.type(torch.FloatTensor).to(x.device)
        batch_m = torch.matmul(m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, reduction='none')

    def forward(self, x, target):
        loss = 0
        if target.shape == x.shape: # to match mixup
            _, idx_ = torch.topk(target, k=2, dim=1, largest=True)
            i1, i2 = idx_[:,0], idx_[:,1]
            v1 = target[torch.tensor([i for i in range(x.shape[0])]), i1]
            v2 = target[torch.tensor([i for i in range(x.shape[0])]), i2]
            loss_y1 = self.forward_oneway(x, i1)
            loss_y2 = self.forward_oneway(x, i2)
            loss = v1.mul(loss_y1) + v2.mul(loss_y2)
        else:
            loss = self.forward_oneway(x, target)
        return loss.mean()


class CB_CE_loss(nn.Module):
    '''
        Paper: Class-Balanced Loss Based on Effective Number of Samples
        Code: https://github.com/richardaecn/class-balanced-loss
    '''
    def __init__(self, args):
        super(CB_CE_loss, self).__init__()
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, args.cls_num)
        weight = (1.0 - beta) / np.array(effective_num)
        weight = weight / np.sum(weight) * len(args.cls_num)
        self.weight = torch.FloatTensor(weight)

    def forward(self, x, target):
        weight = self.weight.to(x.device)
        return F.cross_entropy(input = x, target = target, weight = weight)

'''
class CADEL_CE_loss(nn.Module):
    """
        CADEL loss, work for mixup and non-mixup
    """
    def __init__(self):
        super(CADEL_CE_loss, self).__init__()

    def forward(self, pred, target, tempture=1.0, weights=None):
        batch_size, feat_dim = pred.shape[0], pred.shape[1]

        logpt = F.log_softmax(pred, dim=-1)
        pt = torch.exp(logpt)
        
        #1: mixup
        if pred.shape==target.shape:
            loss1 = torch.sum(-target * logpt, dim=-1)
            loss1 = loss1.mean()
        #2: non-mixup
        else:
            loss1 = F.nll_loss(logpt, target)

        if weights is None:
            weights = pt.clone().detach().unsqueeze(2)
            loss2 = 0
        else:
            weights = torch.cat((weights, pt.detach().clone().unsqueeze(2)), 2)
            soft_label = torch.mean(weights, dim=2).view(batch_size, -1)
            soft_label = torch.softmax(soft_label, dim=1)
            loss2 = -tempture*torch.sum(soft_label*logpt)/batch_size
            
        loss = loss1 + loss2
        return loss, weights


class CADEL_BCE_loss(nn.Module):
    """
        CADEL Binary Cross-Entropy loss, work for mixup and non-mixup
    """
    def __init__(self, smoothing=0.1):
        super(CADEL_BCE_loss, self).__init__()
        self.smoothing = smoothing
    
    def _one_hot(self, x, target):
        num_classes = x.shape[-1]
        off_value = self.smoothing / num_classes
        on_value = 1. - self.smoothing + off_value
        target = target.long().view(-1, 1)
        target = torch.full((target.size()[0], num_classes),
            off_value, device=x.device, 
            dtype=x.dtype).scatter_(1, target, on_value)
        return target

    def forward(self, pred, target, tempture=1.0, weights=None):
        batch_size, _ = pred.shape[0], pred.shape[1]

        pt = F.sigmoid(pred)
        logpt = torch.log(pt)
        
        #1: non-mixup
        if pred.shape!=target.shape:
            target = self._one_hot(pred, target)
        
        loss1 = torch.sum(-target * logpt, dim=-1)
        loss1 = loss1.mean()
        if weights is None:
            weights = pt.clone().detach().unsqueeze(2)
            loss2 = 0
        else:
            weights = torch.cat((weights, pt.detach().clone().unsqueeze(2)), 2)
            soft_label = torch.mean(weights, dim=2).view(batch_size, -1)
            soft_label = torch.softmax(soft_label, dim=1)
            loss2 = -tempture*torch.sum(soft_label*logpt)/batch_size
            
        loss = loss1 + loss2
        return loss, weights
'''

def CADEL_CE_loss(pred, target, target2=None, lam=1.0, tempture=1.0, weights=None):
    #step 1: calculate mixup loss
    batch_size, feat_dim = pred.shape[0], pred.shape[1]

    logpt = F.log_softmax(pred, dim=1)
    pt = torch.exp(logpt)
    
    loss1 = F.nll_loss(logpt, target)
    if target2 is not None:
        loss1 = lam*loss1 + (1-lam)*F.nll_loss(logpt, target2)
    
    #step 2: calcalte self knowledge distalliation loss
    if weights is None:
        weights = pt.clone().detach().unsqueeze(2)
        loss2 = 0
    else:
        weights = torch.cat((weights, pt.detach().clone().unsqueeze(2)), 2)
        soft_label = torch.mean(weights, dim=2).view(batch_size, feat_dim)
        soft_label = torch.softmax(soft_label, dim=1)
        loss2 = -tempture*torch.sum(soft_label*logpt)/batch_size
        
    loss = loss1 + loss2

    return loss, weights


def CADEL_CE_loss_v2(pred, target, target2=None, lam=1.0, alpha=1.0, weights=None, gamma=2.0, down_weight=True):
    #step 1: calculate mixup loss
    batch_size, feat_dim = pred.shape[0], pred.shape[1]

    logpt = F.log_softmax(pred, dim=1)
    pt = torch.exp(logpt)
    
    loss1 = F.nll_loss(logpt, target)
    if target2 is not None:
        loss1 = lam*loss1 + (1-lam)*F.nll_loss(logpt, target2)
    
    #step 2: calcalte reguralization term
    if weights is None:
        weights = pt.clone().detach().unsqueeze(2)
        loss2 = 0
    else:
        weights = torch.cat((weights, pt.detach().clone().unsqueeze(2)), 2)
        soft_label = torch.mean(weights, dim=2).view(batch_size, feat_dim)
        soft_label = torch.softmax(soft_label, dim=1)
        index1 = target.view(-1, 1).long()
        modulating_factor1 = soft_label.gather(1, index1)
        modulating_factor1 = (1.-modulating_factor1)**gamma if down_weight else modulating_factor1**gamma
        loss2 = F.nll_loss(modulating_factor1*logpt, target)

        if target2 is not None:
            index2 = target2.view(-1, 1).long()
            modulating_factor2 = soft_label.gather(1, index2)
            modulating_factor2 = (1.-modulating_factor2)**gamma if down_weight else modulating_factor2**gamma
            loss2 = lam*loss2+ (1-lam)* F.nll_loss(modulating_factor2*logpt, target2)
           
    loss = alpha*loss1 + (1-alpha)*loss2

    return loss, weights

def CADEL_CE_loss_v3(pred, target, target2=None, lam=1.0, tempture=1.0, weights=None):
    #step 1: calculate mixup loss
    batch_size, feat_dim = pred.shape[0], pred.shape[1]

    logpt = F.log_softmax(pred, dim=1)
    pt = torch.exp(logpt)
    
    loss1 = F.nll_loss(logpt, target)
    if target2 is not None:
        loss1 = lam*loss1 + (1-lam)*F.nll_loss(logpt, target2)
    
    #step 2: calcalte self knowledge distalliation loss
    if weights is None:
        weights = pt.clone().detach().unsqueeze(2)
        loss2 = 0
    else:
        weights = torch.cat((weights, pt.detach().clone().unsqueeze(2)), 2)
        soft_label = torch.mean(weights, dim=2).view(batch_size, feat_dim)
        soft_label = torch.softmax(soft_label, dim=1)
        loss2 = -tempture*torch.sum(soft_label*logpt)/batch_size
        
    loss = loss1 + loss2

    return loss, weights

def CADEL_BCE_loss(pred, target, target2=None, lam=1.0, tempture=1.0, weights=None, num_classes=1000):
    #step 1: calculate mixup loss
    batch_size, _ = pred.shape[0], pred.shape[1]
    #print("taget1.type=", target.dtype)
    if pred.shape != target.shape:
        target = F.one_hot(target, num_classes=num_classes).float()
        #print("taget2.type=", target.dtype)

    pt = torch.sigmoid(pred)
    
    loss1 = F.binary_cross_entropy_with_logits(pred, target)
    if target2 is not None:
        if pred.shape != target2.shape:
            target2 = F.one_hot(target2, num_classes=num_classes).float()
        loss1 = lam*loss1 + (1-lam)*F.binary_cross_entropy_with_logits(pred, target2)
    
    #step 2: calcalte self knowledge distalliation loss
    if weights is None:
        weights = pt.clone().detach().unsqueeze(2)
        loss2 = 0
    else:
        logpt = torch.log(pt)
        weights = torch.cat((weights, pt.detach().clone().unsqueeze(2)), 2)
        soft_label = torch.mean(weights, dim=2).view(batch_size, -1)
        soft_label = torch.softmax(soft_label, dim=1)
        loss2 = -tempture*torch.sum(soft_label*logpt)/batch_size
        #print("loss1.shape=", loss1.shape, "   loss2.shape=", loss2.shape)
    
    loss = loss1 + loss2

    return loss, weights

def ensemble_loss(pred, target, target2=None, lam=1, weight1=None, weight2=None, bins=10, gamma=1.0, base_weight=2, down_weight=True):
    """ Args:
    pred [batch_num, class_num]:
        The direct prediction of classification fc layer.
    target [batch_num, 1]:
        class label.
    """
    #print('pred.shape=', pred.shape)
    batch_size = pred.shape[0]
    edges = [float(x) / bins for x in range(bins + 1)]
    edges[-1] += 1e-6

    logpt = F.log_softmax(pred, dim=1)
    pt = torch.exp(logpt)

    index1 = target.view(pred.shape[0], 1).long()
    p1 = pt.gather(1, index1).clone().detach()

    if weight1 is None:
        weight1 = p1
    else:
        weight1 = torch.cat((weight1, p1), 1)

    modulating_factor1 = torch.mean(weight1, 1)
    w1 = torch.ones_like(modulating_factor1)
    for i in range(bins):
        inds = (modulating_factor1>=edges[i]) & (modulating_factor1<edges[i+1])
        num_in_bin = inds.sum().item()
        if num_in_bin>0:
            w1[inds] = base_weight - gamma*num_in_bin/batch_size if down_weight else base_weight + gamma*num_in_bin/batch_size
           
    #weights = weights.view(-1, 1)**gamma
    loss1 = F.nll_loss(w1.view(-1, 1)*logpt, target)
    #print("loss1=", loss1)


    #for mixup
    loss2 = 0
    if target2 is not None:
        index2 = target2.view(-1, 1).long()
        p2 = pt.gather(1, index2).clone().detach()
        if weight2 is None:
            weight2 = p2
        else:
            weight2 = torch.cat((weight2, p2), 1)

        modulating_factor2 = torch.mean(weight2, 1)
        w2 = torch.ones_like(modulating_factor2)
        for i in range(bins):
            inds = (modulating_factor2>=edges[i]) & (modulating_factor2<edges[i+1])
            num_in_bin = inds.sum().item()
            if num_in_bin>0:
                w2[inds] = base_weight - gamma*num_in_bin/batch_size if down_weight else base_weight + gamma*num_in_bin/batch_size 
            
        #weights = weights.view(-1, 1)**gamma
        loss2 = F.nll_loss(w2.view(-1, 1)*logpt, target2)

    #print("loss2=", loss2)
    loss = lam*loss1 + (1-lam)*loss2
    return loss, weight1, weight2

def ensemble_loss_v2(pred, target, target2=None, lam=1, weight1=None, weight2=None, bins=10, gamma=1.0, base_weight=2, tempture=1.0):
    """ Args:
    pred [batch_num, class_num]:
        The direct prediction of classification fc layer.
    target [batch_num, 1]:
        class label.
    """
    #print('pred.shape=', pred.shape)
    batch_size = pred.shape[0]
    edges = [float(x) / bins for x in range(bins + 1)]
    edges[-1] += 1e-6

    logpt = F.log_softmax(pred, dim=1)
    pt = torch.exp(logpt)

    index1 = target.view(pred.shape[0], 1).long()
    p1 = pt.gather(1, index1).clone().detach()
    #print("pt.shape=", pt.shape, "    p1.shape=", p1.shape)

    if weight1 is None:
        weight1 = p1
    else:
        weight1 = torch.cat((weight1, p1), 1)

    modulating_factor1 = torch.mean(weight1, 1)
    w1 = torch.ones_like(modulating_factor1)
    for i in range(bins):
        inds = (modulating_factor1>=edges[i]) & (modulating_factor1<edges[i+1])
        num_in_bin = inds.sum().item()
        if num_in_bin>0:
            w1[inds] = base_weight + gamma*num_in_bin/batch_size
           
    #weights = weights.view(-1, 1)**gamma
    #temp_loss1 = F.nll_loss(w1.view(-1, 1)*logpt, target)
    temp_loss1 = logpt.gather(1, index1)
    loss1 = tempture*torch.mean(-temp_loss1) + (1-tempture)*torch.mean(-temp_loss1*w1)
    #print("loss1=", loss1)


    #for mixup
    loss2 = 0
    if target2 is not None:
        index2 = target2.view(-1, 1).long()
        p2 = pt.gather(1, index2).clone().detach()
        if weight2 is None:
            weight2 = p2
        else:
            weight2 = torch.cat((weight2, p2), 1)

        modulating_factor2 = torch.mean(weight2, 1)
        w2 = torch.ones_like(modulating_factor2)
        for i in range(bins):
            inds = (modulating_factor2>=edges[i]) & (modulating_factor2<edges[i+1])
            num_in_bin = inds.sum().item()
            if num_in_bin>0:
                w2[inds] = base_weight + gamma*num_in_bin/batch_size
            
        #weights = weights.view(-1, 1)**gamma
        #loss2 = F.nll_loss(w2.view(-1, 1)*logpt, target2)
        temp_loss2 = logpt.gather(1, index2)
        loss2 = tempture*torch.mean(-temp_loss2) + (1-tempture)*torch.mean(-temp_loss2*w2)

    #print("loss2=", loss2)
    loss = lam*loss1 + (1-lam)*loss2
    return loss, weight1, weight2


def Ensemble_BCE_loss(pred, target, target2=None, lam=1, weight1=None, weight2=None, bins=10, gamma=1.0, base_weight=2, tempture=1.0):
    """ Args:
    pred [batch_num, class_num]:
        The direct prediction of classification fc layer.
    target [batch_num, 1]:
        class label.
    """
    #print('pred.shape=', pred.shape)
    batch_size = pred.shape[0]
    edges = [float(x) / bins for x in range(bins + 1)]
    edges[-1] += 1e-6

    #logpt = F.log_softmax(pred, dim=1)
    #pt = torch.exp(logpt)
    pt = torch.sigmoid(pred)
    #print('pt=', pt)
    #logpt = torch.log(pt)

    index1 = target.view(-1, 1).long()
    #print("index1=", index1)
    target_one_hot = F.one_hot(index1, num_classes=pred.shape[1]).view(index1.shape[0], -1).float()
    #print("pt.shape=", pt.shape, "   index1=", torch.max(index1))
    p1 = pt.gather(1, index1).clone().detach()
    

    if weight1 is None:
        weight1 = p1
    else:
        weight1 = torch.cat((weight1, p1), 1)
    #print("weights=", weight1)
    modulating_factor1 = torch.mean(weight1, 1)
    #print('modulating_factors=', modulating_factor1)
    w1 = torch.ones_like(modulating_factor1)
    for i in range(bins):
        inds = (modulating_factor1>=edges[i]) & (modulating_factor1<edges[i+1])
        num_in_bin = inds.sum()
        #print("num_in_bin=", num_in_bin)
        if num_in_bin>0:
            w1[inds] = base_weight + gamma*num_in_bin/batch_size
           
    #weights = weights.view(-1, 1)**gamma
    #temp_loss1 = F.nll_loss(w1.view(-1, 1)*logpt, target)
    #temp_loss1 = logpt.gather(1, index1)
    #temp_loss1 = torch.sum(target_one_hot*torch.log(pt)+(1-target_one_hot)*torch.log(1-pt), dim=-1)/2
    temp_loss1 = F.binary_cross_entropy_with_logits(pred, target_one_hot.view(pred.shape[0], -1), reduction='none')
    temp_loss1 = torch.mean(temp_loss1, dim=-1)
    loss1 = tempture*torch.mean(temp_loss1) + (1-tempture)*torch.mean(temp_loss1*w1)
    #print("loss1=", loss1)


    #for mixup
    loss2 = 0
    if target2 is not None:
        index2 = target2.view(-1, 1).long()
        target_one_hot2 = F.one_hot(index2, num_classes=pred.shape[1]).view(index2.shape[0], -1).float()
        p2 = pt.gather(1, index2).clone().detach()
        if weight2 is None:
            weight2 = p2
        else:
            weight2 = torch.cat((weight2, p2), 1)

        modulating_factor2 = torch.mean(weight2, 1)
        w2 = torch.ones_like(modulating_factor2)
        for i in range(bins):
            inds = (modulating_factor2>=edges[i]) & (modulating_factor2<edges[i+1])
            num_in_bin = inds.sum().item()
            if num_in_bin>0:
                w2[inds] = base_weight + gamma*num_in_bin/batch_size
            
        #weights = weights.view(-1, 1)**gamma
        #loss2 = F.nll_loss(w2.view(-1, 1)*logpt, target2)
        #temp_loss2 = logpt.gather(1, index2)
        #temp_loss2 = torch.sum(target_one_hot2*torch.log(pt) + (1-target_one_hot2)*torch.log(1-pt), dim=-1)/2
        temp_loss2 = F.binary_cross_entropy_with_logits(pred, target_one_hot2, reduction='none')
        temp_loss2 = torch.mean(temp_loss2, dim=-1)
        loss2 = tempture*torch.mean(temp_loss2) + (1-tempture)*torch.mean(temp_loss2*w2)

    #print("loss2=", loss2)
    loss = lam*loss1 + (1-lam)*loss2
    return loss, weight1, weight2

def Ensemble_BCE_loss_v2(pred, target, target2=None, lam=1, weight1=None, weight2=None, bins=10, gamma=1.0, base_weight=2, tempture=1.0):
    """ Args:
    pred [batch_num, class_num]:
        The direct prediction of classification fc layer.
    target [batch_num, 1]:
        class label.
    """
    #print('pred.shape=', pred.shape)
    batch_size = pred.shape[0]
    edges = [float(x) / bins for x in range(bins + 1)]
    edges[-1] += 1e-6

    #logpt = F.log_softmax(pred, dim=1)
    #pt = torch.exp(logpt)
    pt = torch.sigmoid(pred)
    #print('pt=', pt)
    #logpt = torch.log(pt)

    index1 = target.view(-1, 1).long()
    #print("index1=", index1)
    target_one_hot = F.one_hot(index1, num_classes=pred.shape[1]).view(index1.shape[0], -1).float()
    #print("pt.shape=", pt.shape, "   index1=", torch.max(index1))
    p1 = pt.gather(1, index1).clone().detach()
    

    if weight1 is None:
        weight1 = p1
    else:
        weight1 = torch.cat((weight1, p1), 1)
    #print("weights=", weight1)
    modulating_factor1 = torch.mean(weight1, 1)
    #print('modulating_factors=', modulating_factor1)
    w1 = torch.ones_like(modulating_factor1)
    for i in range(bins):
        inds = (modulating_factor1>=edges[i]) & (modulating_factor1<edges[i+1])
        num_in_bin = inds.sum()
        #print("num_in_bin=", num_in_bin)
        if num_in_bin>0:
            w1[inds] = base_weight + gamma*num_in_bin/batch_size
           
    #weights = weights.view(-1, 1)**gamma
    #temp_loss1 = F.nll_loss(w1.view(-1, 1)*logpt, target)
    #temp_loss1 = logpt.gather(1, index1)
    #temp_loss1 = torch.sum(target_one_hot*torch.log(pt)+(1-target_one_hot)*torch.log(1-pt), dim=-1)/2
    temp_loss1 = F.binary_cross_entropy_with_logits(pred, target_one_hot.view(pred.shape[0], -1), reduction='none')
    #temp_loss1 = torch.sum(temp_loss1, dim=-1)
    temp_loss1 = temp_loss1.gather(1, index1).view(-1, 1)
    loss1 = tempture*torch.mean(temp_loss1) + (1-tempture)*torch.mean(temp_loss1*w1)
    #print("loss1=", loss1)


    #for mixup
    loss2 = 0
    if target2 is not None:
        index2 = target2.view(-1, 1).long()
        target_one_hot2 = F.one_hot(index2, num_classes=pred.shape[1]).view(index2.shape[0], -1).float()
        p2 = pt.gather(1, index2).clone().detach()
        if weight2 is None:
            weight2 = p2
        else:
            weight2 = torch.cat((weight2, p2), 1)

        modulating_factor2 = torch.mean(weight2, 1)
        w2 = torch.ones_like(modulating_factor2)
        for i in range(bins):
            inds = (modulating_factor2>=edges[i]) & (modulating_factor2<edges[i+1])
            num_in_bin = inds.sum().item()
            if num_in_bin>0:
                w2[inds] = base_weight + gamma*num_in_bin/batch_size
            
        #weights = weights.view(-1, 1)**gamma
        #loss2 = F.nll_loss(w2.view(-1, 1)*logpt, target2)
        #temp_loss2 = logpt.gather(1, index2)
        #temp_loss2 = torch.sum(target_one_hot2*torch.log(pt) + (1-target_one_hot2)*torch.log(1-pt), dim=-1)/2
        temp_loss2 = F.binary_cross_entropy_with_logits(pred, target_one_hot2, reduction='none')
        #temp_loss2 = torch.sum(temp_loss2, dim=-1)
        temp_loss2 = temp_loss2.gather(1, index2).view(-1, 1)
        loss2 = tempture*torch.mean(temp_loss2) + (1-tempture)*torch.mean(temp_loss2*w2)

    #print("loss2=", loss2)
    loss = lam*loss1 + (1-lam)*loss2
    return loss, weight1, weight2

def Ensemble_BCE_loss_v3(pred, target, target2=None, lam=1, weight1=None, weight2=None, bins=10, gamma=1.0, base_weight=2, tempture=1.0):
    """ Args:
    pred [batch_num, class_num]:
        The direct prediction of classification fc layer.
    target [batch_num, 1]:
        class label.
    """
    #print('pred.shape=', pred.shape)
    batch_size = pred.shape[0]
    edges = [float(x) / bins for x in range(bins + 1)]
    edges[-1] += 1e-6

    #logpt = F.log_softmax(pred, dim=1)
    #pt = torch.exp(logpt)
    pt = torch.sigmoid(pred)
    #print('pt=', pt)
    #logpt = torch.log(pt)

    index1 = target.view(-1, 1).long()
    #print("index1=", index1)
    target_one_hot = F.one_hot(index1, num_classes=pred.shape[1]).view(index1.shape[0], -1).float()
    #print("pt.shape=", pt.shape, "   index1=", torch.max(index1))
    p1 = pt.gather(1, index1).clone().detach()
    

    if weight1 is None:
        weight1 = p1
    else:
        weight1 = torch.cat((weight1, p1), 1)
    #print("weights=", weight1)
    modulating_factor1 = torch.mean(weight1, 1)
    #print('modulating_factors=', modulating_factor1)
    w1 = torch.ones_like(modulating_factor1)
    for i in range(bins):
        inds = (modulating_factor1>=edges[i]) & (modulating_factor1<edges[i+1])
        num_in_bin = inds.sum()
        #print("num_in_bin=", num_in_bin)
        if num_in_bin>0:
            w1[inds] = base_weight + gamma*num_in_bin/batch_size
           
    #weights = weights.view(-1, 1)**gamma
    #temp_loss1 = F.nll_loss(w1.view(-1, 1)*logpt, target)
    #temp_loss1 = logpt.gather(1, index1)
    temp_loss1 = target_one_hot*torch.log(pt)+(1-target_one_hot)*torch.log(1-pt)
    temp_loss1 = torch.sum(temp_loss1, dim=-1)
    #temp_loss1 = F.binary_cross_entropy_with_logits(pred, target_one_hot.view(pred.shape[0], -1), reduction='none')
    #temp_loss1 = torch.sum(temp_loss1, dim=-1)
    #temp_loss1 = temp_loss1.gather(1, index1).view(-1, 1)
    loss1 = tempture*torch.mean(-temp_loss1) + (1-tempture)*torch.mean(-temp_loss1*w1)
    #print("loss1=", loss1)


    #for mixup
    loss2 = 0
    if target2 is not None:
        index2 = target2.view(-1, 1).long()
        target_one_hot2 = F.one_hot(index2, num_classes=pred.shape[1]).view(index2.shape[0], -1).float()
        p2 = pt.gather(1, index2).clone().detach()
        if weight2 is None:
            weight2 = p2
        else:
            weight2 = torch.cat((weight2, p2), 1)

        modulating_factor2 = torch.mean(weight2, 1)
        w2 = torch.ones_like(modulating_factor2)
        for i in range(bins):
            inds = (modulating_factor2>=edges[i]) & (modulating_factor2<edges[i+1])
            num_in_bin = inds.sum().item()
            if num_in_bin>0:
                w2[inds] = base_weight + gamma*num_in_bin/batch_size
            
        #weights = weights.view(-1, 1)**gamma
        #loss2 = F.nll_loss(w2.view(-1, 1)*logpt, target2)
        #temp_loss2 = logpt.gather(1, index2)
        temp_loss2 = target_one_hot2*torch.log(pt) + (1-target_one_hot2)*torch.log(1-pt)
        temp_loss2 = torch.sum(temp_loss2, dim=-1)
        #temp_loss2 = F.binary_cross_entropy_with_logits(pred, target_one_hot2, reduction='none')
        #temp_loss2 = torch.sum(temp_loss2, dim=-1)
        #temp_loss2 = temp_loss2.gather(1, index2).view(-1, 1)
        loss2 = tempture*torch.mean(-temp_loss2) + (1-tempture)*torch.mean(-temp_loss2*w2)

    #print("loss2=", loss2)
    loss = lam*loss1 + (1-lam)*loss2
    return loss, weight1, weight2

def Ensemble_BCE_loss_v4(pred, target, weights=None, bins=10, gamma=1.0, base_weight=2, tempture=1.0):
    """ Args:
    pred [batch_num, class_num]:
        The direct prediction of classification fc layer.
    target [batch_num, class_num]:
        class label.
    """
    #print('pred.shape=', pred.shape)
    batch_size = pred.shape[0]
    edges = [float(x) / bins for x in range(bins + 1)]
    edges[-1] += 1e-6

    pt = torch.sigmoid(pred)
    pt_selected, _ = torch.max(pt, dim=-1)
    pt_selected = pt_selected.clone().detach().view(-1, 1)
    if weights is None:
        weights = pt_selected
    else:
        weights = torch.cat((weights, pt_selected), 1)
    #print("weights=", weight1)
    modulating_factor = torch.mean(weights, 1)
    #print('modulating_factors=', modulating_factor1)
    w = torch.ones_like(modulating_factor)
    for i in range(bins):
        inds = (modulating_factor>=edges[i]) & (modulating_factor<edges[i+1])
        num_in_bin = inds.sum()
        #print("num_in_bin=", num_in_bin)
        if num_in_bin>0:
            w[inds] = base_weight + gamma*num_in_bin/batch_size
           
    temp_loss = target*torch.log(pt)+(1-target)*torch.log(1-pt)
    temp_loss = torch.sum(temp_loss, dim=-1)
    loss = tempture*torch.mean(-temp_loss) + (1-tempture)*torch.mean(-temp_loss*w)
    
    return loss, weights
