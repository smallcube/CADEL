# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# LiVT: https://github.com/XuZhengzhuo/LiVT
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer
from timm.models.vision_transformer import Block

from models.CosNormClassifier import CosNorm_Classifier

class AUX_Layer(nn.Module):
    def __init__(self, global_pool='token',
            embed_dim=768,
            num_heads=12,
            mlp_ratio=4.,
            depth=1,
            qkv_bias=True,
            qk_scale=None,
            fc_norm=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            norm_layer=None,
            act_layer=None,
            block_fn=Block, 
            normalized=True,
            num_classes=1000,
            scale=30):
        super().__init__()

        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        
        self.num_classes = num_classes
        self.global_pool = global_pool
        #self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        #self.num_prefix_tokens = 1 if class_token else 0
        #self.no_embed_class = no_embed_class
        #self.grad_checkpointing = False

        self.block_fn = nn.Sequential(*[
                        block_fn(dim=embed_dim, 
                            num_heads=num_heads, 
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            drop=drop_rate,
                            attn_drop=attn_drop_rate,
                            drop_path=drop_path_rate,
                            norm_layer=norm_layer
                        )
                        for i in range(depth)])
        
        
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        if normalized:
            self.head = CosNorm_Classifier(embed_dim, num_classes, scale=scale)
            #self.FC = NormedLinear(channel, num_classes, scale=scale)
        else:
            self.head = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        #feature
        x = self.block_fn(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            feat = self.fc_norm(x)
        else:
            x = self.norm(x)
            feat = x[:, 0]
        
        #head
        x = self.head(feat)
        return x
        


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, classifier_num=3, selected_layers=[7, 9], normalized=True, aux_depth=1, scale=30, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm
        depth = kwargs['depth']
        drop_path_rate = kwargs['drop_path_rate']
        num_classes = kwargs['num_classes']
        #scale = kwargs['scale']
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        #self.selected_layers = [int(i*depth//classifier_num) for i in range(1, classifier_num)]
        self.selected_layers = selected_layers

        self.aux_layer1 = AUX_Layer(global_pool=global_pool,
                                    embed_dim=kwargs['embed_dim'],
                                    num_heads=kwargs['num_heads'],
                                    mlp_ratio=kwargs['mlp_ratio'],
                                    normalized=normalized,
                                    num_classes=num_classes,
                                    drop_path_rate=dpr[self.selected_layers[0]],
                                    depth=aux_depth,
                                    scale=scale)
        
        self.aux_layer2 = AUX_Layer(global_pool=global_pool,
                                    embed_dim=kwargs['embed_dim'],
                                    num_heads=kwargs['num_heads'],
                                    mlp_ratio=kwargs['mlp_ratio'],
                                    normalized=normalized,
                                    num_classes=num_classes,
                                    drop_path_rate=dpr[self.selected_layers[1]],
                                    depth=aux_depth,
                                    scale=scale)

        if normalized:
            self.head = CosNorm_Classifier(embed_dim, num_classes, scale=scale)
            #self.FC = NormedLinear(channel, num_classes, scale=scale)
        else:
            self.head = nn.Linear(embed_dim, num_classes)
        
        
    def get_selected_layers(self,):
        return self.selected_layers
        
    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        features = []
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            if idx in self.selected_layers:
                features.append(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome, features

    def forward(self, x):
        feat, features = self.forward_features(x)
        logit1 = self.aux_layer1(features[0])
        logit2 = self.aux_layer2(features[1])
        logit = self.head(feat)
        
        return [logit1]+[logit2]+[logit]

def vit_tiny_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_small_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

