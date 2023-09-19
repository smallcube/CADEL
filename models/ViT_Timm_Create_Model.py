from models.ViT_Feature_Timm import *
from utils import *
from os import path
from models.utils.pos_embed import interpolate_pos_embed
from timm.models.layers import trunc_normal_
        
def create_model(img_size=224, patch_size=16, num_classes=1000, drop_path=0.1, global_pool=True, 
                 embed_dim=768, depth=12, num_heads=12, checkpoint=None, **kwargs):
    
    print('Loading Scratch ViT Feature Model.')
    model = VisionTransformer(img_size=img_size, num_classes=num_classes, drop_path_rate=drop_path, global_pool=global_pool,
                            patch_size=patch_size, embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=4, 
                            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    if checkpoint is not None:
        print('==> Loading weights from %s' % checkpoint)
        checkpoint = torch.load(checkpoint, map_location='cpu')
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        
        if global_pool:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        else:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # manually initialize fc layer
        trunc_normal_(model.head.weight, std=2e-5)
    else:
        print('No Pretrained Weights For Feature Model.')

    return model