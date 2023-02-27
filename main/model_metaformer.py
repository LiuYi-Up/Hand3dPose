import torch
import torch.nn as nn
from torch.nn import functional as F
from nets.backbone import FPN
from nets.transformer import Transformer
from nets.regressor import Regressor
from nets.metaformer import make_metaformer
from config import cfg
import math
from data.HO3D.HO3D import make_heatmap
import numpy as np


class Model(nn.Module):
    def __init__(self, backbone, MetaFormer, regressor):
        super(Model, self).__init__()
        self.backbone = backbone
        self.metaformer = MetaFormer
        self.regressor = regressor
        
    
    def forward(self, inputs, targets, meta_info, mode):
        p_feats = self.backbone(inputs['img']) # primary, secondary feats
        # feats = self.FIT(s_feats, p_feats)
        # feats = self.SET(feats, feats)
        feats = p_feats
        meta_out = self.metaformer(feats)

        pre_joints_hm = meta_out[:, :21, :, :]
        pre_feature = meta_out[:, 21:, :, :]

        pred_joints3d = self.regressor(pre_feature, pre_joints_hm)
       
        if mode == 'train':
            # heatmap = []
            # for b in range(pre_joints_hm.shape[0]):
            #     hm = make_heatmap(targets['joints_img'][b, :, :])
            #     heatmap.append(hm)
            # heatmap = np.array(heatmap)
            # loss functions
            loss = {}
            loss['joints3d'] = cfg.lambda_joints * F.mse_loss(pred_joints3d, targets['joints_coord_cam'])
            loss['joints_2d_hm'] = cfg.lambda_joints_img * F.mse_loss(pre_joints_hm, targets['joints_hm'])
            return loss

        else:
            # test output
            out = {}
            out['joints_coord_cam'] = pred_joints3d
            return out

def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight,std=0.001)
    elif type(m) == nn.Conv2d:
        nn.init.normal_(m.weight,std=0.001)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias,0)
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)
        nn.init.constant_(m.bias,0)

def get_model(mode):
    backbone = FPN(pretrained=True)
    metaformer = make_metaformer()
    regressor = Regressor()
    
    if mode == 'train':
        # FIT.apply(init_weights)
        # SET.apply(init_weights)
        regressor.apply(init_weights)
        
    model = Model(backbone, metaformer, regressor)
    
    return model