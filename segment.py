from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_spixel_prob(logits, dim=1,tau=0.7):
    return F.softmax(logits / tau, dim=dim)

def _conv_in(in_c, out_c, k=3, dilation=1, pad=True, relu=True):
    layers = []
    if pad and k == 3:
        padding = dilation  # Dilated conv için padding ayarla
        layers.append(nn.ReflectionPad2d(padding))
    layers.append(nn.Conv2d(in_c, out_c, k, dilation=dilation, bias=False))
    layers.append(nn.InstanceNorm2d(out_c, affine=True))
    if relu:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)

class EdgeAwareSPModule(nn.Module):
    def __init__(self, in_c=5, num_feat=32, num_layers=4, num_spixels=200):
        super().__init__()
        feats = []
        c = in_c
        dilations = [1, 1, 2, 4]  
        
        for i in range(num_layers):
            co = num_feat * (2**i)
            feats.append(_conv_in(c, co, 3, dilation=dilations[i], pad=True, relu=True))
            c = co
        
        self.feat_conv = nn.Sequential(*feats)
        self.c_feat = c
        self.head_logits = _conv_in(self.c_feat, num_spixels, 1, dilation=1, pad=False, relu=False)
        
    def forward(self, x, get_prob=get_spixel_prob):
        feat = self.feat_conv(x)
        logits = self.head_logits(feat)
        P = get_prob(logits)
        return {'feat': feat, 'logits': logits, 'P': P}
    
def sp_soft_pool_avg(X,P,eps=1e-8):
    B,C,H,W=X.shape; K=P.shape[1]
    Xf=X.view(B,C,-1); Pf=P.view(B,K,-1)
    mass=Pf.sum(-1,keepdim=True).clamp_min(eps)
    Z=(Pf@Xf.transpose(1,2))/mass
    return Z

def sp_project(Z,P):
    B,K,C=Z.shape; H,W=P.shape[-2:]
    return (Z.transpose(1,2)@P.view(B,K,-1)).view(B,C,H,W)
