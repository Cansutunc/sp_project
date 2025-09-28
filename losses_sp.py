import torch
import torch.nn as nn
import torch.nn.functional as F

def sobel_gx_gy(gray):
    kx=torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=gray.dtype,device=gray.device).view(1,1,3,3)
    ky=torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]],dtype=gray.dtype,device=gray.device).view(1,1,3,3)
    return F.conv2d(gray,kx,padding=1), F.conv2d(gray,ky,padding=1)

def rgb_to_gray(x):
    if x.shape[1]==3:
        r,g,b=x[:,0:1],x[:,1:2],x[:,2:3]
        return 0.2989*r+0.5870*g+0.1140*b
    return x[:,0:1]

class MutualInfoLoss(nn.Module):
    def __init__(self,coef_card=2.0): super().__init__(); self.coef=coef_card
    def forward(self,P):
        B,K,H,W=P.shape
        pix=-(P*(P.add(1e-16)).log()).sum(1).mean()
        m=P.view(B,K,-1).mean(-1)
        ent=-(m*(m.add(1e-16)).log()).sum(1).mean()
        return pix - self.coef*ent

class SmoothnessLoss(nn.Module):
    def __init__(self,sigma=10.0): super().__init__(); self.sigma=sigma
    def forward(self,P,I):
        dx=P[:,:,:, :-1]-P[:,:,:, 1:]; dx=F.pad(dx,(0,1,0,0))
        dy=P[:,:, :-1,:]-P[:,:, 1:,:]; dy=F.pad(dy,(0,0,0,1))
        l1x=dx.abs().sum(1,keepdim=True); l1y=dy.abs().sum(1,keepdim=True)
        gray = rgb_to_gray(I)
        gx, gy = sobel_gx_gy(gray * 255.0)   # <--- ÖNEMLİ
        wx = torch.exp(-(gx**2)/(2*(self.sigma**2)))
        wy = torch.exp(-(gy**2)/(2*(self.sigma**2)))
        return (l1x*wx + l1y*wy).mean()

def laplacian_edge_map(x):
    lap=torch.tensor([[0,1,0],[1,-4,1],[0,1,0]],dtype=x.dtype,device=x.device).view(1,1,3,3)
    if x.shape[1]>1:
        e=sum(F.conv2d(x[:,i:i+1],lap,padding=1).abs() for i in range(x.shape[1]))
    else:
        e=F.conv2d(x,lap,padding=1).abs()
    e=e.flatten(2); return F.softmax(e,dim=-1)

class EdgeAwareKLLoss(nn.Module):
    def forward(self,I, Ihat):
        EI=laplacian_edge_map(I); EP=laplacian_edge_map(Ihat)
        return (EI*(EI.add(1e-12).log() - EP.add(1e-12).log())).sum(-1).mean()
