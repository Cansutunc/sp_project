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

class MutualInfoLoss2(nn.Module):
    def __init__(self,coef_card=0.4): super().__init__(); self.coef=coef_card
    def forward(self,P):
        B,K,H,W=P.shape
        pix=-(P*(P.add(1e-16)).log()).sum(1).mean()
        m=P.view(B,K,-1).mean(-1)
        ent=-(m*(m.add(1e-16)).log()).sum(1).mean()
        return pix - self.coef*ent
class MutualInfoLoss(nn.Module):
    def __init__(self, coef_card=0.5):
        super().__init__()
        self.coef = coef_card
    
    def forward(self, P):
        B, K, H, W = P.shape
        
        # Pixel certainty
        pix_entropy = -(P * (P.clamp(min=1e-16)).log()).sum(1).mean()
        
        # SP balance (variance-based)
        m = P.view(B, K, -1).mean(-1)
        target = 1.0 / K
        variance = ((m - target) ** 2).sum(1).mean()
        
        # Death penalty
        inactive = (m < 0.005).float().sum(1).mean()
        
        # All terms ADDED
        return pix_entropy + self.coef * variance + 0.1 * inactive
# class MutualInfoLoss(nn.Module):
#     def __init__(self, coef_card=2.0):
#         super().__init__()
#         self.coef = coef_card
    
#     def forward(self, P):
#         B, K, H, W = P.shape
        
#         # Pixel entropy: minimize (pixels should be certain)
#         pix_entropy = -(P * (P.add(1e-16)).log()).sum(1).mean()
        
#         # Superpixel mass variance (penalty for imbalance)
#         m = P.view(B, K, -1).mean(-1)  # [B, K]
#         target = 1.0 / K  # Uniform target
#         variance = ((m - target) ** 2).sum(1).mean()
        
#         # Penalize unused superpixels
#         inactive_penalty = (m < 0.01).float().sum(1).mean()
        
#         return pix_entropy + self.coef * variance + 0.1 * inactive_penalty
    #BU LOSSU POZTIF EDİYOR
# class MutualInfoLoss(nn.Module):
#     def __init__(self, coef_card=2.0):
#         super().__init__()
#         self.coef = coef_card
    
#     def forward(self, P):
#         B, K, H, W = P.shape
        
#         # Pixel entropy
#         pix = -(P * (P.add(1e-16)).log()).sum(1).mean()
        
#         # Superpixel mass variance (penalty for imbalance)
#         m = P.view(B, K, -1).mean(-1)  # [B, K]
#         target = 1.0 / K  # Uniform target
#         variance = ((m - target) ** 2).sum(1).mean()
        
#         return pix + self.coef * variance
    
class SmoothnessLoss(nn.Module):
    def __init__(self, sigma=5.0):
        super().__init__()
        self.sigma = 2 * sigma**2
    
    def forward(self, P, I):
        dx = P[:, :, :, :-1] - P[:, :, :, 1:]
        dx = F.pad(dx, (0, 1, 0, 0))
        dy = P[:, :, :-1, :] - P[:, :, 1:, :]
        dy = F.pad(dy, (0, 0, 0, 1))
        
        l1x = dx.abs().sum(1, keepdim=True)
        l1y = dy.abs().sum(1, keepdim=True)
        
        gray = rgb_to_gray(I)
        gx, gy = sobel_gx_gy(gray * 255.0)
        
        # FIXED: Normalize per-pixel, not per-image
        edge_mag = torch.sqrt(gx**2 + gy**2)  # [B, 1, H, W]
        
        # Normalize to [0, 1] per-image
        edge_mag_max = edge_mag.view(edge_mag.shape[0], -1).max(dim=1)[0]
        edge_mag_max = edge_mag_max.view(-1, 1, 1, 1).clamp(min=1e-8)
        edge_norm = edge_mag / edge_mag_max
        
        # Weight: high at smooth areas (0), low at edges (1)
        w = torch.exp(-edge_norm * self.sigma)
        
        return (l1x * w + l1y * w).mean()
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
    
class ReconstructionLoss(nn.Module):
    def __init__(self): super().__init__(); self.mse=nn.MSELoss()
    def forward(self, pred_img, target_img): return self.mse(pred_img, target_img)

class EdgeGuidedLoss(nn.Module):
    """Forces superpixels to align with image edges."""
    def __init__(self, sigma=10.0):
        super().__init__()
        self.sigma = sigma
    
    def forward(self, P, I):
        # Compute image edges (Sobel)
        gray = rgb_to_gray(I) * 255.0
        gx, gy = sobel_gx_gy(gray)
        edge_strength = torch.sqrt(gx**2 + gy**2)  # [B, 1, H, W]
        
        # Compute superpixel boundary probability
        dx = P[:,:,:,:-1] - P[:,:,:,1:]
        dy = P[:,:,:-1,:] - P[:,:,1:,:]
        dx = F.pad(dx, (0,1,0,0))
        dy = F.pad(dy, (0,0,0,1))
        
        sp_boundary = (dx.abs() + dy.abs()).sum(1, keepdim=True)  # [B, 1, H, W]
        
        # Penalize boundaries NOT aligned with edges
        edge_map = (edge_strength > edge_strength.mean()).float()
        penalty = sp_boundary * (1 - edge_map)
        
        return penalty.mean()
    

def feature_reconstruction_loss(Fmap, P, eps=1e-8):
    """
    Feature-level reconstruction loss (RGB yerine)
    Fmap: [B, C, H, W] - CNN features from spcnn
    P: [B, K, H, W] - superpixel assignment probabilities
    """
    from segment import sp_soft_pool_avg, sp_project
    
    B, C, H, W = Fmap.shape
    K = P.shape[1]
    
    # Pool features per superpixel
    Z = sp_soft_pool_avg(Fmap, P)  # [B, K, C]
    
    # Project back to pixel space
    Fhat = sp_project(Z, P)  # [B, C, H, W]
    
    # MSE loss
    return F.mse_loss(Fmap, Fhat)