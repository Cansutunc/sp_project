import torch
import torch.nn as nn
import torch.nn.functional as F

def sobel_gx_gy(gray):
    """Sobel edge detection kernels."""
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=gray.dtype, device=gray.device).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=gray.dtype, device=gray.device).view(1,1,3,3)
    return F.conv2d(gray, kx, padding=1), F.conv2d(gray, ky, padding=1)

def rgb_to_gray(x):
    """Convert RGB to grayscale."""
    if x.shape[1] == 3:
        r, g, b = x[:,0:1], x[:,1:2], x[:,2:3]
        return 0.2989*r + 0.5870*g + 0.1140*b
    return x[:,0:1]


class MutualInfoLoss(nn.Module):
    """
    Mutual Information Loss.
    Encourages sparse assignment per pixel and balanced superpixel sizes.
    
    Args:
        coef_card: Coefficient for cardinality term (entropy over superpixels)
    """
    def __init__(self, coef_card=2.0):
        super().__init__()
        self.coef = coef_card
    
    def forward(self, P):
        """
        Args:
            P: [B, K, H, W] - soft probability map (already softmaxed)
        
        Returns:
            loss: scalar (negative value - this is normal for entropy-based losses!)
        """
        B, K, H, W = P.shape
        
        # Pixel-wise entropy (encourages sparse assignment per pixel)
        pix = -(P * (P.add(1e-16)).log()).sum(1).mean()
        
        # Marginal entropy (encourages balanced superpixel sizes)
        m = P.view(B, K, -1).mean(-1)  # [B, K]
        ent = -(m * (m.add(1e-16)).log()).sum(1).mean()
        
        return pix - self.coef * ent


class SmoothnessLoss(nn.Module):
    """
    Smoothness Loss with edge-awareness.
    Penalizes changes in superpixel assignment, weighted by image gradients.
    
    Args:
        sigma: Gaussian kernel parameter for edge weighting
    """
    def __init__(self, sigma=10.0):
        super().__init__()
        self.sigma = sigma
    
    def forward(self, P, I):
        """
        Args:
            P: [B, K, H, W] - soft probability map
            I: [B, 3, H, W] - RGB image
        
        Returns:
            loss: scalar
        """
        # Compute gradients of probability map
        dx = P[:,:,:, :-1] - P[:,:,:, 1:]
        dx = F.pad(dx, (0,1,0,0))
        dy = P[:,:, :-1,:] - P[:,:, 1:,:]
        dy = F.pad(dy, (0,0,0,1))
        
        # L1 norm of gradients
        l1x = dx.abs().sum(1, keepdim=True)  # [B, 1, H, W]
        l1y = dy.abs().sum(1, keepdim=True)
        
        # Compute image gradients
        gray = rgb_to_gray(I)
        gx, gy = sobel_gx_gy(gray * 255.0)  # Scale to [0, 255] for better gradients
        
        # Edge-aware weights (lower weight at edges)
        wx = torch.exp(-(gx**2) / (2 * (self.sigma**2)))
        wy = torch.exp(-(gy**2) / (2 * (self.sigma**2)))
        
        # Weighted smoothness loss
        return (l1x * wx + l1y * wy).mean()


class EdgeAwareKLLoss(nn.Module):
    """
    Edge-Aware KL Divergence Loss.
    Encourages superpixel boundaries to align with image edges.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, I, Ihat):
        """
        Args:
            I: [B, 3, H, W] - original image
            Ihat: [B, 3, H, W] - reconstructed image from superpixels
        
        Returns:
            loss: scalar
        """
        EI = laplacian_edge_map(I)      # Edge map of original
        EP = laplacian_edge_map(Ihat)   # Edge map of reconstruction
        
        # KL divergence between edge distributions
        return (EI * (EI.add(1e-12).log() - EP.add(1e-12).log())).sum(-1).mean()


class ReconstructionLoss(nn.Module):
    """
    Image Reconstruction Loss.
    Ensures that features capture enough information to reconstruct the original image.
    
    This is crucial for learning meaningful superpixel features!
    """
    def __init__(self, loss_type='mse'):
        super().__init__()
        self.loss_type = loss_type
        
        if loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        elif loss_type == 'l1':
            self.loss_fn = nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(self, pred_img, target_img):
        """
        Args:
            pred_img: [B, 3, H, W] - reconstructed image from CNN
            target_img: [B, 3, H, W] - original RGB image
        
        Returns:
            loss: scalar
        """
        return self.loss_fn(pred_img, target_img)


def laplacian_edge_map(x):
    """
    Compute edge map using Laplacian operator.
    Returns a probability distribution over pixel locations.
    """
    lap = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=x.dtype, device=x.device).view(1,1,3,3)
    
    if x.shape[1] > 1:
        # Multi-channel: sum absolute values across channels
        e = sum(F.conv2d(x[:,i:i+1], lap, padding=1).abs() for i in range(x.shape[1]))
    else:
        e = F.conv2d(x, lap, padding=1).abs()
    
    # Flatten spatial dimensions and convert to probability distribution
    e = e.flatten(2)  # [B, 1, H*W]
    return F.softmax(e, dim=-1)


# ==================== Combined Loss (Optional) ====================

class CombinedSuperpixelLoss(nn.Module):
    """
    Combined loss for superpixel segmentation.
    Combines MI, Smoothness, Edge-Aware, and Reconstruction losses.
    
    Usage:
        loss_fn = CombinedSuperpixelLoss(
            w_mi=1.0, w_smooth=1.0, w_edge=1.0, w_recon=10.0
        )
        loss = loss_fn(P, I, I_recon)
    """
    def __init__(self, w_mi=1.0, w_smooth=1.0, w_edge=1.0, w_recon=10.0,
                 mi_coef=2.0, smooth_sigma=10.0):
        super().__init__()
        
        self.w_mi = w_mi
        self.w_smooth = w_smooth
        self.w_edge = w_edge
        self.w_recon = w_recon
        
        self.mi_loss = MutualInfoLoss(mi_coef)
        self.smooth_loss = SmoothnessLoss(smooth_sigma)
        self.edge_loss = EdgeAwareKLLoss()
        self.recon_loss = ReconstructionLoss('mse')
    
    def forward(self, P, I, I_recon=None, return_components=False):
        """
        Args:
            P: [B, K, H, W] - soft probability map
            I: [B, 3, H, W] - original image
            I_recon: [B, 3, H, W] - reconstructed image (optional)
            return_components: if True, return dict with individual losses
        
        Returns:
            loss: scalar or dict
        """
        # Compute individual losses
        L_mi = self.mi_loss(P)
        L_smooth = self.smooth_loss(P, I)
        
        # Edge-aware loss (needs reconstruction from superpixels)
        from segment import sp_soft_pool_avg, sp_project
        Zrgb = sp_soft_pool_avg(I, P)
        Ihat = sp_project(Zrgb, P)
        L_edge = self.edge_loss(I, Ihat)
        
        # Reconstruction loss (if head is available)
        L_recon = torch.tensor(0.0, device=P.device)
        if I_recon is not None:
            L_recon = self.recon_loss(I_recon, I)
        
        # Combined loss
        total_loss = (self.w_mi * L_mi + 
                     self.w_smooth * L_smooth + 
                     self.w_edge * L_edge +
                     self.w_recon * L_recon)
        
        if return_components:
            return {
                'total': total_loss,
                'mi': L_mi.item(),
                'smooth': L_smooth.item(),
                'edge': L_edge.item(),
                'recon': L_recon.item()
            }
        
        return total_loss


# ==================== Utility Functions ====================

def compute_superpixel_metrics(P, image, return_dict=True):
    """
    Compute various metrics for superpixel quality.
    Useful for debugging and analysis.
    
    Args:
        P: [B, K, H, W] - soft probability map
        image: [B, 3, H, W] - original image
        return_dict: if True, return dict; else return tuple
    
    Returns:
        metrics: dict or tuple of metrics
    """
    B, K, H, W = P.shape
    
    # 1. Assignment entropy (lower = more confident)
    entropy_per_pixel = -(P * torch.log(P + 1e-10)).sum(dim=1)  # [B, H, W]
    mean_entropy = entropy_per_pixel.mean().item()
    
    # 2. Number of active superpixels
    seg = P.argmax(dim=1)  # [B, H, W]
    unique_per_batch = [len(torch.unique(seg[b])) for b in range(B)]
    mean_unique = sum(unique_per_batch) / B
    
    # 3. Superpixel size distribution
    sizes = []
    for b in range(B):
        for k in range(K):
            size = (seg[b] == k).sum().item()
            if size > 0:
                sizes.append(size)
    
    mean_size = sum(sizes) / len(sizes) if sizes else 0
    std_size = torch.tensor(sizes, dtype=torch.float32).std().item() if len(sizes) > 1 else 0
    
    if return_dict:
        return {
            'mean_entropy': mean_entropy,
            'mean_unique_sp': mean_unique,
            'mean_sp_size': mean_size,
            'std_sp_size': std_size,
        }
    else:
        return mean_entropy, mean_unique, mean_size, std_size


if __name__ == "__main__":
    """Test the loss functions."""
    print("Testing Superpixel Loss Functions")
    print("=" * 50)
    
    # Create dummy data
    B, K, H, W = 2, 50, 128, 128
    P = torch.softmax(torch.randn(B, K, H, W), dim=1)
    I = torch.rand(B, 3, H, W)
    I_recon = torch.rand(B, 3, H, W)
    
    # Test individual losses
    mi_loss = MutualInfoLoss(2.0)
    smooth_loss = SmoothnessLoss(10.0)
    edge_loss = EdgeAwareKLLoss()
    recon_loss = ReconstructionLoss()
    
    print(f"MI Loss: {mi_loss(P).item():.4f}")
    print(f"Smoothness Loss: {smooth_loss(P, I).item():.4f}")
    
    from segment import sp_soft_pool_avg, sp_project
    Zrgb = sp_soft_pool_avg(I, P)
    Ihat = sp_project(Zrgb, P)
    print(f"Edge Loss: {edge_loss(I, Ihat).item():.4f}")
    print(f"Recon Loss: {recon_loss(I_recon, I).item():.4f}")
    
    # Test combined loss
    print("\n" + "=" * 50)
    print("Testing Combined Loss")
    combined = CombinedSuperpixelLoss(w_mi=1.0, w_smooth=1.0, w_edge=1.0, w_recon=10.0)
    loss_dict = combined(P, I, I_recon, return_components=True)
    
    print(f"Total Loss: {loss_dict['total'].item():.4f}")
    print(f"  - MI: {loss_dict['mi']:.4f}")
    print(f"  - Smooth: {loss_dict['smooth']:.4f}")
    print(f"  - Edge: {loss_dict['edge']:.4f}")
    print(f"  - Recon: {loss_dict['recon']:.4f}")
    
    # Test metrics
    print("\n" + "=" * 50)
    print("Testing Metrics")
    metrics = compute_superpixel_metrics(P, I)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    
    print("\n✓ All tests passed!")