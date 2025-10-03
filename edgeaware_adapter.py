"""
Adapter to integrate EdgeAwareSpixel into your pipeline.
This maintains compatibility with your existing GNN training code.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Assuming you've cloned and can import from EdgeAwareSpixel repo
# from EdgeAwareSpixel.model import EdgeAwareSpixelNet  # Adjust import path

class EdgeAwareSpixelAdapter(nn.Module):
    """
    Wrapper around EdgeAwareSpixel to make it compatible with your pipeline.
    Converts hard assignments to soft probabilities for gradient flow.
    """
    def __init__(self, num_spixels=200, feature_dim=256, tau=0.5):
        super().__init__()
        self.num_spixels = num_spixels
        self.feature_dim = feature_dim
        self.tau = tau  # Temperature for soft assignment
        
        # Initialize EdgeAwareSpixel model (adjust parameters as needed)
        # You'll need to import the actual model from the repo
        # self.spixel_net = EdgeAwareSpixelNet(
        #     n_spixels=num_spixels,
        #     feature_dim=feature_dim
        # )
        
        # Feature extraction backbone (similar to your current one)
        self.feat_conv = self._build_feature_backbone(in_c=3, num_feat=32, num_layers=4)
        self.c_feat = 32 * (2**(4-1))  # 256
        
    def _build_feature_backbone(self, in_c, num_feat, num_layers):
        """Build CNN backbone for feature extraction."""
        layers = []
        c = in_c
        for i in range(num_layers):
            co = num_feat * (2**i)
            layers.append(nn.Conv2d(c, co, 3, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(co, affine=True))
            layers.append(nn.ReLU(inplace=True))
            c = co
        return nn.Sequential(*layers)
    
    def hard_to_soft_assignment(self, spixel_indices, spixel_centers, pixel_features):
        """
        Convert hard assignments to soft probabilities for gradient flow.
        
        Args:
            spixel_indices: [B, H, W] - hard assignments (integer labels)
            spixel_centers: [B, K, C] - superpixel feature centroids
            pixel_features: [B, C, H, W] - per-pixel features
        
        Returns:
            P: [B, K, H, W] - soft probability assignment
        """
        B, C, H, W = pixel_features.shape
        K = spixel_centers.shape[1]
        
        # Reshape for computation
        pixel_feat_flat = pixel_features.view(B, C, -1).transpose(1, 2)  # [B, HW, C]
        
        # Compute similarity between pixels and superpixel centers
        # Using cosine similarity
        pixel_norm = F.normalize(pixel_feat_flat, dim=-1)  # [B, HW, C]
        center_norm = F.normalize(spixel_centers, dim=-1)  # [B, K, C]
        
        # Similarity matrix: [B, HW, K]
        similarity = torch.bmm(pixel_norm, center_norm.transpose(1, 2))
        
        # Convert to soft assignment with temperature
        P_flat = F.softmax(similarity / self.tau, dim=-1)  # [B, HW, K]
        
        # Reshape to [B, K, H, W]
        P = P_flat.transpose(1, 2).view(B, K, H, W)
        
        return P
    
    def forward(self, x, get_prob=None):
        """
        Forward pass compatible with your existing pipeline.
        
        Args:
            x: [B, 5, H, W] - RGB + coordinates (we'll ignore coordinates)
            get_prob: unused, kept for compatibility
        
        Returns:
            dict with keys: 'P', 'feat', 'logits' (for compatibility)
        """
        # Extract RGB channels only (ignore coordinates)
        rgb = x[:, :3, :, :] if x.shape[1] == 5 else x
        
        # Extract features
        feat = self.feat_conv(rgb)  # [B, C, H, W]
        
        # Run EdgeAwareSpixel (you need to implement this part)
        # spixel_indices, spixel_centers = self.spixel_net(rgb, feat)
        
        # PLACEHOLDER: Simulate EdgeAwareSpixel output for now
        B, C, H, W = feat.shape
        K = self.num_spixels
        
        # Simulate hard assignment (replace with actual EdgeAwareSpixel output)
        spixel_indices = torch.randint(0, K, (B, H, W), device=x.device)
        
        # Simulate superpixel centers by pooling features
        spixel_centers = torch.randn(B, K, C, device=x.device)  # Replace with actual centers
        
        # Convert hard assignment to soft probabilities
        P = self.hard_to_soft_assignment(spixel_indices, spixel_centers, feat)
        
        # For compatibility, create dummy logits (P is already softmaxed)
        logits = torch.log(P + 1e-10) * self.tau
        
        return {
            'P': P,           # [B, K, H, W] soft assignment
            'feat': feat,     # [B, C, H, W] feature map
            'logits': logits, # [B, K, H, W] log probabilities
            'hard_assignment': spixel_indices  # [B, H, W] hard labels (for visualization)
        }


class EdgeAwareSpixelDirect(nn.Module):
    """
    Direct implementation inspired by EdgeAwareSpixel paper.
    This is a simpler alternative if you can't integrate the full repo.
    """
    def __init__(self, in_c=5, num_feat=32, num_layers=4, num_spixels=200, 
                 grid_size=14, feature_dim=256):
        super().__init__()
        self.num_spixels = num_spixels
        self.grid_size = grid_size
        self.K_side = int(np.sqrt(num_spixels))  # Assume square grid
        
        # Feature extraction
        self.feat_conv = self._build_backbone(in_c, num_feat, num_layers)
        self.c_feat = num_feat * (2**(num_layers-1))
        
        # Learnable superpixel offsets (from grid positions)
        self.sp_offsets = nn.Parameter(torch.zeros(1, num_spixels, 2))
        
        # Feature projection for similarity computation
        self.feat_proj = nn.Conv2d(self.c_feat, feature_dim, 1)
        
    def _build_backbone(self, in_c, num_feat, num_layers):
        layers = []
        c = in_c
        for i in range(num_layers):
            co = num_feat * (2**i)
            layers.append(nn.Conv2d(c, co, 3, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(co, affine=True))
            layers.append(nn.ReLU(inplace=True))
            c = co
        return nn.Sequential(*layers)
    
    def get_spixel_centers(self, H, W, device):
        """Initialize superpixel centers on a grid with learnable offsets."""
        K_side = self.K_side
        
        # Grid positions
        y_grid = torch.linspace(0, H-1, K_side, device=device)
        x_grid = torch.linspace(0, W-1, K_side, device=device)
        yy, xx = torch.meshgrid(y_grid, x_grid, indexing='ij')
        
        # Flatten to [K, 2]
        grid_pos = torch.stack([xx.flatten(), yy.flatten()], dim=-1)  # [K, 2]
        
        # Add learnable offsets (scaled)
        offsets_scaled = self.sp_offsets.squeeze(0) * (H / K_side)  # [K, 2]
        centers = grid_pos + offsets_scaled
        
        # Clamp to image boundaries
        centers = torch.clamp(centers, min=0, max=torch.tensor([W-1, H-1], device=device))
        
        return centers.unsqueeze(0)  # [1, K, 2]
    
    def compute_soft_assignment(self, pixel_features, sp_centers, pixel_coords):
        """
        Compute soft assignment based on feature and spatial similarity.
        
        Args:
            pixel_features: [B, C, H, W]
            sp_centers: [B, K, 2] - (x, y) coordinates
            pixel_coords: [B, 2, H, W] - per-pixel coordinates
        
        Returns:
            P: [B, K, H, W] - soft assignment probabilities
        """
        B, C, H, W = pixel_features.shape
        K = sp_centers.shape[1]
        
        # Reshape
        feat_flat = pixel_features.view(B, C, -1).transpose(1, 2)  # [B, HW, C]
        coord_flat = pixel_coords.view(B, 2, -1).transpose(1, 2)  # [B, HW, 2]
        
        # Compute spatial distances to each superpixel center
        # coord_flat: [B, HW, 2], sp_centers: [B, K, 2]
        spatial_dist = torch.cdist(coord_flat, sp_centers, p=2)  # [B, HW, K]
        
        # Normalize spatial distances
        spatial_weight = torch.exp(-spatial_dist / (H * 0.1))  # [B, HW, K]
        
        # Feature similarity (cosine)
        # For simplicity, compute feature centroids by spatial proximity
        # This is a simplified version - full implementation would iterate
        
        # Use spatial proximity as soft assignment (simplified)
        P_flat = F.softmax(-spatial_dist * 10, dim=-1)  # [B, HW, K]
        
        # Reshape
        P = P_flat.transpose(1, 2).view(B, K, H, W)
        
        return P
    
    def forward(self, x, get_prob=None):
        """Forward pass."""
        B, _, H, W = x.shape
        
        # Extract features
        feat = self.feat_conv(x)  # [B, C, H, W]
        feat_proj = self.feat_proj(feat)  # [B, feature_dim, H, W]
        
        # Get superpixel centers
        sp_centers = self.get_spixel_centers(H, W, x.device).repeat(B, 1, 1)  # [B, K, 2]
        
        # Create pixel coordinate grid
        yy, xx = torch.meshgrid(
            torch.arange(H, device=x.device),
            torch.arange(W, device=x.device),
            indexing='ij'
        )
        pixel_coords = torch.stack([xx, yy], dim=0).unsqueeze(0).float()  # [1, 2, H, W]
        pixel_coords = pixel_coords.repeat(B, 1, 1, 1)
        
        # Compute soft assignment
        P = self.compute_soft_assignment(feat_proj, sp_centers, pixel_coords)
        
        # Create logits for compatibility
        logits = torch.log(P + 1e-10)
        
        return {
            'P': P,
            'feat': feat,
            'logits': logits,
            'sp_centers': sp_centers
        }


# Example usage
if __name__ == "__main__":
    import numpy as np
    
    # Test adapter
    model = EdgeAwareSpixelDirect(
        in_c=5, 
        num_feat=32, 
        num_layers=4, 
        num_spixels=196,  # 14x14 grid
        grid_size=14
    )
    
    x = torch.randn(2, 5, 320, 320)
    
    with torch.no_grad():
        output = model(x)
        print("Output shapes:")
        for k, v in output.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape}")
