import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyGAT(nn.Module):
    def __init__(self,in_dim,hid_dim=128,out_dim=128,heads=4):
        super().__init__(); self.proj=nn.Linear(in_dim,hid_dim,bias=False); self.attn=nn.Parameter(torch.randn(heads,hid_dim)); self.out=nn.Linear(hid_dim,out_dim)
    def forward(self, Z, A_soft):
        """
        Z: [B, K, C]      node features
        A_soft: [B, K, K] soft adjacency mask in [0,1]
        """
        H = self.proj(Z)                    # [B, K, Hd]
        B, K, Hd = H.shape

        # node score per head: T[b,k,a] = <H[b,k,:], attn[a,:]>  -> [B, K, heads]
        H_heads = torch.einsum('bkh,ah->bka', H, self.attn)   # (k: node, a: head)

        # pairwise scores per head: s[b,a,i,j] = T[b,i,a] + T[b,j,a]
        s = H_heads.unsqueeze(2) + H_heads.unsqueeze(1)       # [B, K, K, heads]
        s = s.permute(0, 3, 1, 2).contiguous()                # [B, heads, K, K]

        # mask with A_soft
        s = s.masked_fill(A_soft.unsqueeze(1) <= 0, -1e9)

        # attention weights per head
        alpha = torch.softmax(s, dim=-1)                      # [B, heads, K, K]

        # aggregate per head: alpha[b,a,i,j] * H[b,j,c]  sum over j -> [B, heads, K, Hd]
        H_per_head = torch.einsum('baij,bjc->baic', alpha, H).contiguous()

        # average over heads -> [B, K, Hd]
        H_agg = H_per_head.mean(dim=1)

        return torch.nn.functional.elu(self.out(H_agg))       # [B, K, out_dim]


class ClusterHead(nn.Module):
    def __init__(self,in_dim,n_clusters): super().__init__(); self.head=nn.Linear(in_dim,n_clusters)
    def forward(self,Z): return F.softmax(self.head(Z),dim=-1)

# def soft_adjacency(Z,tau=0.5):
#     ZN=F.normalize(Z,dim=-1); S=ZN@ZN.transpose(1,2); S=(S+1)*0.5; 
    
#     return F.softmax(S/tau,dim=-1)

# def modularity_loss(A_soft,Y):
#     d=A_soft.sum(-1); m=(d.sum(-1,keepdim=True)/2.0).clamp_min(1e-8)
#     P=(d.unsqueeze(-1)*d.unsqueeze(-2))/(2.0*m.unsqueeze(-1))
#     YY=Y@Y.transpose(1,2); Q=((A_soft-P)*YY).sum((1,2))/(2.0*m.squeeze(-1))
#     return (-Q).mean()

def soft_adjacency(Z, tau=0.5):
    ZN = F.normalize(Z, dim=-1)
    S = ZN @ ZN.transpose(1,2)  # Cosine similarity [-1, 1]
    S = (S + 1) * 0.5           # Map to [0, 1]
    S = torch.sigmoid((S - 0.5) / tau)  # Use sigmoid instead of softmax
    # Zero out self-loops
    mask = torch.eye(S.shape[1], device=S.device).unsqueeze(0)
    S = S * (1 - mask)
    return S

# def modularity_loss(A, Y):
#     """
#     A: [B, K, K] - unnormalized adjacency matrix)
#     Y: [B, K, C] - soft cluster assignments
#     """
#     # Compute degrees
#     d = A.sum(dim=-1)  # [B, K]
#     m = d.sum(dim=-1, keepdim=True).clamp_min(1e-8)  # [B, 1]
    
#     # Expected edges under null model
#     P = (d.unsqueeze(-1) * d.unsqueeze(-2)) / (2.0 * m.unsqueeze(-1))  # [B, K, K]
    
#     # Soft cluster indicator: YY[i,j] = probability i and j in same cluster
#     YY = Y @ Y.transpose(1, 2)  # [B, K, K]
    
#     # Modularity: Q = (1/2m) Σᵢⱼ [Aᵢⱼ - Pᵢⱼ] * YY[i,j]
#     Q = ((A - P) * YY).sum(dim=(1, 2)) / (2.0 * m.squeeze(-1))
    
#     # Return negative (we minimize loss)
#     return -Q.mean()

def cluster_reconstruction_loss(Z, A_soft, Y):
    """
    Encourages nodes in same cluster to be similar.
    Z: [B, K, C] - node features
    A_soft: [B, K, K] - soft adjacency (row-normalized is OK here)
    Y: [B, K, n_clusters] - cluster assignments
    """
    # Compute cluster centroids
    cluster_sizes = Y.sum(dim=1, keepdim=True).clamp_min(1e-8)  # [B, 1, C]
    centroids = (Y.transpose(1, 2) @ Z) / cluster_sizes.transpose(1, 2)  # [B, n_clusters, feat_dim]
    
    # Reconstruct features from cluster assignments
    Z_recon = Y @ centroids  # [B, K, feat_dim]
    
    # MSE loss
    recon_loss = F.mse_loss(Z, Z_recon)
    
    # Entropy regularization (prevent collapse)
    cluster_probs = Y.mean(dim=1)  # [B, n_clusters]
    entropy = -(cluster_probs * torch.log(cluster_probs + 1e-10)).sum(dim=-1).mean()
    
    return recon_loss - 0.1 * entropy

def contrastive_cluster_loss(Z, A, Y, temperature=0.1):
    """
    Graph-aware contrastive loss
    """
    cluster_sizes = Y.sum(dim=1, keepdim=True).clamp_min(1e-8)
    centroids = (Y.transpose(1, 2) @ Z) / cluster_sizes.transpose(1, 2)
    
    Z_norm = F.normalize(Z, dim=-1)
    cent_norm = F.normalize(centroids, dim=-1)
    sim = Z_norm @ cent_norm.transpose(1, 2) / temperature  # [B, K, C]
    
    # Cross-entropy
    loss_ce = -(Y * F.log_softmax(sim, dim=-1)).sum(dim=-1).mean()
    
    # Graph-aware: connected nodes should be in same cluster
    Y_sim = Y @ Y.transpose(1, 2)  # [B, K, K]
    graph_penalty = -(A * Y_sim).sum(dim=(1, 2)) / (A.sum(dim=(1, 2)) + 1e-8)
    graph_penalty = graph_penalty.mean()
    
    return loss_ce + 0.1 * graph_penalty  # ← A kullanılıyor!

def graph_cluster_loss(Z, A_soft, Y):
    """
    Clusters should respect graph structure + be balanced.
    """
    # Smoothness: connected nodes should have similar cluster assignments
    YY = Y @ Y.transpose(1, 2)  # [B, K, K] - cluster agreement matrix
    smoothness = -(A_soft * YY).sum(dim=(1, 2)).mean()
    
    # Balance: encourage uniform cluster distribution
    cluster_sizes = Y.sum(dim=1)  # [B, n_clusters]
    balance = cluster_sizes.std(dim=-1).mean()
    
    # Entropy: prevent collapse to single cluster
    entropy = -(Y * torch.log(Y + 1e-10)).sum(dim=-1).mean()
    
    return smoothness + 0.5 * balance - 0.1 * entropy

def spatial_adjacency(P, k=8):
    """
    Spatial adjacency based on superpixel centers.
    P: [B, K, H, W] - superpixel probabilities
    k: number of nearest neighbors
    Returns: [B, K, K] - binary adjacency matrix
    """
    B, K, H, W = P.shape
    device = P.device
    
    # Create coordinate grid
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    coords = torch.stack([xx.float(), yy.float()], dim=0)  # [2, H, W]
    
    # Compute superpixel centers
    P_flat = P.view(B, K, -1)  # [B, K, H*W]
    coords_flat = coords.view(2, -1).unsqueeze(0).expand(B, -1, -1)  # [B, 2, H*W]
    
    # Weighted average: center[k] = Σ P[k,i] * coord[i] / Σ P[k,i]
    centers = torch.einsum('bkn,bdn->bkd', P_flat, coords_flat)  # [B, K, 2]
    centers = centers / P_flat.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    
    # Compute pairwise distances
    dist = torch.cdist(centers, centers)  # [B, K, K]
    
    # K-NN graph: keep k nearest neighbors
    _, idx = dist.topk(k+1, largest=False, dim=-1)  # [B, K, k+1] (includes self)
    
    # Create adjacency matrix
    A = torch.zeros(B, K, K, device=device)
    for b in range(B):
        for i in range(K):
            neighbors = idx[b, i, 1:]  # Skip self (first element)
            A[b, i, neighbors] = 1.0
    
    # Make symmetric
    A = (A + A.transpose(1, 2)) / 2.0
    A = (A > 0).float()
    
    return A


def hybrid_adjacency(P, Z, k=8, alpha=0.7, tau=0.5):
    """
    Hybrid graph: spatial + feature similarity
    P: [B, K, H, W] - superpixel probabilities
    Z: [B, K, C] - node features
    alpha: weight for spatial (1-alpha for features)
    """
    A_spatial = spatial_adjacency(P, k=k)
    A_feature = soft_adjacency(Z, tau=tau)
    
    # Combine
    A = alpha * A_spatial + (1 - alpha) * A_feature
    
    # Row-normalize
    A = A / (A.sum(dim=-1, keepdim=True) + 1e-8)
    
    return A


def cluster_quality_loss(Y, A):
    """
    Multi-objective cluster quality loss:
    1. Smoothness: connected nodes should agree
    2. Balance: clusters should have similar sizes
    3. Entropy: prevent collapse
    
    Y: [B, K, C] - soft cluster assignments
    A: [B, K, K] - adjacency matrix
    """
    # 1. Graph smoothness: neighbors should be in same cluster
    Y_sim = Y @ Y.transpose(1, 2)  # [B, K, K] - cluster agreement matrix
    smoothness_loss = -(A * Y_sim).sum(dim=(1, 2)) / (A.sum(dim=(1, 2)) + 1e-8)
    smoothness_loss = smoothness_loss.mean()
    
    # 2. Cluster balance: prevent size imbalance
    cluster_sizes = Y.sum(dim=1)  # [B, C]
    target_size = cluster_sizes.mean(dim=-1, keepdim=True)
    balance_loss = ((cluster_sizes - target_size) ** 2).mean()
    
    # 3. Entropy regularization: prevent collapse to single cluster
    cluster_probs = Y.mean(dim=1)  # [B, C]
    entropy = -(cluster_probs * torch.log(cluster_probs + 1e-10)).sum(dim=-1).mean()
    
    return smoothness_loss + 0.2 * balance_loss - 0.1 * entropy


def adaptive_loss_weights(L_sp, L_gnn, alpha=0.12):
    """
    GradNorm-inspired adaptive loss weighting.
    Balances loss magnitudes to prevent one dominating.
    
    Returns: (w_sp, w_gnn) - loss weights
    """
    # Initialize on first call
    if not hasattr(adaptive_loss_weights, 'initialized'):
        adaptive_loss_weights.L0_sp = L_sp.detach().clone()
        adaptive_loss_weights.L0_gnn = L_gnn.detach().clone()
        adaptive_loss_weights.initialized = True
    
    # Compute relative loss decrease
    r_sp = L_sp.detach() / (adaptive_loss_weights.L0_sp + 1e-8)
    r_gnn = L_gnn.detach() / (adaptive_loss_weights.L0_gnn + 1e-8)
    
    # Inverse training rate weighting
    w_sp = 1.0
    w_gnn = ((r_sp / (r_gnn + 1e-8)) ** alpha).clamp(0.1, 10.0)
    
    return w_sp, w_gnn.item()