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

def modularity_loss(A, Y):
    """
    A: [B, K, K] - unnormalized adjacency matrix)
    Y: [B, K, C] - soft cluster assignments
    """
    # Compute degrees
    d = A.sum(dim=-1)  # [B, K]
    m = d.sum(dim=-1, keepdim=True).clamp_min(1e-8)  # [B, 1]
    
    # Expected edges under null model
    P = (d.unsqueeze(-1) * d.unsqueeze(-2)) / (2.0 * m.unsqueeze(-1))  # [B, K, K]
    
    # Soft cluster indicator: YY[i,j] = probability i and j in same cluster
    YY = Y @ Y.transpose(1, 2)  # [B, K, K]
    
    # Modularity: Q = (1/2m) Σᵢⱼ [Aᵢⱼ - Pᵢⱼ] * YY[i,j]
    Q = ((A - P) * YY).sum(dim=(1, 2)) / (2.0 * m.squeeze(-1))
    
    # Return negative (we minimize loss)
    return -Q.mean()

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

def contrastive_cluster_loss(Z, A_soft, Y, temperature=0.1):
    """
    Nodes in same cluster should be similar, different clusters dissimilar.
    """
    # Compute cluster centroids
    cluster_sizes = Y.sum(dim=1, keepdim=True).clamp_min(1e-8)
    centroids = (Y.transpose(1, 2) @ Z) / cluster_sizes.transpose(1, 2)  # [B, C, D]
    
    # Similarity of each node to each centroid
    Z_norm = F.normalize(Z, dim=-1)
    cent_norm = F.normalize(centroids, dim=-1)
    sim = Z_norm @ cent_norm.transpose(1, 2) / temperature  # [B, K, C]
    
    # Cross-entropy with soft targets
    loss = -(Y * F.log_softmax(sim, dim=-1)).sum(dim=-1).mean()
    
    return loss

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