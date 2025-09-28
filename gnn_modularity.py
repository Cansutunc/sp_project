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

def soft_adjacency(Z,tau=0.5):
    ZN=F.normalize(Z,dim=-1); S=ZN@ZN.transpose(1,2); S=(S+1)*0.5; return F.softmax(S/tau,dim=-1)

def modularity_loss(A_bin,Y):
    d=A_bin.sum(-1); m=(d.sum(-1,keepdim=True)/2.0).clamp_min(1e-8)
    P=(d.unsqueeze(-1)*d.unsqueeze(-2))/(2.0*m.unsqueeze(-1))
    YY=Y@Y.transpose(1,2); Q=((A_bin-P)*YY).sum((1,2))/(2.0*m.squeeze(-1))
    return (-Q).mean()
