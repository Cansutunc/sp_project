import os, argparse
import torch
from torch import optim
from tqdm import tqdm
from data_utils import ImagesFolder
from segment import EdgeAwareSPModule, get_spixel_prob, sp_soft_pool_avg, sp_project
from losses_sp import MutualInfoLoss, SmoothnessLoss, EdgeAwareKLLoss
from gnn_modularity import TinyGAT, ClusterHead, soft_adjacency, modularity_loss

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data', type=str, required=True)
    ap.add_argument('--size', type=int, nargs=2, default=[320,320])
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--batch_size', type=int, default=2)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--K', type=int, default=200)
    ap.add_argument('--C', type=int, default=6)
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--gamma', type=float, default=1.0)
    args=ap.parse_args()

    device = args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu'

    ds=ImagesFolder(args.data, tuple(args.size))
    dl=torch.utils.data.DataLoader(ds,batch_size=args.batch_size,shuffle=True,num_workers=0,drop_last=True)

    spcnn=EdgeAwareSPModule(in_c=3,num_feat=32,num_layers=4,num_spixels=args.K,add_recon=False).to(device)
    mi, sm, ed = MutualInfoLoss(2.0), SmoothnessLoss(10.0), EdgeAwareKLLoss()
    Cf = 32*(2**(4-1))
    gat=TinyGAT(in_dim=Cf,hid_dim=128,out_dim=128,heads=4).to(device)
    head=ClusterHead(in_dim=128,n_clusters=args.C).to(device)

    opt=optim.AdamW(list(spcnn.parameters())+list(gat.parameters())+list(head.parameters()), lr=args.lr)
    os.makedirs('checkpoints', exist_ok=True)

    for ep in range(1, args.epochs+1):
        spcnn.train(); gat.train(); head.train()
        tot=0.0
        for x,_ in tqdm(dl, ncols=80, desc=f'Epoch {ep}'):
            x=x.to(device)
            outs=spcnn(x, get_spixel_prob)
            P, Fmap = outs['P'], outs['feat']
            Zrgb = sp_soft_pool_avg(x,P); Ihat = sp_project(Zrgb,P)
            L_mi, L_sm, L_ed = mi(P), sm(P,x), ed(x,Ihat)
            L_sp = L_mi + L_sm + L_ed
            Z = sp_soft_pool_avg(Fmap, P)
            A_soft = soft_adjacency(Z, tau=0.5)
            with torch.no_grad():
                A_bin = (A_soft > (A_soft.mean(dim=(-1,-2), keepdim=True))).float()
            H = gat(Z, A_soft)
            Y = head(H)
            L_mod = modularity_loss(A_bin, Y)
            L = L_sp + args.gamma * L_mod
            opt.zero_grad(set_to_none=True)
            L.backward(); opt.step()
            tot += float(L.detach())
        print(f'[ep {ep}] total={tot/len(dl):.4f}')
        torch.save({'spcnn': spcnn.state_dict(), 'gat': gat.state_dict(), 'head': head.state_dict()}, 'checkpoints/spcnn_gnn.pt')

if __name__=='__main__': main()
