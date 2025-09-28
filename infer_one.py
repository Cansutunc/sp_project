import os, argparse, numpy as np
from PIL import Image
import torch
from segment import EdgeAwareSPModule, get_spixel_prob
from segment import sp_soft_pool_avg
from gnn_modularity import TinyGAT, ClusterHead, soft_adjacency

def load_img(path,size=None):
    img=Image.open(path).convert('RGB')
    if size is not None: img=img.resize((size[1], size[0]), Image.BILINEAR)
    x=torch.from_numpy((np.array(img).astype('float32')/255.).transpose(2,0,1))
    return x, img

def colorize(labels, K):
    import matplotlib.pyplot as plt
    cmap=plt.get_cmap('tab20', K)
    H,W=labels.shape
    out=np.zeros((H,W,3), dtype=np.float32)
    for k in range(K): out[labels==k]=np.array(cmap(k)[:3])
    return (out*255).astype(np.uint8)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--img', type=str, required=True)
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--size', type=int, nargs=2, default=[320,320])
    ap.add_argument('--K', type=int, default=200)
    ap.add_argument('--C', type=int, default=6)
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--outdir', type=str, default='./outputs')
    args=ap.parse_args()
    device = args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu'
    os.makedirs(args.outdir, exist_ok=True)
    x, img_pil = load_img(args.img, tuple(args.size)); x=x.unsqueeze(0).to(device)
    spcnn=EdgeAwareSPModule(3,32,4,args.K,False).to(device)
    gat=TinyGAT(in_dim=32*(2**(4-1)), hid_dim=128, out_dim=128, heads=4).to(device)
    head=ClusterHead(in_dim=128, n_clusters=args.C).to(device)
    ckpt=torch.load(args.ckpt, map_location=device)
    spcnn.load_state_dict(ckpt['spcnn']); gat.load_state_dict(ckpt['gat']); head.load_state_dict(ckpt['head'])
    spcnn.eval(); gat.eval(); head.eval()

    with torch.no_grad():
        outs=spcnn(x, get_spixel_prob); P=outs['P']
        seg=P.argmax(1)[0].cpu().numpy()
        Z=sp_soft_pool_avg(outs['feat'], P)
        A_soft=soft_adjacency(Z,tau=0.5)
        H=gat(Z,A_soft); Y=head(H)[0]
        labels=Y.argmax(-1).cpu().numpy()
    Hx,Wx=seg.shape; lab_img=np.zeros((Hx,Wx),dtype=np.int64)
    for k in range(args.K): lab_img[seg==k]=labels[k]
    Image.fromarray(colorize(lab_img, args.C)).save(os.path.join(args.outdir,'clusters.png'))
    img_pil.save(os.path.join(args.outdir,'input.png'))
    print('Saved to', args.outdir)
    # SP-CNN sonrası:
    seg_sp = P.argmax(dim=1)[0].cpu().numpy()
    Image.fromarray((seg_sp / seg_sp.max() * 255).astype(np.uint8)).save('./outputs/superpixels_idmap.png')
if __name__=='__main__': main()
