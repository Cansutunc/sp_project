import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from scipy.optimize import linear_sum_assignment

from data_utils import CocoStuffSupervised
from segment import EdgeAwareSPModule, get_spixel_prob, sp_soft_pool_avg
from gnn_modularity import TinyGAT, ClusterHead, soft_adjacency

def calculate_miou(pred_mask, gt_mask, num_classes):
    intersection = np.zeros(num_classes)
    union = np.zeros(num_classes)
    for c in range(num_classes):
        pred_inds = pred_mask == c
        target_inds = gt_mask == c
        intersection[c] = (pred_inds & target_inds).sum()
        union[c] = (pred_inds | target_inds).sum()
    
    # NaN bölünmelerini önle
    iou = intersection / (union + 1e-8)
    # Sadece ground truth'ta var olan sınıfların IoU'sunu dikkate al
    present_classes = [c for c in range(num_classes) if union[c] > 0]
    return np.mean(iou[present_classes]) if present_classes else 0.0

def main(args):
    device = args.device if torch.cuda.is_available() else 'cpu'

    eval_transform = transforms.Compose([
        transforms.Resize(tuple(args.size)),
        transforms.ToTensor(),
    ])
    
    # Maskeler için resize yaparken interpolasyonu NEAREST yapmalıyız
    target_transform = transforms.Compose([
        transforms.Resize(tuple(args.size), interpolation=Image.NEAREST),
        transforms.Lambda(lambda x: torch.from_numpy(np.array(x)).long()),
    ])

    dataset = CocoStuffSupervised(
        root=args.data_root, 
        annFile=args.ann_file, 
        transform=eval_transform, 
        target_transform=target_transform
    )
    
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    spcnn = EdgeAwareSPModule(3, 32, 4, args.K, False).to(device)
    gat = TinyGAT(in_dim=32 * (2**(4 - 1)), hid_dim=128, out_dim=128, heads=4).to(device)
    head = ClusterHead(in_dim=128, n_clusters=args.C).to(device)

    print(f"Loading checkpoint from: {args.ckpt}")
    checkpoint = torch.load(args.ckpt, map_location=device)
    spcnn.load_state_dict(checkpoint['spcnn'])
    gat.load_state_dict(checkpoint['gat'])
    head.load_state_dict(checkpoint['head'])
    
    spcnn.eval()
    gat.eval()
    head.eval()

    total_miou = 0.0
    with torch.no_grad():
        for img, gt_mask in tqdm(loader, desc="Evaluating mIoU"):
            img = img.to(device)
            
            outs = spcnn(img, get_spixel_prob)
            P, Fmap = outs['P'], outs['feat']
            
            Z = sp_soft_pool_avg(Fmap, P)
            A_soft = soft_adjacency(Z, tau=0.5)
            H = gat(Z, A_soft)
            Y = head(H)
            
            pred_labels = Y.argmax(-1).cpu().numpy()
            spixel_map = P.argmax(1).cpu().numpy()
            
            for i in range(img.size(0)):
                pred_seg = np.zeros_like(spixel_map[i])
                for k in range(args.K):
                    pred_seg[spixel_map[i] == k] = pred_labels[i, k]

                gt_seg = gt_mask[i].squeeze().numpy()
                
                # Unsupervised -> Supervised eşleştirme için Hungarian algoritması
                present_gt_labels = np.unique(gt_seg)
                present_pred_labels = np.unique(pred_seg)
                num_gt_labels = len(present_gt_labels)
                num_pred_labels = len(present_pred_labels)
                
                cost_matrix = np.zeros((num_pred_labels, num_gt_labels))
                for pred_idx, pred_id in enumerate(present_pred_labels):
                    for gt_idx, gt_id in enumerate(present_gt_labels):
                        intersection = np.sum((pred_seg == pred_id) & (gt_seg == gt_id))
                        cost_matrix[pred_idx, gt_idx] = -intersection
                
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                
                # En iyi eşleşmeye göre tahmin maskesini yeniden haritala
                remapped_pred_seg = np.zeros_like(pred_seg)
                for pred_idx, gt_idx in zip(row_ind, col_ind):
                    remapped_pred_seg[pred_seg == present_pred_labels[pred_idx]] = present_gt_labels[gt_idx]
                
                miou = calculate_miou(remapped_pred_seg, gt_seg, num_classes=256) # COCO'da 255'e kadar label var
                total_miou += miou

    avg_miou = total_miou / len(dataset)
    print(f"\n--- Evaluation Result ---")
    print(f"Mean IoU on COCO-Stuff val: {avg_miou:.4f}")
    print(f"-------------------------")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Unsupervised Segmentation Model")
    parser.add_argument('--ckpt', type=str, required=True, help="Path to the model checkpoint (.pt file)")
    parser.add_argument('--data_root', type=str, default='./coco/val2017', help="Path to COCO validation images")
    parser.add_argument('--ann_file', type=str, default='./coco/annotations/instances_val2017.json', help="Path to COCO annotation file")
    parser.add_argument('--size', type=int, nargs=2, default=[320, 320])
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--K', type=int, default=200, help="Number of superpixels (must match model)")
    parser.add_argument('--C', type=int, default=6, help="Number of clusters (must match model)")
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    main(args)