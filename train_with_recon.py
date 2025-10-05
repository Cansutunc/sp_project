import sys
import os
import argparse
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from data_utils import ImagesFolder
from segment import EdgeAwareSPModule, get_spixel_prob, sp_soft_pool_avg, sp_project
from losses_sp import EdgeGuidedLoss, MutualInfoLoss, SmoothnessLoss, EdgeAwareKLLoss
from gnn_loss import TinyGAT, ClusterHead, soft_adjacency, contrastive_cluster_loss
from skimage.segmentation import mark_boundaries
import torch.nn.functional as F 

def visualize_superpixels(image_tensor, p_map, out_path):
    image_pil = Image.fromarray((image_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    seg = p_map.argmax(dim=1)[0].cpu().numpy()
    boundary_img = mark_boundaries(np.array(image_pil), seg)
    plt.imsave(out_path, boundary_img)

def create_coord_grid_normalized(x):
    """Normalize RGB and coordinates together - CNNRIM style."""
    B, _, H, W = x.shape
    device = x.device
    
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )
    
    coords = torch.stack([xx, yy], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)
    x_with_coords = torch.cat([x, coords], dim=1)
    
    # NORMALIZE TOGETHER 
    mean = x_with_coords.mean(dim=(2, 3), keepdim=True)
    std = x_with_coords.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    x_with_coords = (x_with_coords - mean) / std
    
    return x_with_coords

def main(args):
    device = args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu'
    
    # --- Data Loading ---
    if args.dataset == 'cocostuff':
        print(f"Loading COCO-Stuff dataset, using {args.subset_fraction*100:.1f}% of the data.")
        train_dataset = ImagesFolder(root='./coco/train2017', size=tuple(args.size))
        val_dataset = ImagesFolder(root='./coco/val2017', size=tuple(args.size))
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, drop_last=True, pin_memory=True)
    else:
        print(f"Loading custom dataset from: {args.data}")
        dataset = ImagesFolder(args.data, tuple(args.size))
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(args.val_split * dataset_size))
        np.random.seed(42)
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=2, drop_last=True)
        val_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=2, drop_last=True)

    # --- Model ---
    # SPCNN with reconstruction head enabled
    spcnn = EdgeAwareSPModule(
        in_c=5, 
        num_feat=32, 
        num_layers=4, 
        num_spixels=args.K
    ).to(device)

    # Loss functions
    mi = MutualInfoLoss(args.mi_coef)
    sm = SmoothnessLoss(args.smooth_sigma)
    ed = EdgeAwareKLLoss()
  
    
    Cf = 32 * (2**(4 - 1))
    gat = TinyGAT(in_dim=Cf, hid_dim=128, out_dim=128, heads=4).to(device)
    head = ClusterHead(in_dim=128, n_clusters=args.C).to(device)

    # Optimizer
    optimizer = optim.AdamW(
        list(spcnn.parameters()) + list(gat.parameters()) + list(head.parameters()), 
        lr=args.lr
    )
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=args.patience // 2)
    
    # --- Checkpoint Setup ---
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.vis_dir, exist_ok=True)
    history = {
        'train_loss': [], 'train_loss_sp': [], 'train_loss_recon': [], 'train_loss_gnn': [],
        'val_loss': [], 'val_loss_sp': [], 'val_loss_recon': [], 'val_loss_gnn': []
    }
    best_val_loss = float('inf')
    epochs_no_improve = 0
    start_epoch = 1

    # --- Resume ---
    last_checkpoint_path = os.path.join(args.checkpoint_dir, 'last.pt')
    if args.resume and os.path.exists(last_checkpoint_path):
        print(f"Resuming training from checkpoint: {last_checkpoint_path}")
        checkpoint = torch.load(last_checkpoint_path, map_location=device)
        spcnn.load_state_dict(checkpoint['spcnn'])
        gat.load_state_dict(checkpoint['gat'])
        head.load_state_dict(checkpoint['head'])
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
        if 'best_val_loss' in checkpoint:
            best_val_loss = checkpoint['best_val_loss']
        print(f"Resumed successfully. Starting from epoch {start_epoch}.")

    # --- Training Loop ---
    for epoch in range(start_epoch, args.epochs + 1):
        # === TRAINING ===
        spcnn.train(); gat.train(); head.train()
        train_loss, train_loss_sp, train_loss_recon, train_loss_gnn = 0.0, 0.0, 0.0, 0.0
        
        for batch_idx, (x, _) in enumerate(tqdm(train_loader, ncols=80, desc=f'Epoch {epoch} [Train]')):
            x = x.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            # Forward SPCNN
            x_with_coords = create_coord_grid_normalized(x)  # Zaten [B, 5, H, W] döndürür

            outs = spcnn(x_with_coords, get_spixel_prob)
            P, Fmap = outs['P'], outs['feat']

            # Superpixel losses
            L_mi = mi(P)
            L_smooth = sm(P, x)

            # RGB reconstruction from superpixels
            Zrgb = sp_soft_pool_avg(x, P)
            Ihat = sp_project(Zrgb, P)

            L_edge = ed(x, Ihat)
            L_recon = F.mse_loss(Ihat, x)  # ← Yeni: basit MSE

            # Combined
            L_sp = args.w_mi*L_mi + args.w_smooth*L_smooth + args.w_edge*L_edge + args.w_recon*L_recon
            # Combined superpixel loss with weights (matching paper)

            # GNN Loss
            Z = sp_soft_pool_avg(Fmap, P)
            A_soft = soft_adjacency(Z, tau=args.tau)
            H = gat(Z, A_soft)
            Y = head(H)
            L_gnn = contrastive_cluster_loss(Z, A_soft, Y)
            
            # Total loss
            loss = L_sp + args.gamma * L_gnn
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_loss_sp += L_sp.item()
            train_loss_recon += L_recon.item()
            train_loss_gnn += L_gnn.item()
            # ========== DETAILED LOSS TRACKING ==========
            # Print detailed loss breakdown every 100 batches
            if (batch_idx % 100 == 0 and batch_idx > 0) or batch_idx == len(train_loader) - 1:
                # Superpixel statistics
                sp_mass = P.view(P.shape[0], P.shape[1], -1).mean(-1)  # [B, K]
                active_sps = (sp_mass > 0.01).sum().item() / P.shape[0]
                mean_sp_mass = sp_mass.mean().item()
                std_sp_mass = sp_mass.std().item()
                
                print(f"\n  Superpixel Stats:")
                print(f"    └─ Active SPs:     {active_sps:.1f}/{args.K}")
                print(f"    └─ Mean mass:      {mean_sp_mass:.4f} (target: {1.0/args.K:.4f})")
                print(f"    └─ Std mass:       {std_sp_mass:.4f} (lower = more balanced)")
                print(f"{'='*70}\n")
            # Print detailed loss breakdown every 100 batches
            if batch_idx % 100 == 0 and batch_idx > 0:
                print(f"\n[Batch {batch_idx}] Loss breakdown:")
                print(f"  MI: {L_mi.item():.4f}, Smooth: {L_smooth.item():.4f}, "
                      f"Edge: {L_edge.item():.4f}, Recon: {L_recon.item():.4f}")
                print(f"  SP Total: {L_sp.item():.4f}, GNN: {L_gnn.item():.4f}, "
                      f"Total: {loss.item():.4f}")
        
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_loss_sp'].append(train_loss_sp / len(train_loader))
        history['train_loss_recon'].append(train_loss_recon / len(train_loader))
        history['train_loss_gnn'].append(train_loss_gnn / len(train_loader))

        # === VALIDATION ===
        spcnn.eval(); gat.eval(); head.eval()
        val_loss, val_loss_sp, val_loss_recon, val_loss_gnn = 0.0, 0.0, 0.0, 0.0
        
        with torch.no_grad():
            for i, (x, _) in enumerate(tqdm(val_loader, ncols=80, desc=f'Epoch {epoch} [Val]')):
                x = x.to(device)
                x_with_coords = create_coord_grid_normalized(x)
                outs = spcnn(x_with_coords, get_spixel_prob)
                P, Fmap = outs['P'], outs['feat']
                
                # Superpixel losses
                L_mi = mi(P)
                L_smooth = sm(P, x)
                Zrgb = sp_soft_pool_avg(x, P)
                Ihat = sp_project(Zrgb, P)
                L_edge = ed(x, Ihat)
                L_recon = F.mse_loss(Ihat, x) 
                edge_guided = EdgeGuidedLoss()
                L_edge_guided = edge_guided(P, x)

                L_sp = (args.w_mi * L_mi + 
                       args.w_smooth * L_smooth + 
                       args.w_edge * L_edge +
                       args.w_recon * L_recon +
                       args.w_edge_guided * L_edge_guided)
                
                # GNN loss
                Z = sp_soft_pool_avg(Fmap, P)
                A_soft = soft_adjacency(Z, tau=args.tau)
                H = gat(Z, A_soft)
                Y = head(H)
                L_gnn = contrastive_cluster_loss(Z, A_soft, Y)
                
                loss = L_sp + args.gamma * L_gnn
                val_loss += loss.item()
                val_loss_sp += L_sp.item()
                val_loss_recon += L_recon.item()
                val_loss_gnn += L_gnn.item()
                
                # Visualize
                if i == 0 and epoch % args.vis_interval == 0:
                    visualize_superpixels(x[0], P[0].unsqueeze(0), 
                                        os.path.join(args.vis_dir, f"epoch_{epoch}_val_sp.png"))
                    
                    # Save reconstruction if available
                    if 'recon' in outs:
                        recon_img = outs['recon'][0].detach().cpu().numpy().transpose(1, 2, 0)
                        recon_img = np.clip(recon_img, 0, 1)
                        plt.imsave(os.path.join(args.vis_dir, f"epoch_{epoch}_recon.png"), recon_img)
        
        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        history['val_loss_sp'].append(val_loss_sp / len(val_loader))
        history['val_loss_recon'].append(val_loss_recon / len(val_loader))
        history['val_loss_gnn'].append(val_loss_gnn / len(val_loader))
        
        print(f'\n{"="*80}')
        print(f'[Epoch {epoch}] Summary:')
        print(f'  Train - Total: {history["train_loss"][-1]:.4f}, '
              f'SP: {history["train_loss_sp"][-1]:.4f}, '
              f'Recon: {history["train_loss_recon"][-1]:.4f}, '
              f'GNN: {history["train_loss_gnn"][-1]:.4f}')
        print(f'  Val   - Total: {avg_val_loss:.4f}, '
              f'SP: {history["val_loss_sp"][-1]:.4f}, '
              f'Recon: {history["val_loss_recon"][-1]:.4f}, '
              f'GNN: {history["val_loss_gnn"][-1]:.4f}')
        print(f'{"="*80}\n')
        
        # --- Checkpoint Saving ---
        checkpoint_data = {
            'epoch': epoch, 'spcnn': spcnn.state_dict(), 'gat': gat.state_dict(),
            'head': head.state_dict(), 'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(), 'best_val_loss': best_val_loss 
        }
        
        torch.save(checkpoint_data, os.path.join(args.checkpoint_dir, f'epoch_{epoch}.pt'))
        torch.save(checkpoint_data, os.path.join(args.checkpoint_dir, 'last.pt'))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_data['best_val_loss'] = best_val_loss
            torch.save(checkpoint_data, os.path.join(args.checkpoint_dir, 'best.pt'))
            epochs_no_improve = 0
            print("✓ Validation loss improved, saving best model.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break
        
        scheduler.step(avg_val_loss)

    # --- Plot Results ---
    plt.figure(figsize=(20, 5))
    
    plt.subplot(1, 4, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.title('Total Loss'); plt.xlabel('Epochs'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True)

    plt.subplot(1, 4, 2)
    plt.plot(history['train_loss_sp'], label='Train SP')
    plt.plot(history['val_loss_sp'], label='Val SP')
    plt.title('Superpixel Loss'); plt.xlabel('Epochs'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True)
    
    plt.subplot(1, 4, 3)
    plt.plot(history['train_loss_recon'], label='Train Recon')
    plt.plot(history['val_loss_recon'], label='Val Recon')
    plt.title('Reconstruction Loss'); plt.xlabel('Epochs'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True)
    
    plt.subplot(1, 4, 4)
    plt.plot(history['train_loss_gnn'], label='Train GNN')
    plt.plot(history['val_loss_gnn'], label='Val GNN')
    plt.title('GNN Loss'); plt.xlabel('Epochs'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.checkpoint_dir, 'loss_curves.png'))
    print(f"Saved loss curves to {args.checkpoint_dir}/loss_curves.png")

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="Unsupervised Superpixel Segmentation with Reconstruction")
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--dataset', type=str, default='cocostuff', choices=['custom', 'cocostuff'])
    ap.add_argument('--subset_fraction', type=float, default=1.0)
    ap.add_argument('--data', type=str, default='./data')
    ap.add_argument('--size', type=int, nargs=2, default=[320, 320])
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch_size', type=int, default=2)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--K', type=int, default=200)
    ap.add_argument('--C', type=int, default=6)
    ap.add_argument('--device', type=str, default='cuda')
    
    # Loss weights (matching paper: [1, 2, 10] for [MI, Smooth+Contour, Recon])
    ap.add_argument('--w_mi', type=float, default=1.0, help='Weight for Mutual Info loss')
    ap.add_argument('--w_smooth', type=float, default=1.0, help='Weight for Smoothness loss')
    ap.add_argument('--w_edge', type=float, default=1.0, help='Weight for Edge-Aware loss')
    ap.add_argument('--w_recon', type=float, default=10.0, help='Weight for Reconstruction loss')
    ap.add_argument('--w_edge_guided', type=float, default=2.0, help='Weight for Edge-Guided loss')
    ap.add_argument('--gamma', type=float, default=0.5, help='Weight for GNN loss')
    
    # Loss hyperparameters
    ap.add_argument('--mi_coef', type=float, default=2.0, help='MI loss cardinality coefficient')
    ap.add_argument('--smooth_sigma', type=float, default=10.0, help='Smoothness loss sigma')
    ap.add_argument('--tau', type=float, default=0.5, help='Temperature for soft adjacency')
    
    ap.add_argument('--val_split', type=float, default=0.2)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--checkpoint_dir', type=str, default='./checkpoints_recon')
    ap.add_argument('--vis_dir', type=str, default='./visualizations_recon')
    ap.add_argument('--vis_interval', type=int, default=1)
    
    args = ap.parse_args()
    main(args)
