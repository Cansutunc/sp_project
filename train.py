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
import subprocess
import pandas as pd
from data_utils import ImagesFolder
from segment import EdgeAwareSPModule, get_spixel_prob, sp_soft_pool_avg, sp_project
from skimage.segmentation import mark_boundaries
from losses_sp import MutualInfoLoss, SmoothnessLoss, feature_reconstruction_loss
from gnn_loss import (TinyGAT, ClusterHead, soft_adjacency,
                      contrastive_cluster_loss, spatial_adjacency, hybrid_adjacency,
                      cluster_quality_loss, adaptive_loss_weights)

def visualize_superpixels(image_tensor, p_map, out_path):
    image_pil = Image.fromarray((image_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    seg = p_map.argmax(dim=1)[0].cpu().numpy()
    # PIL objesini NumPy array'ine çevirerek hatayı düzelt
    boundary_img = mark_boundaries(np.array(image_pil), seg)
    
    plt.imsave(out_path, boundary_img)

def create_coord_grid(x, scale=0.1): 
    B, _, H, W = x.shape
    device = x.device
    yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    # Coordinates [-scale, scale] aralığına normalleştir
    coords = torch.stack([
        ((xx.float() / (W - 1)) * 2 - 1) * scale, # <-- scale ile çarptık
        ((yy.float() / (H - 1)) * 2 - 1) * scale  # <-- scale ile çarptık
    ], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)
    
    return coords

def main(args):
    device = args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu'
    if args.dataset == 'cocostuff':
        print(f"Loading COCO-Stuff dataset, using {args.subset_fraction*100:.1f}% of the data.")
        train_dataset = ImagesFolder(root='./coco/train2017', size=tuple(args.size), subset_fraction=args.subset_fraction)
        val_dataset = ImagesFolder(root='./coco/val2017', size=tuple(args.size), subset_fraction=args.subset_fraction)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, drop_last=True, pin_memory=True)
    else: # Kendi özel veri setiniz
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

    # Model,Loss functions and optimizer 
    spcnn = EdgeAwareSPModule(in_c=5, num_feat=32, num_layers=4, num_spixels=args.K).to(device)

    mi, sm = MutualInfoLoss(0.3), SmoothnessLoss(10.0)
    
    Cf = 32 * (2**(4 - 1))
    gat = TinyGAT(in_dim=Cf, hid_dim=128, out_dim=128, heads=4).to(device)
    head = ClusterHead(in_dim=128, n_clusters=args.C).to(device)
    # # Eğer sadece GNN eğitilecekse, en iyi spcnn ağırlıklarını yükle ve dondur
    # if args.train_gnn_only:
    #     print("--- Loading best spcnn weights and freezing it. ---")
    #     best_ckpt_path = os.path.join(args.checkpoint_dir, 'best.pt')
    #     if os.path.exists(best_ckpt_path):
    #         checkpoint = torch.load(best_ckpt_path, map_location=device)
    #         spcnn.load_state_dict(checkpoint['spcnn'])
    #         print("Best spcnn weights loaded.")
    #     else:
    #         print("WARNING: best.pt not found. GNN will be trained with random spcnn weights.")
        
    #     # spcnn'in gradyanlarını kapat
    #     for param in spcnn.parameters():
    #         param.requires_grad = False
    # # spcnn parametreleri için bir grup
    # params_spcnn = {
    #     'params': spcnn.parameters(),
    #     'lr': args.lr_spcnn 
    # }

    # # gat ve head (GNN) parametreleri için ayrı bir grup
    # params_gnn = {
    #     'params': list(gat.parameters()) + list(head.parameters()),
    #     'lr': args.lr_gnn
    # }

    #optimizer = optim.AdamW([params_spcnn, params_gnn])
    #optimizer = optim.AdamW(list(spcnn.parameters()) + list(gat.parameters()) + list(head.parameters()), lr=args.lr)
    optimizer = optim.AdamW([
    {'params': spcnn.parameters(), 'lr': args.lr_spcnn},
    {'params': gat.parameters(), 'lr': args.lr_gnn},
    {'params': head.parameters(), 'lr': args.lr_gnn}
])
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=args.patience // 2)
    
    # --- Checkpoint, Loglama ve Erken Durdurma Ayarları ---
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.vis_dir, exist_ok=True)
    batch_losses = []
    history = {
        'train_loss': [],
        'train_loss_sp': [],
        'train_loss_gnn': [],
        'val_loss': [],
        'val_loss_sp': [],
        'val_loss_gnn': []
    }
    best_val_loss = float('inf')
    epochs_no_improve = 0
    start_epoch = 1

    # (Resume training) 
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

    # Train loop
    for epoch in range(start_epoch, args.epochs + 1):
        # --- Eğitim Aşaması ---
        spcnn.train(); gat.train(); head.train()
        train_loss, train_loss_sp, train_loss_gnn = 0.0, 0.0, 0.
        
        for batch_idx, (x, _) in enumerate(tqdm(train_loader, ncols=80, desc=f'Epoch {epoch} [Train]')):
            x = x.to(device)
            optimizer.zero_grad(set_to_none=True)
            coords = create_coord_grid(x, scale=0.1)
            x_with_coords = torch.cat([x, coords], dim=1)  # [B, 5, H, W]
            
            outs = spcnn(x_with_coords, get_spixel_prob)
            P = outs['P']        # [B, K, H, W]
            Fmap = outs['feat']  # [B, C, H, W]
            
            #  SUPERPIXEL LOSSES
            # ========== SUPERPIXEL LOSSES ==========
            L_mi = mi(P)              # Pixel certainty + SP balance
            L_sm = sm(P, x)           # Edge-aware smoothness
            L_feat_recon = feature_reconstruction_loss(Fmap, P)

            L_sp = L_mi + 10*L_sm 

            #  GRAPH CONSTRUCTION
            Z = sp_soft_pool_avg(Fmap, P)  # [B, K, C]
            
            # Hybrid adjacency: spatial + feature
            A_hybrid = hybrid_adjacency(P, Z, k=8, alpha=0.7, tau=0.5)
            
            # GNN CLUSTERING
            H = gat(Z, A_hybrid)
            Y = head(H)
            
            # CLUSTERING LOSSES 
            if not args.spcnn_only:
                L_contrast = contrastive_cluster_loss(Z, A_hybrid, Y)
                L_quality = cluster_quality_loss(Y, A_hybrid)
                
                L_gnn = L_contrast + 0.5 * L_quality
                
                #  ADAPTIVE LOSS WEIGHTING 
                w_sp, w_gnn = adaptive_loss_weights(L_sp, L_gnn, alpha=0.12)
                
                loss = w_sp * L_sp + w_gnn * L_gnn
            else:
                # SPCNN-only training
                L_gnn = torch.tensor(0.0, device=device)
                w_sp, w_gnn = 1.0, 0.0 
                loss = L_sp
                
            loss.backward(); optimizer.step()
            train_loss += loss.item()
            train_loss_sp += L_sp.item()
            train_loss_gnn += L_gnn.item()
            # print(f"\n[Batch {batch_idx}] Loss breakdown:")
            # print(f"  MI: {L_mi.item():.4f}, Smooth: {L_sm.item():.4f}, "
            #           f"Recon: {L_feat_recon.item():.4f}")
            # print(f"  SP Total: {L_sp.item():.4f}, GNN: {L_gnn.item():.4f}, "
            #           f"Total: {loss.item():.4f}")
            # # Epoch 1, batch 10 civarında
            # if epoch == 1 and batch_idx == 10:
            #     print(f"\n=== LOSS DEBUG ===")
            #     print(f"L_mi:   {L_mi.item():+.4f}")
            #     print(f"L_sm:   {L_sm.item():+.4f}")
            #     print(f"L_feat: {L_feat_recon.item():+.4f}")
            #     print(f"L_sp:   {L_sp.item():+.4f}")
            #     print(f"==================\n")
            # ADD THIS: Record every 10th batch to save memory
            if batch_idx % 10 == 0:
                batch_losses.append({
                    'epoch': epoch,
                    'batch': batch_idx,
                    'iteration': (epoch - 1) * len(train_loader) + batch_idx,
                    'total_loss': loss.item(),
                    'L_mi': L_mi.item(),
                    'L_sm': L_sm.item(),
                    'L_sp': L_sp.item(),
                    'L_gnn': L_gnn.item(),
                    'lr_spcnn': optimizer.param_groups[0]['lr'],
                    'lr_gnn': optimizer.param_groups[1]['lr']
                })
            
        
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_loss_sp'].append(train_loss_sp / len(train_loader))
        history['train_loss_gnn'].append(train_loss_gnn / len(train_loader))

        # Validation 
        spcnn.eval(); gat.eval(); head.eval()
        val_loss, val_loss_sp, val_loss_gnn = 0.0, 0.0, 0.0
        with torch.no_grad():
            for i, (x, _) in enumerate(tqdm(val_loader, ncols=80, desc=f'Epoch {epoch} [Val]')):
                x = x.to(device)
                # addin g coordinate channels
                # ========== SUPERPIXEL GENERATION ==========
                coords = create_coord_grid(x, scale=0.1)
                x_with_coords = torch.cat([x, coords], dim=1)
                
                outs = spcnn(x_with_coords, get_spixel_prob)
                P = outs['P']
                Fmap = outs['feat']
                
                # ========== SUPERPIXEL LOSSES ==========
                L_mi = mi(P)
                L_sm = sm(P, x)
                L_feat_recon = feature_reconstruction_loss(Fmap, P)

                L_sp = L_mi + 10*L_sm 
                # ========== GRAPH CONSTRUCTION ==========
                Z = sp_soft_pool_avg(Fmap, P)
                A_hybrid = hybrid_adjacency(P, Z, k=8, alpha=0.7, tau=0.5)
                
                # ========== GNN CLUSTERING ==========
                H = gat(Z, A_hybrid)
                Y = head(H)
                
                # ========== CLUSTERING LOSSES ==========
                if not args.spcnn_only:
                    L_contrast = contrastive_cluster_loss(Z, A_hybrid, Y)
                    L_quality = cluster_quality_loss(Y, A_hybrid)
                    
                    L_gnn = L_contrast + 0.5 * L_quality
                    
                    # ========== ADAPTIVE LOSS WEIGHTING ==========
                    w_sp, w_gnn = adaptive_loss_weights(L_sp, L_gnn, alpha=0.12)
                    
                    loss = w_sp * L_sp + w_gnn * L_gnn
                else:
                    # SPCNN-only training
                    L_gnn = torch.tensor(0.0, device=device)
                    w_sp, w_gnn = 1.0, 0.0
                    loss = L_sp
                val_loss += loss.item()
                val_loss_sp += L_sp.item()
                val_loss_gnn += L_gnn.item()
                
                if i == 0 and epoch % args.vis_interval == 0:
                    visualize_superpixels(x[0], P[0].unsqueeze(0), os.path.join(args.vis_dir, f"epoch_{epoch}_val_sp.png"))
                    
                    # ========== VALIDATION SAMPLE ANALYSIS ==========
                    print(f"\n{'─'*70}")
                    print(f"[Validation Sample Analysis - Epoch {epoch}]")
                    print(f"{'─'*70}")
                    
                    # Single sample detailed metrics
                    sample_idx = 0
                    P_sample = P[sample_idx]  # [K, H, W]
                    
                    # Superpixel assignment entropy (per pixel)
                    pix_entropy = -(P_sample * (P_sample + 1e-16).log()).sum(0).mean().item()
                    
                    # Superpixel usage
                    sp_mass = P_sample.view(P_sample.shape[0], -1).mean(-1)  # [K]
                    active_count = (sp_mass > 0.01).sum().item()
                    
                    # Boundaries compactness (approximate)
                    sp_labels = P_sample.argmax(0)  # [H, W]
                    boundaries = (sp_labels[:, :-1] != sp_labels[:, 1:]).float().sum().item()
                    boundaries += (sp_labels[:-1, :] != sp_labels[1:, :]).float().sum().item()
                    boundary_density = boundaries / (2 * P_sample.shape[1] * P_sample.shape[2])
                    
                    print(f"  Pixel Entropy:        {pix_entropy:.4f} (lower = more certain)")
                    print(f"  Active Superpixels:   {active_count}/{args.K}")
                    print(f"  Boundary Density:     {boundary_density:.4f} (lower = more compact)")
                    print(f"  Saved visualization:  epoch_{epoch}_val_sp.png")
                    print(f"{'─'*70}\n")
        
        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        history['val_loss_sp'].append(val_loss_sp / len(val_loader))
        history['val_loss_gnn'].append(val_loss_gnn / len(val_loader))
        
        # Print detailed metrics
        print(f'[Epoch {epoch}] Train Loss: {history["train_loss"][-1]:.4f}, Val Loss: {avg_val_loss:.4f}')
        print(f'  └─ Train: L_sp={history["train_loss_sp"][-1]:.4f}, L_gnn={history["train_loss_gnn"][-1]:.4f}')
        print(f'  └─ Val:   L_sp={history["val_loss_sp"][-1]:.4f}, L_gnn={history["val_loss_gnn"][-1]:.4f}')
        print(f'  └─ Weights: w_sp={w_sp:.2f}, w_gnn={w_gnn:.2f}')
        
        checkpoint_data = {
            'epoch': epoch, 'spcnn': spcnn.state_dict(), 'gat': gat.state_dict(),
            'head': head.state_dict(), 'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(), 'best_val_loss': best_val_loss 
        }
        # Her epoch için ayrı checkpoint
        torch.save(checkpoint_data, os.path.join(args.checkpoint_dir, f'epoch_{epoch}.pt'))
        # Kolay devam etme için 'last.pt'
        torch.save(checkpoint_data, os.path.join(args.checkpoint_dir, 'last.pt'))

        # En iyi modele karar ver ve kaydet
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_data['best_val_loss'] = best_val_loss
            torch.save(checkpoint_data, os.path.join(args.checkpoint_dir, 'best.pt'))
            epochs_no_improve = 0
            print("Validation loss improved, saving best model.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break
        
        scheduler.step(avg_val_loss)

        #  Periyodik mIoU Değerlendirmesi 
        if epoch % 5 == 0:
            # (code to save temp checkpoint is the same) 
            temp_ckpt_path = os.path.join(args.checkpoint_dir, 'temp_eval.pt')
            # The command now explicitly uses the same python as the training script
            command = [
                sys.executable, 'evaluate.py',  # Use sys.executable instead of 'python'
                '--ckpt', temp_ckpt_path,
                '--data_root', './coco/val2017',
                '--ann_file', './coco/annotations/instances_val2017.json',
                '--K', str(args.K),
                '--C', str(args.C),
                '--device', args.device,
                '--batch_size', str(args.batch_size)
            ]


    # Epoch-level plots
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Total Train Loss')
    plt.plot(history['val_loss'], label='Total Validation Loss')
    plt.title('Total Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(history['train_loss_sp'], label='Superpixel Loss (L_sp)')
    plt.plot(history['train_loss_gnn'], label='Modularity Loss (L_gnn)')
    plt.title('Training Loss Components')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(history['val_loss_sp'], label='Superpixel Loss (L_sp)')
    plt.plot(history['val_loss_gnn'], label='Modularity Loss (L_gnn)')
    plt.title('Validation Loss Components')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.checkpoint_dir, 'loss_curves_detailed.png'))
    print("Saved detailed loss curves to loss_curves_detailed.png")
    
    # Batch-level plots
    if len(batch_losses) > 0:
        df = pd.DataFrame(batch_losses)
        
        window = 50
        df['total_smooth'] = df['total_loss'].rolling(window, min_periods=1).mean()
        df['L_mi_smooth'] = df['L_mi'].rolling(window, min_periods=1).mean()
        df['L_sm_smooth'] = df['L_sm'].rolling(window, min_periods=1).mean()
        df['L_gnn_smooth'] = df['L_gnn'].rolling(window, min_periods=1).mean()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        axes[0, 0].plot(df['iteration'], df['total_loss'], alpha=0.2, color='gray', label='Raw')
        axes[0, 0].plot(df['iteration'], df['total_smooth'], linewidth=2, label='Smoothed (50-batch MA)')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Total Loss')
        axes[0, 0].set_title('Total Loss per Batch')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        axes[0, 1].plot(df['iteration'], df['L_mi_smooth'], label='MI Loss', linewidth=2)
        axes[0, 1].plot(df['iteration'], df['L_sm_smooth'], label='Smoothness Loss', linewidth=2)
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('SPCNN Loss Components')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        axes[1, 0].plot(df['iteration'], df['L_gnn_smooth'], color='green', linewidth=2)
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('GNN Loss')
        axes[1, 0].set_title('GNN Loss per Batch')
        axes[1, 0].grid(alpha=0.3)
        
        axes[1, 1].plot(df['iteration'], df['lr_spcnn'], label='LR SPCNN', linewidth=2)
        axes[1, 1].plot(df['iteration'], df['lr_gnn'], label='LR GNN', linewidth=2)
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.checkpoint_dir, 'batch_loss_curves.png'), dpi=300)
        print("Saved batch-level loss curves to batch_loss_curves.png")
        
        df.to_csv(os.path.join(args.checkpoint_dir, 'batch_losses.csv'), index=False)
        print(f"Saved batch loss data to {args.checkpoint_dir}/batch_losses.csv")

    

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="Unsupervised Superpixel Segmentation Training")
    ap.add_argument('--spcnn_only', action='store_true', help='Train only SPCNN (ignore GNN losses)')
    ap.add_argument('--resume', action='store_true', help='Resume training from the last checkpoint')
    ap.add_argument('--dataset', type=str, default='custom', choices=['custom', 'cocostuff'], help="Which dataset to use.")
    ap.add_argument('--subset_fraction', type=float, default=1.0, help="Fraction of COCO to use.")
    ap.add_argument('--data', type=str, default='./data', help="Path for custom dataset.")
    ap.add_argument('--size', type=int, nargs=2, default=[320, 320])
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch_size', type=int, default=2)
    ap.add_argument('--lr_spcnn', type=float, default=1e-5, help="Learning rate for SPCNN module.")
    ap.add_argument('--lr_gnn', type=float, default=5e-4, help="Learning rate for GNN module (gat + head).")
    #ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--K', type=int, default=100)
    ap.add_argument('--C', type=int, default=6)
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--gamma', type=float, default=1.0)
    ap.add_argument('--val_split', type=float, default=0.2)
    ap.add_argument('--patience', type=int, default=5, help="Early stopping patience.")
    ap.add_argument('--checkpoint_dir', type=str, default='./checkpoints_final')
    ap.add_argument('--vis_dir', type=str, default='./visualizations_final')
    ap.add_argument('--vis_interval', type=int, default=1)
     #ap.add_argument('--train_gnn_only', action='store_true', help="If set, only train the GNN part, keeping spcnn fixed.")
    args = ap.parse_args()
    main(args)