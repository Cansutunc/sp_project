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

# Proje dosyalarınızdan gerekli modülleri import edin
from data_utils import ImagesFolder
from segment import EdgeAwareSPModule, get_spixel_prob, sp_soft_pool_avg, sp_project
from losses_sp import MutualInfoLoss, SmoothnessLoss, EdgeAwareKLLoss
from gnn_modularity import TinyGAT, ClusterHead, soft_adjacency, modularity_loss

def visualize_superpixels(image_tensor, p_map, out_path):
    """Süperpiksel sınırlarını resim üzerinde görselleştirir."""
    image_pil = Image.fromarray((image_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    seg = p_map.argmax(dim=1)[0].cpu().numpy()
    
    # Gerekli kütüphaneyi import et
    from skimage.segmentation import mark_boundaries
    # PIL objesini NumPy array'ine çevirerek hatayı düzelt
    boundary_img = mark_boundaries(np.array(image_pil), seg)
    
    plt.imsave(out_path, boundary_img)

def main(args):
    device = args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu'

    # --- Veri Yükleme ---
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

    # --- Model, Kayıp Fonksiyonları ve Optimizatör ---
    spcnn = EdgeAwareSPModule(in_c=3, num_feat=32, num_layers=4, num_spixels=args.K, add_recon=False).to(device)
    mi, sm, ed = MutualInfoLoss(2.0), SmoothnessLoss(10.0), EdgeAwareKLLoss()
    Cf = 32 * (2**(4 - 1))
    gat = TinyGAT(in_dim=Cf, hid_dim=128, out_dim=128, heads=4).to(device)
    head = ClusterHead(in_dim=128, n_clusters=args.C).to(device)

    optimizer = optim.AdamW(list(spcnn.parameters()) + list(gat.parameters()) + list(head.parameters()), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=args.patience // 2)
    
    # --- Checkpoint, Loglama ve Erken Durdurma Ayarları ---
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.vis_dir, exist_ok=True)
    history = {
        'train_loss': [],
        'train_loss_sp': [],
        'train_loss_mod': [],
        'val_loss': [],
        'val_loss_sp': [],
        'val_loss_mod': []
    }
    best_val_loss = float('inf')
    epochs_no_improve = 0
    start_epoch = 1

    # --- Eğitim Kaldığı Yerden Devam Etme (Resume) ---
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

    # --- Ana Eğitim Döngüsü ---
    for epoch in range(start_epoch, args.epochs + 1):
        # --- Eğitim Aşaması ---
        spcnn.train(); gat.train(); head.train()
        train_loss, train_loss_sp, train_loss_mod = 0.0, 0.0, 0.
        for x, _ in tqdm(train_loader, ncols=80, desc=f'Epoch {epoch} [Train]'):
            x = x.to(device)
            optimizer.zero_grad(set_to_none=True)

            outs = spcnn(x, get_spixel_prob); P, Fmap = outs['P'], outs['feat']
            Zrgb = sp_soft_pool_avg(x, P); Ihat = sp_project(Zrgb, P)
            L_sp = mi(P) + sm(P, x) + ed(x, Ihat)
            
            Z = sp_soft_pool_avg(Fmap, P); A_soft = soft_adjacency(Z, tau=0.5)
            
            A_bin = (A_soft > (A_soft.mean(dim=(-1,-2), keepdim=True))).float()
            
            H = gat(Z, A_soft); Y = head(H)
            L_mod = modularity_loss(A_bin, Y)
            
            loss = L_sp + args.gamma * L_mod
            loss.backward(); optimizer.step()
            train_loss += loss.item()
            train_loss_sp += L_sp.item()
            train_loss_mod += L_mod.item()
        
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_loss_sp'].append(train_loss_sp / len(train_loader))
        history['train_loss_mod'].append(train_loss_mod / len(train_loader))

        # --- Doğrulama Aşaması ---
        spcnn.eval(); gat.eval(); head.eval()
        val_loss, val_loss_sp, val_loss_mod = 0.0, 0.0, 0.0
        with torch.no_grad():
            for i, (x, _) in enumerate(tqdm(val_loader, ncols=80, desc=f'Epoch {epoch} [Val]')):
                x = x.to(device)
                
                outs = spcnn(x, get_spixel_prob); P, Fmap = outs['P'], outs['feat']
                Zrgb = sp_soft_pool_avg(x, P); Ihat = sp_project(Zrgb, P)
                L_sp = mi(P) + sm(P, x) + ed(x, Ihat)
                
                Z = sp_soft_pool_avg(Fmap, P); A_soft = soft_adjacency(Z, tau=0.5)
                A_bin = (A_soft > (A_soft.mean(dim=(-1,-2), keepdim=True))).float()
                
                H = gat(Z, A_soft); Y = head(H)
                L_mod = modularity_loss(A_bin, Y)
                
                loss = L_sp + args.gamma * L_mod
                val_loss += loss.item()
                val_loss_sp += L_sp.item()
                val_loss_mod += L_mod.item()
                
                if i == 0 and epoch % args.vis_interval == 0:
                    visualize_superpixels(x[0], P[0].unsqueeze(0), os.path.join(args.vis_dir, f"epoch_{epoch}_val_sp.png"))
        
        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        history['val_loss_sp'].append(val_loss_sp / len(val_loader))
        history['val_loss_mod'].append(val_loss_mod / len(val_loader))
        
        print(f'[Epoch {epoch}] Train Loss: {history["train_loss"][-1]:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        # --- Checkpoint Kaydetme ---
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

        # --- Periyodik mIoU Değerlendirmesi ---
        if epoch % 5 == 0:
            # ... (code to save temp checkpoint is the same) ...
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

    plt.figure(figsize=(18, 5))
    
    # Plot 1: Total Losses
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Total Train Loss')
    plt.plot(history['val_loss'], label='Total Validation Loss')
    plt.title('Total Loss Over Epochs'); plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)

    # Plot 2: Training Loss Components
    plt.subplot(1, 3, 2)
    plt.plot(history['train_loss_sp'], label='Superpixel Loss (L_sp)')
    plt.plot(history['train_loss_mod'], label='Modularity Loss (L_mod)')
    plt.title('Training Loss Components'); plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
    
    # Plot 3: Validation Loss Components
    plt.subplot(1, 3, 3)
    plt.plot(history['val_loss_sp'], label='Superpixel Loss (L_sp)')
    plt.plot(history['val_loss_mod'], label='Modularity Loss (L_mod)')
    plt.title('Validation Loss Components'); plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('loss_curves_detailed.png')
    print("Saved detailed loss curves to loss_curves_detailed.png")
    

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="Unsupervised Superpixel Segmentation Training")
    ap.add_argument('--resume', action='store_true', help='Resume training from the last checkpoint')
    ap.add_argument('--dataset', type=str, default='custom', choices=['custom', 'cocostuff'], help="Which dataset to use.")
    ap.add_argument('--subset_fraction', type=float, default=1.0, help="Fraction of COCO to use.")
    ap.add_argument('--data', type=str, default='./data', help="Path for custom dataset.")
    ap.add_argument('--size', type=int, nargs=2, default=[320, 320])
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch_size', type=int, default=2)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--K', type=int, default=200)
    ap.add_argument('--C', type=int, default=6)
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--gamma', type=float, default=1.0)
    ap.add_argument('--val_split', type=float, default=0.2)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--checkpoint_dir', type=str, default='./checkpoints_final')
    ap.add_argument('--vis_dir', type=str, default='./visualizations_final')
    ap.add_argument('--vis_interval', type=int, default=1)
    args = ap.parse_args()
    main(args)