"""
SPCNN Kalite Analiz Scripti
Bu script şunları kontrol eder:
1. Superpixel kalitesi (over-segmentation, under-segmentation)
2. Feature derinliği ve çeşitliliği
3. GNN'in öğrenebileceği yeterli bilgi var mı?
4. Superpixeller arası correlation matrix analizi
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.segmentation import mark_boundaries
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from scipy import ndimage
import os
import argparse

from segment import EdgeAwareSPModule, get_spixel_prob, sp_soft_pool_avg
from data_utils import ImagesFolder
from torch.utils.data import DataLoader


def create_coord_grid(x, scale=0.1):
    B, _, H, W = x.shape
    device = x.device
    yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    coords = torch.stack([
        ((xx.float() / (W - 1)) * 2 - 1) * scale,
        ((yy.float() / (H - 1)) * 2 - 1) * scale
    ], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)
    return coords


class SPCNNQualityAnalyzer:
    """SPCNN çıktılarını analiz eden sınıf."""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.metrics = {
            'feature_stats': [],
            'superpixel_stats': [],
            'boundary_adherence': [],
            'feature_diversity': []
        }
    
    def analyze_superpixel_quality(self, P, image):
        """
        Superpixel segmentasyon kalitesini analiz et.
        
        Args:
            P: [B, K, H, W] - soft probability map
            image: [B, 3, H, W] - original image
        """
        B, K, H, W = P.shape
        
        # Hard assignment
        seg = P.argmax(dim=1).cpu().numpy()  # [B, H, W]
        
        metrics = {}
        
        for b in range(B):
            seg_b = seg[b]
            
            # 1. Kaç tane unique superpixel var?
            unique_sp = len(np.unique(seg_b))
            metrics['unique_superpixels'] = unique_sp
            
            # 2. Ortalama superpixel boyutu
            sp_sizes = []
            for sp_id in range(K):
                size = (seg_b == sp_id).sum()
                if size > 0:
                    sp_sizes.append(size)
            
            metrics['mean_sp_size'] = np.mean(sp_sizes) if sp_sizes else 0
            metrics['std_sp_size'] = np.std(sp_sizes) if sp_sizes else 0
            metrics['min_sp_size'] = np.min(sp_sizes) if sp_sizes else 0
            metrics['max_sp_size'] = np.max(sp_sizes) if sp_sizes else 0
            
            # 3. Compactness (ne kadar düzenli?)
            compactness_scores = []
            for sp_id in range(K):
                mask = (seg_b == sp_id).astype(np.uint8)
                if mask.sum() < 5:
                    continue
                
                # Perimeter hesapla
                boundary = mask - ndimage.binary_erosion(mask).astype(np.uint8)
                perimeter = boundary.sum()
                area = mask.sum()
                
                if perimeter > 0:
                    compactness = 4 * np.pi * area / (perimeter ** 2)
                    compactness_scores.append(compactness)
            
            metrics['mean_compactness'] = np.mean(compactness_scores) if compactness_scores else 0
            
            # 4. Soft assignment entropy
            P_b = P[b].cpu()  # [K, H, W]
            entropy_per_pixel = -(P_b * torch.log(P_b + 1e-10)).sum(dim=0)
            metrics['mean_assignment_entropy'] = entropy_per_pixel.mean().item()
            metrics['std_assignment_entropy'] = entropy_per_pixel.std().item()
            
        return metrics
    
    def analyze_feature_depth(self, Z, P):
        """
        Feature derinliğini ve çeşitliliğini analiz et.
        
        Args:
            Z: [B, K, C] - superpixel features
            P: [B, K, H, W] - probability map
        """
        B, K, C = Z.shape
        
        metrics = {}
        
        for b in range(B):
            Z_b = Z[b].cpu().numpy()  # [K, C]
            P_b = P[b].cpu().numpy()  # [K, H, W]
            
            # 1. Feature magnitude
            feat_norms = np.linalg.norm(Z_b, axis=1)
            metrics['mean_feature_norm'] = feat_norms.mean()
            metrics['std_feature_norm'] = feat_norms.std()
            
            # 2. Feature diversity (cosine similarity)
            Z_norm = Z_b / (np.linalg.norm(Z_b, axis=1, keepdims=True) + 1e-8)
            similarity = Z_norm @ Z_norm.T
            
            mask = ~np.eye(K, dtype=bool)
            off_diagonal_sim = similarity[mask]
            
            metrics['mean_feature_similarity'] = off_diagonal_sim.mean()
            metrics['std_feature_similarity'] = off_diagonal_sim.std()
            
            # 3. Feature clustering
            from sklearn.cluster import KMeans
            n_clusters = min(10, K)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(Z_b)
            
            cluster_counts = np.bincount(cluster_labels, minlength=n_clusters)
            cluster_probs = cluster_counts / cluster_counts.sum()
            cluster_entropy = -(cluster_probs * np.log(cluster_probs + 1e-10)).sum()
            
            metrics['feature_cluster_entropy'] = cluster_entropy
            metrics['max_cluster_entropy'] = np.log(n_clusters)
            
            # 4. Spatial smoothness
            P_hard = P_b.argmax(axis=0)
            
            spatial_consistency = []
            for k in range(K):
                if (P_hard == k).sum() == 0:
                    continue
                
                mask_k = (P_hard == k)
                dilated = ndimage.binary_dilation(mask_k)
                boundary = dilated & ~mask_k
                neighbor_ids = np.unique(P_hard[boundary])
                neighbor_ids = neighbor_ids[neighbor_ids != k]
                
                if len(neighbor_ids) == 0:
                    continue
                
                k_feat = Z_b[k]
                neighbor_feats = Z_b[neighbor_ids]
                
                k_norm = k_feat / (np.linalg.norm(k_feat) + 1e-8)
                neighbor_norms = neighbor_feats / (np.linalg.norm(neighbor_feats, axis=1, keepdims=True) + 1e-8)
                
                similarities = neighbor_norms @ k_norm
                spatial_consistency.append(similarities.mean())
            
            metrics['spatial_feature_consistency'] = np.mean(spatial_consistency) if spatial_consistency else 0
            
        return metrics
    
    def check_gnn_learning_capacity(self, Z, P):
        """GNN'in bu feature'lardan öğrenebilecek yeterli bilgi var mı?"""
        B, K, C = Z.shape
        
        metrics = {}
        
        for b in range(B):
            Z_b = Z[b].cpu().numpy()  # [K, C]
            
            # 1. Intrinsic dimensionality
            pca = PCA(n_components=min(K, C))
            pca.fit(Z_b)
            
            cumsum = np.cumsum(pca.explained_variance_ratio_)
            n_dims_95 = np.argmax(cumsum >= 0.95) + 1
            
            metrics['intrinsic_dim_95'] = n_dims_95
            metrics['total_dims'] = C
            metrics['dim_efficiency'] = n_dims_95 / C
            
            # 2. Rank of feature matrix
            rank = np.linalg.matrix_rank(Z_b)
            metrics['feature_matrix_rank'] = rank
            metrics['rank_ratio'] = rank / min(K, C)
            
        return metrics
    
    def plot_correlation_matrices(self, Z, P, image, save_dir='./analysis'):
        """Superpixeller arası correlation matrix'leri çiz."""
        os.makedirs(save_dir, exist_ok=True)
        
        B, K, C = Z.shape
        
        for b in range(min(B, 3)):
            Z_b = Z[b].cpu().numpy()  # [K, C]
            P_b = P[b].cpu().numpy()  # [K, H, W]
            img_b = image[b].cpu().numpy().transpose(1, 2, 0)
            seg = P_b.argmax(axis=0)
            
            fig = plt.figure(figsize=(20, 12))
            gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
            
            # 1. Feature Correlation (Pearson)
            ax1 = fig.add_subplot(gs[0, 0])
            corr_pearson = np.corrcoef(Z_b)
            im1 = ax1.imshow(corr_pearson, cmap='RdBu_r', vmin=-1, vmax=1)
            ax1.set_title('Feature Correlation (Pearson)', fontweight='bold')
            ax1.set_xlabel('Superpixel ID')
            ax1.set_ylabel('Superpixel ID')
            plt.colorbar(im1, ax=ax1, fraction=0.046)
            
            # 2. Cosine Similarity
            ax2 = fig.add_subplot(gs[0, 1])
            Z_norm = Z_b / (np.linalg.norm(Z_b, axis=1, keepdims=True) + 1e-8)
            cosine_sim = Z_norm @ Z_norm.T
            im2 = ax2.imshow(cosine_sim, cmap='RdBu_r', vmin=-1, vmax=1)
            ax2.set_title('Cosine Similarity', fontweight='bold')
            ax2.set_xlabel('Superpixel ID')
            ax2.set_ylabel('Superpixel ID')
            plt.colorbar(im2, ax=ax2, fraction=0.046)
            
            # 3. Spatial Adjacency
            ax3 = fig.add_subplot(gs[0, 2])
            H, W = seg.shape
            adjacency = np.zeros((K, K))
            for i in range(H-1):
                for j in range(W-1):
                    k1 = seg[i, j]
                    k2 = seg[i, j+1]
                    if k1 != k2:
                        adjacency[k1, k2] += 1
                        adjacency[k2, k1] += 1
                    k3 = seg[i+1, j]
                    if k1 != k3:
                        adjacency[k1, k3] += 1
                        adjacency[k3, k1] += 1
            
            adjacency = adjacency / (adjacency.max() + 1e-8)
            im3 = ax3.imshow(adjacency, cmap='YlOrRd')
            ax3.set_title('Spatial Adjacency', fontweight='bold')
            ax3.set_xlabel('Superpixel ID')
            ax3.set_ylabel('Superpixel ID')
            plt.colorbar(im3, ax=ax3, fraction=0.046)
            
            # 4. Feature-Weighted Adjacency
            ax4 = fig.add_subplot(gs[0, 3])
            feature_weighted = adjacency * cosine_sim
            im4 = ax4.imshow(feature_weighted, cmap='viridis')
            ax4.set_title('Feature-Weighted Adjacency\n(What GNN sees)', fontweight='bold')
            ax4.set_xlabel('Superpixel ID')
            ax4.set_ylabel('Superpixel ID')
            plt.colorbar(im4, ax=ax4, fraction=0.046)
            
            # 5. Correlation Distribution
            ax5 = fig.add_subplot(gs[1, 0])
            mask = ~np.eye(K, dtype=bool)
            corr_values = corr_pearson[mask]
            ax5.hist(corr_values, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
            ax5.axvline(corr_values.mean(), color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {corr_values.mean():.3f}')
            ax5.set_title('Correlation Distribution', fontweight='bold')
            ax5.set_xlabel('Correlation Coefficient')
            ax5.set_ylabel('Frequency')
            ax5.legend()
            ax5.grid(alpha=0.3)
            
            # 6. Similarity Distribution
            ax6 = fig.add_subplot(gs[1, 1])
            sim_values = cosine_sim[mask]
            ax6.hist(sim_values, bins=50, alpha=0.7, color='coral', edgecolor='black')
            ax6.axvline(sim_values.mean(), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {sim_values.mean():.3f}')
            ax6.set_title('Cosine Similarity Distribution', fontweight='bold')
            ax6.set_xlabel('Cosine Similarity')
            ax6.set_ylabel('Frequency')
            ax6.legend()
            ax6.grid(alpha=0.3)
            
            # 7. Correlation vs Spatial Distance
            ax7 = fig.add_subplot(gs[1, 2])
            centroids = np.zeros((K, 2))
            for k in range(K):
                mask_k = (seg == k)
                if mask_k.sum() > 0:
                    ys, xs = np.where(mask_k)
                    centroids[k] = [xs.mean(), ys.mean()]
            
            spatial_dist = cdist(centroids, centroids, metric='euclidean')
            
            n_samples = min(5000, K * K)
            indices = np.random.choice(K*K, n_samples, replace=False)
            rows, cols = np.unravel_index(indices, (K, K))
            valid = rows != cols
            
            scatter = ax7.scatter(spatial_dist[rows[valid], cols[valid]], 
                                 corr_pearson[rows[valid], cols[valid]],
                                 alpha=0.3, s=10, c=adjacency[rows[valid], cols[valid]],
                                 cmap='coolwarm')
            ax7.set_title('Correlation vs Spatial Distance', fontweight='bold')
            ax7.set_xlabel('Spatial Distance (pixels)')
            ax7.set_ylabel('Feature Correlation')
            ax7.grid(alpha=0.3)
            plt.colorbar(scatter, ax=ax7, label='Adjacency', fraction=0.046)
            
            # 8. Spatial vs Feature Correlation
            ax8 = fig.add_subplot(gs[1, 3])
            spatial_corr = adjacency[mask]
            feature_corr = cosine_sim[mask]
            
            ax8.scatter(spatial_corr, feature_corr, alpha=0.3, s=10, c='purple')
            z = np.polyfit(spatial_corr, feature_corr, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(spatial_corr.min(), spatial_corr.max(), 100)
            ax8.plot(x_trend, p(x_trend), "r--", linewidth=2, 
                    label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
            ax8.set_title('Spatial vs Feature Correlation', fontweight='bold')
            ax8.set_xlabel('Spatial Adjacency')
            ax8.set_ylabel('Feature Similarity')
            ax8.legend()
            ax8.grid(alpha=0.3)
            
            # 9. Original Image with Superpixels
            ax9 = fig.add_subplot(gs[2, 0])
            boundary_img = mark_boundaries(img_b, seg, color=(1, 0, 0))
            ax9.imshow(boundary_img)
            ax9.set_title('Superpixel Segmentation', fontweight='bold')
            ax9.axis('off')
            
            # 10. High Correlation Pairs
            ax10 = fig.add_subplot(gs[2, 1])
            corr_no_diag = corr_pearson.copy()
            np.fill_diagonal(corr_no_diag, -2)
            top_k = 10
            top_indices = np.argpartition(corr_no_diag.ravel(), -top_k)[-top_k:]
            top_pairs = np.unravel_index(top_indices, corr_no_diag.shape)
            
            vis_img = img_b.copy()
            for i, (idx1, idx2) in enumerate(zip(top_pairs[0], top_pairs[1])):
                mask1 = (seg == idx1)
                mask2 = (seg == idx2)
                color = plt.cm.Set3(i / top_k)[:3]
                vis_img[mask1] = vis_img[mask1] * 0.5 + np.array(color) * 0.5
                vis_img[mask2] = vis_img[mask2] * 0.5 + np.array(color) * 0.5
            
            ax10.imshow(vis_img)
            ax10.set_title(f'Top {top_k} Correlated Pairs', fontweight='bold')
            ax10.axis('off')
            
            # 11. Low Correlation Pairs
            ax11 = fig.add_subplot(gs[2, 2])
            bottom_indices = np.argpartition(corr_no_diag.ravel(), top_k)[:top_k]
            bottom_pairs = np.unravel_index(bottom_indices, corr_no_diag.shape)
            
            vis_img2 = img_b.copy()
            for i, (idx1, idx2) in enumerate(zip(bottom_pairs[0], bottom_pairs[1])):
                mask1 = (seg == idx1)
                mask2 = (seg == idx2)
                color = plt.cm.Set1(i / top_k)[:3]
                vis_img2[mask1] = vis_img2[mask1] * 0.5 + np.array(color) * 0.5
                vis_img2[mask2] = vis_img2[mask2] * 0.5 + np.array(color) * 0.5
            
            ax11.imshow(vis_img2)
            ax11.set_title(f'Top {top_k} Anti-Correlated Pairs', fontweight='bold')
            ax11.axis('off')
            
            # 12. Statistics Summary
            ax12 = fig.add_subplot(gs[2, 3])
            ax12.axis('off')
            
            spatial_feat_corr = np.corrcoef(spatial_corr, feature_corr)[0,1]
            stats_text = f"""
CORRELATION STATISTICS

Feature Correlation (Pearson):
  Mean: {corr_values.mean():.4f}
  Std:  {corr_values.std():.4f}
  Min:  {corr_values.min():.4f}
  Max:  {corr_values.max():.4f}

Cosine Similarity:
  Mean: {sim_values.mean():.4f}
  Std:  {sim_values.std():.4f}

Spatial-Feature Correlation:
  Pearson r: {spatial_feat_corr:.4f}
  
INTERPRETATION:
"""
            
            if sim_values.mean() > 0.7:
                stats_text += "\n⚠️  HIGH similarity\n    Features too similar!"
            elif sim_values.mean() < 0.3:
                stats_text += "\n✓  GOOD diversity"
            else:
                stats_text += "\n✓  Moderate similarity"
            
            if spatial_feat_corr > 0.3:
                stats_text += "\n✓  Spatial coherence good"
            else:
                stats_text += "\n⚠️  Weak spatial coherence"
            
            ax12.text(0.05, 0.95, stats_text, transform=ax12.transAxes,
                     fontsize=10, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.savefig(os.path.join(save_dir, f'correlation_analysis_batch_{b}.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ Correlation matrix saved for batch {b}")
    
    def visualize_features(self, Z, P, image, save_dir='./analysis'):
        """Feature'ları görselleştir."""
        os.makedirs(save_dir, exist_ok=True)
        
        B, K, C = Z.shape
        
        for b in range(min(B, 3)):
            Z_b = Z[b].cpu().numpy()
            P_b = P[b].cpu().numpy()
            img_b = image[b].cpu().numpy().transpose(1, 2, 0)
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # 1. Original image
            axes[0, 0].imshow(img_b)
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')
            
            # 2. Superpixel segmentation
            seg = P_b.argmax(axis=0)
            boundary_img = mark_boundaries(img_b, seg)
            axes[0, 1].imshow(boundary_img)
            axes[0, 1].set_title(f'Superpixels (K={K})')
            axes[0, 1].axis('off')
            
            # 3. Assignment entropy map
            entropy_map = -(P_b * np.log(P_b + 1e-10)).sum(axis=0)
            im = axes[0, 2].imshow(entropy_map, cmap='viridis')
            axes[0, 2].set_title('Assignment Entropy')
            axes[0, 2].axis('off')
            plt.colorbar(im, ax=axes[0, 2])
            
            # 4. Feature PCA
            if C >= 3:
                pca = PCA(n_components=3)
                Z_pca = pca.fit_transform(Z_b)
                
                H, W = P_b.shape[1:]
                pca_img = np.zeros((H, W, 3))
                for k in range(K):
                    mask = (seg == k)
                    pca_img[mask] = Z_pca[k]
                
                pca_img = (pca_img - pca_img.min()) / (pca_img.max() - pca_img.min() + 1e-8)
                axes[1, 0].imshow(pca_img)
                axes[1, 0].set_title('PCA Features (RGB=PC1,2,3)')
                axes[1, 0].axis('off')
            
            # 5. Feature similarity heatmap
            Z_norm = Z_b / (np.linalg.norm(Z_b, axis=1, keepdims=True) + 1e-8)
            similarity = Z_norm @ Z_norm.T
            im = axes[1, 1].imshow(similarity, cmap='RdBu_r', vmin=-1, vmax=1)
            axes[1, 1].set_title('Feature Similarity Matrix')
            plt.colorbar(im, ax=axes[1, 1])
            
            # 6. Feature norm map
            feat_norms = np.linalg.norm(Z_b, axis=1)
            norm_img = np.zeros((H, W))
            for k in range(K):
                mask = (seg == k)
                norm_img[mask] = feat_norms[k]
            
            im = axes[1, 2].imshow(norm_img, cmap='plasma')
            axes[1, 2].set_title('Feature Magnitude')
            axes[1, 2].axis('off')
            plt.colorbar(im, ax=axes[1, 2])
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'feature_analysis_batch_{b}.png'), dpi=150)
            plt.close()
    
    def print_summary(self):
        """Analiz sonuçlarını özetle."""
        print("\n" + "="*80)
        print("SPCNN KALİTE ANALİZİ SONUÇLARI")
        print("="*80)
        
        if not self.metrics['feature_stats']:
            print("Henüz analiz yapılmadı!")
            return
        
        all_metrics = {}
        for key in self.metrics['feature_stats'][0].keys():
            values = [m[key] for m in self.metrics['feature_stats']]
            all_metrics[key] = (np.mean(values), np.std(values))
        
        print("\n📊 FEATURE KALİTESİ:")
        print(f"  Feature Norm: {all_metrics['mean_feature_norm'][0]:.4f} ± {all_metrics['mean_feature_norm'][1]:.4f}")
        print(f"  Feature Similarity: {all_metrics['mean_feature_similarity'][0]:.4f} ± {all_metrics['mean_feature_similarity'][1]:.4f}")
        print(f"  Spatial Consistency: {all_metrics['spatial_feature_consistency'][0]:.4f} ± {all_metrics['spatial_feature_consistency'][1]:.4f}")
        
        sim = all_metrics['mean_feature_similarity'][0]
        if sim > 0.8:
            print("  ⚠️  UYARI: Feature'lar çok benzer! (similarity > 0.8)")
            print("     GNN'in öğrenmesi zor olacak.")
        elif sim < 0.3:
            print("  ✓ İYİ: Feature'lar yeterince çeşitli.")
        
        consistency = all_metrics['spatial_feature_consistency'][0]
        if consistency < 0.3:
            print("  ⚠️  UYARI: Spatial consistency düşük!")
        elif consistency > 0.7:
            print("  ⚠️  UYARI: Spatial consistency çok yüksek!")
        else:
            print("  ✓ İYİ: Spatial consistency dengeli.")
        
        print("\n📐 SUPERPIXEL KALİTESİ:")
        sp_metrics = self.metrics['superpixel_stats'][0]
        print(f"  Unique Superpixels: {sp_metrics['unique_superpixels']}")
        print(f"  Mean Size: {sp_metrics['mean_sp_size']:.1f} pixels")
        print(f"  Compactness: {sp_metrics['mean_compactness']:.4f}")
        print(f"  Assignment Entropy: {sp_metrics['mean_assignment_entropy']:.4f}")
        
        print("\n🧠 GNN ÖĞRENME KAPASİTESİ:")
        gnn_metrics = self.metrics['feature_diversity'][0]
        print(f"  Intrinsic Dimensionality: {gnn_metrics['intrinsic_dim_95']}/{gnn_metrics['total_dims']}")
        print(f"  Dimension Efficiency: {gnn_metrics['dim_efficiency']:.2%}")
        print(f"  Feature Rank: {gnn_metrics['feature_matrix_rank']}")
        print(f"  Rank Ratio: {gnn_metrics['rank_ratio']:.2%}")
        
        if gnn_metrics['dim_efficiency'] < 0.3:
            print("  ⚠️  UYARI: Feature'lar düşük boyutlu!")
        elif gnn_metrics['rank_ratio'] < 0.8:
            print("  ⚠️  UYARI: Feature matrix rank düşük!")
        else:
            print("  ✓ İYİ: Feature space yeterince zengin.")
        
        print("\n💡 ÖNERİLER:")
        if sim > 0.7:
            print("  1. SPCNN'i daha fazla eğitin")
            print("  2. Feature extraction katmanlarını derinleştirin")
        if consistency < 0.4:
            print("  3. Smoothness loss weight'ini artırın")
        
        print("="*80 + "\n")


def main(args):
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    # Load model
    spcnn = EdgeAwareSPModule(in_c=5, num_feat=32, num_layers=4, num_spixels=args.K).to(device)
    
    if args.ckpt:
        print(f"Loading checkpoint: {args.ckpt}")
        checkpoint = torch.load(args.ckpt, map_location=device)
        spcnn.load_state_dict(checkpoint['spcnn'])
        print("Checkpoint loaded successfully.")
    else:
        print("⚠️  No checkpoint provided - analyzing random initialization!")
    
    spcnn.eval()
    
    # Load data
    dataset = ImagesFolder(args.data, tuple(args.size))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # Analyzer
    analyzer = SPCNNQualityAnalyzer(device)
    
    print(f"\n🔍 Analyzing {args.num_samples} images...")
    
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= args.num_samples // args.batch_size:
                break
            
            x = x.to(device)
            
            # Add coordinates
            coords = create_coord_grid(x)
            x_with_coords = torch.cat([x, coords], dim=1)
            
            # Forward pass
            outs = spcnn(x_with_coords, get_spixel_prob)
            P = outs['P']
            Fmap = outs['feat']
            
            # Pool features to superpixels
            Z = sp_soft_pool_avg(Fmap, P)
            
            # Analyze
            sp_quality = analyzer.analyze_superpixel_quality(P, x)
            feat_depth = analyzer.analyze_feature_depth(Z, P)
            gnn_capacity = analyzer.check_gnn_learning_capacity(Z, P)
            
            analyzer.metrics['superpixel_stats'].append(sp_quality)
            analyzer.metrics['feature_stats'].append(feat_depth)
            analyzer.metrics['feature_diversity'].append(gnn_capacity)
            
            # Visualize first batch
            if i == 0:
                print("\n📊 Generating visualizations...")
                analyzer.visualize_features(Z, P, x, args.output_dir)
                analyzer.plot_correlation_matrices(Z, P, x, args.output_dir)
            
            print(f"  Processed batch {i+1}/{args.num_samples // args.batch_size}")
    
    # Print summary
    analyzer.print_summary()
    
    print(f"\n✓ Görselleştirmeler kaydedildi: {args.output_dir}")
    print(f"  - feature_analysis_batch_0.png")
    print(f"  - correlation_analysis_batch_0.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SPCNN Quality Analysis")
    parser.add_argument('--ckpt', type=str, default=None, help="Checkpoint path (optional)")
    parser.add_argument('--data', type=str, default='./coco/val2017', help="Data path")
    parser.add_argument('--size', type=int, nargs=2, default=[320, 320])
    parser.add_argument('--K', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='./spcnn_analysis')
    
    args = parser.parse_args()
    main(args)