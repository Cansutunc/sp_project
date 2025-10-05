# gradient_check.py
import torch
from train_with_recon import *
import argparse

def check_gradients_full_system(args):
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    # Load models
    spcnn = EdgeAwareSPModule(in_c=5, num_feat=32, num_layers=4, num_spixels=args.K).to(device)
    gat = TinyGAT(in_dim=256, hid_dim=128, out_dim=128, heads=4).to(device)
    head = ClusterHead(in_dim=128, n_clusters=args.C).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.ckpt, map_location=device)
    spcnn.load_state_dict(checkpoint['spcnn'])
    gat.load_state_dict(checkpoint['gat'])
    head.load_state_dict(checkpoint['head'])
    
    # Loss functions
    mi = MutualInfoLoss(3.0)
    sm = SmoothnessLoss(15.0)
    ed = EdgeAwareKLLoss()
    recon_loss = torch.nn.MSELoss()
    
    # Dummy data
    x = torch.rand(2, 3, 320, 320, requires_grad=False).to(device)
    
    print("="*80)
    print("GRADIENT FLOW ANALYSIS")
    print("="*80)
    
    # Forward pass
    x_with_coords = create_coord_grid_normalized(x)
    outs = spcnn(x_with_coords, get_spixel_prob)
    P, Fmap = outs['P'], outs['feat']
    
    # SP losses
    L_mi = mi(P)
    L_smooth = sm(P, x)
    Zrgb = sp_soft_pool_avg(x, P)
    Ihat = sp_project(Zrgb, P)
    L_edge = ed(x, Ihat)
    L_recon = recon_loss(outs['recon'], x) if 'recon' in outs else torch.tensor(0.0)
    L_sp = 1.5*L_mi + 0.5*L_smooth + 1.0*L_edge + 8.0*L_recon
    
    # GNN forward
    Z = sp_soft_pool_avg(Fmap, P)
    A_soft = soft_adjacency(Z, tau=0.5)
    H = gat(Z, A_soft)
    Y = head(H)
    L_gnn = contrastive_cluster_loss(Z, A_soft, Y)
    
    # Total loss
    loss = L_sp + args.gamma * L_gnn
    
    print(f"\nLoss Values:")
    print(f"  L_sp: {L_sp.item():.4f}")
    print(f"  L_gnn: {L_gnn.item():.4f}")
    print(f"  Total: {loss.item():.4f}")
    
    # === TEST 1: Check if P receives gradients ===
    print("\n" + "="*80)
    print("TEST 1: Does P receive gradients from total loss?")
    print("="*80)
    
    loss.backward(retain_graph=True)
    
    if P.grad is not None:
        print(f"✓ P receives gradients")
        print(f"  P grad norm: {P.grad.norm().item():.6f}")
        print(f"  P grad mean: {P.grad.mean().item():.6f}")
        print(f"  P grad max: {P.grad.max().item():.6f}")
    else:
        print("✗ P does NOT receive gradients!")
    
    # === TEST 2: Separate gradient contributions ===
    print("\n" + "="*80)
    print("TEST 2: Gradient contributions from L_sp vs L_gnn")
    print("="*80)
    
    # Zero out gradients
    spcnn.zero_grad()
    gat.zero_grad()
    head.zero_grad()
    
    # Gradient from L_sp only
    L_sp.backward(retain_graph=True)
    grad_P_from_sp = P.grad.clone() if P.grad is not None else None
    grad_Fmap_from_sp = Fmap.grad.clone() if Fmap.grad is not None else None
    
    spcnn.zero_grad()
    
    # Gradient from L_gnn only
    L_gnn.backward(retain_graph=True)
    grad_P_from_gnn = P.grad.clone() if P.grad is not None else None
    grad_Fmap_from_gnn = Fmap.grad.clone() if Fmap.grad is not None else None
    
    print("\nFrom L_sp:")
    print(f"  P grad norm: {grad_P_from_sp.norm().item():.6f if grad_P_from_sp is not None else 0}")
    print(f"  Fmap grad norm: {grad_Fmap_from_sp.norm().item():.6f if grad_Fmap_from_sp is not None else 0}")
    
    print("\nFrom L_gnn:")
    print(f"  P grad norm: {grad_P_from_gnn.norm().item():.6f if grad_P_from_gnn is not None else 0}")
    print(f"  Fmap grad norm: {grad_Fmap_from_gnn.norm().item():.6f if grad_Fmap_from_gnn is not None else 0}")
    
    if grad_P_from_gnn is not None:
        ratio = grad_P_from_gnn.norm() / (grad_P_from_sp.norm() + 1e-8)
        print(f"\n  Ratio (GNN/SP): {ratio.item():.4f}")
        if ratio < 0.01:
            print("  ⚠ GNN gradients are very weak compared to SP")
        elif ratio > 0.1:
            print("  ✓ GNN gradients are significant")
    else:
        print("\n  ✗ NO gradients from L_gnn to P!")
    
    # === TEST 3: SPNN parameter gradients ===
    print("\n" + "="*80)
    print("TEST 3: SPNN parameter gradient magnitudes")
    print("="*80)
    
    spcnn.zero_grad()
    gat.zero_grad()
    head.zero_grad()
    
    loss.backward()
    
    total_norm_spcnn = 0
    total_norm_gnn = 0
    
    print("\nSPCNN parameters:")
    for name, param in spcnn.named_parameters():
        if param.grad is not None:
            pnorm = param.grad.norm().item()
            total_norm_spcnn += pnorm ** 2
            if 'head_logits' in name or 'head_recon' in name:
                print(f"  {name}: {pnorm:.6f}")
    
    print(f"\nTotal SPNN grad norm: {total_norm_spcnn**0.5:.6f}")
    
    print("\nGNN parameters:")
    for name, param in list(gat.named_parameters())[:3] + list(head.named_parameters())[:2]:
        if param.grad is not None:
            pnorm = param.grad.norm().item()
            total_norm_gnn += pnorm ** 2
            print(f"  {name}: {pnorm:.6f}")
    
    print(f"\nTotal GNN grad norm: {total_norm_gnn**0.5:.6f}")
    
    # === TEST 4: Gradient through sp_soft_pool_avg ===
    print("\n" + "="*80)
    print("TEST 4: Gradient flow through sp_soft_pool_avg")
    print("="*80)
    
    spcnn.zero_grad()
    
    # Isolated test
    Fmap_test = Fmap.detach().requires_grad_(True)
    P_test = P.detach().requires_grad_(True)
    
    Z_test = sp_soft_pool_avg(Fmap_test, P_test)
    loss_test = Z_test.sum()
    loss_test.backward()
    
    print(f"\nIsolated test (Z.sum() backward):")
    print(f"  P_test grad norm: {P_test.grad.norm().item():.6f}")
    print(f"  Fmap_test grad norm: {Fmap_test.grad.norm().item():.6f}")
    
    if P_test.grad.norm() < 1e-6:
        print("  ⚠ WARNING: Very weak gradient through pooling!")
    else:
        print("  ✓ Gradient flows through sp_soft_pool_avg")
    
    # === SUMMARY ===
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    issues = []
    
    if grad_P_from_gnn is None or grad_P_from_gnn.norm() < 1e-8:
        issues.append("GNN gradients not reaching P (expected if gamma=0)")
    
    if P_test.grad.norm() < 1e-6:
        issues.append("Weak gradient through sp_soft_pool_avg - numerical issue")
    
    if total_norm_spcnn**0.5 < 1e-6:
        issues.append("SPNN not receiving gradients - critical issue")
    
    if len(issues) == 0:
        print("✓ All gradient flows look healthy!")
    else:
        print("Issues found:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    
    print("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--K', type=int, default=50)
    parser.add_argument('--C', type=int, default=6)
    parser.add_argument('--gamma', type=float, default=0.0)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    check_gradients_full_system(args)