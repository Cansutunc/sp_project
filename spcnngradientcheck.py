# simple_gradient_check.py
import torch
from train_with_recon import *

checkpoint = torch.load('./checkpoints_diversity_fix/epoch_1.pt')
spcnn = EdgeAwareSPModule(in_c=5, num_feat=32, num_layers=4, num_spixels=50)
spcnn.load_state_dict(checkpoint['spcnn'])

x = torch.rand(2, 3, 320, 320)
x_with_coords = create_coord_grid_normalized(x)

outs = spcnn(x_with_coords, get_spixel_prob)
P = outs['P']

# Sadece SPNN parametrelerini kontrol et
loss = P.sum()
loss.backward()

print("SPNN Gradient Check:")
total_norm = 0
for name, param in spcnn.named_parameters():
    if param.grad is not None:
        norm = param.grad.norm().item()
        total_norm += norm ** 2
        if 'head' in name:
            print(f"  {name}: {norm:.6f}")

print(f"\nTotal grad norm: {total_norm**0.5:.6f}")
print("✓ SPNN gradients flow correctly" if total_norm > 0 else "✗ Problem!")