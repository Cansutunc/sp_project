import os, glob
from typing import Tuple, Optional
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
import torchvision.transforms as transforms

class ImagesFolder(Dataset):
    def __init__(self, root: str, size: Optional[Tuple[int,int]] = None, subset_fraction=1.0):
        self.files=[]
        for ext in ('*.jpg','*.jpeg','*.png','*.bmp'):
            # NEW corrected line
            self.files+=glob.glob(os.path.join(root, '**', ext), recursive=True)
        
        self.files.sort()
        self.size=size
        self.transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
        ])

        if subset_fraction < 1.0:
            num_samples = int(len(self.files) * subset_fraction)
            indices = np.random.choice(len(self.files), num_samples, replace=False)
            self.files = [self.files[i] for i in indices]

    def __len__(self): 
        return len(self.files)

    def __getitem__(self, idx):
        path=self.files[idx]
        img=Image.open(path).convert('RGB')
        
        # Apply transformations
        img_tensor = self.transform(img)
        
        return img_tensor, os.path.basename(path)