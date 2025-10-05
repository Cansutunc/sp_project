import os, glob
from typing import Tuple, Optional
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
import torchvision.transforms as transforms
from torchvision.datasets import VisionDataset
from pycocotools.coco import COCO
import random

class ImagesFolder(Dataset):
    def __init__(self, root: str, size: Optional[Tuple[int,int]] = None, subset_fraction: float = 1.0):
        self.files = []
        # Search recursively through all subfolders for images
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
            self.files += glob.glob(os.path.join(root, '**', ext), recursive=True)
        
        self.files.sort()
        
        # Apply subset_fraction
        if subset_fraction < 1.0:
            n_samples = int(len(self.files) * subset_fraction)
            # Use random sampling with fixed seed for reproducibility
            import random
            random.seed(42)
            self.files = random.sample(self.files, n_samples)
            self.files.sort()  # Re-sort after sampling
        
        self.size = size
        
    def __len__(self): 
        return len(self.files)
        
    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert('RGB')
        if self.size is not None: 
            img = img.resize((self.size[1], self.size[0]), Image.BILINEAR)
        x = torch.from_numpy((np.array(img).astype('float32')/255.).transpose(2,0,1))
        return x, os.path.basename(path)

class CocoStuffSupervised(VisionDataset):
    def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None):
        super(CocoStuffSupervised, self).__init__(root, transforms, transform, target_transform)
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        # PNG maskelerinin bulunduğu kök dizini otomatik bulur
        self.mask_root = os.path.join(os.path.dirname(os.path.dirname(root)), 'stuffthingmaps', os.path.basename(root))

    def __getitem__(self, index):
        id = self.ids[index]
        # Resim yolunu al
        path = self.coco.loadImgs(id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        
        # Segmentasyon maskesi yolunu al (.jpg -> .png)
        mask_path = os.path.join(self.mask_root, path.replace('.jpg', '.png'))
        mask = Image.open(mask_path)

        # Transformları uygula
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        return img, mask

    def __len__(self):
        return len(self.ids)