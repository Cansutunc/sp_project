import os, glob
from typing import Tuple, Optional
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
import torchvision.datasets as dset
import torchvision.transforms as transforms

# --- Original ImagesFolder Class (unchanged) ---
class ImagesFolder(Dataset):
    def __init__(self, root: str, size: Optional[Tuple[int,int]] = None):
        self.files=[]
        for ext in ('*.jpg','*.jpeg','*.png','*.bmp'):
            self.files+=glob.glob(os.path.join(root,ext))
        self.files.sort(); self.size=size
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        path=self.files[idx]; img=Image.open(path).convert('RGB')
        if self.size is not None: img=img.resize((self.size[1], self.size[0]), Image.BILINEAR)
        x=torch.from_numpy((np.array(img).astype('float32')/255.).transpose(2,0,1))
        return x, os.path.basename(path)

# --- New UNupervised CocoStuff Class ---
class CocoStuffUnsupervised(Dataset):
    def __init__(self, root, annFile, size=(320, 320), subset_fraction=1.0):
        self.coco = dset.CocoDetection(root, annFile)
        self.size = size
        self.transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
        ])

        if subset_fraction < 1.0:
            num_samples = int(len(self.coco) * subset_fraction)
            indices = np.random.choice(len(self.coco), num_samples, replace=False)
            self.coco = Subset(self.coco, indices)

    def __len__(self):
        return len(self.coco)

    def __getitem__(self, index):
        # We only care about the image, so we ignore the target '_'
        img, _ = self.coco[index] 
        return self.transform(img), f"coco_img_{index}" # Return a dummy filename