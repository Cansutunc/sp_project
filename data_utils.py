import os, glob
from typing import Tuple, Optional
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

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
