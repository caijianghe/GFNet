from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from data.common import get_patch,arugment
from utils import make_coord
import torch

class NYU_v2_datset(Dataset):
    """NYUDataset."""

    def __init__(self, root_dir, scale=8, train=True, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            scale (float): dataset scale
            train (bool): train or test
            transform (callable, optional): Optional transform to be applied on a sample.
            
        """
        self.root_dir = root_dir
        self.transform = transform
        self.scale = scale
        self.train = train
        
        if train:
            self.depths = np.load('%s/train_depth_split.npy'%root_dir)
            self.images = np.load('%s/train_images_split.npy'%root_dir)
        else:
            self.depths = np.load('%s/test_depth.npy'%root_dir)
            self.images = np.load('%s/test_images_v2.npy'%root_dir)

    def __len__(self):
        return self.depths.shape[0]

    def __getitem__(self, idx):
        depth = self.depths[idx]
        image = self.images[idx]
        if self.train:
            image, depth = get_patch(img=image, gt=np.expand_dims(depth,2), patch_size=256)
            image, depth = arugment(img=image, gt=depth)
        h, w = depth.shape[:2]
        s = self.scale
        lr = np.array(Image.fromarray(depth.squeeze()).resize((w//s,h//s), Image.BICUBIC))
        # depth_lr = np.array(Image.fromarray(depth).resize((w // s, h // s), Image.BICUBIC))

        if self.transform:
            image = self.transform(image).float()
            depth = self.transform(depth).float()
            lr = self.transform(np.expand_dims(lr,2)).float()
            # depth_lr = self.transform(np.expand_dims(depth_lr,2)).float()

        image = image.contiguous()
        depth = depth.contiguous()
        lr = lr.contiguous()
        # depth_lr = depth_lr.contiguous()

        hr_coord = make_coord(depth.shape[-2:], flatten=True)
        # cell = torch.tensor([2 / depth.shape[-2], 2 / depth.shape[-1]], dtype=torch.float32)
        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / depth.shape[-2]
        cell[:, 1] *= 2 / depth.shape[-1]

        sample = {'guidance': image, 'lr': lr, 'gt': depth, 'hr_coord': hr_coord, 'cell': cell}
        
        return sample