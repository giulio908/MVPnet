from torch.utils.data import Dataset
import os
import cv2
import numpy as np
import torch
from utils import get_inverse_depth_viz

class MirrorDataset(Dataset):
    def __init__(self, base_dir, base_names_list, transform=None):
        self.base_dir = base_dir
        self.base_names = base_names_list
        self.transform = transform 

        self.rgb_dir = f"{self.base_dir}/image"
        self.mask_sloppy_dir = f"{self.base_dir}/mask_sloppy"
        self.depth_dir = f"{self.base_dir}/depth"
        self.target_perfect_dir = f"{self.base_dir}/depth_gt"
        self.mask_dir = f"{self.base_dir}/mask"

    def __len__(self):
        return len(self.base_names)

    def __getitem__(self, idx):
        name_no_ext = self.base_names[idx]

        rgb_path = os.path.join(self.rgb_dir, f"{name_no_ext}.jpg")
        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB) # Forma: (H, W, 3)

        mask_path = os.path.join(self.mask_sloppy_dir, f"{name_no_ext}.png")
        mask_sloppy = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask_sloppy = (mask_sloppy > 127).astype(np.float32)

        depth_path = os.path.join(self.depth_dir, f"{name_no_ext}.npy")
        depth_raw = np.load(depth_path).astype(np.float32)
        depth_norm, min_inv, max_inv = get_inverse_depth_viz(depth_raw)

        target_path = os.path.join(self.target_perfect_dir, f"{name_no_ext}.npy")
        t_perfect = np.load(target_path).astype(np.float32)
        t_perfect_norm, _, _ = get_inverse_depth_viz(t_perfect, min_inv=min_inv, max_inv=max_inv)

        mask_perf_path = os.path.join(self.mask_dir, name_no_ext + '.png')
        mask_perfect = cv2.imread(mask_perf_path, cv2.IMREAD_GRAYSCALE)
        mask_perfect = (mask_perfect > 127).astype(np.float32)

        if self.transform is not None:
            augmented = self.transform(
                image=rgb,
                mask=mask_perfect,
                mask_sloppy=mask_sloppy,
                depth=depth_norm,
                t_perfect=t_perfect_norm
            )
            
            rgb = augmented['image']
            mask_perfect = augmented['mask']
            mask_sloppy = augmented['mask_sloppy']
            depth_norm = augmented['depth']
            t_perfect_norm = augmented['t_perfect']

        rgb = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)

        return {
            'rgb': torch.from_numpy(rgb),
            'depth': torch.from_numpy(depth_norm).unsqueeze(0),
            'mask_sloppy': torch.from_numpy(mask_sloppy).unsqueeze(0),
            'target_perfect': torch.from_numpy(t_perfect_norm).unsqueeze(0),
            'mask_perfect': torch.from_numpy(mask_perfect).unsqueeze(0)
        }