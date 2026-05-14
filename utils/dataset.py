import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import tifffile as tiff


class ChangeDetectionDataset(Dataset):
    def __init__(self, root_dir, image_size=128):
        self.root_dir = root_dir
        self.image_size = image_size

        self.pre_dir = os.path.join(root_dir, "pre-event")
        self.post_dir = os.path.join(root_dir, "post-event")
        self.target_dir = os.path.join(root_dir, "target")

        self.files = sorted(os.listdir(self.pre_dir))

    def __len__(self):
        return len(self.files)

    def remap_labels(self, mask):
        """
        Label Remapping:
        0,1 -> 0 (No Change)
        2,3 -> 1 (Change)
        """
        mask = np.where(mask <= 1, 0, 1)
        return mask.astype(np.float32)

    def __getitem__(self, idx):

        file_name = self.files[idx]

        # -----------------------------
        # Load Images
        # -----------------------------
        pre_img = tiff.imread(
            os.path.join(self.pre_dir, file_name)
        )

        post_img = tiff.imread(
            os.path.join(self.post_dir, file_name)
        )

        mask = tiff.imread(
            os.path.join(self.target_dir, file_name)
        )

        # -----------------------------
        # Remap Labels
        # -----------------------------
        mask = self.remap_labels(mask)

        # -----------------------------
        # Normalize Images
        # -----------------------------
        pre_img = pre_img.astype(np.float32) / 255.0
        post_img = post_img.astype(np.float32) / 255.0

        # -----------------------------
        # Ensure 3D
        # -----------------------------
        if len(pre_img.shape) == 2:
            pre_img = np.expand_dims(pre_img, axis=-1)

        if len(post_img.shape) == 2:
            post_img = np.expand_dims(post_img, axis=-1)

        # -----------------------------
        # Concatenate Channels
        # -----------------------------
        x = np.concatenate(
            [pre_img, post_img],
            axis=-1
        )

        # -----------------------------
        # Resize
        # -----------------------------
        x = cv2.resize(
            x,
            (self.image_size, self.image_size)
        )

        mask = cv2.resize(
            mask,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST
        )

        # -----------------------------
        # Convert to Tensor
        # -----------------------------
        x = torch.tensor(
            x,
            dtype=torch.float32
        ).permute(2, 0, 1)

        y = torch.tensor(
            mask,
            dtype=torch.float32
        ).unsqueeze(0)

        return x, y