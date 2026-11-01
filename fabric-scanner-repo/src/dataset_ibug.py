"""
dataset.py  (iBUG Fabrics Dataset version)

Matches the REAL structure on disk:

    archive/Cotton/<sample_id>/im_1.*  im_2.*  im_3.*  im_4.*  tag
    archive/Denim/<sample_id>/...
    archive/Silk/<sample_id>/...
    archive/Wool/<sample_id>/...
    archive/Polyester/<sample_id>/...
    archive/Leather/<sample_id>/...

Only the 6 folders above are used (TARGET_CLASSES below) — every other
folder in archive/ (Acrylic, Velvet, Nylon, Unclassified, etc.) is
simply ignored, no need to move or delete anything.

Point root_dir at your archive/ folder directly when you run training:
    python train.py --data_dir "archive"
"""

import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

IMG_SIZE = 224

# The 6 class folders we actually want, exactly as named on disk
# (capitalization matches the real folder names).
TARGET_CLASSES = ["Cotton", "Denim", "Silk", "Wool", "Polyester", "Leather"]


def find_illum_images(sample_dir):
    """Finds im_1..im_4 regardless of file extension (.jpg/.png/.bmp/etc)."""
    paths = []
    for i in range(1, 5):
        matches = glob.glob(os.path.join(sample_dir, f"im_{i}.*"))
        if not matches:
            return None  # incomplete sample, skip it
        paths.append(matches[0])
    return paths


def brightness(img: np.ndarray) -> float:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean()


def align_and_diff(off_img: np.ndarray, on_img: np.ndarray) -> np.ndarray:
    """Aligns the brighter frame to the darker frame, then returns
    the difference map (this is your 'sheen' signal)."""
    off_gray = cv2.cvtColor(off_img, cv2.COLOR_BGR2GRAY)
    on_gray = cv2.cvtColor(on_img, cv2.COLOR_BGR2GRAY)

    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-6)
    try:
        _, warp_matrix = cv2.findTransformECC(
            off_gray, on_gray, warp_matrix, cv2.MOTION_EUCLIDEAN, criteria
        )
        aligned_on = cv2.warpAffine(
            on_img, warp_matrix, (off_img.shape[1], off_img.shape[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
        )
    except cv2.error:
        aligned_on = on_img

    return cv2.absdiff(aligned_on, off_img)


class FabricIlluminationDataset(Dataset):
    """
    root_dir/                  (this is your "archive" folder)
        Cotton/<sample_id>/im_1.*  im_2.*  im_3.*  im_4.*  tag
        Denim/<sample_id>/...
        Silk/<sample_id>/...
        Wool/<sample_id>/...
        Polyester/<sample_id>/...
        Leather/<sample_id>/...
        (any other folders, e.g. Acrylic, Velvet, Unclassified, are ignored)
    """

    def __init__(self, root_dir, class_names=None, transform=None):
        self.root_dir = root_dir
        self.class_names = class_names or TARGET_CLASSES
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}
        self.samples = []  # list of (sample_folder_path, label)
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ])

        skipped = 0
        for cls in self.class_names:
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir):
                print(f"WARNING: folder not found, skipping: {cls_dir}")
                continue
            for sample_id in sorted(os.listdir(cls_dir)):
                sample_dir = os.path.join(cls_dir, sample_id)
                if not os.path.isdir(sample_dir):
                    continue
                if find_illum_images(sample_dir) is None:
                    skipped += 1
                    continue
                self.samples.append((sample_dir, self.class_to_idx[cls]))

        print(f"Loaded {len(self.samples)} samples across {len(self.class_names)} classes "
              f"({skipped} incomplete samples skipped).")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_dir, label = self.samples[idx]

        img_paths = find_illum_images(sample_dir)
        imgs = []
        for path in img_paths:
            img = cv2.imread(path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            imgs.append(img)

        # Pick darkest as "off", brightest as "on" — approximates your
        # original flash-off/flash-on pair using the 4 lighting shots.
        brightnesses = [brightness(im) for im in imgs]
        off_img = imgs[int(np.argmin(brightnesses))]
        on_img = imgs[int(np.argmax(brightnesses))]

        diff_img = align_and_diff(off_img, on_img)

        off_rgb = cv2.cvtColor(off_img, cv2.COLOR_BGR2RGB)
        diff_rgb = cv2.cvtColor(diff_img, cv2.COLOR_BGR2RGB)

        return {
            "off_image": self.transform(off_rgb),
            "diff_image": self.transform(diff_rgb),
            "label": torch.tensor(label, dtype=torch.long),
        }
