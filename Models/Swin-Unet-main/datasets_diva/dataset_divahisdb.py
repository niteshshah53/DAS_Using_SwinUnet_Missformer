import os
import glob
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


def _png_to_multichannel_mask(png_path, bit_to_class=None, add_background=True):
    """Convert a DIVAHisDB palette PNG to a multi-channel binary mask.

    Args:
        png_path: path to the ground-truth PNG
        bit_to_class: dict mapping bit index -> class index (e.g. {0:0,1:1,2:2})
        add_background: whether to append a background channel

    Returns:
        np.ndarray shape (C,H,W) dtype float32 with values 0.0/1.0
    """
    if bit_to_class is None:
        # default mapping: bit0->text, bit1->comment, bit2->decoration
        bit_to_class = {0: 0, 1: 1, 2: 2}
    img = np.array(Image.open(png_path).convert("RGB"))
    H, W = img.shape[:2]
    B = img[..., 2].astype(np.uint8)
    R = img[..., 0].astype(np.uint8)
    num_classes = max(bit_to_class.values()) + 1
    masks = np.zeros((num_classes, H, W), dtype=np.uint8)
    for bit_idx, cls_idx in bit_to_class.items():
        bit_val = (1 << bit_idx)
        m = ((B & bit_val) != 0)
        # include R==128 variants (annotation tool sometimes sets R=128)
        m = m | ((R == 128) & ((B & bit_val) == 0))
        masks[cls_idx] = m.astype(np.uint8)
    if add_background:
        bg = (~np.any(masks, axis=0)).astype(np.uint8)
        masks = np.vstack([masks, bg[None, ...]])
    return masks.astype(np.float32)


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label, angle_range=(-20, 20)):
    from scipy import ndimage
    angle = np.random.randint(angle_range[0], angle_range[1])
    image = ndimage.rotate(image, angle, order=3, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class DivaHisDB_dataset(Dataset):
    """Dataset for DIVA-HisDB arranged under `DivaHisDB/img/<manuscript>/<split>` and
    `DivaHisDB/pixel-level-gt/<manuscript>/<split>`.

    Returns samples with keys: 'image' (torch.FloatTensor [1,H,W]) and 'label' (torch.FloatTensor [C,H,W]).
    """

    def __init__(self, root_dir, manuscript=None, split="training", transforms=None, bit_to_class=None, add_background=True, patch_size=None, augment=False):
        self.root_dir = root_dir
        self.img_root = os.path.join(root_dir, "img")
        self.gt_root = os.path.join(root_dir, "pixel-level-gt")
        self.split = split
        self.transforms = transforms
        self.bit_to_class = bit_to_class if bit_to_class is not None else {0: 0, 1: 1, 2: 2}
        self.add_background = add_background
        self.patch_size = patch_size
        self.augment = augment

        # collect pairs
        self.samples = []
        manuscripts = [manuscript] if manuscript else sorted(os.listdir(self.img_root))
        for m in manuscripts:
            img_folder = os.path.join(self.img_root, m, split)
            gt_folder = os.path.join(self.gt_root, m, split)
            if not os.path.isdir(img_folder) or not os.path.isdir(gt_folder):
                continue
            for img_path in sorted(glob.glob(os.path.join(img_folder, "*.jpg"))):
                basename = os.path.splitext(os.path.basename(img_path))[0]
                # matching gt png has same base name
                gt_path = os.path.join(gt_folder, basename + ".png")
                if not os.path.exists(gt_path):
                    # skip if no gt
                    continue
                self.samples.append((img_path, gt_path, m))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, gt_path, case = self.samples[idx]
        img = Image.open(img_path).convert("L")  # grayscale like other dataset loader
        img_np = np.array(img).astype(np.float32)
        # normalize to 0-1 optionally later in transforms
        label = _png_to_multichannel_mask(gt_path, bit_to_class=self.bit_to_class, add_background=self.add_background)

        H, W = img_np.shape

        # If patch_size provided and training, sample random patch and corresponding label
        if self.patch_size and self.split == 'training':
            ph, pw = self.patch_size, self.patch_size
            if H <= ph or W <= pw:
                # fallback to resize
                from scipy.ndimage import zoom
                zh = ph / max(H, 1)
                zw = pw / max(W, 1)
                img_patch = zoom(img_np, (zh, zw), order=3)
                label_patch = np.zeros((label.shape[0], ph, pw), dtype=label.dtype)
                for c in range(label.shape[0]):
                    label_patch[c] = zoom(label[c], (zh, zw), order=0)
            else:
                y = np.random.randint(0, H - ph + 1)
                x = np.random.randint(0, W - pw + 1)
                img_patch = img_np[y:y+ph, x:x+pw]
                label_patch = label[:, y:y+ph, x:x+pw]
            # augment
            if self.augment and np.random.rand() > 0.5:
                img_patch, label_patch = random_rot_flip(img_patch, label_patch)
            elif self.augment and np.random.rand() > 0.5:
                img_patch, label_patch = random_rotate(img_patch, label_patch)
            sample = {"image": img_patch, "label": label_patch}
        elif self.patch_size and self.split in ['validation', 'test']:
            # center crop for validation/test
            ph = pw = self.patch_size
            cy = max(0, (H - ph) // 2)
            cx = max(0, (W - pw) // 2)
            img_patch = img_np[cy:cy+ph, cx:cx+pw]
            label_patch = label[:, cy:cy+ph, cx:cx+pw]
            sample = {"image": img_patch, "label": label_patch}
        else:
            sample = {"image": img_np, "label": label}

        if self.transforms:
            sample = self.transforms(sample)
        else:
            sample["image"] = torch.from_numpy(sample["image"]).unsqueeze(0)
            sample["label"] = torch.from_numpy(sample["label"]).float()
        sample["case_name"] = case + "/" + os.path.basename(img_path)
        return sample
