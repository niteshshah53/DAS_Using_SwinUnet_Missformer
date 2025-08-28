"""Small helper to convert DIVAHisDB masks to .npz (optional) and run a loader smoke test."""
import os
import sys
from pathlib import Path
# ensure parent folder (Swin-Unet-main) is on sys.path so local package imports work when running this file directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets_diva.dataset_divahisdb import DivaHisDB_dataset, _png_to_multichannel_mask
import numpy as np


def smoke_test(root_dir):
    ds = DivaHisDB_dataset(root_dir, manuscript=None, split="training", transforms=None)
    print('dataset length', len(ds))
    if len(ds) == 0:
        print('no samples found; check paths')
        return
    s = ds[0]
    print('sample keys:', list(s.keys()))
    print('image shape', s['image'].shape)
    print('label shape', s['label'].shape)


if __name__ == '__main__':
    root = os.path.join(os.path.dirname(__file__), '..', 'DivaHisDB')
    root = os.path.abspath(root)
    smoke_test(root)
