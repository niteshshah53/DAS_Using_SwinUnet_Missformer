import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from matplotlib.colors import ListedColormap

# Directory containing .npy prediction files
your_pred_dir = './model_out/udiadsbib/predictions/'  # Change if needed
# Directory containing ground truth masks (optional)
your_gt_dir = 'U-DIADS-Bib-MS'  # Change if you want to compare with GT

# Custom colormap for 6 classes
class_colors = [
    (0, 0, 0),         # 0: Background (black)
    (1, 1, 0),         # 1: Paratext (yellow)
    (0, 1, 1),         # 2: Decoration (cyan)
    (1, 0, 1),         # 3: Main Text (magenta)
    (1, 0, 0),         # 4: Title (red)
    (0, 1, 0),         # 5: Chapter Headings (lime)
]
cmap = ListedColormap(class_colors)
class_names = [
    'Background', 'Paratext', 'Decoration', 'Main Text', 'Title', 'Chapter Headings'
]

# List all prediction files
pred_files = sorted(glob(os.path.join(your_pred_dir, '*.npy')))

print(f"Found {len(pred_files)} prediction files. Press Enter to step, or 'q' then Enter to quit.")
for idx, pred_path in enumerate(pred_files):
    pred = np.load(pred_path)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    im1 = plt.imshow(pred, cmap=cmap, vmin=0, vmax=5)
    plt.title('Prediction')
    plt.axis('off')
    # Add colorbar with class names
    cbar = plt.colorbar(im1, ticks=range(6), fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels(class_names)

    # Try to find and plot the ground truth if available
    base_name = os.path.basename(pred_path).replace('_pred.npy', '.png')
    gt_path = None
    for root, dirs, files in os.walk(your_gt_dir):
        if base_name in files:
            gt_path = os.path.join(root, base_name)
            break
    plt.subplot(1, 2, 2)
    if gt_path:
        from PIL import Image
        gt = np.array(Image.open(gt_path))
        if gt.ndim == 3:
            # Convert RGB mask to class indices if needed
            from collections import defaultdict
            color_map = {
                (0, 0, 0): 0,
                (255, 255, 0): 1,
                (0, 255, 255): 2,
                (255, 0, 255): 3,
                (255, 0, 0): 4,
                (0, 255, 0): 5,
            }
            gt_idx = np.zeros(gt.shape[:2], dtype=np.uint8)
            for rgb, idx in color_map.items():
                matches = np.all(gt == rgb, axis=-1)
                gt_idx[matches] = idx
            gt = gt_idx
        im2 = plt.imshow(gt, cmap=cmap, vmin=0, vmax=5)
        plt.title('Ground Truth')
        plt.axis('off')
    else:
        plt.text(0.5, 0.5, 'No GT found', ha='center', va='center', fontsize=14)
        plt.axis('off')
    plt.suptitle(os.path.basename(pred_path))
    plt.tight_layout()
    plt.show(block=False)
    user_input = input(f'[{idx+1}/{len(pred_files)}] Press Enter for next, or q then Enter to quit: ')
    plt.close()
    if user_input.strip().lower() == 'q':
        print('Exiting visualization.')
        break