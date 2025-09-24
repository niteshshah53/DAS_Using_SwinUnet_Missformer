import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from matplotlib.colors import ListedColormap
import argparse

# Dataset configurations
DATASET_CONFIGS = {
    'UDIADS_BIB': {
        'class_colors': [
            (0, 0, 0),         # 0: Background (black)
            (1, 1, 0),         # 1: Paratext (yellow)
            (0, 1, 1),         # 2: Decoration (cyan)
            (1, 0, 1),         # 3: Main Text (magenta)
            (1, 0, 0),         # 4: Title (red)
            (0, 1, 0),         # 5: Chapter Headings (lime)
        ],
        'class_names': [
            'Background', 'Paratext', 'Decoration', 'Main Text', 'Title', 'Chapter Headings'
        ],
        'color_map': {
            (0, 0, 0): 0,        # Background
            (255, 255, 0): 1,    # Paratext
            (0, 255, 255): 2,    # Decoration
            (255, 0, 255): 3,    # Main Text
            (255, 0, 0): 4,      # Title
            (0, 255, 0): 5,      # Chapter Headings
        },
        'max_classes': 6
    },
    'DIVAHISDB': {
        'class_colors': [
            (0, 0, 0),         # 0: Background (black)
            (0, 1, 0),         # 1: Comment (green)
            (1, 0, 0),         # 2: Decoration (red)
            (0, 0, 1),         # 3: Main Text (blue)
        ],
        'class_names': [
            'Background', 'Comment', 'Decoration', 'Main Text'
        ],
        'color_map': {
            (0, 0, 0): 0,        # Background
            (0, 255, 0): 1,      # Comment
            (255, 0, 0): 2,      # Decoration
            (0, 0, 255): 3,      # Main Text
        },
        'max_classes': 4
    }
}

def get_dataset_config(dataset_type):
    """Get dataset configuration based on type."""
    if dataset_type.upper() not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Supported: {list(DATASET_CONFIGS.keys())}")
    return DATASET_CONFIGS[dataset_type.upper()]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize prediction results')
    parser.add_argument('--dataset', type=str, default='UDIADS_BIB', 
                       choices=['UDIADS_BIB', 'DIVAHISDB'],
                       help='Dataset type (default: UDIADS_BIB)')
    parser.add_argument('--pred_dir', type=str, default='./model_out/udiadsbib/predictions/',
                       help='Directory containing .npy prediction files')
    parser.add_argument('--gt_dir', type=str, default='U-DIADS-Bib-MS',
                       help='Directory containing ground truth masks')
    
    args = parser.parse_args()
    
    # Get dataset configuration
    config = get_dataset_config(args.dataset)
    class_colors = config['class_colors']
    class_names = config['class_names']
    color_map = config['color_map']
    max_classes = config['max_classes']
    
    cmap = ListedColormap(class_colors)
    
    print(f"Dataset: {args.dataset}")
    print(f"Classes: {class_names}")
    print(f"Prediction directory: {args.pred_dir}")
    print(f"Ground truth directory: {args.gt_dir}")
    print("-" * 50)

    # List all prediction files
    pred_files = sorted(glob(os.path.join(args.pred_dir, '*.npy')))

    print(f"Found {len(pred_files)} prediction files. Press Enter to step, or 'q' then Enter to quit.")
    for idx, pred_path in enumerate(pred_files):
        pred = np.load(pred_path)
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        im1 = plt.imshow(pred, cmap=cmap, vmin=0, vmax=max_classes-1)
        plt.title('Prediction')
        plt.axis('off')
        # Add colorbar with class names
        cbar = plt.colorbar(im1, ticks=range(max_classes), fraction=0.046, pad=0.04)
        cbar.ax.set_yticklabels(class_names)

        # Try to find and plot the ground truth if available
        base_name = os.path.basename(pred_path).replace('_pred.npy', '.png')
        gt_path = None
        for root, dirs, files in os.walk(args.gt_dir):
            if base_name in files:
                gt_path = os.path.join(root, base_name)
                break
        plt.subplot(1, 2, 2)
        if gt_path:
            from PIL import Image
            gt = np.array(Image.open(gt_path))
            if gt.ndim == 3:
                # Convert RGB mask to class indices if needed
                gt_idx = np.zeros(gt.shape[:2], dtype=np.uint8)
                for rgb, idx in color_map.items():
                    matches = np.all(gt == rgb, axis=-1)
                    gt_idx[matches] = idx
                gt = gt_idx
            im2 = plt.imshow(gt, cmap=cmap, vmin=0, vmax=max_classes-1)
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