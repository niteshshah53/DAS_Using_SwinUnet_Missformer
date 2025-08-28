import os
import argparse
import glob
import json
import csv
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as TF
from networks.vision_transformer import SwinUnet as ViT_seg
from config import get_config
from datasets_diva.dataset_divahisdb import _png_to_multichannel_mask


def get_model(config, args):
    net = ViT_seg(config, img_size=args.patch_size, num_classes=args.num_classes).cuda()
    try:
        net.load_from(config)
    except Exception:
        pass
    return net


def load_checkpoint_for_test(net, args):
    candidates = []
    if args.output_dir and os.path.isdir(args.output_dir):
        best = os.path.join(args.output_dir, 'best_model_latest.pth')
        if os.path.exists(best):
            candidates.append(best)
        else:
            for p in sorted(glob.glob(os.path.join(args.output_dir, '*.pth'))):
                candidates.append(p)
    if not candidates:
        for p in sorted(glob.glob(os.path.join('pretrained_ckpt', '*.pth'))):
            candidates.append(p)
    if not candidates:
        return None
    ckpt = candidates[0]
    try:
        state = torch.load(ckpt, map_location='cpu')
        if 'state_dict' in state:
            state_dict = state['state_dict']
        else:
            state_dict = state
        net.load_state_dict(state_dict, strict=False)
        return ckpt
    except Exception:
        return None


def sliding_inference_on_image(model, img_arr, patch_size, stride):
    h, w = img_arr.shape[:2]
    n_classes = model.num_classes if hasattr(model, 'num_classes') else model.head.out_channels if hasattr(model, 'head') else None
    # We'll infer channel count from model output at runtime
    pred_accum = None
    count_map = np.zeros((h, w), dtype=np.float32)
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = img_arr[y:y+patch_size, x:x+patch_size, :]
            patch_tensor = TF.to_tensor(Image.fromarray(patch)).unsqueeze(0).cuda()
            with torch.no_grad():
                out = model(patch_tensor)
            out_np = out.squeeze(0).cpu().numpy()  # (C, H, W)
            C, ph, pw = out_np.shape
            if pred_accum is None:
                pred_accum = np.zeros((h, w, C), dtype=np.float32)
            out_np_t = np.transpose(out_np, (1, 2, 0))  # (H, W, C)
            pred_accum[y:y+ph, x:x+pw, :] += out_np_t
            count_map[y:y+ph, x:x+pw] += 1
    # handle border regions if image not divisible by patch_size: process last rows/cols
    # right/bottom padding patches
    if (w % patch_size) != 0:
        x = w - patch_size
        for y in range(0, h - patch_size + 1, stride):
            patch = img_arr[y:y+patch_size, x:x+patch_size, :]
            patch_tensor = TF.to_tensor(Image.fromarray(patch)).unsqueeze(0).cuda()
            with torch.no_grad():
                out = model(patch_tensor)
            out_np = out.squeeze(0).cpu().numpy()
            out_np_t = np.transpose(out_np, (1, 2, 0))
            pred_accum[y:y+patch_size, x:x+patch_size, :] += out_np_t
            count_map[y:y+patch_size, x:x+patch_size] += 1
    if (h % patch_size) != 0:
        y = h - patch_size
        for x in range(0, w - patch_size + 1, stride):
            patch = img_arr[y:y+patch_size, x:x+patch_size, :]
            patch_tensor = TF.to_tensor(Image.fromarray(patch)).unsqueeze(0).cuda()
            with torch.no_grad():
                out = model(patch_tensor)
            out_np = out.squeeze(0).cpu().numpy()
            out_np_t = np.transpose(out_np, (1, 2, 0))
            pred_accum[y:y+patch_size, x:x+patch_size, :] += out_np_t
            count_map[y:y+patch_size, x:x+patch_size] += 1
    # corner
    if (w % patch_size) != 0 and (h % patch_size) != 0:
        x = w - patch_size
        y = h - patch_size
        patch = img_arr[y:y+patch_size, x:x+patch_size, :]
        patch_tensor = TF.to_tensor(Image.fromarray(patch)).unsqueeze(0).cuda()
        with torch.no_grad():
            out = model(patch_tensor)
        out_np = out.squeeze(0).cpu().numpy()
        out_np_t = np.transpose(out_np, (1, 2, 0))
        pred_accum[y:y+patch_size, x:x+patch_size, :] += out_np_t
        count_map[y:y+patch_size, x:x+patch_size] += 1

    count_map = np.maximum(count_map, 1)[:, :, None]
    pred_accum = pred_accum / count_map
    return pred_accum  # (H, W, C) floats (logits)


def evaluate_on_diva(args):
    # provide fallback attributes expected by get_config/update_config
    if not hasattr(args, 'opts'):
        args.opts = None
    if not hasattr(args, 'batch_size'):
        args.batch_size = None
    if not hasattr(args, 'zip'):
        args.zip = False
    if not hasattr(args, 'cache_mode'):
        args.cache_mode = None
    if not hasattr(args, 'resume'):
        args.resume = None
    if not hasattr(args, 'accumulation_steps'):
        args.accumulation_steps = None
    if not hasattr(args, 'use_checkpoint'):
        args.use_checkpoint = False
    if not hasattr(args, 'amp_opt_level'):
        args.amp_opt_level = ''
    if not hasattr(args, 'tag'):
        args.tag = 'default'
    if not hasattr(args, 'eval'):
        args.eval = False
    if not hasattr(args, 'throughput'):
        args.throughput = False

    config = get_config(args)
    net = get_model(config, args)
    ckpt = load_checkpoint_for_test(net, args)
    if ckpt:
        print('Loaded checkpoint:', ckpt)
    else:
        print('No checkpoint found; aborting')
        return

    manuscript = args.manuscript
    img_dir = os.path.join(args.diva_root, 'img', manuscript, args.split)
    gt_dir = os.path.join(args.diva_root, 'pixel-level-gt', manuscript, args.split)
    if not os.path.isdir(img_dir) or not os.path.isdir(gt_dir):
        print('Missing dirs', img_dir, gt_dir)
        return

    img_files = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
    if len(img_files) == 0:
        print('No images found in', img_dir)
        return

    # Prepare output directories for predicted images and comparisons
    save_root = args.save_dir or os.path.join(args.output_dir or '.', 'predictions_diva')
    result_dir = os.path.join(save_root, 'result')
    compare_dir = os.path.join(save_root, 'compare')
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(compare_dir, exist_ok=True)

    # default colors for visualization (RGB)
    class_colors = [(0, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255)]

    # Metrics accumulators will be created after detecting number of classes from GT
    TP = None; FP = None; FN = None; IoU_accum = None
    processed = 0
    for img_path in img_files:
        base = os.path.splitext(os.path.basename(img_path))[0]
        gt_path = os.path.join(gt_dir, base + '.png')
        if not os.path.exists(gt_path):
            print('GT missing for', base); continue
        img = np.array(Image.open(img_path).convert('RGB'))
        gt_masks = _png_to_multichannel_mask(gt_path, add_background=True)  # (C,H,W)
        C, H, W = gt_masks.shape

        if args.num_classes is None:
            args.num_classes = C

        if TP is None:
            TP = np.zeros(C, dtype=np.float64)
            FP = np.zeros(C, dtype=np.float64)
            FN = np.zeros(C, dtype=np.float64)
            IoU_accum = np.zeros(C, dtype=np.float64)

        pred_logits = sliding_inference_on_image(net, img, args.patch_size, args.stride)
        # pred_logits is (H,W,C)
        if pred_logits.shape[-1] != C:
            print(f'Warning: model outputs {pred_logits.shape[-1]} channels but GT has {C}; will try to adapt by truncation/padding')
            # truncate or pad
            if pred_logits.shape[-1] > C:
                pred_logits = pred_logits[..., :C]
            else:
                pad = np.zeros((H, W, C - pred_logits.shape[-1]), dtype=pred_logits.dtype)
                pred_logits = np.concatenate([pred_logits, pad], axis=-1)

        if args.multilabel:
            probs = 1.0 / (1.0 + np.exp(-pred_logits))
            preds = (probs >= args.threshold).astype(np.uint8)
        else:
            preds = np.argmax(pred_logits, axis=-1)
            # convert to one-hot per-class
            preds = np.stack([(preds == i).astype(np.uint8) for i in range(pred_logits.shape[-1])], axis=0)
            preds = np.transpose(preds, (1, 2, 0))

        # ensure preds is (H,W,C) and gt_masks is (C,H,W)
        if preds.shape != (H, W, C):
            # maybe GT is C,H,W; convert
            if preds.shape == (C, H, W):
                preds = np.transpose(preds, (1, 2, 0))
            else:
                print('Unexpected shape for preds', preds.shape); continue

        gt = np.transpose(gt_masks, (1, 2, 0)).astype(np.uint8)  # (H,W,C)

        for cls in range(C):
            pred_c = preds[..., cls].astype(bool)
            gt_c = gt[..., cls].astype(bool)
            TP[cls] += np.logical_and(pred_c, gt_c).sum()
            FP[cls] += np.logical_and(pred_c, np.logical_not(gt_c)).sum()
            FN[cls] += np.logical_and(np.logical_not(pred_c), gt_c).sum()
            union = np.logical_or(pred_c, gt_c).sum()
            IoU_accum[cls] += (np.logical_and(pred_c, gt_c).sum() / (union + 1e-8))

        # Save predicted RGB mask and comparison image (original | pred | gt)
        # pred_rgb: additive overlay of per-class colors (clips at 255)
        pred_rgb = np.zeros((H, W, 3), dtype=np.uint16)
        gt_rgb = np.zeros((H, W, 3), dtype=np.uint16)
        for cls in range(C):
            color = class_colors[cls] if cls < len(class_colors) else (255, 255, 255)
            mask_pred = preds[..., cls].astype(np.uint8)
            mask_gt = gt[..., cls].astype(np.uint8)
            for c_idx in range(3):
                pred_rgb[..., c_idx] += mask_pred * color[c_idx]
                gt_rgb[..., c_idx] += mask_gt * color[c_idx]
        # clip to 0-255 and convert to uint8
        pred_rgb = np.clip(pred_rgb, 0, 255).astype(np.uint8)
        gt_rgb = np.clip(gt_rgb, 0, 255).astype(np.uint8)

        orig_arr = img
        # If sizes mismatch, resize orig to prediction
        if orig_arr.shape[0:2] != pred_rgb.shape[0:2]:
            from PIL import Image as PILImage
            orig_arr = np.array(PILImage.fromarray(orig_arr).resize((pred_rgb.shape[1], pred_rgb.shape[0]), PILImage.BILINEAR))

        Image.fromarray(pred_rgb).save(os.path.join(result_dir, f'{base}.png'))
        comp = np.concatenate([orig_arr, pred_rgb, gt_rgb], axis=1)
        Image.fromarray(comp).save(os.path.join(compare_dir, f'{base}_compare.png'))

        processed += 1
        print('Processed', base)

    if processed == 0:
        print('No processed images')
        return

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    mean_iou = IoU_accum / processed

    # prepare class names
    class_names = args.class_names.split(',') if args.class_names else [f'cls{i}' for i in range(len(TP))]
    # Print per-class
    print('\nPer-class metrics:')
    for i in range(len(TP)):
        name = class_names[i] if i < len(class_names) else f'cls{i}'
        print(f"{name}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1[i]:.4f}, IoU={mean_iou[i]:.4f}")

    print('\nMean metrics:')
    print(f"Mean Precision: {np.mean(precision):.4f}")
    print(f"Mean Recall: {np.mean(recall):.4f}")
    print(f"Mean F1: {np.mean(f1):.4f}")
    print(f"Mean IoU: {np.mean(mean_iou):.4f}")

    # Save metrics
    out_dir = args.save_dir or os.path.join(args.output_dir or '.', 'diva_metrics')
    os.makedirs(out_dir, exist_ok=True)
    metrics = {'per_class': {}, 'mean': {}}
    metrics['mean'] = {'precision': float(np.mean(precision)), 'recall': float(np.mean(recall)), 'f1': float(np.mean(f1)), 'iou': float(np.mean(mean_iou))}
    for i in range(len(TP)):
        metrics['per_class'][class_names[i] if i < len(class_names) else f'cls{i}'] = {
            'precision': float(precision[i]), 'recall': float(recall[i]), 'f1': float(f1[i]), 'iou': float(mean_iou[i])
        }
    with open(os.path.join(out_dir, 'metrics_diva.json'), 'w') as jf:
        json.dump(metrics, jf, indent=2)
    with open(os.path.join(out_dir, 'metrics_diva.csv'), 'w', newline='') as cf:
        writer = csv.writer(cf)
        writer.writerow(['class', 'precision', 'recall', 'f1', 'iou'])
        for i in range(len(TP)):
            writer.writerow([class_names[i] if i < len(class_names) else f'cls{i}', f"{precision[i]:.6f}", f"{recall[i]:.6f}", f"{f1[i]:.6f}", f"{mean_iou[i]:.6f}"])
        writer.writerow(['MEAN', f"{np.mean(precision):.6f}", f"{np.mean(recall):.6f}", f"{np.mean(f1):.6f}", f"{np.mean(mean_iou):.6f}"])
    print('Saved metrics to', out_dir)


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/swin_tiny_patch4_window7_224_lite.yaml')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--diva_root', type=str, default='DivaHisDB')
    parser.add_argument('--manuscript', type=str, required=True, help='CB55, CS18, or CS863')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--patch_size', type=int, default=224)
    parser.add_argument('--stride', type=int, default=224)
    parser.add_argument('--num_classes', type=int, default=None)
    parser.add_argument('--multilabel', action='store_true', default=True, help='treat output as independent channels with sigmoid')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--class_names', type=str, default='text,comment,decoration,background')
    parser.add_argument('--save_dir', type=str, default=None)
    return parser


if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    # basic setup
    args.num_classes = args.num_classes
    evaluate_on_diva(args)
