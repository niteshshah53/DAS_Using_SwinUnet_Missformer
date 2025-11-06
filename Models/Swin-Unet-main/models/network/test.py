"""
Testing Script for CNN-Transformer Historical Document Segmentation Models

This script evaluates trained CNN-Transformer models on historical document test datasets by:
- Loading trained model checkpoints
- Running inference on test images using patch-based approach
- Computing segmentation metrics (IoU, Precision, Recall, F1)
- Saving prediction visualizations

Supported datasets: U-DIADS-Bib, DIVAHISDB

Usage:
    # For CNN-Transformer:
    python test.py --output_dir ./models/ --manuscript Latin2 --is_savenii
    
Author: Clean Code Version
"""

import argparse
import logging
import os
import random
import sys
import warnings
import glob

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Add common directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../common'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def parse_arguments():
    """Parse command line arguments for testing script."""
    parser = argparse.ArgumentParser(
        description='Test CNN-Transformer model on historical document datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test on U-DIADS-Bib dataset with CNN-Transformer
  python test.py --output_dir ./models/ \\
                 --dataset UDIADS_BIB --manuscript Latin2 --is_savenii
  
  # Test on DIVAHISDB dataset with CNN-Transformer
  python test.py --dataset DIVAHISDB \\
                 --output_dir ./models/ --manuscript Latin2
        """
    )
    
    # Core arguments
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory containing trained model checkpoints')
    
    # Dataset configuration
    parser.add_argument('--dataset', type=str, default='UDIADS_BIB',
                       choices=['UDIADS_BIB', 'DIVAHISDB'],
                       help='Dataset to test on')
    parser.add_argument('--manuscript', type=str, required=True,
                       help='Manuscript to test (e.g., Latin2, Latin14396, Latin16746, Syr341, Latin2FS, etc.)')
    parser.add_argument('--udiadsbib_root', type=str, default='../../U-DIADS-Bib-MS',
                       help='Root directory for U-DIADS-Bib dataset')
    parser.add_argument('--divahisdb_root', type=str, default='../../DivaHisDB',
                       help='Root directory for DIVAHISDB dataset')
    parser.add_argument('--use_patched_data', action='store_true',
                       help='Use pre-generated patches instead of full images')
    
    # Model parameters
    parser.add_argument('--num_classes', type=int, default=None,
                       help='Number of segmentation classes (auto-detected from dataset)')
    parser.add_argument('--img_size', type=int, default=224,
                       help='Input patch size for inference')
    parser.add_argument('--batch_size', type=int, default=24,
                       help='Batch size for testing')
    
    # Model enhancement flags
    parser.add_argument('--deep_supervision', action='store_true', default=False, 
                       help='Enable deep supervision (must match training configuration)')
    parser.add_argument('--fusion_method', type=str, default='simple',
                       choices=['simple', 'fourier', 'smart'],
                       help='Feature fusion method (must match training configuration)')
    parser.add_argument('--adapter_mode', type=str, default='external', choices=['external', 'streaming'],
                       help='Adapter placement: external adapters or streaming (must match training)')
    parser.add_argument('--bottleneck', action='store_true', default=False,
                       help='Enable bottleneck with 2 Swin blocks (must match training)')
    parser.add_argument('--use_multiscale_agg', action='store_true', default=False,
                       help='Enable multi-scale aggregation (must match training)')
    
    # Freezing configuration
    parser.add_argument('--freeze_encoder', action='store_true', default=False,
                       help='Freeze encoder during testing (for inference speed)')
    
    # Output options
    parser.add_argument('--is_savenii', action="store_true",
                       help='Save prediction results during inference')
    parser.add_argument('--test_save_dir', type=str, default='../predictions',
                       help='Directory to save prediction results')
    
    # Test-time augmentation
    parser.add_argument('--use_tta', action='store_true', default=False,
                       help='Enable test-time augmentation: predict on augmented versions (original, hflip, vflip, rot90) and average')
    
    # CRF post-processing
    parser.add_argument('--use_crf', action='store_true', default=False,
                       help='Enable CRF post-processing: refine predictions using DenseCRF for spatial coherence')
    
    # System configuration
    parser.add_argument('--deterministic', type=int, default=1,
                       help='Use deterministic testing')
    parser.add_argument('--seed', type=int, default=1234,
                       help='Random seed for reproducibility')
    
    # Advanced options (required by config.py)
    parser.add_argument('--opts', nargs='+', default=None,
                       help='Modify config options')
    parser.add_argument('--zip', action='store_true',
                       help='Use zipped dataset')
    parser.add_argument('--cache-mode', type=str, default='part',
                       choices=['no', 'full', 'part'],
                       help='Dataset caching strategy')
    parser.add_argument('--resume', help='Resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int,
                       help='Gradient accumulation steps')
    parser.add_argument('--use-checkpoint', action='store_true',
                       help='Use gradient checkpointing')
    parser.add_argument('--amp-opt-level', type=str, default='O1',
                       choices=['O0', 'O1', 'O2'],
                       help='Mixed precision optimization level')
    parser.add_argument('--tag', help='Experiment tag')
    parser.add_argument('--eval', action='store_true',
                       help='Evaluation only mode')
    parser.add_argument('--throughput', action='store_true',
                       help='Test throughput only')
    
    return parser.parse_args()


def validate_arguments(args):
    """Validate and normalize command line arguments."""
    # Check for suspicious command line tokens
    bad_tokens = [t for t in sys.argv[1:] if t.lstrip('-').startswith('mg_')]
    if bad_tokens:
        print(f"Warning: suspicious argv tokens detected: {bad_tokens}")
        print("Did you accidentally paste a continuation fragment?")
    
    # Validate required paths
    if not os.path.exists(args.output_dir):
        raise ValueError(f"Output directory not found: {args.output_dir}")
    
    # Set dataset-specific parameters
    if args.dataset.upper() == "UDIADS_BIB":
        # Determine number of classes based on manuscript
        if args.manuscript in ['Syr341FS', 'Syr341']:
            args.num_classes = 5
            print("Detected Syriaque341 manuscript: using 5 classes (no Chapter Headings)")
        else:
            args.num_classes = 6
            print(f"Using 6 classes for manuscript: {args.manuscript}")
        
        if not os.path.exists(args.udiadsbib_root):
            raise ValueError(f"U-DIADS-Bib dataset path not found: {args.udiadsbib_root}")
    elif args.dataset.upper() == "DIVAHISDB":
        args.num_classes = 4
        if not os.path.exists(args.divahisdb_root):
            raise ValueError(f"DIVAHISDB dataset path not found: {args.divahisdb_root}")
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    print("All arguments validated successfully!")


def get_model(args, config):
    """
    Create and load the CNN-Transformer model.
    
    Args:
        args: Command line arguments
        config: Model configuration (not used for CNN-Transformer)
        
    Returns:
        torch.nn.Module: Initialized model
    """
    from vision_transformer_cnn import CNNTransformerUnet as ViT_seg
    # Create model with same deep supervision and fusion settings as training
    model = ViT_seg(
        None,
        img_size=args.img_size,
        num_classes=args.num_classes,
        use_deep_supervision=getattr(args, 'deep_supervision', False),
        fusion_method=getattr(args, 'fusion_method', 'simple'),
        use_bottleneck=getattr(args, 'bottleneck', False),
        adapter_mode=getattr(args, 'adapter_mode', 'external'),
        use_multiscale_agg=getattr(args, 'use_multiscale_agg', False)
    )
    
    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
    else:
        print("CUDA not available, using CPU")
    
    # Freeze encoder if requested
    if getattr(args, 'freeze_encoder', False):
        print("🔒 Freezing encoder for testing")
        model.model.freeze_encoder()
    
    model.load_from(None)
    return model


def setup_logging(log_folder, snapshot_name):
    """Set up logging configuration."""
    os.makedirs(log_folder, exist_ok=True)
    log_file = os.path.join(log_folder, f"{snapshot_name}.txt")
    
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def setup_reproducible_testing(args):
    """Set up reproducible testing environment."""
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def load_model_checkpoint(model, args):
    """
    Load trained model checkpoint.
    
    Args:
        model: Model to load weights into
        args: Command line arguments
        
    Returns:
        str: Name of loaded checkpoint file
    """
    checkpoint_path = os.path.join(args.output_dir, 'best_model_latest.pth')
    
    if not os.path.exists(checkpoint_path):
        # Try alternative checkpoint names
        alt_path = os.path.join(args.output_dir, 'best_model.pth')
        if os.path.exists(alt_path):
            checkpoint_path = alt_path
        else:
            raise FileNotFoundError(f"No checkpoint found in {args.output_dir}")
    
    # Load checkpoint with appropriate strictness based on architecture match
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract model_state from checkpoint (checkpoint may be a dict with 'model_state' key or direct state_dict)
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        model_state_dict = checkpoint['model_state']
    else:
        # If checkpoint is already a state_dict, use it directly
        model_state_dict = checkpoint
    
    # Check if checkpoint has deep supervision auxiliary heads
    has_aux_heads = any('aux_heads' in key for key in model_state_dict.keys())
    
    # Check if checkpoint has fusion modules (fourier or smart)
    has_skip_fusions = any('skip_fusions' in key for key in model_state_dict.keys())
    has_smart_skips = any('smart_skips' in key for key in model_state_dict.keys())
    
    # Check if checkpoint has bottleneck layer
    has_bottleneck = any('bottleneck_layer' in key and 'upsample' not in key for key in model_state_dict.keys())
    
    # Check adapter mode: checkpoint has feature_adapters (external) or streaming_proj (streaming)
    has_feature_adapters = any('feature_adapters' in key for key in model_state_dict.keys())
    has_streaming_proj = any('streaming_proj' in key for key in model_state_dict.keys())
    
    # Check multi-scale aggregation
    has_multiscale_agg = any('multiscale_proj' in key or 'multiscale_fusion' in key for key in model_state_dict.keys())
    
    # Determine if there's a mismatch in architecture components
    ds_mismatch = (has_aux_heads and not model.model.use_deep_supervision) or \
                  (not has_aux_heads and model.model.use_deep_supervision)
    
    # Check adapter mode mismatch
    adapter_mismatch = False
    if model.model.adapter_mode == 'external':
        adapter_mismatch = not has_feature_adapters or has_streaming_proj
    elif model.model.adapter_mode == 'streaming':
        adapter_mismatch = not has_streaming_proj or has_feature_adapters
    
    # Check fusion method mismatch
    fusion_mismatch = False
    if model.model.fusion_method == 'fourier':
        fusion_mismatch = not has_skip_fusions
    elif model.model.fusion_method == 'smart':
        fusion_mismatch = not has_smart_skips
    else:  # 'simple'
        fusion_mismatch = (has_skip_fusions or has_smart_skips)
    
    # Check bottleneck mismatch
    bottleneck_mismatch = (has_bottleneck and not model.model.use_bottleneck) or \
                         (not has_bottleneck and model.model.use_bottleneck)
    
    # Check multi-scale aggregation mismatch
    multiscale_mismatch = (has_multiscale_agg and not model.model.use_multiscale_agg) or \
                         (not has_multiscale_agg and model.model.use_multiscale_agg)
    
    # Use strict=False if there's any mismatch
    if ds_mismatch or fusion_mismatch or bottleneck_mismatch or adapter_mismatch or multiscale_mismatch:
        print(f"Note: Checkpoint and model have architecture differences")
        print(f"  - Deep Supervision: checkpoint={has_aux_heads}, model={model.model.use_deep_supervision}")
        print(f"  - Adapter Mode: checkpoint has feature_adapters={has_feature_adapters}, streaming_proj={has_streaming_proj}, model={model.model.adapter_mode}")
        print(f"  - Fusion Method: checkpoint has skip_fusions={has_skip_fusions}, smart_skips={has_smart_skips}, model={model.model.fusion_method}")
        print(f"  - Bottleneck: checkpoint={has_bottleneck}, model={model.model.use_bottleneck}")
        print(f"  - Multi-Scale Aggregation: checkpoint={has_multiscale_agg}, model={model.model.use_multiscale_agg}")
        print(f"  - Loading with strict=False to handle mismatches")
        
        msg = model.load_state_dict(model_state_dict, strict=False)
        
        if msg.unexpected_keys:
            print(f"Warning: Ignored unexpected keys: {msg.unexpected_keys[:3]}... ({len(msg.unexpected_keys)} total)")
        if msg.missing_keys:
            print(f"Warning: Missing keys (may be expected due to architecture differences):")
            missing_fusion = [k for k in msg.missing_keys if 'skip_fusions' in k or 'smart_skips' in k]
            missing_other = [k for k in msg.missing_keys if 'skip_fusions' not in k and 'smart_skips' not in k]
            if missing_fusion:
                print(f"  - Fusion-related: {missing_fusion[0]}... ({len(missing_fusion)} total)")
            if missing_other:
                print(f"  - Other: {missing_other[0]}... ({len(missing_other)} total)")
    else:
        # Perfect match, use strict=True for exact loading
        print(f"Checkpoint and model architecture match - loading with strict=True")
        msg = model.load_state_dict(model_state_dict, strict=True)
    
    print(f"Model checkpoint loaded successfully")
    
    return os.path.basename(checkpoint_path)


def get_dataset_info(dataset_type, manuscript=None):
    """
    Get dataset-specific information.
    
    Args:
        dataset_type (str): Type of dataset
        manuscript (str): Manuscript name for class-specific logic
        
    Returns:
        tuple: (class_colors, class_names, rgb_to_class_function)
    """
    if dataset_type.upper() == "UDIADS_BIB":
        from datasets.dataset_udiadsbib import rgb_to_class
        
        class_colors = [
            (0, 0, 0),         # 0: Background (black)
            (255, 255, 0),     # 1: Paratext (yellow)
            (0, 255, 255),     # 2: Decoration (cyan)  
            (255, 0, 255),     # 3: Main Text (magenta)
            (255, 0, 0),       # 4: Title (red)
            (0, 255, 0),       # 5: Chapter Headings (lime)
        ]
        
        # Adjust class names based on manuscript
        if manuscript in ['Syr341', 'Syr341FS']:
            # Syr341 manuscripts don't have Chapter Headings
            class_names = [
                'Background', 'Paratext', 'Decoration', 
                'Main Text', 'Title'
            ]
            class_colors = class_colors[:5]  # Only first 5 colors
        else:
            class_names = [
                'Background', 'Paratext', 'Decoration', 
                'Main Text', 'Title', 'Chapter Headings'
            ]
        
        return class_colors, class_names, rgb_to_class
        
    elif dataset_type.upper() == "DIVAHISDB":
        try:
            from datasets.dataset_divahisdb import rgb_to_class
        except ImportError:
            print("Warning: DIVAHISDB dataset class not available")
            rgb_to_class = None
        
        class_colors = [
            (0, 0, 0),      # 0: Background (black)
            (0, 255, 0),    # 1: Comment (green)
            (255, 0, 0),    # 2: Decoration (red)
            (0, 0, 255),    # 3: Main Text (blue)
        ]
        
        class_names = ['Background', 'Comment', 'Decoration', 'Main Text']
        
        return class_colors, class_names, rgb_to_class
    
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def get_dataset_paths(args):
    """
    Get dataset-specific file paths.
    
    Args:
        args: Command line arguments
        
    Returns:
        tuple: (patch_dir, mask_dir, original_img_dir, original_mask_dir)
    """
    manuscript_name = args.manuscript
    
    if args.dataset.upper() == "UDIADS_BIB":
        if args.use_patched_data:
            patch_dir = f'{args.udiadsbib_root}/{manuscript_name}/Image/test'
            mask_dir = f'{args.udiadsbib_root}/{manuscript_name}/mask/test_labels'
        else:
            patch_dir = f'{args.udiadsbib_root}/{manuscript_name}/img-{manuscript_name}/test'
            mask_dir = f'{args.udiadsbib_root}/{manuscript_name}/pixel-level-gt-{manuscript_name}/test'
        
        # Use the original dataset directory (before patching) for original images
        # Extract the base directory name from the patched data root
        base_dir = args.udiadsbib_root.replace('_patched', '')
        original_img_dir = f'{base_dir}/{manuscript_name}/img-{manuscript_name}/test'
        original_mask_dir = f'{base_dir}/{manuscript_name}/pixel-level-gt-{manuscript_name}/test'
        
    elif args.dataset.upper() == "DIVAHISDB":
        if args.use_patched_data:
            patch_dir = f'{args.divahisdb_root}/{manuscript_name}/Image/test'
            mask_dir = f'{args.divahisdb_root}/{manuscript_name}/mask/test_labels'
        else:
            patch_dir = f'{args.divahisdb_root}/img/{manuscript_name}/test'
            mask_dir = f'{args.divahisdb_root}/pixel-level-gt/{manuscript_name}/test'
        
        # Use the original dataset directory (before patching) for original images
        # Extract the base directory name from the patched data root
        base_dir = args.divahisdb_root.replace('_patched', '')
        original_img_dir = f'{base_dir}/img/{manuscript_name}/test'
        original_mask_dir = f'{base_dir}/pixel-level-gt/{manuscript_name}/test'
    
    return patch_dir, mask_dir, original_img_dir, original_mask_dir


def process_patch_groups(patch_files):
    """
    Group patch files by their original image names.
    
    Args:
        patch_files (list): List of patch file paths
        
    Returns:
        tuple: (patch_groups, patch_positions) dictionaries
    """
    patch_groups = {}
    patch_positions = {}
    
    for patch_path in patch_files:
        filename = os.path.basename(patch_path)
        parts = filename.split('_')
        
        if len(parts) >= 2:
            original_name = '_'.join(parts[:-1])
            patch_id = int(parts[-1].split('.')[0])
            
            if original_name not in patch_groups:
                patch_groups[original_name] = []
            patch_groups[original_name].append(patch_path)
            patch_positions[patch_path] = patch_id
    
    return patch_groups, patch_positions


def estimate_image_dimensions(original_name, original_img_dir, patches, patch_positions, patch_size=224):
    """
    Estimate original image dimensions from patch information.
    
    Args:
        original_name (str): Name of original image
        original_img_dir (str): Directory containing original images
        patches (list): List of patch paths
        patch_positions (dict): Mapping of patch paths to positions
        patch_size (int): Size of each patch
        
    Returns:
        tuple: (width, height, patches_per_row)
    """
    # Try to find original image for exact dimensions
    for ext in ['.jpg', '.png', '.tif', '.tiff']:
        orig_path = os.path.join(original_img_dir, f"{original_name}{ext}")
        if os.path.exists(orig_path):
            with Image.open(orig_path) as img:
                orig_width, orig_height = img.size
            
            patches_per_row = orig_width // patch_size
            if patches_per_row == 0:
                patches_per_row = 1
            
            max_x = ((orig_width // patch_size) + (1 if orig_width % patch_size else 0)) * patch_size
            max_y = ((orig_height // patch_size) + (1 if orig_height % patch_size else 0)) * patch_size
            
            return max_x, max_y, patches_per_row
    
    # Estimate from patch positions if original not found
    logging.warning(f"Could not find original image for {original_name}, estimating dimensions")
    patches_per_row = 10  # Default fallback
    max_patch_id = max([patch_positions[p] for p in patches])
    max_x = ((max_patch_id % patches_per_row) + 1) * patch_size
    max_y = ((max_patch_id // patches_per_row) + 1) * patch_size
    
    return max_x, max_y, patches_per_row


def predict_patch_with_tta(patch_tensor, model, return_probs=False):
    """
    Predict patch with test-time augmentation.
    
    Applies augmentations (original, horizontal flip, vertical flip, rotation 90°),
    gets predictions for each, reverses augmentations, and averages probabilities.
    
    Args:
        patch_tensor: Input patch tensor [1, 3, H, W]
        model: Neural network model
        return_probs: If True, return averaged probabilities instead of argmax prediction
        
    Returns:
        numpy.ndarray: Averaged prediction map [H, W] or probabilities [C, H, W]
    """
    import torchvision.transforms.functional as TF
    
    device = patch_tensor.device
    augmented_outputs = []
    
    # 1. Original (no augmentation)
    with torch.no_grad():
        output = model(patch_tensor)
        if isinstance(output, tuple):
            output = output[0]  # Use main output for inference
        # Apply softmax to get probabilities
        probs = torch.softmax(output, dim=1)
        augmented_outputs.append(probs)
    
    # 2. Horizontal flip
    patch_hflip = TF.hflip(patch_tensor.squeeze(0)).unsqueeze(0)
    with torch.no_grad():
        output = model(patch_hflip.to(device))
        if isinstance(output, tuple):
            output = output[0]
        probs = torch.softmax(output, dim=1)
        # Reverse horizontal flip on probabilities
        probs_reversed = TF.hflip(probs.squeeze(0)).unsqueeze(0)
        augmented_outputs.append(probs_reversed)
    
    # 3. Vertical flip
    patch_vflip = TF.vflip(patch_tensor.squeeze(0)).unsqueeze(0)
    with torch.no_grad():
        output = model(patch_vflip.to(device))
        if isinstance(output, tuple):
            output = output[0]
        probs = torch.softmax(output, dim=1)
        # Reverse vertical flip on probabilities
        probs_reversed = TF.vflip(probs.squeeze(0)).unsqueeze(0)
        augmented_outputs.append(probs_reversed)
    
    # 4. Rotation 90°
    patch_rot90 = TF.rotate(patch_tensor.squeeze(0), angle=-90).unsqueeze(0)
    with torch.no_grad():
        output = model(patch_rot90.to(device))
        if isinstance(output, tuple):
            output = output[0]
        probs = torch.softmax(output, dim=1)
        # Reverse rotation 90° (rotate back by +90°)
        probs_reversed = TF.rotate(probs.squeeze(0), angle=90).unsqueeze(0)
        augmented_outputs.append(probs_reversed)
    
    # Average all probabilities (all tensors are on the same device)
    averaged_probs = torch.stack(augmented_outputs).mean(dim=0)
    
    if return_probs:
        # Return probabilities [C, H, W]
        return averaged_probs.squeeze(0).cpu().numpy()
    else:
        # Take argmax to get final prediction
        pred_patch = torch.argmax(averaged_probs, dim=1).squeeze(0).cpu().numpy()
        return pred_patch


def apply_crf_postprocessing(prob_map, rgb_image, num_classes=6, 
                              spatial_weight=3.0, spatial_x_stddev=3.0, spatial_y_stddev=3.0,
                              color_weight=10.0, color_stddev=50.0,
                              num_iterations=10):
    """
    Apply DenseCRF post-processing to refine segmentation predictions.
    
    Args:
        prob_map: Probability map [H, W, C] with class probabilities
        rgb_image: Original RGB image [H, W, 3] for pairwise potentials
        num_classes: Number of segmentation classes
        spatial_weight: Weight for spatial pairwise potentials
        spatial_x_stddev: Standard deviation for spatial x dimension
        spatial_y_stddev: Standard deviation for spatial y dimension
        color_weight: Weight for color pairwise potentials
        color_stddev: Standard deviation for color similarity
        num_iterations: Number of CRF iterations
        
    Returns:
        numpy.ndarray: Refined prediction map [H, W] with class indices
    """
    try:
        # pydensecrf2 package installs as 'pydensecrf' module
        try:
            import pydensecrf2.densecrf as dcrf
            from pydensecrf2.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian
        except ImportError:
            # Fallback: try importing as pydensecrf (actual module name)
            import pydensecrf.densecrf as dcrf
            from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian
    except ImportError:
        error_msg = (
            "pydensecrf2 is not installed. CRF post-processing requires this package.\n"
            "Installation options:\n"
            "  1. pip install pydensecrf2\n"
            "  2. conda install -c conda-forge pydensecrf2\n"
            "  3. python3 -m pip install pydensecrf2\n"
            "Note: The package installs as 'pydensecrf' module even though pip package is 'pydensecrf2'.\n"
            "Note: If using conda and encountering symbol errors, try: conda install libgcc"
        )
        logging.error(error_msg)
        raise ImportError("pydensecrf2 is required for CRF post-processing")
    
    H, W = prob_map.shape[:2]
    
    # Ensure probabilities are in correct format and range
    if prob_map.shape[2] != num_classes:
        raise ValueError(f"Probability map has {prob_map.shape[2]} classes but expected {num_classes}")
    
    # Ensure prob_map is C-contiguous (required by pydensecrf)
    prob_map = np.ascontiguousarray(prob_map, dtype=np.float32)
    
    # Resize RGB image if dimensions don't match
    if rgb_image.shape[:2] != (H, W):
        rgb_image_resized = np.array(Image.fromarray(rgb_image).resize((W, H), Image.BILINEAR))
    else:
        rgb_image_resized = rgb_image.copy()
    
    # Ensure RGB image is uint8 and C-contiguous (required by pydensecrf)
    if rgb_image_resized.dtype != np.uint8:
        rgb_image_resized = np.clip(rgb_image_resized, 0, 255).astype(np.uint8)
    rgb_image_resized = np.ascontiguousarray(rgb_image_resized, dtype=np.uint8)
    
    # Transpose probability map to [C, H, W] format for DenseCRF
    prob_map_transposed = prob_map.transpose(2, 0, 1)  # [C, H, W]
    # Ensure transposed array is C-contiguous (required by pydensecrf)
    prob_map_transposed = np.ascontiguousarray(prob_map_transposed, dtype=np.float32)
    
    # Create CRF model
    crf = dcrf.DenseCRF2D(W, H, num_classes)
    
    # Set unary potentials (negative log probabilities)
    unary = unary_from_softmax(prob_map_transposed)
    crf.setUnaryEnergy(unary)
    
    # Add pairwise potentials
    
    # 1. Spatial pairwise potential (encourages nearby pixels to have same label)
    pairwise_gaussian = create_pairwise_gaussian(sdims=(spatial_y_stddev, spatial_x_stddev), shape=(H, W))
    crf.addPairwiseEnergy(pairwise_gaussian, compat=spatial_weight)
    
    # 2. Bilateral pairwise potential (encourages similar colored pixels to have same label)
    pairwise_bilateral = create_pairwise_bilateral(
        sdims=(spatial_y_stddev, spatial_x_stddev),
        schan=(color_stddev, color_stddev, color_stddev),
        img=rgb_image_resized,
        chdim=2
    )
    crf.addPairwiseEnergy(pairwise_bilateral, compat=color_weight)
    
    # Run inference
    Q = crf.inference(num_iterations)
    
    # Get refined probabilities and convert to prediction
    refined_probs = np.array(Q).reshape((num_classes, H, W)).transpose(1, 2, 0)
    refined_pred = np.argmax(refined_probs, axis=2).astype(np.uint8)
    
    return refined_pred


def stitch_patches(patches, patch_positions, max_x, max_y, patches_per_row, patch_size, model, use_tta=False, return_probs=False):
    """
    Stitch together patch predictions into full image.
    
    Args:
        patches (list): List of patch file paths
        patch_positions (dict): Mapping of patch paths to positions
        max_x, max_y (int): Maximum image dimensions
        patches_per_row (int): Number of patches per row
        patch_size (int): Size of each patch
        model: Neural network model
        use_tta (bool): Whether to use test-time augmentation
        return_probs (bool): If True, also return softmax probabilities (needed for CRF)
        
    Returns:
        numpy.ndarray: Stitched prediction map
        Optional[numpy.ndarray]: Stitched probability map [H, W, C] if return_probs=True
    """
    import torchvision.transforms.functional as TF
    
    pred_full = np.zeros((max_y, max_x), dtype=np.int32)
    count_map = np.zeros((max_y, max_x), dtype=np.int32)
    
    if return_probs:
        # Initialize probability map (will be accumulated)
        num_classes = None
        prob_full = None
    
    for patch_path in patches:
        patch_id = patch_positions[patch_path]
        
        # Calculate patch position
        x = (patch_id % patches_per_row) * patch_size
        y = (patch_id // patches_per_row) * patch_size
        
        # Load and process patch
        patch = Image.open(patch_path).convert("RGB")
        patch_tensor = TF.to_tensor(patch).unsqueeze(0).cuda()
        
        # Use TTA if enabled, otherwise use standard prediction
        if use_tta:
            if return_probs:
                # Get averaged probabilities from TTA
                probs = predict_patch_with_tta(patch_tensor, model, return_probs=True)  # [C, H, W]
                if prob_full is None:
                    num_classes = probs.shape[0]
                    prob_full = np.zeros((max_y, max_x, num_classes), dtype=np.float32)
                pred_patch = np.argmax(probs, axis=0)  # [H, W]
            else:
                pred_patch = predict_patch_with_tta(patch_tensor, model, return_probs=False)
        else:
            with torch.no_grad():
                output = model(patch_tensor)
                
                # Handle deep supervision output (tuple) vs regular output (tensor)
                if isinstance(output, tuple):
                    # Deep supervision: output is (main_logits, aux_outputs)
                    output = output[0]  # Use main output for inference
                
                if return_probs:
                    probs = torch.softmax(output, dim=1).cpu().numpy()[0]  # [C, H, W]
                    if prob_full is None:
                        num_classes = probs.shape[0]
                        prob_full = np.zeros((max_y, max_x, num_classes), dtype=np.float32)
                
                pred_patch = torch.argmax(output, dim=1).cpu().numpy()[0]
        
        # Add to prediction map with boundary checking
        if y + patch_size <= pred_full.shape[0] and x + patch_size <= pred_full.shape[1]:
            pred_full[y:y+patch_size, x:x+patch_size] += pred_patch
            count_map[y:y+patch_size, x:x+patch_size] += 1
            if return_probs:
                # Convert probabilities from [C, H, W] to [H, W, C] format
                # probs is always [C, H, W] format from model output
                prob_full[y:y+patch_size, x:x+patch_size, :] += probs.transpose(1, 2, 0)
        else:
            # Handle edge cases
            valid_h = min(patch_size, pred_full.shape[0] - y)
            valid_w = min(patch_size, pred_full.shape[1] - x)
            if valid_h > 0 and valid_w > 0:
                pred_full[y:y+valid_h, x:x+valid_w] += pred_patch[:valid_h, :valid_w]
                count_map[y:y+valid_h, x:x+valid_w] += 1
                if return_probs:
                    # probs is [C, H, W], slice and transpose to [H, W, C]
                    prob_full[y:y+valid_h, x:x+valid_w, :] += probs[:, :valid_h, :valid_w].transpose(1, 2, 0)
    
    # Normalize by count map
    pred_full = np.round(pred_full / np.maximum(count_map, 1)).astype(np.uint8)
    
    if return_probs:
        # Normalize probabilities by count map
        prob_full = prob_full / np.maximum(count_map[:, :, np.newaxis], 1)
        return pred_full, prob_full
    else:
        return pred_full


def save_prediction_results(pred_full, original_name, class_colors, result_dir):
    """Save prediction results as RGB image."""
    if result_dir is not None:
        os.makedirs(result_dir, exist_ok=True)
        
        # Convert class indices to RGB
        rgb_mask = np.zeros((pred_full.shape[0], pred_full.shape[1], 3), dtype=np.uint8)
        for idx, color in enumerate(class_colors):
            rgb_mask[pred_full == idx] = color
        
        pred_png_path = os.path.join(result_dir, f"{original_name}.png")
        Image.fromarray(rgb_mask).save(pred_png_path)


def save_comparison_visualization(pred_full, gt_class, original_name, original_img_dir, 
                                test_save_path, class_colors, class_names):
    """Save side-by-side comparison visualization."""
    compare_dir = os.path.join(test_save_path, 'compare')
    os.makedirs(compare_dir, exist_ok=True)
    
    # Create colormap
    cmap = ListedColormap(class_colors)
    n_classes = len(class_colors)
    
    # Resize ground truth if dimensions don't match
    if gt_class.shape != pred_full.shape:
        logging.warning(f"Resizing ground truth for {original_name}")
        gt_class_resized = np.zeros_like(pred_full)
        min_h = min(gt_class.shape[0], pred_full.shape[0])
        min_w = min(gt_class.shape[1], pred_full.shape[1])
        gt_class_resized[:min_h, :min_w] = gt_class[:min_h, :min_w]
        gt_class = gt_class_resized
    
    # Create visualization
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Find and load original image
    orig_img_path = None
    for ext in ['.jpg', '.png', '.tif', '.tiff']:
        test_path = os.path.join(original_img_dir, f"{original_name}{ext}")
        if os.path.exists(test_path):
            orig_img_path = test_path
            break
    
    if orig_img_path:
        orig_img = Image.open(orig_img_path).convert("RGB")
        if orig_img.size != (pred_full.shape[1], pred_full.shape[0]):
            orig_img = orig_img.resize((pred_full.shape[1], pred_full.shape[0]), Image.BILINEAR)
        axs[0].imshow(np.array(orig_img))
    else:
        axs[0].imshow(np.zeros((pred_full.shape[0], pred_full.shape[1], 3), dtype=np.uint8))
    
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    
    axs[1].imshow(pred_full, cmap=cmap, vmin=0, vmax=(n_classes - 1))
    axs[1].set_title('Prediction')
    axs[1].axis('off')
    
    axs[2].imshow(gt_class, cmap=cmap, vmin=0, vmax=(n_classes - 1))
    axs[2].set_title('Ground Truth')
    axs[2].axis('off')
    
    plt.tight_layout()
    save_img_path = os.path.join(compare_dir, f"{original_name}_compare.png")
    plt.savefig(save_img_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


def compute_segmentation_metrics(pred_full, gt_class, n_classes, TP, FP, FN):
    """
    Compute segmentation metrics for each class.
    
    Args:
        pred_full: Prediction array
        gt_class: Ground truth array
        n_classes: Number of classes
        TP, FP, FN: Arrays to accumulate metrics
    """
    for cls in range(n_classes):
        pred_c = (pred_full == cls)
        gt_c = (gt_class == cls)
        TP[cls] += np.logical_and(pred_c, gt_c).sum()
        FP[cls] += np.logical_and(pred_c, np.logical_not(gt_c)).sum()
        FN[cls] += np.logical_and(np.logical_not(pred_c), gt_c).sum()


def print_final_metrics(TP, FP, FN, class_names, num_processed_images):
    """Print final computed metrics."""
    n_classes = len(class_names)
    
    if num_processed_images > 0:
        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        iou_per_class = TP / (TP + FP + FN + 1e-8)
    else:
        precision = np.zeros(n_classes)
        recall = np.zeros(n_classes)
        f1 = np.zeros(n_classes)
        iou_per_class = np.zeros(n_classes)
    
    logging.info("\nPer-class metrics:")
    logging.info("-" * 80)
    for cls in range(n_classes):
        logging.info(f"{class_names[cls]:<15}: Precision={precision[cls]:.4f}, "
                    f"Recall={recall[cls]:.4f}, F1={f1[cls]:.4f}, IoU={iou_per_class[cls]:.4f}")
    
    logging.info("\nMean metrics:")
    logging.info("-" * 40)
    logging.info(f"Mean Precision: {np.mean(precision):.4f}")
    logging.info(f"Mean Recall: {np.mean(recall):.4f}")
    logging.info(f"Mean F1: {np.mean(f1):.4f}")
    logging.info(f"Mean IoU: {np.mean(iou_per_class):.4f}")


def inference(args, model, test_save_path=None):
    """
    Run inference on historical document dataset.
    
    Args:
        args: Command line arguments
        model: Trained neural network model
        test_save_path: Path to save test results
        
    Returns:
        str: Status message
    """
    logging.info(f"Starting inference on {args.dataset} dataset")
    
    # Get dataset-specific information
    class_colors, class_names, rgb_to_class_func = get_dataset_info(args.dataset, args.manuscript)
    n_classes = len(class_colors)
    # Use the actual patch size that the model was trained on (224x224)
    # args.img_size is the full image size, not the patch size
    patch_size = 224
    
    # Get dataset paths
    patch_dir, mask_dir, original_img_dir, original_mask_dir = get_dataset_paths(args)
    
    # Check if directories exist
    if not os.path.exists(patch_dir):
        logging.error(f"Patch directory not found: {patch_dir}")
        return "Testing Failed!"
    
    # Find patch files
    patch_files = sorted(glob.glob(os.path.join(patch_dir, '*.png')))
    if len(patch_files) == 0:
        logging.info(f"No patch files found in {patch_dir}")
        return "Testing Finished!"
    
    logging.info(f"Found {len(patch_files)} patches for {args.manuscript}")
    
    # Initialize metrics
    TP = np.zeros(n_classes, dtype=np.float64)
    FP = np.zeros(n_classes, dtype=np.float64)  
    FN = np.zeros(n_classes, dtype=np.float64)
    
    # Set up result directory
    result_dir = os.path.join(test_save_path, "result") if test_save_path else None
    
    # Group patches by original image
    patch_groups, patch_positions = process_patch_groups(patch_files)
    
    # Process each original image
    num_processed_images = 0
    
    for original_name, patches in patch_groups.items():
        logging.info(f"Processing: {original_name} ({len(patches)} patches)")
        
        # Estimate image dimensions
        max_x, max_y, patches_per_row = estimate_image_dimensions(
            original_name, original_img_dir, patches, patch_positions, patch_size
        )
        
        # Stitch patches together
        use_tta = getattr(args, 'use_tta', False)
        use_crf = getattr(args, 'use_crf', False)
        
        if use_crf:
            # Get both predictions and probabilities for CRF
            pred_full, prob_full = stitch_patches(
                patches, patch_positions, max_x, max_y, 
                patches_per_row, patch_size, model, use_tta=use_tta, return_probs=True
            )
            
            # Load original RGB image for CRF pairwise potentials
            orig_img_rgb = None
            for ext in ['.jpg', '.png', '.tif', '.tiff']:
                orig_path = os.path.join(original_img_dir, f"{original_name}{ext}")
                if os.path.exists(orig_path):
                    orig_img_pil = Image.open(orig_path).convert("RGB")
                    # Resize to match prediction dimensions
                    if orig_img_pil.size != (max_x, max_y):
                        orig_img_pil = orig_img_pil.resize((max_x, max_y), Image.BILINEAR)
                    orig_img_rgb = np.array(orig_img_pil)
                    break
            
            if orig_img_rgb is not None:
                logging.info(f"Applying CRF post-processing to {original_name}")
                try:
                    # Apply CRF refinement
                    pred_full = apply_crf_postprocessing(
                        prob_full, orig_img_rgb, 
                        num_classes=n_classes,
                        spatial_weight=3.0,
                        spatial_x_stddev=3.0,
                        spatial_y_stddev=3.0,
                        color_weight=10.0,
                        color_stddev=50.0,
                        num_iterations=10
                    )
                    logging.info(f"CRF post-processing completed for {original_name}")
                except Exception as e:
                    logging.warning(f"CRF post-processing failed for {original_name}: {e}")
                    logging.warning("Falling back to non-CRF predictions")
                    # Fall back to argmax if CRF fails
                    pred_full = np.argmax(prob_full, axis=2).astype(np.uint8)
            else:
                logging.warning(f"Original image not found for CRF: {original_name}, using non-CRF predictions")
                pred_full = np.argmax(prob_full, axis=2).astype(np.uint8)
        else:
            # Standard inference without CRF
            pred_full = stitch_patches(
                patches, patch_positions, max_x, max_y, 
                patches_per_row, patch_size, model, use_tta=use_tta, return_probs=False
            )
        
        # Save prediction results
        save_prediction_results(pred_full, original_name, class_colors, result_dir)
        
        # Load ground truth for evaluation
        gt_found = False
        for ext in ['.png', '.jpg', '.tif', '.tiff']:
            gt_path = os.path.join(original_mask_dir, f"{original_name}{ext}")
            if os.path.exists(gt_path):
                gt_pil = Image.open(gt_path).convert("RGB")
                gt_np = np.array(gt_pil)
                
                if rgb_to_class_func:
                    gt_class = rgb_to_class_func(gt_np)
                    gt_found = True
                    break
        
        if not gt_found:
            logging.warning(f"No ground truth found for {original_name}")
            gt_class = np.zeros_like(pred_full)
        
        # Save comparison visualization
        if test_save_path and gt_found:
            save_comparison_visualization(
                pred_full, gt_class, original_name, original_img_dir,
                test_save_path, class_colors, class_names
            )
        
        # Compute metrics
        if gt_found:
            # Ensure ground truth has same dimensions as prediction
            if gt_class.shape != pred_full.shape:
                logging.warning(f"Resizing ground truth for metrics computation: {gt_class.shape} -> {pred_full.shape}")
                gt_class_resized = np.zeros_like(pred_full)
                min_h = min(gt_class.shape[0], pred_full.shape[0])
                min_w = min(gt_class.shape[1], pred_full.shape[1])
                gt_class_resized[:min_h, :min_w] = gt_class[:min_h, :min_w]
                gt_class = gt_class_resized
            
            compute_segmentation_metrics(pred_full, gt_class, n_classes, TP, FP, FN)
            num_processed_images += 1
        
        logging.info(f"Completed: {original_name}")
    
    # Print final metrics
    print_final_metrics(TP, FP, FN, class_names, num_processed_images)
    logging.info(f"Inference completed on {num_processed_images} images")
    
    return "Testing Finished!"


def main():
    """Main testing function."""
    print("=== Historical Document Segmentation Testing ===")
    print()
    
    # Parse and validate arguments
    args = parse_arguments()
    
    try:
        validate_arguments(args)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    # Set up reproducible testing
    setup_reproducible_testing(args)
    
    # Create model (no config needed for CNN-Transformer)
    model = get_model(args, None)
    
    # Load trained model checkpoint
    try:
        checkpoint_name = load_model_checkpoint(model, args)
        print(f"Loaded checkpoint: {checkpoint_name}")
    except Exception as e:
        print(f"ERROR: Failed to load model checkpoint: {e}")
        sys.exit(1)
    
    # Set up logging
    log_folder = './test_log/test_log_'
    setup_logging(log_folder, checkpoint_name)
    
    logging.info(str(args))
    logging.info(f"Testing with checkpoint: {checkpoint_name}")
    
    # Set up test save directory
    if args.is_savenii:
        test_save_path = os.path.join(args.output_dir, "predictions")
        os.makedirs(test_save_path, exist_ok=True)
        logging.info(f"Saving predictions to: {test_save_path}")
    else:
        test_save_path = None
        logging.info("Not saving prediction files")
    
    # Run inference
    print()
    print("=== Starting Testing ===")
    print(f"Dataset: {args.dataset}")
    print(f"Model: CNN-Transformer")
    print(f"Manuscript: {args.manuscript}")
    print(f"Save predictions: {args.is_savenii}")
    print(f"Test-time augmentation: {'Enabled' if args.use_tta else 'Disabled'}")
    if args.use_tta:
        print("  - Augmentations: Original, Horizontal flip, Vertical flip, Rotation 90°")
    print(f"CRF post-processing: {'Enabled' if args.use_crf else 'Disabled'}")
    if args.use_crf:
        print("  - DenseCRF with spatial and color pairwise potentials")
    print()
    
    try:
        result = inference(args, model, test_save_path)
        print()
        print("=== TESTING COMPLETED SUCCESSFULLY ===")
        print(f"Results saved to: {test_save_path if test_save_path else 'No files saved'}")
        print("="*50)
        return result
        
    except Exception as e:
        print(f"ERROR: Testing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()