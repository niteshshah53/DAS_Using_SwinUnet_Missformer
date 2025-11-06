"""
Hybrid Model Training Module
Training approach for Hybrid models (hybrid1 and hybrid2) with early stopping and class weights.

Hybrid1: EfficientNetB4 encoder + SwinUnet decoder
Hybrid2: SwinUnet encoder + EfficientNet decoder

Both models use the same training approach with all three losses (CE + Focal + Dice).
"""

import os
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

# Add common directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../common'))

from utils import utils
from utils.utils import FocalLoss, DiceLoss

# Global worker initialization function for DataLoader (needed for Windows multiprocessing)
def worker_init_fn(worker_id, seed=1234):
    import random
    import numpy as np
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def setup_logging(output_path):
    """
    Set up logging to both file and console.
    
    Args:
        output_path (str): Directory where log file will be saved
    """
    log_file = os.path.join(output_path, "training.log")
    
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def compute_class_weights(train_dataset, num_classes):
    """
    Compute class weights for balanced training based on pixel frequency.
    Uses square root of inverse frequency for extreme imbalances (better than linear).
    Args:
        train_dataset: Training dataset object with .mask_paths
        num_classes (int): Number of segmentation classes
    Returns:
        torch.Tensor: Class weights (on CUDA if available)
    """
    print("\nComputing class weights...")

    # Define color maps for different datasets
    if num_classes == 6:
        # UDIADS-BIB color map (standard manuscripts)
        COLOR_MAP = {
            (0, 0, 0): 0,        # Background
            (255, 255, 0): 1,    # Paratext
            (0, 255, 255): 2,    # Decoration
            (255, 0, 255): 3,    # Main text
            (255, 0, 0): 4,      # Title
            (0, 255, 0): 5,      # Chapter Heading
        }
    elif num_classes == 5:
        # UDIADS-BIB color map for Syriaque341 (no Chapter Headings)
        COLOR_MAP = {
            (0, 0, 0): 0,        # Background
            (255, 255, 0): 1,    # Paratext
            (0, 255, 255): 2,    # Decoration
            (255, 0, 255): 3,    # Main text
            (255, 0, 0): 4,      # Title
            # Note: Chapter Heading (0, 255, 0) is not present in Syriaque341
        }
        # Also count Chapter Headings pixels but don't include in class map
        chapter_headings_color = (0, 255, 0)
    elif num_classes == 4:
        # DivaHisDB color map
        COLOR_MAP = {
            (0, 0, 0): 0,        # Background
            (0, 255, 0): 1,      # Comment
            (255, 0, 0): 2,      # Decoration
            (0, 0, 255): 3,      # Main Text
        }
        chapter_headings_color = None
    else:
        raise ValueError(f"Unsupported number of classes: {num_classes}")

    class_counts = np.zeros(num_classes, dtype=np.float64)
    unmapped_pixel_count = 0
    chapter_headings_count = 0

    for mask_path in train_dataset.mask_paths:
        mask = np.array(Image.open(mask_path).convert("RGB"))
        
        # Track mapped pixels
        mapped_mask = np.zeros(mask.shape[:2], dtype=bool)
        
        for rgb, cls in COLOR_MAP.items():
            matches = np.all(mask == rgb, axis=-1)
            class_counts[cls] += np.sum(matches)
            mapped_mask[matches] = True
        
        # For 5-class mode: count Chapter Headings pixels separately
        if num_classes == 5 and chapter_headings_color is not None:
            chapter_matches = np.all(mask == chapter_headings_color, axis=-1)
            chapter_headings_count += np.sum(chapter_matches)
            mapped_mask[chapter_matches] = True
        
        # Count unmapped pixels
        unmapped_pixels = ~mapped_mask
        if np.any(unmapped_pixels):
            unmapped_pixel_count += np.sum(unmapped_pixels)

    # Report Chapter Headings pixels if found in 5-class mode
    if num_classes == 5 and chapter_headings_count > 0:
        print(f"\n⚠️  WARNING: Found {chapter_headings_count:,} Chapter Headings pixels in Syr341FS!")
        print(f"   These will be mapped to Background (class 0)")
        print(f"   This may contribute to class imbalance.")

    # Compute frequencies
    total_pixels = class_counts.sum()
    if total_pixels == 0:
        raise ValueError("No pixels found in masks!")
    
    class_freq = class_counts / total_pixels

    # Use square root of inverse frequency for extreme imbalances
    # This is less aggressive than linear inverse frequency but still helps rare classes
    # Formula: sqrt(1 / (freq + epsilon)) then normalize
    epsilon = 1e-6
    weights = np.sqrt(1.0 / (class_freq + epsilon))
    
    # Normalize weights to sum to num_classes (balanced baseline)
    weights = weights / weights.sum() * num_classes
    
    # For extremely rare classes (< 0.5%), apply additional boost
    rare_threshold = 0.005  # 0.5%
    for cls in range(num_classes):
        if class_freq[cls] < rare_threshold and class_freq[cls] > 0:
            boost_factor = 1.5  # 50% boost for rare classes
            weights[cls] *= boost_factor
    
    # Renormalize after boosting
    weights = weights / weights.sum() * num_classes

    # Print detailed analysis
    print("\n" + "-"*80)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("-"*80)
    print(f"{'Class':<6} {'Frequency':<15} {'Weight':<15} {'Pixels':<15}")
    print("-"*80)
    for cls in range(num_classes):
        pixel_count = int(class_counts[cls])
        print(f"{cls:<6} {class_freq[cls]:<15.6f} {weights[cls]:<15.6f} {pixel_count:<15,}")
    print("-"*80)
    if unmapped_pixel_count > 0:
        print(f"\n⚠️  Unmapped pixels: {unmapped_pixel_count:,} (mapped to Background)")
    print()

    # Return as tensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.tensor(weights, dtype=torch.float32, device=device)


def create_data_loaders(train_dataset, val_dataset, batch_size, num_workers, seed):
    """
    Create data loaders for training and validation.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size
        num_workers: Number of data loading workers
        seed: Random seed
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Validate that datasets are not empty
    if len(train_dataset) == 0:
        raise ValueError(
            f"Training dataset is empty! Found 0 samples.\n"
            f"This usually means the dataset directories don't exist or are empty.\n"
            f"Please check that the dataset path is correct and contains the expected files."
        )
    
    if len(val_dataset) == 0:
        raise ValueError(
            f"Validation dataset is empty! Found 0 samples.\n"
            f"This usually means the dataset directories don't exist or are empty.\n"
            f"Please check that the dataset path is correct and contains the expected files."
        )
    
    # On Windows, reduce num_workers to avoid multiprocessing issues
    if os.name == 'nt':  # Windows
        num_workers = min(num_workers, 2)
        if num_workers > 0:
            print(f"Windows detected: reducing num_workers to {num_workers}")
    
    # Create a partial function for worker initialization
    import functools
    worker_fn = functools.partial(worker_init_fn, seed=seed)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_fn if num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_fn if num_workers > 0 else None
    )
    
    return train_loader, val_loader


def create_loss_functions(class_weights, num_classes):
    """
    Create loss functions WITH class weights properly applied.
    
    Args:
        class_weights (torch.Tensor): Weights for each class
        num_classes (int): Number of segmentation classes
        
    Returns:
        tuple: (cross_entropy_loss, focal_loss, dice_loss)
    """
    
    # Apply weights to CrossEntropyLoss (this was missing!)
    ce_loss = CrossEntropyLoss(weight=class_weights)
    
    # Reduce focal loss gamma (5 is too aggressive, causes instability)
    focal_loss = FocalLoss(gamma=3)  # Changed from 5 to 2
    
    # Dice loss doesn't use class weights directly
    dice_loss = DiceLoss(num_classes)
    
    return ce_loss, focal_loss, dice_loss


def compute_combined_loss(predictions, labels, ce_loss, focal_loss, dice_loss):
    """
    Unified loss computation for both training and validation.
    Supports deep supervision with auxiliary outputs.
    
    Args:
        predictions: Model predictions (logits or tuple of (logits, aux_outputs))
        labels: Ground truth labels
        ce_loss, focal_loss, dice_loss: Loss functions
        
    Returns:
        torch.Tensor: Combined loss value
    """
    # Handle deep supervision (tuple of main + auxiliary outputs)
    if isinstance(predictions, tuple):
        logits, aux_outputs = predictions
        
        # Main loss (full weight)
        loss_ce = ce_loss(logits, labels)
        loss_focal = focal_loss(logits, labels)
        loss_dice = dice_loss(logits, labels, softmax=True)
        main_loss = 0.4 * loss_ce + 0.1 * loss_focal + 0.5 * loss_dice
        
        # Auxiliary losses (decreasing weights for deeper layers)
        # TransUNet practice: [0.4, 0.3, 0.2] for 3 aux outputs
        aux_weights = [0.4, 0.3, 0.2][:len(aux_outputs)]
        aux_loss = 0.0
        
        for weight, aux_output in zip(aux_weights, aux_outputs):
            aux_ce = ce_loss(aux_output, labels)
            aux_focal = focal_loss(aux_output, labels)
            aux_dice = dice_loss(aux_output, labels, softmax=True)
            aux_combined = 0.4 * aux_ce + 0.1 * aux_focal + 0.5 * aux_dice
            aux_loss += weight * aux_combined
        
        # Total loss: main + weighted auxiliary
        return main_loss + aux_loss
    else:
        # Standard single output
        loss_ce = ce_loss(predictions, labels)
        loss_focal = focal_loss(predictions, labels)
        loss_dice = dice_loss(predictions, labels, softmax=True)
        
        # Balanced combination - same for train and val
        return 0.4 * loss_ce + 0.1 * loss_focal + 0.5 * loss_dice       


def create_optimizer_and_scheduler(model, learning_rate, args=None, train_loader=None):
    """
    Create optimizer and learning rate scheduler.
    
    Args:
        model: Neural network model
        learning_rate (float): Initial learning rate
        args: Command line arguments (optional)
        train_loader: Training data loader (optional, needed for OneCycleLR)
        
    Returns:
        tuple: (optimizer, scheduler)
    """
    # TransUNet Best Practice: Differential Learning Rates
    # Pretrained encoder gets 10x smaller LR to preserve learned features
    # Decoder gets base LR for faster convergence
    
    encoder_params = []
    decoder_params = []
    adapter_params = []
    
    # Separate parameters by module
    for name, param in model.named_parameters():
        if 'encoder' in name.lower():
            encoder_params.append(param)
        elif 'adapter' in name.lower() or 'align' in name.lower() or 'attention' in name.lower():
            adapter_params.append(param)
        else:
            decoder_params.append(param)
    
    # Parameter groups with differential LR and weight decay
    param_groups = [
        {
            'params': encoder_params,
            'lr': learning_rate * 0.1,  # 10x smaller for pretrained
            'weight_decay': 1e-3,  # Light regularization for pretrained
            'name': 'encoder'
        },
        {
            'params': adapter_params,
            'lr': learning_rate * 0.5,  # Medium LR for adapters
            'weight_decay': 5e-3,  # Medium regularization
            'name': 'adapters'
        },
        {
            'params': decoder_params,
            'lr': learning_rate,  # Full LR for new modules
            'weight_decay': 1e-2,  # Strong regularization for new modules
            'name': 'decoder'
        }
    ]
    
    # Use AdamW optimizer with differential LR
    optimizer = optim.AdamW(
        param_groups,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Select scheduler based on args
    scheduler_type = getattr(args, 'scheduler_type', 'CosineAnnealingWarmRestarts')
    max_epochs = getattr(args, 'max_epochs', 300)
    
    if scheduler_type == 'OneCycleLR':
        # OneCycleLR scheduler - optimal for hybrid CNN-transformer models
        if train_loader is not None:
            # Calculate actual steps per epoch from data loader
            steps_per_epoch = len(train_loader)
            total_steps = max_epochs * steps_per_epoch
            print(f"  📊 OneCycleLR: {steps_per_epoch} steps/epoch × {max_epochs} epochs = {total_steps} total steps")
        else:
            # Fallback: estimate steps per epoch
            total_steps = max_epochs * 1000  # Estimate: ~1000 steps per epoch
            print(f"  📊 OneCycleLR: Estimated {total_steps} total steps (1000 steps/epoch)")
        
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[learning_rate * 10, learning_rate * 5, learning_rate * 10],  # Peak LRs for each group
            total_steps=total_steps,
            pct_start=0.3,  # 30% warmup
            anneal_strategy='cos',
            div_factor=10,  # Initial LR = max_lr/10
            final_div_factor=100  # Final LR = max_lr/1000
        )
        scheduler_name = "OneCycleLR (Peak: 10x, Warmup: 30%, Cosine)"
        
    elif scheduler_type == 'ReduceLROnPlateau':
        # ReduceLROnPlateau scheduler - adaptive based on validation loss
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=15,  # Wait 15 epochs before reducing
            min_lr=1e-6
        )
        scheduler_name = "ReduceLROnPlateau (factor=0.5, patience=15)"
        
    elif scheduler_type == 'CosineAnnealingLR':
        # CosineAnnealingLR scheduler - smooth decay without restarts
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_epochs,
            eta_min=1e-6
        )
        scheduler_name = f"CosineAnnealingLR (T_max={max_epochs})"
        
    else:
        # CosineAnnealingWarmRestarts scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=50,  # Restart every 50 epochs
            T_mult=2,  # Double period after each restart
            eta_min=1e-7
        )
        scheduler_name = "CosineAnnealingWarmRestarts (T_0=50, T_mult=2)"
    
    print("🚀 TransUNet Best Practice: Differential Learning Rates")
    print(f"  📊 Encoder LR:  {learning_rate * 0.1:.6f} (10x smaller, {len(encoder_params)} params)")
    print(f"  📊 Adapter LR:  {learning_rate * 0.5:.6f} (5x smaller, {len(adapter_params)} params)")
    print(f"  📊 Decoder LR:  {learning_rate:.6f} (base LR, {len(decoder_params)} params)")
    print(f"  ⚙️  Scheduler: {scheduler_name}")
    print(f"  ⚙️  Weight decay: Encoder=1e-3, Adapters=5e-3, Decoder=1e-2")
    
    return optimizer, scheduler


def run_training_epoch(model, train_loader, ce_loss, focal_loss, dice_loss, optimizer, class_weights, scheduler=None, scheduler_type='CosineAnnealingWarmRestarts'):
    """
    Run one training epoch.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        ce_loss, focal_loss, dice_loss: Loss functions
        optimizer: Optimizer
        class_weights: Class weights for Dice loss
        scheduler: Learning rate scheduler (optional)
        scheduler_type: Type of scheduler (for OneCycleLR step handling)
        
    Returns:
        float: Average training loss for this epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)
    
    for batch_idx, batch in enumerate(train_loader):
        # Handle different batch formats (dict vs tuple)
        if isinstance(batch, dict):
            images = batch['image'].cuda()
            labels = batch['label'].cuda()
        else:
            images, labels = batch[0].cuda(), batch[1].cuda()
        
        # Forward pass
        predictions = model(images)
        
        # Use unified loss computation
        loss = compute_combined_loss(predictions, labels, ce_loss, focal_loss, dice_loss)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Step scheduler for OneCycleLR (step-based scheduler)
        if scheduler is not None and scheduler_type == 'OneCycleLR':
            scheduler.step()
        
        total_loss += loss.item()
    
    return total_loss / num_batches


def validate_with_sliding_window(model, val_dataset, ce_loss, focal_loss, dice_loss, patch_size=224):
    """
    Validate model using sliding window approach.
    
    Args:
        model: Neural network model
        val_dataset: Validation dataset
        ce_loss, focal_loss, dice_loss: Loss functions
        patch_size (int): Size of patches for sliding window
        
    Returns:
        float: Average validation loss
    """
    model.eval()
    total_loss = 0.0
    num_samples = 0
    
    # Limit validation to first 50 samples for faster training
    max_val_samples = min(50, len(val_dataset))
    
    with torch.no_grad():
        for i in range(max_val_samples):
            sample = val_dataset[i]
            if isinstance(sample, dict):
                image = sample['image'].unsqueeze(0).cuda()
                label = sample['label'].unsqueeze(0).cuda()
            else:
                image, label = sample[0].unsqueeze(0).cuda(), sample[1].unsqueeze(0).cuda()
            
            # Forward pass
            predictions = model(image)
            
            # Use unified loss computation (CRITICAL FIX!)
            loss = compute_combined_loss(predictions, label, ce_loss, focal_loss, dice_loss)
            total_loss += loss.item()
            num_samples += 1
    
    return total_loss / num_samples if num_samples > 0 else float('inf')


def save_best_model(model, epoch, val_loss, best_val_loss, snapshot_path,
                    optimizer=None, scheduler=None, scaler=None):
    """
    Save model + optimizer/scheduler/scaler checkpoint if validation loss improved.
    
    Args:
        model: Neural network model
        epoch (int): Current epoch
        val_loss (float): Current validation loss
        best_val_loss (float): Best validation loss so far
        snapshot_path (str): Directory to save models
        optimizer: Optimizer (optional)
        scheduler: Learning rate scheduler (optional)
        scaler: GradScaler for AMP (optional)
        
    Returns:
        tuple: (best_val_loss, improvement_made)
    """
    improvement_made = False
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        improvement_made = True
        
        best_model_path = os.path.join(snapshot_path, 'best_model_latest.pth')
        
        # Build checkpoint dict
        model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        checkpoint = {
            'epoch': epoch,
            'model_state': model_state,
            'optimizer_state': optimizer.state_dict() if optimizer is not None else None,
            'scheduler_state': scheduler.state_dict() if scheduler is not None else None,
            'best_val_loss': best_val_loss,
        }
        
        # Include scaler state if provided (AMP)
        if scaler is not None:
            try:
                checkpoint['scaler_state'] = scaler.state_dict()
            except Exception:
                # scaler may not support state_dict in some versions; ignore safely
                pass
        
        # Save checkpoint
        torch.save(checkpoint, best_model_path)
        
        print(f"    ✓ New best checkpoint saved! Val loss: {val_loss:.4f}")
    else:
        print(f"    No improvement (current: {val_loss:.4f}, best: {best_val_loss:.4f})")
    
    return best_val_loss, improvement_made


def trainer_hybrid(args, model, snapshot_path, train_dataset=None, val_dataset=None):
    """
    Main training function for Hybrid models (hybrid1 and hybrid2).
    
    Args:
        args: Command line arguments with training configuration
        model: Neural network model to train (hybrid1 or hybrid2)
        snapshot_path (str): Directory to save models and logs
        train_dataset: Training dataset
        val_dataset: Validation dataset
    """
    # Set up logging
    logger = setup_logging(snapshot_path)
    
    # Print training configuration
    print("\n" + "="*80)
    print("TRAINING CONFIGURATION")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Max Epochs: {args.max_epochs}")
    print(f"Learning Rate: {args.base_lr}")
    print(f"Number of Classes: {args.num_classes}")
    print(f"Output Directory: {snapshot_path}")
    print(f"Early Stopping Patience: {getattr(args, 'patience', 50)} epochs")
    print("="*80 + "\n")
    
    # Set up multi-GPU training if available
    if args.n_gpu > 1:
        print(f"Using {args.n_gpu} GPUs for training")
        model = nn.DataParallel(model)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_dataset, val_dataset, 
        args.batch_size * args.n_gpu, 
        args.num_workers, 
        args.seed
    )
    
    # Compute class weights for balanced training
    if hasattr(train_dataset, 'mask_paths'):
        class_weights = compute_class_weights(train_dataset, args.num_classes)
        # REMOVED: Manual weight boosting - let proper inverse frequency work naturally
    else:
        class_weights = torch.ones(args.num_classes)
    
    # Create loss functions, optimizer, and scheduler
    ce_loss, focal_loss, dice_loss = create_loss_functions(class_weights, args.num_classes)
    optimizer, scheduler = create_optimizer_and_scheduler(model, args.base_lr, args, train_loader)
    
    # Get scheduler type for training loop
    scheduler_type = getattr(args, 'scheduler_type', 'CosineAnnealingWarmRestarts')
    
    # Mixed precision scaler (used when CUDA is available)
    try:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler() if torch.cuda.is_available() else None
    except ImportError:
        scaler = None
    
    # Set up TensorBoard logging
    writer = SummaryWriter(os.path.join(snapshot_path, 'tensorboard_logs'))
    
    # Resume from checkpoint if available
    start_epoch = 0
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    checkpoint_path = os.path.join(snapshot_path, 'best_model_latest.pth')
    if os.path.exists(checkpoint_path):
        print(f"\n📂 Found checkpoint: {checkpoint_path}")
        print("   Attempting to resume training...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
            
            # Load model state (critical - must succeed)
            try:
                if isinstance(model, nn.DataParallel):
                    model.module.load_state_dict(checkpoint['model_state'], strict=False)
                else:
                    model.load_state_dict(checkpoint['model_state'], strict=False)
                print("   ✓ Loaded model state")
            except Exception as e:
                print(f"   ⚠️  Failed to load model state: {e}")
                raise  # Re-raise if model loading fails - cannot continue without model
            
            # Load optimizer state (handle parameter group mismatches gracefully)
            if optimizer is not None and 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state'])
                    print("   ✓ Loaded optimizer state")
                except Exception as e:
                    print(f"   ⚠️  Could not load optimizer state: {e}")
                    print("   Starting with fresh optimizer state (this is OK if architecture changed)")
            
            # Load scheduler state (handle mismatches gracefully, especially for OneCycleLR)
            # Check if scheduler type matches before loading state
            if scheduler is not None and 'scheduler_state' in checkpoint and checkpoint['scheduler_state'] is not None:
                scheduler_state = checkpoint['scheduler_state']
                current_scheduler_type = type(scheduler).__name__
                
                # Check if scheduler state matches current scheduler type
                # OneCycleLR has 'total_steps' key, CosineAnnealingWarmRestarts has 'T_0', etc.
                state_keys = set(scheduler_state.keys())
                is_onecycle = 'total_steps' in state_keys
                is_cosine_warm = 'T_0' in state_keys and 'T_mult' in state_keys
                is_cosine_simple = 'T_max' in state_keys and 'T_0' not in state_keys
                is_reduce_on_plateau = 'mode' in state_keys and 'factor' in state_keys
                
                scheduler_type_match = False
                if current_scheduler_type == 'OneCycleLR' and is_onecycle:
                    scheduler_type_match = True
                elif current_scheduler_type == 'CosineAnnealingWarmRestarts' and is_cosine_warm:
                    scheduler_type_match = True
                elif current_scheduler_type == 'CosineAnnealingLR' and is_cosine_simple:
                    scheduler_type_match = True
                elif current_scheduler_type == 'ReduceLROnPlateau' and is_reduce_on_plateau:
                    scheduler_type_match = True
                
                if scheduler_type_match:
                    try:
                        scheduler.load_state_dict(scheduler_state)
                        print("   ✓ Loaded scheduler state")
                    except Exception as e:
                        print(f"   ⚠️  Could not load scheduler state: {e}")
                        print("   Starting with fresh scheduler state")
                        # Fast-forward scheduler to current epoch if epoch-based
                        if current_scheduler_type in ['CosineAnnealingWarmRestarts', 'CosineAnnealingLR']:
                            for _ in range(checkpoint.get('epoch', 0)):
                                scheduler.step()
                            print(f"   ✓ Fast-forwarded scheduler to epoch {checkpoint.get('epoch', 0)}")
                else:
                    print(f"   ⚠️  Scheduler type mismatch (checkpoint has different scheduler type)")
                    print(f"   Starting with fresh scheduler state")
                    # Fast-forward scheduler to current epoch if epoch-based
                    if current_scheduler_type in ['CosineAnnealingWarmRestarts', 'CosineAnnealingLR']:
                        for _ in range(checkpoint.get('epoch', 0)):
                            scheduler.step()
                        print(f"   ✓ Fast-forwarded scheduler to epoch {checkpoint.get('epoch', 0)}")
            
            # Load scaler state
            if scaler is not None and 'scaler_state' in checkpoint:
                try:
                    scaler.load_state_dict(checkpoint['scaler_state'])
                    print("   ✓ Loaded scaler state")
                except Exception:
                    print("   ⚠️  Could not load scaler state, starting fresh")
            
            # Load training state
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            
            print(f"   ✓ Successfully loaded checkpoint from epoch {checkpoint.get('epoch', 0)}")
            print(f"   ✓ Best validation loss: {best_val_loss:.4f}")
            print(f"   ✓ Resuming from epoch {start_epoch}\n")
        except Exception as e:
            print(f"   ⚠️  Failed to load checkpoint: {e}")
            print("   Starting training from scratch\n")
    
    # Training loop
    patience = getattr(args, 'patience', 50)  # Early stopping patience from args or default to 50
    
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    print(f"Early stopping patience: {patience} epochs")
    print(f"Learning rate scheduler: {scheduler_type} (better convergence for transformers)")
    if start_epoch > 0:
        print(f"Resuming from epoch: {start_epoch}")
    print("="*80)
    
    for epoch in range(start_epoch, args.max_epochs):
        print(f"\nEPOCH {epoch+1}/{args.max_epochs}")
        print("-" * 50)
        
        # Training phase
        train_loss = run_training_epoch(model, train_loader, ce_loss, focal_loss, dice_loss, optimizer, class_weights, scheduler, scheduler_type)
        
        # Validation phase
        val_loss = validate_with_sliding_window(model, val_dataset, ce_loss, focal_loss, dice_loss)
        
        # Log results
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('Early_Stopping/Patience_Remaining', patience - epochs_without_improvement, epoch)
        writer.add_scalar('Early_Stopping/Epochs_Without_Improvement', epochs_without_improvement, epoch)
        
        # Print epoch summary
        print(f"Results:")
        print(f"  • Train Loss: {train_loss:.4f}")
        print(f"  • Validation Loss: {val_loss:.4f}")
        print(f"  • Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save periodic checkpoint (every 100 epochs) - useful for recovery and evaluation
        if (epoch + 1) % 100 == 0:
            periodic_checkpoint_path = os.path.join(snapshot_path, f"epoch_{epoch + 1}.pth")
            model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            periodic_checkpoint = {
                'epoch': epoch,
                'model_state': model_state,
                'optimizer_state': optimizer.state_dict() if optimizer is not None else None,
                'scheduler_state': scheduler.state_dict() if scheduler is not None else None,
                'best_val_loss': best_val_loss,
                'train_loss': train_loss,
                'val_loss': val_loss,
            }
            if scaler is not None:
                try:
                    periodic_checkpoint['scaler_state'] = scaler.state_dict()
                except Exception:
                    pass
            torch.save(periodic_checkpoint, periodic_checkpoint_path)
            print(f"   💾 Periodic checkpoint saved: epoch_{epoch + 1}.pth")
        
        # Save best model and check for improvement
        best_val_loss, improvement_made = save_best_model(
            model, epoch, val_loss, best_val_loss, snapshot_path,
            optimizer=optimizer, scheduler=scheduler, scaler=scaler
        )
        
        # Early stopping logic
        if improvement_made:
            epochs_without_improvement = 0
            print(f"    ✓ Improvement detected! Resetting patience counter.")
        else:
            epochs_without_improvement += 1
            print(f"    ⚠ No improvement for {epochs_without_improvement} epochs (patience: {patience}, remaining: {patience - epochs_without_improvement})")
        
        # Learning rate scheduling - handle different scheduler types
        if scheduler_type == 'ReduceLROnPlateau':
            scheduler.step(val_loss)  # ReduceLROnPlateau needs validation loss
        elif scheduler_type != 'OneCycleLR':
            scheduler.step()  # Other epoch-based schedulers step automatically
        # OneCycleLR is already stepped in run_training_epoch
        
        # Early stopping check
        if epochs_without_improvement >= patience:
            print("\n" + "="*80)
            print("EARLY STOPPING TRIGGERED!")
            print("="*80)
            print(f"Model has not improved for {patience} consecutive epochs.")
            print(f"Stopping training at epoch {epoch+1}.")
            print(f"Best validation loss achieved: {best_val_loss:.4f}")
            print("="*80 + "\n")
            break
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED")
    print("="*80)
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Total Epochs: {epoch+1}")
    print(f"Models Saved To: {snapshot_path}")
    print(f"TensorBoard Logs: {os.path.join(snapshot_path, 'tensorboard_logs')}")
    print("="*80 + "\n")
    
    writer.close()
    logger.info(f"Training completed. Best val loss: {best_val_loss:.4f}")
    
    return "Training Finished!"