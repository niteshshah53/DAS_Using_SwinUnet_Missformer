import os
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from torch.cuda.amp import autocast, GradScaler

# Add common directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../common'))

from utils import DiceLoss, FocalLoss


def setup_logging(output_path):
    """Set up logging to both file and console."""
    log_file = os.path.join(output_path, "training.log")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File and console handlers
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def compute_class_weights(train_dataset, num_classes):
    """
    Compute class weights based on inverse frequency.
    Normalized to sum to num_classes for balanced weighting.
    """
    print("\nComputing class weights...")

    # Define color maps
    if num_classes == 6:
        COLOR_MAP = {
            (0, 0, 0): 0,        # Background
            (255, 255, 0): 1,    # Paratext
            (0, 255, 255): 2,    # Decoration
            (255, 0, 255): 3,    # Main text
            (255, 0, 0): 4,      # Title
            (0, 255, 0): 5,      # Chapter Heading
        }
    elif num_classes == 5:
        COLOR_MAP = {
            (0, 0, 0): 0,        # Background
            (255, 255, 0): 1,    # Paratext
            (0, 255, 255): 2,    # Decoration
            (255, 0, 255): 3,    # Main text
            (255, 0, 0): 4,      # Title
        }
    elif num_classes == 4:
        COLOR_MAP = {
            (0, 0, 0): 0,        # Background
            (0, 255, 0): 1,      # Comment
            (255, 0, 0): 2,      # Decoration
            (0, 0, 255): 3,      # Main Text
        }
    else:
        raise ValueError(f"Unsupported number of classes: {num_classes}")

    # Accumulate pixel counts per class efficiently by converting each
    # RGB mask into a single-label map once per image, then using
    # np.bincount to accumulate counts.
    class_counts = np.zeros(num_classes, dtype=np.int64)

    # Build integer mapping for color tuples -> class index
    mapping = {k: v for k, v in COLOR_MAP.items()}
    map_int = { (r << 16) | (g << 8) | b: cls for (r, g, b), cls in mapping.items() }

    for mask_path in train_dataset.mask_paths:
        mask = np.array(Image.open(mask_path).convert("RGB"))
        # vectorize RGB to single integer per pixel
        rgb_int = (mask[:, :, 0].astype(np.uint32) << 16) | (
            mask[:, :, 1].astype(np.uint32) << 8) | mask[:, :, 2].astype(np.uint32)

        flat = rgb_int.ravel()
        # Initialize flat labels to -1 (unknown)
        label_flat = np.full(flat.shape, -1, dtype=np.int32)

        # Iterate keys (few - number of classes) and set corresponding labels
        for rgb_val, cls in map_int.items():
            if np.any(flat == rgb_val):
                label_flat[flat == rgb_val] = int(cls)

        # Count only valid labels (>=0)
        valid = label_flat >= 0
        if np.any(valid):
            counts = np.bincount(label_flat[valid].astype(np.int64), minlength=num_classes)
            class_counts += counts

    # Avoid zero-total case
    total_pixels = class_counts.sum()
    if total_pixels == 0:
        # fallback to uniform weights
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.ones(num_classes, dtype=torch.float32, device=device)

    # Compute per-class frequency
    class_freq = class_counts.astype(np.float64) / float(total_pixels)

    # Log-scaled inverse-frequency weights to avoid extreme values
    # This helps when one class is <0.01% of data
    eps = 1e-6
    weights = np.log(1 + (1.0 / (class_freq + eps)))
    weights = weights / weights.mean()

    # Rarity-based proportional boost: boost classes whose frequency is below
    # the lower quartile. Scale boost factor up to `max_boost`.
    try:
        lower_q = np.percentile(class_freq, 25)
    except Exception:
        lower_q = class_freq.min()

    max_boost = 4.0
    boost_mask = class_freq < lower_q
    if np.any(boost_mask):
        # factor = min(max_boost, lower_q / freq)
        factors = np.ones_like(weights)
        factors[boost_mask] = np.minimum(max_boost, (lower_q + eps) / (class_freq[boost_mask] + eps))
        weights = weights * factors

    # Re-normalize to mean=1 after boosting
    weights = weights / weights.mean()

    # Print analysis
    print("\n" + "-" * 80)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("-" * 80)
    print(f"{'Class':<6} {'Count':<12} {'Frequency':<15} {'Weight':<15}")
    print("-" * 80)
    for cls in range(num_classes):
        print(f"{cls:<6} {class_counts[cls]:<12d} {class_freq[cls]:<15.6f} {weights[cls]:<15.6f}")
    print("-" * 80 + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.tensor(weights.astype(np.float32), dtype=torch.float32, device=device)


def create_data_loaders(train_dataset, val_dataset, batch_size, num_workers, seed, sampler=None):
    """Create training and validation data loaders.

    If `sampler` is provided, it will be used for the training loader and
    `shuffle` will be disabled (as required by PyTorch DataLoader).
    """
    def worker_init_fn(worker_id):
        import random
        random.seed(seed + worker_id)
    
    if sampler is not None:
        # When providing a sampler, DataLoader requires shuffle=False
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn
        )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )
    
    return train_loader, val_loader


def create_balanced_sampler(train_dataset, num_classes, threshold=0.01, eps=1e-6):
    """
    Create a WeightedRandomSampler that oversamples images containing rare classes.
    
    Uses continuous rarity scores with square-root inverse frequency to prevent
    overly aggressive oversampling that can cause noisy gradients.
    
    Returns None if dataset is invalid.
    """
    if not hasattr(train_dataset, 'mask_paths') or len(train_dataset.mask_paths) == 0:
        return None

    # Build color map for dataset classes (same mapping as compute_class_weights)
    if num_classes == 6:
        COLOR_MAP = {
            (0, 0, 0): 0,
            (255, 255, 0): 1,
            (0, 255, 255): 2,
            (255, 0, 255): 3,
            (255, 0, 0): 4,
            (0, 255, 0): 5,
        }
    elif num_classes == 5:
        COLOR_MAP = {
            (0, 0, 0): 0,
            (255, 255, 0): 1,
            (0, 255, 255): 2,
            (255, 0, 255): 3,
            (255, 0, 0): 4,
        }
    elif num_classes == 4:
        COLOR_MAP = {
            (0, 0, 0): 0,
            (0, 255, 0): 1,
            (255, 0, 0): 2,
            (0, 0, 255): 3,
        }
    else:
        return None

    # compute per-class pixel counts
    class_counts = np.zeros(num_classes, dtype=np.int64)
    mapping = {k: v for k, v in COLOR_MAP.items()}
    map_int = { (r << 16) | (g << 8) | b: cls for (r, g, b), cls in mapping.items() }

    for mask_path in train_dataset.mask_paths:
        mask = np.array(Image.open(mask_path).convert('RGB'))
        rgb_int = (mask[:, :, 0].astype(np.uint32) << 16) | (mask[:, :, 1].astype(np.uint32) << 8) | mask[:, :, 2].astype(np.uint32)
        flat = rgb_int.ravel()
        label_flat = np.full(flat.shape, -1, dtype=np.int32)
        for rgb_val, cls in map_int.items():
            if np.any(flat == rgb_val):
                label_flat[flat == rgb_val] = int(cls)
        valid = label_flat >= 0
        if np.any(valid):
            counts = np.bincount(label_flat[valid].astype(np.int64), minlength=num_classes)
            class_counts += counts

    total = class_counts.sum()
    if total == 0:
        return None

    # Compute class frequencies
    class_freq = class_counts.astype(np.float64) / float(total)

    # Compute continuous rarity scores for all samples
    # Use square-root inverse frequency to prevent overly aggressive oversampling
    sample_weights = []
    for mask_path in train_dataset.mask_paths:
        mask = np.array(Image.open(mask_path).convert('RGB'))
        rgb_int = (mask[:, :, 0].astype(np.uint32) << 16) | (mask[:, :, 1].astype(np.uint32) << 8) | mask[:, :, 2].astype(np.uint32)
        present = set()
        for rgb_val, cls in map_int.items():
            if np.any(rgb_int == rgb_val):
                present.add(cls)
        
        if len(present) == 0:
            # No valid classes found, use uniform weight
            sample_weights.append(1.0)
        else:
            # Compute rarity score: sum of square-root inverse frequency for present classes
            # Square-root provides smoother interpolation than linear inverse frequency
            w = 0.0
            for cls in present:
                w += (1.0 / (class_freq[cls] + eps)) ** 0.5
            sample_weights.append(float(w))

    # Normalize weights to mean=1 for stable sampling probabilities
    sw = np.array(sample_weights, dtype=np.float64)
    sw = sw / (sw.mean() + eps)

    # Create PyTorch sampler
    weights_tensor = torch.DoubleTensor(sw)
    from torch.utils.data import WeightedRandomSampler
    sampler = WeightedRandomSampler(weights=weights_tensor, num_samples=len(weights_tensor), replacement=True)
    print(f"Balanced sampler created (continuous rarity-based oversampling).")
    return sampler


def create_loss_functions(class_weights, num_classes):
    """
    Create loss functions with class weights.
    Based on ablation: Focal Loss is beneficial in full configuration.
    """
    ce_loss = CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    focal_loss = FocalLoss(gamma=3.0, weight=class_weights)
    dice_loss = DiceLoss(num_classes, weight=class_weights, smooth=1e-4)
    
    print("✓ Loss functions initialized")
    print("  - CrossEntropyLoss: weighted, label_smoothing=0.1")
    print("  - FocalLoss: gamma=3.0, weighted")
    print("  - DiceLoss: weighted, smooth=1e-4\n")
    
    return ce_loss, focal_loss, dice_loss


def compute_combined_loss(predictions, labels, ce_loss, focal_loss, dice_loss):
    if isinstance(predictions, tuple):
        logits, aux_outputs = predictions
        loss_ce = ce_loss(logits, labels)
        loss_focal = focal_loss(logits, labels)
        loss_dice = dice_loss(logits, labels, softmax=True)

        main_loss = 0.3 * loss_ce + 0.2 * loss_focal + 0.5 * loss_dice

        aux_weights = [0.4, 0.3, 0.2][:len(aux_outputs)]
        aux_loss = 0.0

        for weight, aux_output in zip(aux_weights, aux_outputs):
            aux_ce = ce_loss(aux_output, labels)
            aux_focal = focal_loss(aux_output, labels)
            aux_dice = dice_loss(aux_output, labels, softmax=True)
            aux_combined = 0.3 * aux_ce + 0.2 * aux_focal + 0.5 * aux_dice
            aux_loss += weight * aux_combined

        total_loss = main_loss + aux_loss
        return total_loss
    else:
        loss_ce = ce_loss(predictions, labels)
        loss_focal = focal_loss(predictions, labels)
        loss_dice = dice_loss(predictions, labels, softmax=True)
        total_loss = 0.3 * loss_ce + 0.2 * loss_focal + 0.5 * loss_dice
        return total_loss



def create_optimizer_and_scheduler(model, learning_rate, max_epochs, steps_per_epoch, encoder_lr_factor=0.1, scheduler_type='OneCycleLR'):
    """
    Create AdamW optimizer with learning rate scheduler.
    
    Args:
        encoder_lr_factor: Multiplier for encoder learning rate (default 0.1 = 10x smaller)
        scheduler_type: Type of scheduler to use
            - 'OneCycleLR': Step per batch (requires total_steps)
            - 'CosineAnnealingWarmRestarts': Step per epoch
            - 'ReduceLROnPlateau': Step per epoch based on validation loss
            - 'CosineAnnealingLR': Step per epoch
    """
    # Separate encoder and decoder parameters
    encoder_decay_params = []
    encoder_no_decay_params = []
    decoder_decay_params = []
    decoder_no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # treat 1D params (biases, LayerNorm/BatchNorm weights) and explicit '.bias' as no_decay
        lname = name.lower()
        is_no_decay = param.dim() == 1 or name.endswith('.bias') or 'norm' in lname or 'bn' in lname or 'ln' in lname
        
        # Check if parameter belongs to encoder
        is_encoder = 'encoder' in name.lower() or 'adapter' in name.lower() or 'streaming_proj' in name.lower() or 'feature_adapters' in name.lower()
        
        if is_encoder:
            if is_no_decay:
                encoder_no_decay_params.append(param)
            else:
                encoder_decay_params.append(param)
        else:
            if is_no_decay:
                decoder_no_decay_params.append(param)
            else:
                decoder_decay_params.append(param)

    # Create parameter groups with differential learning rates
    param_groups = []
    if encoder_decay_params:
        param_groups.append({
            'params': encoder_decay_params, 
            'weight_decay': 0.01,
            'lr': learning_rate * encoder_lr_factor,
            'initial_lr': learning_rate * encoder_lr_factor,
            'name': 'encoder_decay'
        })
    if encoder_no_decay_params:
        param_groups.append({
            'params': encoder_no_decay_params, 
            'weight_decay': 0.0,
            'lr': learning_rate * encoder_lr_factor,
            'initial_lr': learning_rate * encoder_lr_factor,
            'name': 'encoder_no_decay'
        })
    if decoder_decay_params:
        param_groups.append({
            'params': decoder_decay_params, 
            'weight_decay': 0.01,
            'lr': learning_rate,
            'initial_lr': learning_rate,
            'name': 'decoder_decay'
        })
    if decoder_no_decay_params:
        param_groups.append({
            'params': decoder_no_decay_params, 
            'weight_decay': 0.0,
            'lr': learning_rate,
            'initial_lr': learning_rate,
            'name': 'decoder_no_decay'
        })

    optimizer = optim.AdamW(param_groups, betas=(0.9, 0.999))
    
    # Create scheduler based on scheduler_type
    scheduler_name = ""
    if scheduler_type == 'OneCycleLR':
        # Calculate total_steps and validate
        total_steps = max_epochs * steps_per_epoch
        
        # Validate that total_steps matches expected training steps
        if total_steps <= 0:
            raise ValueError(f"Invalid total_steps: {total_steps} (max_epochs={max_epochs}, steps_per_epoch={steps_per_epoch})")
        
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            total_steps=total_steps,
            pct_start=0.3,  # 30% warmup
            anneal_strategy='cos',
            div_factor=25.0,  # Initial LR = max_lr/25
            final_div_factor=10000.0  # Final LR = max_lr/10000
        )
        scheduler_name = f"OneCycleLR (total_steps={total_steps}, warmup=30%)"
        
    elif scheduler_type == 'CosineAnnealingWarmRestarts':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=50,
            T_mult=2,
            eta_min=1e-7
        )
        scheduler_name = "CosineAnnealingWarmRestarts (T_0=50, T_mult=2)"
        
    elif scheduler_type == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=15,
            min_lr=1e-6,
            verbose=False
        )
        scheduler_name = "ReduceLROnPlateau (factor=0.5, patience=15)"
        
    elif scheduler_type == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_epochs,
            eta_min=1e-6
        )
        scheduler_name = f"CosineAnnealingLR (T_max={max_epochs})"
        
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}. "
                        f"Supported: OneCycleLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau, CosineAnnealingLR")
    
    print(f"\n🔧 Optimizer: AdamW (lr={learning_rate}, weight_decay=0.01)")
    if encoder_decay_params or encoder_no_decay_params:
        print(f"   - Encoder LR: {learning_rate * encoder_lr_factor:.6f} ({encoder_lr_factor}x)")
    print(f"   - Decoder LR: {learning_rate:.6f} (1.0x)")
    print(f"📈 Scheduler: {scheduler_name}")
    if scheduler_type == 'OneCycleLR':
        print(f"   - Total steps: {max_epochs * steps_per_epoch} ({steps_per_epoch} steps/epoch × {max_epochs} epochs)")
        print(f"   - Max LR: {learning_rate:.6f}")
        print(f"   - Initial LR: {learning_rate/25:.6f}")
        print(f"   - Final LR: {learning_rate/10000:.6f}")
        print(f"   - Warmup: 30% of training")
    print()
    
    return optimizer, scheduler


def run_training_epoch(model, train_loader, ce_loss, focal_loss, dice_loss, 
                       optimizer, scheduler, scaler=None, scheduler_type='OneCycleLR'):
    """Run one training epoch with gradient clipping and optional AMP.

    scaler: torch.cuda.amp.GradScaler or None
    scheduler_type: Type of scheduler - determines when to step
        - 'OneCycleLR': Step per batch
        - Others: Step per epoch (caller handles)
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    valid_batches = 0
    skipped_loss_nan = 0
    skipped_grad_nan = 0
    scheduler_warning_printed = False  # Track if scheduler warning was printed
    
    for batch_idx, batch in enumerate(train_loader):
        # Get data
        if isinstance(batch, dict):
            images = batch['image']
            labels = batch['label']
        else:
            images, labels = batch[0], batch[1]
        
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        
        # Zero grads
        optimizer.zero_grad()

        # Forward (AMP if scaler provided and CUDA available)
        if scaler is not None and torch.cuda.is_available():
            with autocast():
                predictions = model(images)
                loss = compute_combined_loss(predictions, labels, ce_loss, focal_loss, dice_loss)

            # Check for NaN/Inf loss
            if torch.isnan(loss) or torch.isinf(loss):
                skipped_loss_nan += 1
                continue

            # Scaled backward, then unscale for clipping
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            
            # Check for NaN/Inf gradients before clipping
            has_nan_grad = False
            for param in model.parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        has_nan_grad = True
                        break
            
            if has_nan_grad:
                # Gradients contain Inf/NaN - scaler will skip step and reduce scale
                skipped_grad_nan += 1
                scaler.update()  # This reduces the scale factor automatically
                continue
            
            # Gradients are valid, proceed with clipping and step
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            # Scheduler step per batch (OneCycleLR expects step every batch)
            # Only step if scheduler hasn't reached max steps
            # Other schedulers are stepped per epoch by the caller
            if scheduler_type == 'OneCycleLR':
                if hasattr(scheduler, 'total_steps') and scheduler.last_epoch + 1 < scheduler.total_steps:
                    scheduler.step()
                elif hasattr(scheduler, 'total_steps') and not scheduler_warning_printed:
                    if scheduler.last_epoch + 1 >= scheduler.total_steps:
                        print("⚠️  Scheduler reached max steps, stopping LR updates.")
                        scheduler_warning_printed = True
            
            # Track valid loss
            loss_val = float(loss.item())
            if not (np.isnan(loss_val) or np.isinf(loss_val)):
                total_loss += loss_val
                valid_batches += 1
        else:
            predictions = model(images)
            loss = compute_combined_loss(predictions, labels, ce_loss, focal_loss, dice_loss)
            
            # Check for NaN/Inf loss
            if torch.isnan(loss) or torch.isinf(loss):
                skipped_loss_nan += 1
                continue
            
            loss.backward()
            
            # Check for NaN/Inf gradients before clipping
            has_nan_grad = False
            for param in model.parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        has_nan_grad = True
                        break
            
            if has_nan_grad:
                skipped_grad_nan += 1
                optimizer.zero_grad()  # Clear gradients
                continue
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Scheduler step per batch (OneCycleLR expects step every batch)
            # Only step if scheduler hasn't reached max steps
            # Other schedulers are stepped per epoch by the caller
            if scheduler_type == 'OneCycleLR':
                if hasattr(scheduler, 'total_steps') and scheduler.last_epoch + 1 < scheduler.total_steps:
                    scheduler.step()
                elif hasattr(scheduler, 'total_steps') and not scheduler_warning_printed:
                    if scheduler.last_epoch + 1 >= scheduler.total_steps:
                        print("⚠️  Scheduler reached max steps, stopping LR updates.")
                        scheduler_warning_printed = True
            
            # Track valid loss
            loss_val = float(loss.item())
            if not (np.isnan(loss_val) or np.isinf(loss_val)):
                total_loss += loss_val
                valid_batches += 1

        num_batches += 1
    
    # Print summary only if batches were skipped
    if skipped_loss_nan > 0 or skipped_grad_nan > 0:
        total_skipped = skipped_loss_nan + skipped_grad_nan
        print(f"  ⚠️  Skipped {total_skipped} batches ({skipped_loss_nan} NaN/Inf loss, {skipped_grad_nan} NaN/Inf gradients)")
    
    return total_loss / valid_batches if valid_batches > 0 else float('inf')


def validate_model(model, val_loader, ce_loss, focal_loss, dice_loss):
    """
    Validate model on full validation set using data loader.
    More efficient than processing samples individually.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    valid_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            # Get data
            if isinstance(batch, dict):
                images = batch['image']
                labels = batch['label']
            else:
                images, labels = batch[0], batch[1]
            
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            
            # Forward pass (use autocast on GPU for validation too)
            if torch.cuda.is_available():
                with autocast():
                    predictions = model(images)
                    loss = compute_combined_loss(predictions, labels, ce_loss, focal_loss, dice_loss)
            else:
                predictions = model(images)
                loss = compute_combined_loss(predictions, labels, ce_loss, focal_loss, dice_loss)
            
            # Skip NaN/Inf losses
            loss_val = loss.item()
            if not (np.isnan(loss_val) or np.isinf(loss_val)):
                total_loss += loss_val
                valid_batches += 1
            num_batches += 1
    
    return total_loss / valid_batches if valid_batches > 0 else float('inf')


def save_best_model(model, epoch, val_loss, best_val_loss, snapshot_path,
                    optimizer=None, scheduler=None, scaler=None):
    """Save model + optimizer/scheduler/scaler checkpoint if validation loss improved.

    Returns (best_val_loss, improvement_made)
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


def trainer_synapse(args, model, snapshot_path, train_dataset=None, val_dataset=None):
    """
    Main training function optimized based on ablation study.
    
    Best configuration (F1=0.6919, IoU=0.5831):
    - BL (Baseline encoder-decoder)
    - ASH (Alternative Segmentation Head: Conv3x3-ReLU-Conv1x1)
    - DS (Deep Supervision with auxiliary outputs)
    - AFF (Attention Feature Fusion)
    - Bo (Bottleneck with 2 Swin blocks)
    - FL (Focal Loss in combination)
    """
    # Set random seeds for reproducibility
    import random
    seed = getattr(args, 'seed', 1234)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Setup
    logger = setup_logging(snapshot_path)
    patience = getattr(args, 'patience', 25)
    
    # Print configuration
    print("\n" + "="*80)
    print("TRAINING CONFIGURATION - FINAL OPTIMIZED VERSION")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Manuscript: {getattr(args, 'manuscript', 'N/A')}")
    print(f"Model: CNN-Transformer (EfficientNet-B4 + Swin-UNet Decoder)")
    print(f"Configuration: BL + ASH + DS + AFF + Bo + FL (Best from ablation)")
    print(f"Batch Size: {args.batch_size}")
    print(f"Max Epochs: {args.max_epochs}")
    print(f"Learning Rate: {args.base_lr}")
    print(f"Number of Classes: {args.num_classes}")
    print(f"Early Stopping Patience: {patience} epochs")
    print(f"Output Directory: {snapshot_path}")
    print("="*80 + "\n")
    
    # Multi-GPU setup
    if args.n_gpu > 1:
        print(f"🖥️  Using {args.n_gpu} GPUs for training\n")
        model = nn.DataParallel(model)
    
    # Freeze encoder if requested
    if getattr(args, 'freeze_encoder', False):
        print(f"🔒 Freezing encoder for training")
        if isinstance(model, nn.DataParallel):
            model.module.model.freeze_encoder()
        else:
            model.model.freeze_encoder()
        
        freeze_epochs = getattr(args, 'freeze_epochs', 0)
        if freeze_epochs > 0:
            print(f"   Will unfreeze after {freeze_epochs} epochs")
        else:
            print(f"   Encoder will remain frozen for entire training")
        print()
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_dataset, val_dataset,
        args.batch_size * args.n_gpu,
        args.num_workers,
        args.seed
    )
    
    print(f"📊 Dataset Statistics:")
    print(f"   - Training samples: {len(train_dataset)}")
    print(f"   - Validation samples: {len(val_dataset)}")
    print(f"   - Batch size: {args.batch_size * args.n_gpu}")
    print(f"   - Steps per epoch: {len(train_loader)}\n")
    
    # Compute class weights
    if hasattr(train_dataset, 'mask_paths'):
        class_weights = compute_class_weights(train_dataset, args.num_classes)
        # compute_class_weights already applies a bounded rarity-based boost
        print(f"📈 Class weights computed with rarity-based boosting (mean scaled)")
        print(f"   Final weights: {class_weights.cpu().numpy()}\n")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        class_weights = torch.ones(args.num_classes, device=device)
    
    # Create loss functions, optimizer, scheduler
    ce_loss, focal_loss, dice_loss = create_loss_functions(class_weights, args.num_classes)
    
    # Validate dataloader length matches expected steps per epoch
    steps_per_epoch = len(train_loader)
    if steps_per_epoch == 0:
        raise ValueError(f"Train loader is empty! Cannot create scheduler.")
    
    # Get scheduler type from args
    scheduler_type = getattr(args, 'scheduler_type', 'OneCycleLR')
    
    optimizer, scheduler = create_optimizer_and_scheduler(
        model, args.base_lr, args.max_epochs, steps_per_epoch,
        encoder_lr_factor=getattr(args, 'encoder_lr_factor', 0.1),
        scheduler_type=scheduler_type
    )
    
    # Verify scheduler total_steps matches expected training steps (only for OneCycleLR)
    if scheduler_type == 'OneCycleLR' and hasattr(scheduler, 'total_steps'):
        expected_steps = args.max_epochs * steps_per_epoch
        if scheduler.total_steps != expected_steps:
            print(f"⚠️  Warning: Scheduler total_steps ({scheduler.total_steps}) != expected ({expected_steps})")
            print(f"   This may cause LR misalignment. Check dataloader length consistency.")
    # Mixed precision scaler (used when CUDA is available)
    scaler = GradScaler() if torch.cuda.is_available() else None
    
    # TensorBoard
    writer = SummaryWriter(os.path.join(snapshot_path, 'tensorboard_logs'))
    
    # Resume from checkpoint if available
    start_epoch = 0
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    encoder_unfrozen_epoch = None
    
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
            
            # Determine encoder unfrozen epoch if encoder was unfrozen
            if getattr(args, 'freeze_encoder', False) and getattr(args, 'freeze_epochs', 0) > 0:
                if start_epoch > args.freeze_epochs:
                    encoder_unfrozen_epoch = args.freeze_epochs
            
            print(f"   ✓ Successfully loaded checkpoint from epoch {checkpoint.get('epoch', 0)}")
            print(f"   ✓ Best validation loss: {best_val_loss:.4f}")
            print(f"   ✓ Resuming from epoch {start_epoch}\n")
        except Exception as e:
            print(f"   ⚠️  Failed to load checkpoint: {e}")
            print("   Starting training from scratch\n")
    
    # Training loop
    print("="*80)
    print("🚀 STARTING TRAINING")
    print("="*80)
    print(f"Loss: 0.3*CE + 0.2*Focal + 0.5*Dice (with Deep Supervision)")
    print(f"Early stopping: {patience} epochs patience")
    if start_epoch > 0:
        print(f"Resuming from epoch: {start_epoch}")
    print("="*80 + "\n")
    
    for epoch in range(start_epoch, args.max_epochs):
        print(f"EPOCH {epoch+1}/{args.max_epochs}")
        print("-" * 50)
        
        # Train
        train_loss = run_training_epoch(
            model, train_loader, ce_loss, focal_loss, dice_loss,
            optimizer, scheduler, scaler=scaler, scheduler_type=scheduler_type
        )
        
        # Validate
        val_loss = validate_model(model, val_loader, ce_loss, focal_loss, dice_loss)
        
        # Log
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Step scheduler per epoch (for schedulers that step per epoch, not per batch)
        # OneCycleLR is stepped per batch in run_training_epoch
        if scheduler_type != 'OneCycleLR':
            if scheduler_type == 'ReduceLROnPlateau':
                # ReduceLROnPlateau steps based on validation loss
                scheduler.step(val_loss)
            else:
                # Other epoch-based schedulers
                scheduler.step()
        
        # Print results
        print(f"Results:")
        print(f"  • Train Loss: {train_loss:.4f}")
        print(f"  • Val Loss: {val_loss:.4f}")
        print(f"  • Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Unfreeze encoder if requested and epoch reached
        if getattr(args, 'freeze_encoder', False) and getattr(args, 'freeze_epochs', 0) > 0:
            if epoch + 1 == args.freeze_epochs:
                print(f"\n🔓 Unfreezing encoder at epoch {epoch + 1}")
                if isinstance(model, nn.DataParallel):
                    model.module.model.unfreeze_encoder()
                else:
                    model.model.unfreeze_encoder()
                
                # CRITICAL: Reconfigure optimizer with differential learning rates
                # Encoder needs much lower LR to prevent gradient explosion
                # When encoder is frozen, its params aren't in optimizer, so we need to recreate it
                encoder_lr_factor = getattr(args, 'encoder_lr_factor', 0.1)
                encoder_unfrozen_epoch = epoch + 1  # Track when encoder was unfrozen
                print(f"🔄 Reconfiguring optimizer with differential learning rates")
                print(f"   - Encoder LR factor: {encoder_lr_factor:.4f}x (encoder LR will be {args.base_lr * encoder_lr_factor:.6f})")
                print(f"   - Encoder LR will decay: 0.95^epochs_since_unfreeze")
                
                # Save current optimizer state (for decoder params)
                optimizer_state = optimizer.state_dict()
                
                # Recreate optimizer with all parameters (now including encoder)
                encoder_decay_params = []
                encoder_no_decay_params = []
                decoder_decay_params = []
                decoder_no_decay_params = []

                for name, param in model.named_parameters():
                    if not param.requires_grad:
                        continue
                    lname = name.lower()
                    is_no_decay = param.dim() == 1 or name.endswith('.bias') or 'norm' in lname or 'bn' in lname or 'ln' in lname
                    is_encoder = 'encoder' in name.lower() or 'adapter' in name.lower() or 'streaming_proj' in name.lower() or 'feature_adapters' in name.lower()
                    
                    if is_encoder:
                        if is_no_decay:
                            encoder_no_decay_params.append(param)
                        else:
                            encoder_decay_params.append(param)
                    else:
                        if is_no_decay:
                            decoder_no_decay_params.append(param)
                        else:
                            decoder_decay_params.append(param)

                param_groups = []
                if encoder_decay_params:
                    param_groups.append({
                        'params': encoder_decay_params, 
                        'weight_decay': 0.01,
                        'lr': args.base_lr * encoder_lr_factor,
                        'initial_lr': args.base_lr * encoder_lr_factor,
                        'name': 'encoder_decay'
                    })
                if encoder_no_decay_params:
                    param_groups.append({
                        'params': encoder_no_decay_params, 
                        'weight_decay': 0.0,
                        'lr': args.base_lr * encoder_lr_factor,
                        'initial_lr': args.base_lr * encoder_lr_factor,
                        'name': 'encoder_no_decay'
                    })
                if decoder_decay_params:
                    param_groups.append({
                        'params': decoder_decay_params, 
                        'weight_decay': 0.01,
                        'lr': args.base_lr,
                        'initial_lr': args.base_lr,
                        'name': 'decoder_decay'
                    })
                if decoder_no_decay_params:
                    param_groups.append({
                        'params': decoder_no_decay_params, 
                        'weight_decay': 0.0,
                        'lr': args.base_lr,
                        'initial_lr': args.base_lr,
                        'name': 'decoder_no_decay'
                    })

                # Create new optimizer
                optimizer = optim.AdamW(param_groups, betas=(0.9, 0.999))
                
                # Try to load optimizer state (decoder params should match)
                try:
                    optimizer.load_state_dict(optimizer_state)
                    print("   ✓ Preserved optimizer state for decoder parameters")
                except Exception as e:
                    print(f"   ⚠️  Could not fully load optimizer state: {e}")
                    print("   Starting fresh optimizer state (this is OK for encoder)")
                
                # Recreate scheduler for new optimizer structure
                # OneCycleLR needs to be recreated when parameter groups change
                scheduler_type = getattr(args, 'scheduler_type', 'OneCycleLR')
                steps_per_epoch = len(train_loader)
                
                if scheduler_type == 'OneCycleLR':
                    # Track current scheduler step to maintain continuity
                    # OneCycleLR.last_epoch tracks the number of steps taken (since it steps per batch)
                    # At epoch 30, we've completed 30 epochs, so steps = 30 * steps_per_epoch
                    current_step = scheduler.last_epoch if hasattr(scheduler, 'last_epoch') else (epoch * steps_per_epoch)
                    total_steps = args.max_epochs * steps_per_epoch
                    
                    # Create max_lr list matching new parameter groups
                    # Encoder groups get encoder_lr_factor * base_lr, decoder groups get base_lr
                    max_lrs = []
                    for group in param_groups:
                        max_lrs.append(group['lr'])
                    
                    # Create scheduler fresh (without last_epoch) to properly initialize max_lr in param groups
                    # OneCycleLR initialization calls _initial_step() which sets last_epoch to -1 and then steps once
                    # So after initialization, last_epoch will be 0
                    scheduler = optim.lr_scheduler.OneCycleLR(
                        optimizer,
                        max_lr=max_lrs if len(max_lrs) > 1 else max_lrs[0],
                        total_steps=total_steps,
                        pct_start=0.3,
                        anneal_strategy='cos',
                        div_factor=25.0,
                        final_div_factor=10000.0
                    )
                    
                    # After initialization, last_epoch is 0 (initialization already stepped once)
                    # We need to fast-forward to current_step
                    # Set last_epoch to current_step - 1, then step once to reach current_step
                    if current_step > 0:
                        scheduler.last_epoch = current_step - 1  # Set to step before current
                        scheduler.step()  # Step once to advance to current_step and update LRs
                    # If current_step == 0, scheduler is already at the right position (last_epoch=0 after init)
                    print(f"   ✓ Recreated OneCycleLR scheduler (resumed from step {current_step}/{total_steps})")
                elif scheduler_type == 'CosineAnnealingWarmRestarts':
                    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        optimizer,
                        T_0=50,
                        T_mult=2,
                        eta_min=1e-7
                    )
                    # Fast-forward to current epoch
                    for _ in range(epoch):
                        scheduler.step()
                    print(f"   ✓ Recreated CosineAnnealingWarmRestarts scheduler (fast-forwarded to epoch {epoch})")
                elif scheduler_type == 'ReduceLROnPlateau':
                    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        mode='min',
                        factor=0.5,
                        patience=15,
                        min_lr=1e-6,
                        verbose=False
                    )
                    print(f"   ✓ Recreated ReduceLROnPlateau scheduler")
                elif scheduler_type == 'CosineAnnealingLR':
                    scheduler = optim.lr_scheduler.CosineAnnealingLR(
                        optimizer,
                        T_max=args.max_epochs,
                        eta_min=1e-6
                    )
                    # Fast-forward to current epoch
                    for _ in range(epoch):
                        scheduler.step()
                    print(f"   ✓ Recreated CosineAnnealingLR scheduler (fast-forwarded to epoch {epoch})")
                
                print()
            elif encoder_unfrozen_epoch is not None:
                # Encoder was unfrozen in a previous epoch - decay encoder LR factor
                epochs_since_unfreeze = (epoch + 1) - encoder_unfrozen_epoch
                base_encoder_lr_factor = getattr(args, 'encoder_lr_factor', 0.1)
                encoder_lr_factor = base_encoder_lr_factor * (0.95 ** epochs_since_unfreeze)
                
                # Update encoder parameter groups with decayed learning rate
                encoder_lr = args.base_lr * encoder_lr_factor
                for param_group in optimizer.param_groups:
                    if 'encoder' in param_group.get('name', ''):
                        param_group['lr'] = encoder_lr
                
                # Log encoder LR decay (only every 10 epochs to avoid clutter)
                if epochs_since_unfreeze % 10 == 0 or epochs_since_unfreeze == 1:
                    print(f"   🔄 Encoder LR decay: {encoder_lr_factor:.4f}x (epochs since unfreeze: {epochs_since_unfreeze})")
        
        # Save periodic checkpoint (every 10 epochs) - useful for recovery and evaluation
        # Sometimes the best model by loss isn't best by Dice/IoU
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
        
        # Save best and check early stopping (save full checkpoint)
        best_val_loss, improved = save_best_model(
            model, epoch, val_loss, best_val_loss, snapshot_path,
            optimizer=optimizer, scheduler=scheduler, scaler=scaler
        )
        
        if improved:
            epochs_without_improvement = 0
            print(f"    ✅ Improvement! Patience reset.")
        else:
            epochs_without_improvement += 1
            remaining = patience - epochs_without_improvement
            print(f"    ⚠️  No improvement ({epochs_without_improvement}/{patience} epochs, {remaining} remaining)")
        
        # Early stopping
        if epochs_without_improvement >= patience:
            print("\n" + "="*80)
            print("⏸️  EARLY STOPPING TRIGGERED")
            print("="*80)
            print(f"No improvement for {patience} consecutive epochs.")
            print(f"Best validation loss: {best_val_loss:.4f}")
            print(f"Stopped at epoch {epoch+1}/{args.max_epochs}")
            print("="*80 + "\n")
            break
        
        print()  # Blank line between epochs
    
    # Training complete
    print("="*80)
    print("✅ TRAINING COMPLETED")
    print("="*80)
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Total Epochs: {epoch+1}")
    print(f"Models Saved: {snapshot_path}")
    print(f"TensorBoard: {os.path.join(snapshot_path, 'tensorboard_logs')}")
    print("="*80 + "\n")
    
    writer.close()
    logger.info(f"Training completed. Best val loss: {best_val_loss:.4f}")
    
    return "Training Finished!"