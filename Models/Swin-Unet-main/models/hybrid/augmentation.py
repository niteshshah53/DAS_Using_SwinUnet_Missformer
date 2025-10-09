"""
Enhanced data augmentation for hybrid models.

This module provides comprehensive augmentation strategies specifically designed
for historical document segmentation, including:
- Random rotations (±5°)
- Color jittering for varying manuscript conditions
- Random erasing for occlusion robustness
- MixUp/CutMix for better generalization
"""

import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms as tvtf


class HybridAugmentation:
    """
    Enhanced data augmentation pipeline for hybrid models.
    
    Provides comprehensive augmentation strategies for historical document segmentation:
    - Random rotations (±5°)
    - Color jittering for varying manuscript conditions
    - Random erasing for occlusion robustness
    - MixUp/CutMix for better generalization
    """
    
    def __init__(self, patch_size=224, is_training=True):
        """
        Initialize hybrid augmentation.
        
        Args:
            patch_size (int): Target patch size
            is_training (bool): Whether to apply augmentations (True for training, False for validation/test)
        """
        self.patch_size = patch_size
        self.is_training = is_training
        
    def __call__(self, image, mask):
        """
        Apply transforms to image and mask.
        
        Args:
            image (PIL.Image): Input image
            mask (numpy.ndarray): Input mask as numpy array
            
        Returns:
            tuple: (transformed_image_tensor, transformed_mask_tensor)
        """
        # Convert mask to PIL Image for easier manipulation
        if isinstance(mask, np.ndarray):
            mask_pil = Image.fromarray(mask.astype(np.uint8))
        else:
            mask_pil = mask
            
        # Resize to target patch size
        image = image.resize((self.patch_size, self.patch_size), Image.BILINEAR)
        mask_pil = mask_pil.resize((self.patch_size, self.patch_size), Image.NEAREST)
        
        if self.is_training:
            # Apply comprehensive augmentation during training
            image, mask_pil = self._apply_augmentations(image, mask_pil)
        
        # Convert to tensor
        image_tensor = TF.to_tensor(image)
        
        # Convert mask back to numpy and then to tensor
        mask_np = np.array(mask_pil)
        mask_tensor = torch.from_numpy(mask_np).long()
        
        return image_tensor, mask_tensor
    
    def _apply_augmentations(self, image, mask):
        """
        Apply comprehensive data augmentations to image and mask.
        
        Args:
            image (PIL.Image): Input image
            mask (PIL.Image): Input mask
            
        Returns:
            tuple: (augmented_image, augmented_mask)
        """
        # Random horizontal flip (p=0.5)
        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        
        # Random vertical flip (p=0.5)
        if random.random() < 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        
        # Random rotation (±5°) - REDUCED for faster training
        if random.random() < 0.2:  # Reduced from 30% to 20%
            angle = random.uniform(-5, 5)
            image = TF.rotate(image, angle, fill=0)
            mask = TF.rotate(mask, angle, fill=0)
        
        # Random 90-degree rotations (multiples of 90°) - REDUCED
        if random.random() < 0.1:  # Reduced from 20% to 10%
            angle = random.choice([90, 180, 270])
            image = TF.rotate(image, angle, fill=0)
            mask = TF.rotate(mask, angle, fill=0)
        
        # Color jittering for varying manuscript conditions
        if random.random() < 0.4:  # 40% chance
            # Brightness adjustment
            brightness_factor = random.uniform(0.8, 1.2)
            image = TF.adjust_brightness(image, brightness_factor)
            
            # Contrast adjustment
            contrast_factor = random.uniform(0.8, 1.2)
            image = TF.adjust_contrast(image, contrast_factor)
            
            # Saturation adjustment (moderate)
            saturation_factor = random.uniform(0.9, 1.1)
            image = TF.adjust_saturation(image, saturation_factor)
        
        # Random Gaussian blur (light) - REDUCED for faster training
        if random.random() < 0.05:  # Reduced from 10% to 5%
            sigma = random.uniform(0, 0.3)  # Reduced sigma range
            radius = sigma * 2.5
            image = image.filter(ImageFilter.GaussianBlur(radius=max(0.1, radius)))
        
        # Random erasing for occlusion robustness - REDUCED
        if random.random() < 0.1:  # Reduced from 20% to 10%
            image, mask = self._random_erase(image, mask)
        
        # Random affine transformations - REDUCED for faster training
        if random.random() < 0.2:  # Reduced from 30% to 20%
            # Random rotation (-3° to 3°) - MORE CONSERVATIVE
            angle = random.uniform(-3, 3)
            
            # Random scaling (0.95 to 1.05) - MORE CONSERVATIVE
            scale = random.uniform(0.95, 1.05)
            
            # Random shear (-2° to 2°) - MORE CONSERVATIVE
            shear = random.uniform(-2, 2)
            
            # Random translation (-3% to 3% of image size) - MORE CONSERVATIVE
            translate_x = random.uniform(-0.03, 0.03) * self.patch_size
            translate_y = random.uniform(-0.03, 0.03) * self.patch_size
            translate = (translate_x, translate_y)
            
            # Apply affine transformation
            image = TF.affine(image, angle=angle, translate=translate, scale=scale, shear=shear, fill=0)
            mask = TF.affine(mask, angle=angle, translate=translate, scale=scale, shear=shear, fill=0)
        
        return image, mask
    
    def _random_erase(self, image, mask):
        """
        Apply random erasing for occlusion robustness.
        
        Args:
            image (PIL.Image): Input image
            mask (PIL.Image): Input mask
            
        Returns:
            tuple: (erased_image, erased_mask)
        """
        # Convert to numpy arrays
        img_array = np.array(image)
        mask_array = np.array(mask)
        
        # Get image dimensions
        h, w = img_array.shape[:2]
        
        # Random erasing parameters
        erase_prob = 0.5  # Probability of erasing
        erase_ratio = random.uniform(0.02, 0.33)  # Area ratio to erase
        
        if random.random() < erase_prob:
            # Calculate erase area
            erase_area = int(h * w * erase_ratio)
            erase_h = int(np.sqrt(erase_area * random.uniform(0.3, 3.3)))
            erase_w = int(erase_area / erase_h)
            
            # Ensure erase dimensions fit within image
            erase_h = min(erase_h, h)
            erase_w = min(erase_w, w)
            
            # Random position
            erase_x = random.randint(0, w - erase_w)
            erase_y = random.randint(0, h - erase_h)
            
            # Apply erasing (fill with random values)
            if len(img_array.shape) == 3:  # Color image
                img_array[erase_y:erase_y+erase_h, erase_x:erase_x+erase_w] = np.random.randint(0, 256, (erase_h, erase_w, 3))
            else:  # Grayscale image
                img_array[erase_y:erase_y+erase_h, erase_x:erase_x+erase_w] = np.random.randint(0, 256, (erase_h, erase_w))
            
            # Don't erase mask - keep original values
            # mask_array[erase_y:erase_y+erase_h, erase_x:erase_x+erase_w] = 0  # Optional: set to background
        
        # Convert back to PIL Images
        if len(img_array.shape) == 3:
            image = Image.fromarray(img_array.astype(np.uint8))
        else:
            image = Image.fromarray(img_array.astype(np.uint8))
        
        mask = Image.fromarray(mask_array.astype(np.uint8))
        
        return image, mask


def hybrid_training_transform(patch_size=224):
    """
    Create hybrid training transform with comprehensive augmentation.
    
    Args:
        patch_size (int): Target patch size
        
    Returns:
        HybridAugmentation: Training transform with augmentation
    """
    return HybridAugmentation(patch_size=patch_size, is_training=True)


def hybrid_validation_transform(patch_size=224):
    """
    Create hybrid validation/test transform without augmentation.
    
    Args:
        patch_size (int): Target patch size
        
    Returns:
        HybridAugmentation: Validation transform without augmentation
    """
    return HybridAugmentation(patch_size=patch_size, is_training=False)


def hybrid_identity_transform(image, mask):
    """
    Identity transform for compatibility with existing code.
    
    Args:
        image (PIL.Image): Input image
        mask (numpy.ndarray): Input mask
        
    Returns:
        tuple: (image_tensor, mask_tensor)
    """
    # Convert mask to PIL Image
    if isinstance(mask, np.ndarray):
        mask_pil = Image.fromarray(mask.astype(np.uint8))
    else:
        mask_pil = mask
    
    # Convert to tensor
    image_tensor = TF.to_tensor(image)
    
    # Convert mask to tensor
    mask_np = np.array(mask_pil)
    mask_tensor = torch.from_numpy(mask_np).long()
    
    return image_tensor, mask_tensor


# MixUp and CutMix implementations for advanced augmentation
class MixUpCutMix:
    """
    MixUp and CutMix augmentation for better generalization.
    
    Note: These are more complex to implement for segmentation tasks
    as they require careful handling of masks. For now, we focus on
    the simpler but effective augmentations above.
    """
    
    def __init__(self, alpha=0.2, mixup_prob=0.5, cutmix_prob=0.5):
        """
        Initialize MixUp/CutMix augmentation.
        
        Args:
            alpha (float): Mixing parameter
            mixup_prob (float): Probability of applying MixUp
            cutmix_prob (float): Probability of applying CutMix
        """
        self.alpha = alpha
        self.mixup_prob = mixup_prob
        self.cutmix_prob = cutmix_prob
    
    def __call__(self, images, masks):
        """
        Apply MixUp/CutMix augmentation.
        
        Args:
            images (torch.Tensor): Batch of images
            masks (torch.Tensor): Batch of masks
            
        Returns:
            tuple: (augmented_images, augmented_masks)
        """
        # This is a placeholder for future implementation
        # MixUp/CutMix for segmentation requires careful mask handling
        return images, masks


# Export the main functions
__all__ = [
    'HybridAugmentation',
    'hybrid_training_transform',
    'hybrid_validation_transform', 
    'hybrid_identity_transform',
    'MixUpCutMix'
]
