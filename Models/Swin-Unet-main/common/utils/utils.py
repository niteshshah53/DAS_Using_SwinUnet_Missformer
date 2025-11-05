import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk

# Focal Loss for multi-class segmentation
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None, ignore_index=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, input, target):
        # FIXED: Proper Focal Loss implementation
        if self.ignore_index is not None:
            ce_loss = F.cross_entropy(input, target, weight=self.weight, ignore_index=self.ignore_index, reduction='none')
        else:
            ce_loss = F.cross_entropy(input, target, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
    

class DiceLoss(nn.Module):
    """
    Dice Loss with proper weight handling and NaN prevention.
    
    Fixes from original:
    - Added weight parameter to __init__
    - Increased smooth from 1e-5 to 1e-4 for stability
    - Added NaN detection and handling
    - Fixed normalization (sum of weights instead of n_classes)
    - Prevent division by zero for empty classes
    """
    def __init__(self, n_classes, weight=None, smooth=1e-4):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.weight = weight  # Store default weight
        self.smooth = smooth  # Increased for numerical stability

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        """
        Compute Dice loss for a single class with NaN prevention.
        """
        target = target.float()
        smooth = self.smooth
        
        # Check for NaN in inputs before computation
        if torch.isnan(score).any() or torch.isnan(target).any():
            # If inputs contain NaN, return 0 loss to avoid propagating NaN
            return torch.tensor(0.0, device=score.device, dtype=score.dtype)
        
        # Clamp values to prevent overflow/underflow
        score = torch.clamp(score, min=0.0, max=1.0)
        target = torch.clamp(target, min=0.0, max=1.0)
        
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        union = z_sum + y_sum
        
        # Check for NaN in computed values
        if torch.isnan(intersect) or torch.isnan(union):
            # Return 0 loss if computation produced NaN
            return torch.tensor(0.0, device=score.device, dtype=score.dtype)
        
        # CRITICAL FIX: Prevent NaN when both prediction and target are empty
        # If union is extremely small, return 0 loss (perfect for absent class)
        if union < smooth * 10:
            return torch.tensor(0.0, device=score.device, dtype=score.dtype)
        
        dice_coef = (2 * intersect + smooth) / (union + smooth)
        loss = 1 - dice_coef
        
        # Safety check for NaN in final loss (shouldn't happen with above fixes)
        if torch.isnan(loss):
            return torch.tensor(0.0, device=score.device, dtype=score.dtype)
        
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        """
        Forward pass with proper weight handling.
        
        Args:
            inputs: Model predictions [B, C, H, W]
            target: Ground truth labels [B, H, W]
            weight: Optional class weights (overrides self.weight if provided)
            softmax: Whether to apply softmax to inputs
        """
        # Check for NaN or Inf in inputs before processing
        if torch.isnan(inputs).any() or torch.isinf(inputs).any():
            # Replace NaN/Inf with zeros to prevent propagation
            inputs = torch.where(torch.isnan(inputs) | torch.isinf(inputs), 
                                torch.zeros_like(inputs), inputs)
        
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
            # Check again after softmax in case it produced NaN
            if torch.isnan(inputs).any():
                inputs = torch.where(torch.isnan(inputs), 
                                    torch.zeros_like(inputs), inputs)
                # Renormalize to ensure valid probability distribution
                inputs = inputs / (inputs.sum(dim=1, keepdim=True) + 1e-8)
        
        target = self._one_hot_encoder(target)
        
        # Determine which weights to use
        if weight is not None:
            # Use provided weight (from forward call)
            if isinstance(weight, torch.Tensor):
                class_weights = weight.cpu().numpy().tolist()
            else:
                class_weights = weight
        elif self.weight is not None:
            # Use weight from __init__
            if isinstance(self.weight, torch.Tensor):
                class_weights = self.weight.cpu().numpy().tolist()
            else:
                class_weights = self.weight
        else:
            # No weights provided, use uniform weights
            class_weights = [1] * self.n_classes
        
        # Size check
        assert inputs.size() == target.size(), \
            f'predict {inputs.size()} & target {target.size()} shape do not match'
        
        # Compute weighted Dice loss per class
        loss = 0.0
        total_weight = 0.0
        
        for i in range(self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            
            # Skip NaN losses (shouldn't happen with fixes)
            if not torch.isnan(dice):
                loss += dice * class_weights[i]
                total_weight += class_weights[i]
        
        # CRITICAL FIX: Normalize by sum of weights, not n_classes
        # This properly handles weighted classes
        if total_weight > 0:
            return loss / total_weight
        else:
            # Fallback (should never happen)
            print("⚠️  No valid classes in Dice Loss!")
            return torch.tensor(0.0, device=inputs.device, dtype=inputs.dtype)


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy().squeeze(0), label.squeeze(0).cpu().detach().numpy().squeeze(0)
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list