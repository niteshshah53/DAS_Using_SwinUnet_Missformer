import torch
import torch.nn as nn

from .efficientnet_encoder import EfficientNetEncoderWithAdapters
from .swin_decoder import SwinDecoder


# ============================================================================
# HYBRID MODEL: EfficientNet-B4 Encoder + Swin-Unet Decoder
# ============================================================================

class HybridEfficientNetB4SwinDecoder(nn.Module):
    """
    Hybrid model that combines EfficientNet-B4 encoder with Swin-Unet decoder.
    
    This model replaces the Swin-Unet encoder with an EfficientNet-B4 CNN encoder
    while keeping the Swin-Unet decoder intact for segmentation.
    
    Pipeline:
      1. EfficientNet-B4 backbone extracts multi-scale features (strides 4, 8, 16, 32)
      2. Channel adapters map CNN channels to Swin decoder expected dimensions [96, 192, 384, 768]
      3. Features are converted to token sequences and fed into Swin-Unet decoder
      4. Swin decoder performs upsampling with skip connections to produce segmentation masks
    
    Args:
        num_classes: Number of segmentation classes (4, 5, or 6, default: 6)
        img_size: Input image size (default: 224)
        pretrained: Whether to use pretrained EfficientNet weights (default: True)
    """
    
    def __init__(self, num_classes: int = 6, img_size: int = 224, pretrained: bool = True):
        super().__init__()
        
        # Validate num_classes - support 4, 5, and 6 classes
        if num_classes not in [4, 5, 6]:
            raise ValueError(f"num_classes must be 4, 5, or 6, got {num_classes}")
        
        self.num_classes = num_classes
        self.img_size = img_size
        
        # 1. EfficientNet-B4 encoder with channel adapters
        # Target dims follow Swin-Tiny progression used by the decoder
        target_dims = [96, 192, 384, 768]
        self.encoder = EfficientNetEncoderWithAdapters(
            target_dims=target_dims, 
            pretrained=pretrained
        )
        
        # 2. Swin-Unet decoder
        self.decoder = SwinDecoder(
            num_classes=num_classes, 
            img_size=img_size, 
            embed_dim=96
        )
        
        print(f"Hybrid model initialized:")
        print(f"  - Encoder: EfficientNet-B4 with adapters")
        print(f"  - Decoder: Swin-Unet with BOTTLENECK LAYER (2 SwinBlocks)")
        print(f"  - Input size: {img_size}x{img_size}")
        print(f"  - Output classes: {num_classes}")
        print(f"  - ✅ Architecture now matches original SwinUnet!")
    
    @staticmethod
    def _to_tokens(feat: torch.Tensor, out_dim: int) -> torch.Tensor:
        """
        Convert a feature map (B, C, H, W) to token sequence (B, H*W, out_dim).
        
        Args:
            feat: Feature map tensor of shape (B, C, H, W)
            out_dim: Expected output dimension (should match C)
            
        Returns:
            Token sequence of shape (B, H*W, out_dim)
        """
        b, c, h, w = feat.shape
        x = feat.flatten(2).transpose(1, 2)  # (B, H*W, C)
        assert c == out_dim, f"Adapter mismatch: got C={c}, expected {out_dim}"
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the hybrid model.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            
        Returns:
            Segmentation logits of shape (B, num_classes, H, W)
        """
        # Get 4 feature levels from EfficientNet encoder with adapters
        # feats: [P2 (stride 4), P3 (stride 8), P4 (stride 16), P5 (stride 32)]
        feats = self.encoder(x)
        assert len(feats) == 4, "Encoder did not return 4 feature levels"
        
        # Prepare tokens for decoder: bottom (P5) as current x, others as skip list
        p2, p3, p4, p5 = feats  # strides 4, 8, 16, 32
        
        # Convert to token sequences (B, L, C)
        target_dims = self.encoder.get_target_dims()
        x_tokens = self._to_tokens(p5, out_dim=target_dims[3])  # (B, 7*7, 768) for 224 input
        
        x_downsample = [
            self._to_tokens(p2, out_dim=target_dims[0]),  # (B, 56*56, 96)
            self._to_tokens(p3, out_dim=target_dims[1]),  # (B, 28*28, 192)
            self._to_tokens(p4, out_dim=target_dims[2]),  # (B, 14*14, 384)
            self._to_tokens(p5, out_dim=target_dims[3])   # (B, 7*7, 768)
        ]
        
        # ✅ BOTTLENECK PROCESSING: Process deepest features through 2 SwinBlocks
        # This was missing from the original hybrid implementation!
        x_tokens = self.decoder.forward_bottleneck(x_tokens)
        
        # Decode using Swin-Unet decoder path
        x_up = self.decoder.forward_up_features(x_tokens, x_downsample)
        logits = self.decoder.up_x4(x_up)
        
        return logits
    
    def get_model_info(self) -> dict:
        """Get model information for debugging and analysis."""
        return {
            'model_type': 'HybridEfficientNetB4SwinDecoder',
            'encoder': 'EfficientNet-B4',
            'decoder': 'Swin-Unet',
            'num_classes': self.num_classes,
            'img_size': self.img_size,
            'target_dims': self.encoder.get_target_dims(),
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


# ============================================================================
# MODEL FACTORY FUNCTION
# ============================================================================

def create_hybrid_model(num_classes: int = 6, img_size: int = 224, pretrained: bool = True) -> HybridEfficientNetB4SwinDecoder:
    """
    Factory function to create a hybrid EfficientNet-Swin model.
    
    Args:
        num_classes: Number of segmentation classes (4, 5, or 6)
        img_size: Input image size
        pretrained: Whether to use pretrained EfficientNet weights
        
    Returns:
        Initialized hybrid model
    """
    model = HybridEfficientNetB4SwinDecoder(
        num_classes=num_classes,
        img_size=img_size,
        pretrained=pretrained
    )
    
    return model


# ============================================================================
# CONVENIENCE ALIASES
# ============================================================================

# Main model class
HybridModel = HybridEfficientNetB4SwinDecoder

# Alternative names for compatibility
EfficientNetSwinUnet = HybridEfficientNetB4SwinDecoder
CNNTransformerHybrid = HybridEfficientNetB4SwinDecoder


# ============================================================================
# CONVENIENCE FACTORY FUNCTIONS FOR SPECIFIC CLASS COUNTS
# ============================================================================

def create_hybrid_4class(img_size: int = 224, pretrained: bool = True) -> HybridEfficientNetB4SwinDecoder:
    """Create hybrid model with 4 classes."""
    return create_hybrid_model(num_classes=4, img_size=img_size, pretrained=pretrained)


def create_hybrid_5class(img_size: int = 224, pretrained: bool = True) -> HybridEfficientNetB4SwinDecoder:
    """Create hybrid model with 5 classes."""
    return create_hybrid_model(num_classes=5, img_size=img_size, pretrained=pretrained)


def create_hybrid_6class(img_size: int = 224, pretrained: bool = True) -> HybridEfficientNetB4SwinDecoder:
    """Create hybrid model with 6 classes."""
    return create_hybrid_model(num_classes=6, img_size=img_size, pretrained=pretrained)
