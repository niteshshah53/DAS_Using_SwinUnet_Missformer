import torch
import torch.nn as nn

from .swin_encoder import SwinEncoder
from .efficientnet_decoder import ImprovedEfficientNetDecoder, create_improved_efficientnet_decoder


# ============================================================================
# HYBRID MODEL: SwinUnet Encoder + EfficientNet Decoder
# ============================================================================

class HybridSwinEncoderEfficientNetDecoder(nn.Module):
    """
    Hybrid model that combines SwinUnet encoder with EfficientNet decoder.
    
    This model replaces the SwinUnet decoder with an EfficientNet-style CNN decoder
    while keeping the SwinUnet encoder intact for feature extraction.
    
    Pipeline:
      1. SwinUnet encoder extracts multi-scale features (strides 4, 8, 16, 32)
      2. Features are converted to CNN format and fed into EfficientNet decoder
      3. EfficientNet decoder performs upsampling with skip connections to produce segmentation masks
    
    Args:
        num_classes: Number of segmentation classes (4, 5, or 6, default: 6)
        img_size: Input image size (default: 224)
        embed_dim: Swin encoder embedding dimension (default: 96)
        depths: Swin encoder layer depths (default: [2, 2, 2, 2])
        num_heads: Swin encoder attention heads (default: [3, 6, 12, 24])
        efficientnet_variant: EfficientNet decoder variant (default: 'b4')
    """
    
    def __init__(self, num_classes: int = 6, img_size: int = 224, embed_dim: int = 96,
                 depths: list = [2, 2, 2, 2], num_heads: list = [3, 6, 12, 24],
                 efficientnet_variant: str = 'b4', use_deep_supervision: bool = False):
        super().__init__()
        
        # Validate num_classes - support 4, 5, and 6 classes
        if num_classes not in [4, 5, 6]:
            raise ValueError(f"num_classes must be 4, 5, or 6, got {num_classes}")
        
        self.num_classes = num_classes
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.efficientnet_variant = efficientnet_variant
        self.use_deep_supervision = use_deep_supervision
        
        # 1. SwinUnet encoder
        self.encoder = SwinEncoder(
            img_size=img_size,
            patch_size=4,
            in_chans=3,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=7,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
            ape=False,
            patch_norm=True,
            use_checkpoint=False
        )
        
        # Get encoder channels for decoder initialization
        encoder_channels = self.encoder.get_channels()
        
        # 2. Improved EfficientNet decoder
        self.decoder = create_improved_efficientnet_decoder(
            encoder_channels=encoder_channels,
            num_classes=num_classes,
            variant=efficientnet_variant,
            use_deep_supervision=use_deep_supervision
        )
        
        print(f"Hybrid2 model initialized:")
        print(f"  - Encoder: SwinUnet (embed_dim={embed_dim}, depths={depths})")
        print(f"  - Decoder: Improved EfficientNet-{efficientnet_variant.upper()}")
        print(f"  - Input size: {img_size}x{img_size}")
        print(f"  - Output classes: {num_classes}")
        print(f"  - Encoder channels: {encoder_channels}")
        print(f"  - ✅ Architecture: SwinUnet Encoder + Improved EfficientNet Decoder!")
        print(f"  - 🚀 Improvements: CBAM Attention, Smart Skip Connections, Deep Decoder Blocks")
        if use_deep_supervision:
            print(f"  - 🎯 Deep Supervision: Enabled for better gradient flow")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the hybrid2 model.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            
        Returns:
            Segmentation logits of shape (B, num_classes, H, W)
        """
        # Get 4 feature levels from SwinUnet encoder
        # feats: [P1 (stride 4), P2 (stride 8), P3 (stride 16), P4 (stride 32)]
        encoder_features = self.encoder(x)
        assert len(encoder_features) == 4, "Encoder did not return 4 feature levels"
        
        # Decode using Improved EfficientNet decoder
        if self.use_deep_supervision:
            logits, aux_outputs = self.decoder(encoder_features)
            return logits, aux_outputs
        else:
            logits = self.decoder(encoder_features)
            return logits
    
    def get_model_info(self) -> dict:
        """Get model information for debugging and analysis."""
        encoder_info = {
            'encoder_type': 'SwinUnet',
            'embed_dim': self.embed_dim,
            'depths': self.encoder.layers,
            'num_heads': [layer.num_heads for layer in self.encoder.layers],
            'channels': self.encoder.get_channels(),
            'strides': self.encoder.get_strides()
        }
        
        decoder_info = self.decoder.get_model_info()
        
        return {
            'model_type': 'HybridSwinEncoderEfficientNetDecoder',
            'num_classes': self.num_classes,
            'img_size': self.img_size,
            'efficientnet_variant': self.efficientnet_variant,
            'encoder': encoder_info,
            'decoder': decoder_info,
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


# ============================================================================
# MODEL FACTORY FUNCTION
# ============================================================================

def create_hybrid2_model(num_classes: int = 6, img_size: int = 224, embed_dim: int = 96,
                        depths: list = [2, 2, 2, 2], num_heads: list = [3, 6, 12, 24],
                        efficientnet_variant: str = 'b4', use_deep_supervision: bool = False) -> HybridSwinEncoderEfficientNetDecoder:
    """
    Factory function to create a hybrid2 SwinUnet-EfficientNet model.
    
    Args:
        num_classes: Number of segmentation classes (4, 5, or 6)
        img_size: Input image size
        embed_dim: Swin encoder embedding dimension
        depths: Swin encoder layer depths
        num_heads: Swin encoder attention heads
        efficientnet_variant: EfficientNet decoder variant
        use_deep_supervision: Enable deep supervision for better training
        
    Returns:
        Initialized hybrid2 model
    """
    model = HybridSwinEncoderEfficientNetDecoder(
        num_classes=num_classes,
        img_size=img_size,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        efficientnet_variant=efficientnet_variant,
        use_deep_supervision=use_deep_supervision
    )
    
    return model


# ============================================================================
# CONVENIENCE ALIASES
# ============================================================================

# Main model class
Hybrid2Model = HybridSwinEncoderEfficientNetDecoder

# Alternative names for compatibility
SwinEfficientNetHybrid = HybridSwinEncoderEfficientNetDecoder
TransformerCNNHybrid = HybridSwinEncoderEfficientNetDecoder


# ============================================================================
# CONVENIENCE FACTORY FUNCTIONS FOR SPECIFIC CLASS COUNTS
# ============================================================================

def create_hybrid2_4class(img_size: int = 224, efficientnet_variant: str = 'b4') -> HybridSwinEncoderEfficientNetDecoder:
    """Create hybrid2 model with 4 classes."""
    return create_hybrid2_model(num_classes=4, img_size=img_size, efficientnet_variant=efficientnet_variant)


def create_hybrid2_5class(img_size: int = 224, efficientnet_variant: str = 'b4') -> HybridSwinEncoderEfficientNetDecoder:
    """Create hybrid2 model with 5 classes."""
    return create_hybrid2_model(num_classes=5, img_size=img_size, efficientnet_variant=efficientnet_variant)


def create_hybrid2_6class(img_size: int = 224, efficientnet_variant: str = 'b4') -> HybridSwinEncoderEfficientNetDecoder:
    """Create hybrid2 model with 6 classes."""
    return create_hybrid2_model(num_classes=6, img_size=img_size, efficientnet_variant=efficientnet_variant)


# ============================================================================
# EFFICIENTNET VARIANT FACTORY FUNCTIONS
# ============================================================================

def create_hybrid2_b0(num_classes: int = 6, img_size: int = 224) -> HybridSwinEncoderEfficientNetDecoder:
    """Create hybrid2 model with EfficientNet-B0 decoder."""
    return create_hybrid2_model(num_classes=num_classes, img_size=img_size, efficientnet_variant='b0')


def create_hybrid2_b4(num_classes: int = 6, img_size: int = 224) -> HybridSwinEncoderEfficientNetDecoder:
    """Create hybrid2 model with EfficientNet-B4 decoder."""
    return create_hybrid2_model(num_classes=num_classes, img_size=img_size, efficientnet_variant='b4')


def create_hybrid2_b5(num_classes: int = 6, img_size: int = 224) -> HybridSwinEncoderEfficientNetDecoder:
    """Create hybrid2 model with EfficientNet-B5 decoder."""
    return create_hybrid2_model(num_classes=num_classes, img_size=img_size, efficientnet_variant='b5')
