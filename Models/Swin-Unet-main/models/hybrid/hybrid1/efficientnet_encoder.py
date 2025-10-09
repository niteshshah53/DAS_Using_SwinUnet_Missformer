import torch
import torch.nn as nn
import timm


# ============================================================================
# EFFICIENTNET ENCODER
# ============================================================================

class Conv1x1BNAct(nn.Module):
    """1x1 Convolution with BatchNorm and GELU activation for channel adaptation."""
    
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EfficientNetEncoder(nn.Module):
    """
    EfficientNet-B4 encoder that extracts multi-scale features for segmentation.
    
    This encoder uses EfficientNet-B4 as backbone and provides 4 feature levels
    at different scales (strides 4, 8, 16, 32) suitable for U-Net style decoders.
    """
    
    def __init__(self, pretrained: bool = True):
        super().__init__()
        
        # Build EfficientNet-B4 backbone with features_only=True
        # This returns feature maps at 4 scales: strides 4, 8, 16, 32
        self.backbone = timm.create_model(
            'tf_efficientnet_b4_ns', 
            pretrained=pretrained, 
            features_only=True, 
            out_indices=(1, 2, 3, 4)  # Get features at indices 1,2,3,4 (strides 4,8,16,32)
        )
        
        # Get channel dimensions from the backbone
        self.channels = self.backbone.feature_info.channels()
        assert len(self.channels) == 4, f"Expected 4 feature levels, got {self.channels}"
        
        # Print feature info for debugging
        print(f"EfficientNet-B4 feature channels: {self.channels}")
        print(f"Feature strides: {[info['reduction'] for info in self.backbone.feature_info.get_dicts()]}")
    
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Forward pass through EfficientNet encoder.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            
        Returns:
            List of 4 feature maps at different scales:
            - feat[0]: (B, C1, H/4, W/4)   - stride 4
            - feat[1]: (B, C2, H/8, W/8)   - stride 8  
            - feat[2]: (B, C3, H/16, W/16) - stride 16
            - feat[3]: (B, C4, H/32, W/32) - stride 32
        """
        features = self.backbone(x)
        return features
    
    def get_channels(self) -> list[int]:
        """Get the number of channels for each feature level."""
        return self.channels
    
    def get_strides(self) -> list[int]:
        """Get the stride for each feature level."""
        return [info['reduction'] for info in self.backbone.feature_info.get_dicts()]


class EfficientNetEncoderWithAdapters(nn.Module):
    """
    EfficientNet encoder with channel adapters for Swin decoder compatibility.
    
    This version includes 1x1 conv adapters that map EfficientNet channels
    to the expected Swin decoder channel dimensions [96, 192, 384, 768].
    """
    
    def __init__(self, target_dims: list[int] = [96, 192, 384, 768], pretrained: bool = True):
        super().__init__()
        
        # Build EfficientNet encoder
        self.encoder = EfficientNetEncoder(pretrained=pretrained)
        
        # Get source channels from EfficientNet
        source_channels = self.encoder.get_channels()
        
        # Build channel adapters
        self.adapters = nn.ModuleList([
            Conv1x1BNAct(in_ch=source_channels[i], out_ch=target_dims[i]) 
            for i in range(4)
        ])
        
        self.target_dims = target_dims
        
        print(f"EfficientNet channels: {source_channels}")
        print(f"Target dimensions: {target_dims}")
    
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Forward pass with channel adaptation.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            
        Returns:
            List of 4 adapted feature maps with target channel dimensions
        """
        # Get features from EfficientNet
        features = self.encoder(x)
        
        # Adapt channels to target dimensions
        adapted_features = [self.adapters[i](features[i]) for i in range(4)]
        
        return adapted_features
    
    def get_target_dims(self) -> list[int]:
        """Get the target channel dimensions."""
        return self.target_dims
