import torch
import torch.nn as nn
import timm


# ============================================================================
# STREAMING EFFICIENTNET ENCODER
# ============================================================================

class Conv1x1BNAct(nn.Module):
    """1x1 Convolution with BatchNorm and GELU activation for channel adaptation."""
    
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, bias=False),
            #nn.BatchNorm2d(out_ch),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EfficientNetStreamingEncoder(nn.Module):
    """
    Streaming EfficientNet-B4 encoder that processes each stage individually.
    
    This encoder applies channel adaptation and tokenization immediately after each stage,
    making skip connections immediately ready for the decoder. This approach is more
    memory efficient and provides cleaner architecture.
    
    Pipeline:
    1. Stage 1: Extract C1 → Channel Adaptation → Tokenization → Skip Ready
    2. Stage 2: Extract C2 → Channel Adaptation → Tokenization → Skip Ready  
    3. Stage 3: Extract C3 → Channel Adaptation → Tokenization → Skip Ready
    4. Stage 4: Extract C4 → Channel Adaptation → Tokenization → Bottleneck Ready
    """
    
    def __init__(self, target_dims: list[int] = [96, 192, 384, 768], pretrained: bool = True):
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
        self.source_channels = self.backbone.feature_info.channels()
        assert len(self.source_channels) == 4, f"Expected 4 feature levels, got {self.source_channels}"
        
        # Channel adaptation layers for each stage
        self.channel_adapters = nn.ModuleList([
            Conv1x1BNAct(in_ch=self.source_channels[i], out_ch=target_dims[i]) 
            for i in range(4)  # C1, C2, C3, C4
        ])
        
        self.target_dims = target_dims
        
        print(f"✅ STREAMING ENCODER MODE:")
        print(f"   EfficientNet channels: {self.source_channels}")
        print(f"   Target dimensions: {target_dims}")
        print(f"   Channel adapters: {self.source_channels} → {target_dims}")
        print(f"   Processing: Stage → Adapt → Tokenize → Skip Ready")
    
    def _to_tokens(self, feat: torch.Tensor, out_dim: int) -> torch.Tensor:
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
    
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Streaming forward pass: process each stage individually.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            
        Returns:
            List of 4 tokenized feature maps ready for skip connections:
            - tokens[0]: (B, 56*56, 96)   - C1 tokens (stride 4)
            - tokens[1]: (B, 28*28, 192)  - C2 tokens (stride 8)
            - tokens[2]: (B, 14*14, 384)  - C3 tokens (stride 16)
            - tokens[3]: (B, 7*7, 768)    - C4 tokens (stride 32)
        """
        # Get all features from EfficientNet backbone
        features = self.backbone(x)
        
        # Process each stage individually: Extract → Adapt → Tokenize
        tokenized_features = []
        for i, (feat, adapter) in enumerate(zip(features, self.channel_adapters)):
            # Channel adaptation
            adapted_feat = adapter(feat)
            
            # Tokenization
            tokens = self._to_tokens(adapted_feat, self.target_dims[i])
            tokenized_features.append(tokens)
        
        return tokenized_features
    
    def get_target_dims(self) -> list[int]:
        """Get the target channel dimensions."""
        return self.target_dims
    
    def get_source_channels(self) -> list[int]:
        """Get the source channel dimensions."""
        return self.source_channels