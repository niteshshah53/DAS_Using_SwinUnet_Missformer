import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# IMPROVED COMPONENTS
# ============================================================================

class ChannelAttention(nn.Module):
    """Channel Attention Module to weight feature importance."""
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out


class SpatialAttention(nn.Module):
    """Spatial Attention Module to focus on important regions."""
    
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(out))
        return x * out


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention()
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class FeatureRefinementBlock(nn.Module):
    """
    Enhanced feature adaptation with:
    - Gradual channel reduction
    - Batch normalization
    - Residual connections
    - Attention mechanisms
    """
    
    def __init__(self, in_channels, out_channels, use_attention=True):
        super().__init__()
        
        mid_channels = (in_channels + out_channels) // 2
        
        self.adapt = nn.Sequential(
            # First stage: in_channels -> mid_channels
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            
            # Second stage: mid_channels -> out_channels
            nn.Conv2d(mid_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Optional attention
        self.attention = CBAM(out_channels) if use_attention else nn.Identity()
        
        # Residual connection if dimensions match
        self.use_residual = (in_channels == out_channels)
        if not self.use_residual and in_channels > out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
    
    def forward(self, x):
        identity = x
        
        out = self.adapt(x)
        out = self.attention(out)
        
        # Add residual connection
        if self.use_residual:
            out = out + identity
        elif hasattr(self, 'residual_conv'):
            out = out + self.residual_conv(identity)
        
        return out


class EnhancedConvBlock(nn.Module):
    """Enhanced convolution block with residual connection."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Residual connection
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(identity)
        
        out += identity
        out = self.relu(out)
        
        return out


class SmartSkipConnection(nn.Module):
    """
    Intelligent skip connection that:
    - Aligns feature dimensions
    - Applies attention to highlight important features
    - Uses fusion instead of simple concatenation
    """
    
    def __init__(self, encoder_channels, decoder_channels, fusion_type='concat'):
        super().__init__()
        
        self.fusion_type = fusion_type
        
        # Align encoder features to decoder channels
        self.align = nn.Sequential(
            nn.Conv2d(encoder_channels, decoder_channels, 1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True)
        )
        
        # Attention for skip features
        self.attention = CBAM(decoder_channels)
        
        if fusion_type == 'add':
            self.fuse = None
        elif fusion_type == 'concat':
            self.fuse = nn.Sequential(
                nn.Conv2d(decoder_channels * 2, decoder_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(decoder_channels),
                nn.ReLU(inplace=True)
            )
        elif fusion_type == 'attention_fusion':
            # Learnable fusion weights
            self.fuse = nn.Sequential(
                nn.Conv2d(decoder_channels * 2, decoder_channels, 1, bias=False),
                nn.BatchNorm2d(decoder_channels),
                nn.Sigmoid()
            )
    
    def forward(self, encoder_feat, decoder_feat):
        # Align and enhance encoder features
        skip_feat = self.align(encoder_feat)
        skip_feat = self.attention(skip_feat)
        
        if self.fusion_type == 'add':
            return decoder_feat + skip_feat
        elif self.fusion_type == 'concat':
            fused = torch.cat([decoder_feat, skip_feat], dim=1)
            return self.fuse(fused)
        elif self.fusion_type == 'attention_fusion':
            weights = self.fuse(torch.cat([decoder_feat, skip_feat], dim=1))
            return decoder_feat * weights + skip_feat * (1 - weights)


class DeepDecoderBlock(nn.Module):
    """
    Deep decoder block with:
    - Multiple convolution layers
    - Residual connections
    - Attention mechanisms
    """
    
    def __init__(self, in_channels, out_channels, num_blocks=2):
        super().__init__()
        
        blocks = []
        for i in range(num_blocks):
            if i == 0:
                blocks.append(EnhancedConvBlock(in_channels, out_channels))
            else:
                blocks.append(EnhancedConvBlock(out_channels, out_channels))
        
        self.blocks = nn.Sequential(*blocks)
        self.attention = CBAM(out_channels)
    
    def forward(self, x):
        x = self.blocks(x)
        x = self.attention(x)
        return x


# ============================================================================
# IMPROVED EFFICIENTNET DECODER
# ============================================================================

class ImprovedEfficientNetDecoder(nn.Module):
    """
    Enhanced EfficientNet-style decoder with:
    - Better feature adaptation (gradual channel reduction)
    - Attention mechanisms (CBAM)
    - Smart skip connections with feature fusion
    - Deeper decoder blocks
    - Multi-scale feature aggregation
    """
    
    def __init__(self, encoder_channels, num_classes=6, decoder_channels=[256, 128, 64, 32],
                 use_deep_supervision=False):
        super().__init__()
        
        self.num_classes = num_classes
        self.encoder_channels = encoder_channels  # [96, 192, 384, 768]
        self.decoder_channels = decoder_channels  # [256, 128, 64, 32]
        self.use_deep_supervision = use_deep_supervision
        
        # Enhanced feature adaptation with attention
        self.encoder_refinement = nn.ModuleList([
            FeatureRefinementBlock(enc_ch, dec_ch, use_attention=True)
            for enc_ch, dec_ch in zip(encoder_channels, decoder_channels)
        ])
        
        # Smart skip connections
        self.skip_connections = nn.ModuleList([
            SmartSkipConnection(dec_ch, dec_ch, fusion_type='attention_fusion')
            for dec_ch in decoder_channels[:-1]
        ])
        
        # Deep decoder blocks with upsampling
        # Block 1: H/32 -> H/16 (deepest)
        self.decoder1 = nn.Sequential(
            DeepDecoderBlock(decoder_channels[3], decoder_channels[3], num_blocks=2),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(decoder_channels[3], decoder_channels[2], 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels[2]),
            nn.ReLU(inplace=True)
        )
        
        # Block 2: H/16 -> H/8
        self.decoder2 = nn.Sequential(
            DeepDecoderBlock(decoder_channels[2], decoder_channels[2], num_blocks=2),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(decoder_channels[2], decoder_channels[1], 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels[1]),
            nn.ReLU(inplace=True)
        )
        
        # Block 3: H/8 -> H/4
        self.decoder3 = nn.Sequential(
            DeepDecoderBlock(decoder_channels[1], decoder_channels[1], num_blocks=2),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(decoder_channels[1], decoder_channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels[0]),
            nn.ReLU(inplace=True)
        )
        
        # Block 4: H/4 -> H (final upsampling)
        self.decoder4 = nn.Sequential(
            DeepDecoderBlock(decoder_channels[0], decoder_channels[0], num_blocks=2),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(decoder_channels[0], decoder_channels[0] // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels[0] // 2),
            nn.ReLU(inplace=True)
        )
        
        # Final classification with refinement
        self.classifier = nn.Sequential(
            EnhancedConvBlock(decoder_channels[0] // 2, decoder_channels[0] // 4),
            nn.Conv2d(decoder_channels[0] // 4, num_classes, 1, bias=True)
        )
        
        # Optional deep supervision heads
        if use_deep_supervision:
            self.aux_heads = nn.ModuleList([
                nn.Conv2d(dec_ch, num_classes, 1) 
                for dec_ch in decoder_channels
            ])
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, encoder_features):
        """
        Forward pass with improved feature processing.
        
        Args:
            encoder_features: [feat0, feat1, feat2, feat3] from Swin encoder
                            Strides: [4, 8, 16, 32]
        
        Returns:
            Main segmentation logits (and auxiliary outputs if deep supervision)
        """
        # Refine encoder features
        refined_features = []
        for feat, refine_block in zip(encoder_features, self.encoder_refinement):
            refined_feat = refine_block(feat)
            refined_features.append(refined_feat)
        
        aux_outputs = [] if self.use_deep_supervision else None
        
        # Start with deepest feature (stride 32)
        x = refined_features[3]  # (B, 32, H/32, W/32)
        
        if self.use_deep_supervision:
            aux_outputs.append(self.aux_heads[3](x))
        
        # Decoder path with smart skip connections
        # Block 1: H/32 -> H/16
        x = self.decoder1(x)
        x = self.skip_connections[2](refined_features[2], x)
        
        if self.use_deep_supervision:
            aux_outputs.append(self.aux_heads[2](x))
        
        # Block 2: H/16 -> H/8
        x = self.decoder2(x)
        x = self.skip_connections[1](refined_features[1], x)
        
        if self.use_deep_supervision:
            aux_outputs.append(self.aux_heads[1](x))
        
        # Block 3: H/8 -> H/4
        x = self.decoder3(x)
        x = self.skip_connections[0](refined_features[0], x)
        
        if self.use_deep_supervision:
            aux_outputs.append(self.aux_heads[0](x))
        
        # Block 4: H/4 -> H
        x = self.decoder4(x)
        
        # Final classification
        logits = self.classifier(x)
        
        if self.use_deep_supervision:
            return logits, aux_outputs
        return logits
    
    def get_model_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'ImprovedEfficientNetDecoder',
            'encoder_channels': self.encoder_channels,
            'decoder_channels': self.decoder_channels,
            'num_classes': self.num_classes,
            'use_deep_supervision': self.use_deep_supervision,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'improvements': [
                'Feature Refinement Blocks',
                'CBAM Attention',
                'Smart Skip Connections',
                'Deep Decoder Blocks',
                'Residual Connections',
                'Optional Deep Supervision'
            ]
        }


# ============================================================================
# BACKWARD COMPATIBILITY
# ============================================================================

# Keep original EfficientNetDecoder for backward compatibility
class EfficientNetDecoder(nn.Module):
    """
    Original EfficientNet-style decoder (kept for backward compatibility).
    Use ImprovedEfficientNetDecoder for better performance.
    """
    
    def __init__(self, encoder_channels, num_classes=6, decoder_channels=[256, 128, 64, 32]):
        super().__init__()
        
        self.num_classes = num_classes
        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels
        
        # Channel adaptation layers for encoder features
        self.encoder_adapters = nn.ModuleList([
            nn.Conv2d(enc_ch, dec_ch, 1, bias=False) 
            for enc_ch, dec_ch in zip(encoder_channels, decoder_channels)
        ])
        
        # Simple decoder blocks
        self.decoder1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(decoder_channels[3], decoder_channels[2], 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels[2]),
            nn.ReLU(inplace=True)
        )
        
        self.decoder2 = nn.Sequential(
            nn.Conv2d(decoder_channels[2] + decoder_channels[2], decoder_channels[1], 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels[1]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        self.decoder3 = nn.Sequential(
            nn.Conv2d(decoder_channels[1] + decoder_channels[1], decoder_channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels[0]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        self.decoder4 = nn.Sequential(
            nn.Conv2d(decoder_channels[0] + decoder_channels[0], decoder_channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels[0]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(decoder_channels[0], decoder_channels[0] // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels[0] // 2),
            nn.ReLU(inplace=True)
        )
        
        # Final classification head
        self.classifier = nn.Sequential(
            nn.Conv2d(decoder_channels[0] // 2, decoder_channels[0] // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels[0] // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels[0] // 4, num_classes, 1, bias=True)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, encoder_features):
        # Adapt encoder features to decoder channels
        adapted_features = []
        for feat, adapter in zip(encoder_features, self.encoder_adapters):
            adapted_feat = adapter(feat)
            adapted_features.append(adapted_feat)
        
        # Start with deepest feature (stride 32)
        x = adapted_features[3]
        
        # Decoder path with skip connections
        x = self.decoder1(x)
        
        skip_feat = adapted_features[2]
        x = torch.cat([x, skip_feat], dim=1)
        x = self.decoder2(x)
        
        skip_feat = adapted_features[1]
        x = torch.cat([x, skip_feat], dim=1)
        x = self.decoder3(x)
        
        skip_feat = adapted_features[0]
        x = torch.cat([x, skip_feat], dim=1)
        x = self.decoder4(x)
        
        # Final classification
        logits = self.classifier(x)
        
        return logits
    
    def get_model_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'EfficientNetDecoder',
            'encoder_channels': self.encoder_channels,
            'decoder_channels': self.decoder_channels,
            'num_classes': self.num_classes,
            'total_params': total_params,
            'trainable_params': trainable_params
        }


# ============================================================================
# CONVENIENCE FACTORY FUNCTIONS
# ============================================================================

def create_improved_efficientnet_decoder(encoder_channels, num_classes=6, 
                                        variant='b4', use_deep_supervision=False):
    """
    Factory function for improved decoder.
    
    Args:
        encoder_channels: Channels from encoder
        num_classes: Number of classes
        variant: Decoder size (b0/b4/b5)
        use_deep_supervision: Enable auxiliary outputs
    """
    if variant == 'b0':
        decoder_channels = [128, 64, 32, 16]
    elif variant == 'b4':
        decoder_channels = [256, 128, 64, 32]
    elif variant == 'b5':
        decoder_channels = [512, 256, 128, 64]
    else:
        raise ValueError(f"Unsupported variant: {variant}")
    
    return ImprovedEfficientNetDecoder(
        encoder_channels=encoder_channels,
        num_classes=num_classes,
        decoder_channels=decoder_channels,
        use_deep_supervision=use_deep_supervision
    )


def create_efficientnet_decoder(encoder_channels, num_classes=6, variant='b4'):
    """
    Factory function for original decoder (backward compatibility).
    
    Args:
        encoder_channels: Channels from encoder
        num_classes: Number of classes
        variant: Decoder size (b0/b4/b5)
    """
    if variant == 'b0':
        decoder_channels = [128, 64, 32, 16]
    elif variant == 'b4':
        decoder_channels = [256, 128, 64, 32]
    elif variant == 'b5':
        decoder_channels = [512, 256, 128, 64]
    else:
        raise ValueError(f"Unsupported variant: {variant}")
    
    return EfficientNetDecoder(
        encoder_channels=encoder_channels,
        num_classes=num_classes,
        decoder_channels=decoder_channels
    )