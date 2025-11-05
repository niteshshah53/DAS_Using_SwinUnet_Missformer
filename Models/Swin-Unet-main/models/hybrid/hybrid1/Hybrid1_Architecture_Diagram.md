# Hybrid1 Model Architecture - Detailed Diagram

## Overview
Hybrid1 combines a **Streaming EfficientNet-B4 Encoder** with a **Swin-Unet Decoder** for historical document segmentation. The model uses immediate channel adaptation and tokenization after each encoder stage, making skip connections immediately ready for the decoder.

## Complete Architecture Flow

```
                    INPUT IMAGE (224×224×3 RGB)
                              │
                              ▼
                    ┌─────────────────────────┐
                    │ STREAMING EFFICIENTNET-B4│
                    │ ENCODER                 │
                    │                         │
                    │ ┌─────────────────────┐ │
                    │ │ EfficientNet-B4     │ │
                    │ │ Backbone            │ │
                    │ │ (timm)              │ │
                    │ │ out_indices=(1,2,3,4)│ │
                    │ └─────────────────────┘ │
                    │           │             │
                    │           ▼             │
                    │ ┌─────────────────────┐ │
                    │ │ STAGE 1: C1         │ │
                    │ │ Source: 32ch        │ │
                    │ │ Spatial: 56×56      │ │
                    │ │ Stride: 4           │ │
                    │ │                     │ │
                    │ │ Conv1×1 + GELU      │ │
                    │ │ 32ch → 96ch         │ │
                    │ │                     │ │
                    │ │ Tokenization        │ │
                    │ │ (B, 56×56, 96)      │ │
                    │ └─────────────────────┘ │
                    │           │             │
                    │           ▼             │
                    │ ┌─────────────────────┐ │
                    │ │ STAGE 2: C2         │ │
                    │ │ Source: 56ch        │ │
                    │ │ Spatial: 28×28      │ │
                    │ │ Stride: 8           │ │
                    │ │                     │ │
                    │ │ Conv1×1 + GELU      │ │
                    │ │ 56ch → 192ch        │ │
                    │ │                     │ │
                    │ │ Tokenization        │ │
                    │ │ (B, 28×28, 192)     │ │
                    │ └─────────────────────┘ │
                    │           │             │
                    │           ▼             │
                    │ ┌─────────────────────┐ │
                    │ │ STAGE 3: C3         │ │
                    │ │ Source: 160ch       │ │
                    │ │ Spatial: 14×14      │ │
                    │ │ Stride: 16          │ │
                    │ │                     │ │
                    │ │ Conv1×1 + GELU      │ │
                    │ │ 160ch → 384ch       │ │
                    │ │                     │ │
                    │ │ Tokenization        │ │
                    │ │ (B, 14×14, 384)     │ │
                    │ └─────────────────────┘ │
                    │           │             │
                    │           ▼             │
                    │ ┌─────────────────────┐ │
                    │ │ STAGE 4: C4         │ │
                    │ │ Source: 272ch       │ │
                    │ │ Spatial: 7×7        │ │
                    │ │ Stride: 32          │ │
                    │ │                     │ │
                    │ │ Conv1×1 + GELU      │ │
                    │ │ 272ch → 768ch       │ │
                    │ │                     │ │
                    │ │ Tokenization        │ │
                    │ │ (B, 7×7, 768)       │ │
                    │ └─────────────────────┘ │
                    └─────────────────────────┘
                              │
                              ▼
                    ┌─────────────────────────┐
                    │ SWIN-UNET DECODER       │
                    │                         │
                    │ ┌─────────────────────┐ │
                    │ │ BOTTLENECK LAYER    │ │
                    │ │                     │ │
                    │ │ Input: C4 Tokens    │ │
                    │ │ (B, 7×7, 768)       │ │
                    │ │                     │ │
                    │ │ Multi-Scale Agg     │ │
                    │ │ (Optional)          │ │
                    │ │ ┌─────────────────┐ │ │
                    │ │ │ Project C1→768  │ │ │
                    │ │ │ Project C2→768  │ │ │
                    │ │ │ Project C3→768  │ │ │
                    │ │ │ Project C4→768  │ │ │
                    │ │ │                 │ │ │
                    │ │ │ Fusion MLP      │ │ │
                    │ │ │ 768×4 → 768     │ │ │
                    │ │ └─────────────────┘ │ │
                    │ │                     │ │
                    │ │ 2× SwinBlocks       │ │
                    │ │ - WindowAttention   │ │
                    │ │ - MLP               │ │
                    │ │ - LayerNorm         │ │
                    │ │ - DropPath          │ │
                    │ │                     │ │
                    │ │ Output: (B,7×7,768) │ │
                    │ └─────────────────────┘ │
                    │           │             │
                    │           ▼             │
                    │ ┌─────────────────────┐ │
                    │ │ UPSAMPLING LAYER 1  │ │
                    │ │                     │ │
                    │ │ PatchExpand         │ │
                    │ │ 768ch → 384ch       │ │
                    │ │ 7×7 → 14×14         │ │
                    │ │                     │ │
                    │ │ SMART SKIP CONN     │ │
                    │ │ (Optional)          │ │
                    │ │ ┌─────────────────┐ │ │
                    │ │ │ Attention Fusion│ │ │
                    │ │ │ C3 + Decoder    │ │ │
                    │ │ │ 384ch + 384ch   │ │ │
                    │ │ │ → 384ch         │ │ │
                    │ │ └─────────────────┘ │ │
                    │ │                     │ │
                    │ │ OR Naive Concat     │ │
                    │ │ (Baseline)          │ │
                    │ │ C3 + Decoder        │ │
                    │ │ 384ch + 384ch       │ │
                    │ │ → 384ch             │ │
                    │ │                     │ │
                    │ │ 1× SwinBlock        │ │
                    │ │                     │ │
                    │ │ Aux Output 1        │ │
                    │ │ (Deep Supervision)   │ │
                    │ │ (B, 14×14, num_cls) │ │
                    │ └─────────────────────┘ │
                    │           │             │
                    │           ▼             │
                    │ ┌─────────────────────┐ │
                    │ │ UPSAMPLING LAYER 2  │ │
                    │ │                     │ │
                    │ │ PatchExpand         │ │
                    │ │ 384ch → 192ch       │ │
                    │ │ 14×14 → 28×28       │ │
                    │ │                     │ │
                    │ │ SMART SKIP CONN     │ │
                    │ │ (Optional)          │ │
                    │ │ ┌─────────────────┐ │ │
                    │ │ │ Attention Fusion│ │ │
                    │ │ │ C2 + Decoder    │ │ │
                    │ │ │ 192ch + 192ch   │ │ │
                    │ │ │ → 192ch         │ │ │
                    │ │ └─────────────────┘ │ │
                    │ │                     │ │
                    │ │ OR Naive Concat     │ │
                    │ │ (Baseline)          │ │
                    │ │ C2 + Decoder        │ │
                    │ │ 192ch + 192ch       │ │
                    │ │ → 192ch             │ │
                    │ │                     │ │
                    │ │ 2× SwinBlocks       │ │
                    │ │                     │ │
                    │ │ Aux Output 2        │ │
                    │ │ (Deep Supervision)   │ │
                    │ │ (B, 28×28, num_cls) │ │
                    │ └─────────────────────┘ │
                    │           │             │
                    │           ▼             │
                    │ ┌─────────────────────┐ │
                    │ │ UPSAMPLING LAYER 3  │ │
                    │ │                     │ │
                    │ │ PatchExpand         │ │
                    │ │ 192ch → 96ch        │ │
                    │ │ 28×28 → 56×56       │ │
                    │ │                     │ │
                    │ │ SMART SKIP CONN     │ │
                    │ │ (Optional)          │ │
                    │ │ ┌─────────────────┐ │ │
                    │ │ │ Attention Fusion│ │ │
                    │ │ │ C1 + Decoder    │ │ │
                    │ │ │ 96ch + 96ch     │ │ │
                    │ │ │ → 96ch          │ │ │
                    │ │ └─────────────────┘ │ │
                    │ │                     │ │
                    │ │ OR Naive Concat     │ │
                    │ │ (Baseline)          │ │
                    │ │ C1 + Decoder        │ │
                    │ │ 96ch + 96ch         │ │
                    │ │ → 96ch              │ │
                    │ │                     │ │
                    │ │ 2× SwinBlocks       │ │
                    │ │                     │ │
                    │ │ Aux Output 3        │ │
                    │ │ (Deep Supervision)   │ │
                    │ │ (B, 56×56, num_cls) │ │
                    │ └─────────────────────┘ │
                    │           │             │
                    │           ▼             │
                    │ ┌─────────────────────┐ │
                    │ │ UPSAMPLING LAYER 4  │ │
                    │ │                     │ │
                    │ │ PatchExpand         │ │
                    │ │ 96ch → 48ch         │ │
                    │ │ 56×56 → 112×112    │ │
                    │ │                     │ │
                    │ │ 2× SwinBlocks       │ │
                    │ └─────────────────────┘ │
                    │           │             │
                    │           ▼             │
                    │ ┌─────────────────────┐ │
                    │ │ FINAL UPSAMPLING    │ │
                    │ │                     │ │
                    │ │ FinalPatchExpand_X4 │ │
                    │ │ 48ch → 12ch         │ │
                    │ │ 112×112 → 224×224   │ │
                    │ │                     │ │
                    │ │ LayerNorm           │ │
                    │ └─────────────────────┘ │
                    │           │             │
                    │           ▼             │
                    │ ┌─────────────────────┐ │
                    │ │ SEGMENTATION HEAD   │ │
                    │ │                     │ │
                    │ │ Conv3×3 + BN + ReLU │ │
                    │ │ 12ch → 12ch         │ │
                    │ │                     │ │
                    │ │ Conv1×1             │ │
                    │ │ 12ch → num_classes  │ │
                    │ │                     │ │
                    │ │ Output: (B, num_cls,│ │
                    │ │          224, 224)   │ │
                    │ └─────────────────────┘ │
                    └─────────────────────────┘
                              │
                              ▼
                    ┌─────────────────────────┐
                    │ MAIN OUTPUT             │
                    │                         │
                    │ Segmentation Mask       │
                    │ (B, num_classes, 224, 224)│
                    │                         │
                    │ Classes:                │
                    │ - DIVAHISDB: 4 classes  │
                    │ - UDIADS_BIB: 5-6 classes│
                    └─────────────────────────┘

                    ┌─────────────────────────┐
                    │ AUXILIARY OUTPUTS       │
                    │ (Deep Supervision)      │
                    │                         │
                    │ Aux Output 1:           │
                    │ (B, num_classes, 14×14)  │
                    │                         │
                    │ Aux Output 2:           │
                    │ (B, num_classes, 28×28)  │
                    │                         │
                    │ Aux Output 3:           │
                    │ (B, num_classes, 56×56)  │
                    └─────────────────────────┘
```

## Key Components Detail

### 1. Streaming EfficientNet-B4 Encoder

**Pipeline**: `Stage → Channel Adaptation → Tokenization → Skip Ready`

- **EfficientNet-B4 Backbone**: Uses `timm` library with `features_only=True`
- **Source Channels**: [32, 56, 160, 272] (from EfficientNet-B4)
- **Target Channels**: [96, 192, 384, 768] (aligned with Swin-Tiny)
- **Channel Adapters**: Conv1×1 + GELU activation for each stage
- **Tokenization**: Flatten spatial dimensions to token sequence

### 2. Swin-Unet Decoder

**Bottleneck Layer**:
- **Input**: C4 tokens (B, 7×7, 768)
- **Multi-Scale Aggregation** (Optional): Projects all 4 scales to bottleneck dimension
- **Processing**: 2× SwinBlocks with WindowAttention + MLP
- **Output**: Enhanced bottleneck features

**Upsampling Layers** (4 stages):
- **PatchExpand**: Upsamples spatial resolution and adjusts channels
- **Smart Skip Connections** (Optional): Attention-based fusion with encoder features
- **SwinBlocks**: Window-based self-attention + MLP layers
- **Deep Supervision**: Auxiliary outputs at 3 scales

### 3. Smart Skip Connections (Optional Enhancement)

**Components**:
- **Channel Alignment**: Linear projection to match decoder dimensions
- **Self-Attention**: Enhances important skip features
- **Spatial Alignment**: Interpolation to match decoder resolution
- **Fusion MLP**: Concatenates and fuses encoder + decoder features

**Process**:
```
Encoder Skip → Align → Self-Attention → Spatial Align → Concat → Fusion MLP → Output
```

### 4. Multi-Scale Aggregation (Optional Enhancement)

**Process**:
- Projects all 4 encoder scales to bottleneck dimension (768)
- Resizes all features to bottleneck spatial size (7×7)
- Concatenates projected features
- Applies fusion MLP: 768×4 → 768
- Adds residual connection to original bottleneck features

### 5. Deep Supervision (Optional Enhancement)

**Auxiliary Outputs**:
- **Aux1**: After Layer 1 (B, num_classes, 14×14)
- **Aux2**: After Layer 2 (B, num_classes, 28×28)  
- **Aux3**: After Layer 3 (B, num_classes, 56×56)

**Processing**:
- LayerNorm + Linear projection for each auxiliary feature
- Upsampling to full resolution (224×224) for loss computation

## Model Configurations

### Baseline Configuration
- **Skip Connections**: Naive concatenation
- **Multi-Scale Aggregation**: Disabled
- **Deep Supervision**: Disabled
- **Smart Skip**: Disabled

### Enhanced Configuration (`--use_enhanced`)
- **Skip Connections**: Naive concatenation (baseline)
- **Multi-Scale Aggregation**: Enabled
- **Deep Supervision**: Enabled
- **Smart Skip**: Optional (`--use_smart_skip`)

## Data Flow Summary

1. **Input**: RGB image (224×224×3)
2. **Encoder**: EfficientNet-B4 extracts 4 feature scales
3. **Streaming Processing**: Each stage → adapt → tokenize → skip ready
4. **Bottleneck**: Process deepest features with optional multi-scale aggregation
5. **Decoder**: 4-stage upsampling with skip connections
6. **Output**: Main segmentation mask + optional auxiliary outputs

## Key Benefits

- **Memory Efficient**: Streaming processing eliminates intermediate storage
- **Immediate Adaptation**: Skip connections ready without additional processing
- **Flexible Enhancement**: Optional smart skip connections and deep supervision
- **Reference Compliant**: Follows standard segmentation head architecture
- **Multi-Scale Support**: Optional aggregation of all encoder scales

## Parameter Counts

- **Total Parameters**: ~38M
- **Encoder**: EfficientNet-B4 backbone + channel adapters
- **Decoder**: Swin-Unet with bottleneck layer
- **Enhancements**: +0.5% parameters for deep supervision
