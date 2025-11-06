# Hybrid Models for Historical Document Segmentation

This directory contains two hybrid models that combine different encoder-decoder architectures:

## Models

### Hybrid1: EfficientNet-Swin
- **Encoder**: Streaming EfficientNet-B4 (CNN-based with immediate adaptation)
- **Decoder**: SwinUnet (Transformer-based)
- **Architecture**: Streaming CNN Encoder + Transformer Decoder
- **Enhanced Version**: Uses `--use_enhanced` flag for Deep Supervision + Multi-scale Aggregation

#### Hybrid1 Architecture Diagram (Streaming Approach)

```
                    Input Image (224×224 RGB)
                              │
                              ▼
                    ┌─────────────────────────┐
                    │ Streaming EfficientNet-B4│
                    │                         │
                    │ Stage 1: C1 → Adapt → Token (96ch)
                    │ Stage 2: C2 → Adapt → Token (192ch)  
                    │ Stage 3: C3 → Adapt → Token (384ch)
                    │ Stage 4: C4 → Adapt → Token (768ch)
                    └─────────────────────────┘
                              │
                              ▼
                    ┌─────────────────────────┐
                    │ Swin-Unet Decoder        │
                    │                         │
                    │ Bottleneck: 2× SwinBlocks│
                    │   └─ Uses: C4 Tokens (768ch)
                    │                         │
                    │ Upsampling Layer 1:      │
                    │   └─ Uses: C3 Tokens (384ch) Skip
                    │   └─ Aux Head 1 (Enhanced) │
                    │                         │
                    │ Upsampling Layer 2:      │
                    │   └─ Uses: C2 Tokens (192ch) Skip
                    │   └─ Aux Head 2 (Enhanced) │
                    │                         │
                    │ Upsampling Layer 3:      │
                    │   └─ Uses: C1 Tokens (96ch) Skip
                    │   └─ Aux Head 3 (Enhanced) │
                    └─────────────────────────┘
                              │
                              ▼
                    ┌─────────────────────────┐
                    │ Segmentation Head        │
                    │ Conv3×3 + ReLU + Conv1×1 │
                    └─────────────────────────┘
                              │
                              ▼
                    ┌─────────────────────────┐
                    │ Main Output Mask         │
                    │ (4-6 classes)           │
                    └─────────────────────────┘

Enhanced Version (--use_enhanced):
                    ┌─────────────────────────┐
                    │ Auxiliary Outputs       │
                    │                         │
                    │ Aux Output 1: 14×14     │
                    │ Aux Output 2: 28×28     │
                    │ Aux Output 3: 56×56     │
                    └─────────────────────────┘

Skip Connection Flow:
C1 Tokens (96ch)  ──────────────────────────────┐
                                                 ├─→ Upsampling Layer 3
                                                 ├─→ Aux Head 3 (Enhanced)
C2 Tokens (192ch) ──────────────────────────────┤
                                                 ├─→ Upsampling Layer 2  
                                                 ├─→ Aux Head 2 (Enhanced)
C3 Tokens (384ch) ──────────────────────────────┤
                                                 ├─→ Upsampling Layer 1
                                                 ├─→ Aux Head 1 (Enhanced)
C4 Tokens (768ch) ──────────────────────────────┘
                                                 ├─→ Bottleneck Processing

Key Benefits:
• Memory Efficient: No intermediate feature storage
• Immediate Processing: Adapt → Tokenize → Skip Ready
• Clean Architecture: Each stage processed independently
• Ready Skip Connections: Direct use in decoder
• Clear Skip Flow: Each decoder layer uses specific skip tokens
• Deep Supervision: 3 auxiliary outputs at different scales (Enhanced)
```

**Detailed Component Breakdown:**

**Streaming EfficientNet-B4 Encoder Components:**
- **Stem**: Conv3×3 + BN + Swish (stride 2)
- **MBConv Blocks**: Mobile inverted bottleneck convolutions
- **SE Modules**: Squeeze-and-Excitation attention
- **Streaming Processing**: Extract → Adapt → Tokenize → Skip Ready
- **Immediate Adaptation**: Channel adaptation applied after each stage
- **Ready Tokens**: Skip connections are immediately ready for decoder

**Swin-Unet Decoder Components:**
- **SwinTransformerBlock**: Window-based self-attention + MLP
- **BasicLayer_up**: Patch expanding + SwinTransformerBlocks
- **Skip Connections**: Direct use of ready tokenized features
- **Deep Supervision**: Auxiliary outputs at 3 scales (Enhanced version)

**Streaming Processing Details:**
- **Immediate Adaptation**: Channel adaptation applied after each stage
- **Conv1×1**: Linear projection for channel alignment
- **BatchNorm**: Normalization for stable training
- **GELU**: Gaussian Error Linear Unit activation
- **Tokenization**: Spatial flattening for transformer input
- **Memory Efficient**: No intermediate storage of raw features

### Hybrid2: Swin-EfficientNet
- **Encoder**: SwinUnet (Transformer-based)
- **Decoder**: Enhanced EfficientNet-style (CNN-based)
- **Architecture**: Transformer Encoder + CNN Decoder
- **Key Features**: CBAM Attention, Smart Skip Connections, Deep Decoder Blocks

#### Hybrid2 Architecture Diagram (U-Shaped)

```
                    Input Image (224×224 RGB)
                              │
                              ▼
                    ┌─────────────────────────┐
                    │ SwinUnet Encoder        │
                    │                         │
                    │  ┌─────────────────┐   │
                    │  │ Patch Embedding │   │
                    │  │ 4×4 patches     │   │
                    │  │ 96ch, 56×56     │   │
                    │  └─────────────────┘   │
                    │           │             │
                    │           ▼             │
                    │  ┌─────────────────┐   │
                    │  │ Stage 1         │   │
                    │  │ 96ch, 56×56     │   │
                    │  │ 2× SwinBlocks   │   │
                    │  └─────────────────┘   │
                    │           │             │
                    │           ▼             │
                    │  ┌─────────────────┐   │
                    │  │ Stage 2         │   │
                    │  │ 192ch, 28×28    │   │
                    │  │ 2× SwinBlocks   │   │
                    │  └─────────────────┘   │
                    │           │             │
                    │           ▼             │
                    │  ┌─────────────────┐   │
                    │  │ Stage 3         │   │
                    │  │ 384ch, 14×14    │   │
                    │  │ 2× SwinBlocks   │   │
                    │  └─────────────────┘   │
                    │           │             │
                    │           ▼             │
                    │  ┌─────────────────┐   │
                    │  │ Stage 4         │   │
                    │  │ 768ch, 7×7      │   │
                    │  │ 2× SwinBlocks   │   │
                    │  └─────────────────┘   │
                    └─────────────────────────┘
                              │
                              ▼
                    ┌─────────────────────────┐
                    │ Enhanced EfficientNet    │
                    │ Decoder                  │
                    │                         │
                    │  ┌─────────────────┐   │
                    │  │ Cross-Attention │   │
                    │  │ Bottleneck       │   │
                    │  │ Multi-Scale Agg  │   │
                    │  │ Cross-Attn       │   │
                    │  └─────────────────┘   │
                    │           │             │
                    │           ▼             │
                    │  ┌─────────────────┐   │
                    │  │ Deep Decoder    │   │
                    │  │ Block 1         │   │
                    │  │ CBAM Attention  │   │
                    │  │ Smart Skip: 384ch│   │
                    │  └─────────────────┘   │
                    │           │             │
                    │           ▼             │
                    │  ┌─────────────────┐   │
                    │  │ Deep Decoder    │   │
                    │  │ Block 2         │   │
                    │  │ CBAM Attention  │   │
                    │  │ Smart Skip: 192ch│   │
                    │  └─────────────────┘   │
                    │           │             │
                    │           ▼             │
                    │  ┌─────────────────┐   │
                    │  │ Deep Decoder    │   │
                    │  │ Block 3         │   │
                    │  │ CBAM Attention  │   │
                    │  │ Smart Skip: 96ch │   │
                    │  └─────────────────┘   │
                    └─────────────────────────┘
                              │
                              ▼
                    ┌─────────────────────────┐
                    │ Segmentation Head       │
                    │ Conv3×3 + ReLU + Conv1×1 │
                    └─────────────────────────┘
                              │
                              ▼
                    ┌─────────────────────────┐
                    │ Output Mask (4-6 classes)│
                    └─────────────────────────┘

Skip Connections (U-Shaped Flow):
Stage 1 (96ch) ──────────────────────────────────────┐
                                                     ├─→ Smart Skip Connections ──┐
Stage 2 (192ch) ───────────────────────────────────┤                           ├─→ CBAM Attention
                                                     ├─→ Smart Skip Connections ──┤
Stage 3 (384ch) ───────────────────────────────────┤                           ├─→ Feature Fusion
                                                     ├─→ Smart Skip Connections ──┤
Stage 4 (768ch) ───────────────────────────────────┘                           ├─→ Deep Decoder Blocks
                                                     │                           │
                                                     ▼                           ▼
                                             Cross-Attention Bottleneck    Upsampling Path
                                             Multi-Scale Aggregation      (3× Deep Decoder Blocks)
                                                     │                           │
                                                     ▼                           ▼
                                             Enhanced Feature Fusion      Final Segmentation
```

**Detailed Component Breakdown:**

**SwinUnet Encoder Components:**
- **Patch Embedding**: 4×4 patch projection to 96 channels
- **SwinTransformerBlock**: Window-based self-attention + MLP
- **Patch Merging**: Downsampling with channel expansion
- **Multi-Scale Features**: 4 stages with progressive downsampling

**Enhanced EfficientNet Decoder Components:**
- **Cross-Attention Bottleneck**: Multi-head attention between encoder/decoder
- **CBAM Attention**: Channel + Spatial attention modules
- **Smart Skip Connections**: Attention-based feature fusion
- **Deep Decoder Blocks**: Multi-layer convolutions with residual connections
- **Feature Refinement**: Gradual channel reduction with GroupNorm

**Advanced Features:**
- **Multi-Scale Aggregation**: Feature fusion at bottleneck
- **Positional Embeddings**: 2D sinusoidal position encoding
- **GroupNorm**: Normalization for better training stability
- **Residual Connections**: Skip connections for gradient flow

## Usage

## Directory Structure

```
hybrid/
├── hybrid1/                    # EfficientNet-Swin model
│   ├── hybrid_model.py         # Main model implementation
│   ├── efficientnet_encoder.py # EfficientNet-B4 encoder
│   └── swin_decoder.py         # SwinUnet decoder
├── hybrid2/                    # Swin-EfficientNet model
│   ├── model.py               # Main model classes (Hybrid2Enhanced, Hybrid2Baseline, etc.)
│   ├── components.py          # All building blocks (SwinEncoder, decoders, CBAM, etc.)
│   └── __init__.py            # Package initialization
├── train.py                    # Training script (supports both models)
├── test.py                     # Testing script (supports both models)
├── trainer.py                  # Training logic (shared, supports both models)
├── trainer_optuna.py          # Optuna-optimized training
├── optuna_tune.py              # Hyperparameter optimization
├── augmentation.py             # Advanced data augmentation
├── run.sh                      # Run script for hybrid1 (UDIADS_BIB MS)
├── run0.sh                     # Run script for hybrid1 enhanced (UDIADS_BIB MS)
├── run2.sh                     # Run script for hybrid1 (UDIADS_BIB FS)
├── run3.sh                     # Run script for hybrid1 (DIVAHISDB)
├── run_optuna.sh               # Run script for Optuna optimization
└── README.md                   # This file
```

## Usage

### Training

#### Hybrid1 (EfficientNet-Swin)
```bash
python3 train.py \
    --model hybrid1 \
    --dataset UDIADS_BIB \
    --udiadsbib_root "../../U-DIADS-Bib-MS_patched" \
    --manuscript Latin2 \
    --use_patched_data \
    --batch_size 4 \
    --max_epochs 300 \
    --base_lr 0.0003 \
    --patience 30 \
    --output_dir "./results/hybrid1_latin2"
```

#### Hybrid1 Enhanced (with Deep Supervision + Multi-scale Aggregation)
```bash
python3 train.py \
    --model hybrid1 \
    --use_enhanced \
    --dataset UDIADS_BIB \
    --udiadsbib_root "../../U-DIADS-Bib-MS_patched" \
    --manuscript Latin2 \
    --use_patched_data \
    --batch_size 4 \
    --max_epochs 300 \
    --base_lr 0.0003 \
    --patience 30 \
    --output_dir "./results/hybrid1_enhanced_latin2"
```

#### Hybrid2 (Swin-EfficientNet)
```bash
python3 train.py \
    --model hybrid2 \
    --efficientnet_variant b4 \
    --dataset UDIADS_BIB \
    --udiadsbib_root "../../U-DIADS-Bib-MS_patched" \
    --manuscript Latin2 \
    --use_patched_data \
    --batch_size 4 \
    --max_epochs 300 \
    --base_lr 0.0003 \
    --patience 30 \
    --output_dir "./results/hybrid2_latin2"
```

### Testing

#### Hybrid1
```bash
python3 test.py \
    --model hybrid1 \
    --dataset UDIADS_BIB \
    --udiadsbib_root "../../U-DIADS-Bib-MS_patched" \
    --manuscript Latin2 \
    --use_patched_data \
    --is_savenii \
    --output_dir "./results/hybrid1_latin2"
```

#### Hybrid1 Enhanced
```bash
python3 test.py \
    --model hybrid1 \
    --use_enhanced \
    --dataset UDIADS_BIB \
    --udiadsbib_root "../../U-DIADS-Bib-MS_patched" \
    --manuscript Latin2 \
    --use_patched_data \
    --is_savenii \
    --output_dir "./results/hybrid1_enhanced_latin2"
```

#### Hybrid2
```bash
python3 test.py \
    --model hybrid2 \
    --efficientnet_variant b4 \
    --dataset UDIADS_BIB \
    --udiadsbib_root "../../U-DIADS-Bib-MS_patched" \
    --manuscript Latin2 \
    --use_patched_data \
    --is_savenii \
    --use_tta \
    --output_dir "./results/hybrid2_latin2"
```

## Command Line Arguments

### Common Arguments
- `--model`: Model type (`hybrid1` or `hybrid2`)
- `--dataset`: Dataset to use (`UDIADS_BIB` or `DIVAHISDB`)
- `--manuscript`: Manuscript name (e.g., `Latin2`, `Syr341`)
- `--use_patched_data`: Use pre-generated patches
- `--batch_size`: Batch size for training/testing
- `--max_epochs`: Maximum number of training epochs
- `--base_lr`: Initial learning rate
- `--patience`: Early stopping patience
- `--output_dir`: Directory to save results

### Hybrid1 Specific Arguments
- `--use_enhanced`: Use Enhanced Hybrid1 with Deep Supervision + Multi-scale Aggregation

### Hybrid2 Specific Arguments
- `--efficientnet_variant`: EfficientNet variant for decoder (`b0`, `b4`, `b5`)
- `--use_transunet`: Use Enhanced Hybrid2 decoder (all best practices)
- `--use_efficientnet`: Use Enhanced EfficientNet decoder (Pure CNN + transformer-CNN hybrid improvements)

### Testing Arguments
- `--use_tta`: Enable Test-Time Augmentation for improved accuracy (+2-4% mIoU)
- `--is_savenii`: Save prediction visualizations

## Model Configurations

### Hybrid1 (EfficientNet-Swin)
- **Loss Function**: 0.5 * CE + 0.0 * Focal + 0.5 * Dice (with class weights)
- **Optimizer**: AdamW with differential learning rates
- **Scheduler**: CosineAnnealingWarmRestarts (T_0=50, T_mult=2)
- **Early Stopping**: Yes (patience=30 epochs)
- **Class Weights**: Computed from pixel frequency using inverse frequency weighting
- **Enhanced Version**: Deep Supervision + Multi-scale Aggregation (use `--use_enhanced`)

### Hybrid2 (Swin-EfficientNet)
- **Loss Function**: 0.5 * CE + 0.0 * Focal + 0.5 * Dice (with class weights)
- **Optimizer**: AdamW with differential learning rates
- **Scheduler**: CosineAnnealingWarmRestarts (T_0=50, T_mult=2)
- **Early Stopping**: Yes (patience=30 epochs)
- **Class Weights**: Computed from pixel frequency using inverse frequency weighting
- **EfficientNet Variants**: B0 (lightweight), B4 (balanced), B5 (heavy)
- **Enhanced Features**: CBAM Attention, Smart Skip Connections, Deep Decoder Blocks

## Supported Datasets

### U-DIADS-Bib
- **Classes**: 6 classes (5 for Syriaque341 manuscripts)
- **Classes**: Background, Paratext, Decoration, Main Text, Title, Chapter Headings
- **Note**: Syriaque341 manuscripts don't have Chapter Headings (5 classes)

### DIVAHISDB
- **Classes**: 4 classes
- **Classes**: Background, Comment, Decoration, Main Text

## Run Scripts

### Quick Start
```bash
# Run Hybrid1 Enhanced on U-DIADS-Bib MS dataset (Latin2, Latin14396, Latin16746, Syr341)
./run.sh

# Run Hybrid1 Enhanced on U-DIADS-Bib FS dataset (Latin2FS, Latin14396FS, Latin16746FS, Syr341FS)
./run2.sh

# Run Hybrid1 Enhanced on DIVAHISDB dataset (CB55, CSG18, CSG863)
./run3.sh
```

### Custom Runs
```bash
# Train Hybrid1 with custom parameters
python3 train.py --model hybrid1 --manuscript Latin2 --batch_size 4

# Train Hybrid1 Enhanced (recommended)
python3 train.py --model hybrid1 --use_enhanced --manuscript Latin2 --batch_size 4

# Train Hybrid2 with EfficientNet-B4 decoder (recommended)
python3 train.py --model hybrid2 --efficientnet_variant b4 --manuscript Latin2 --batch_size 4

# Test with TTA enabled for better accuracy
python3 test.py --model hybrid1 --use_enhanced --use_tta --manuscript Latin2
```

## Model Comparison

| Aspect | Hybrid1 | Hybrid1 Enhanced | Hybrid2 (Swin-EfficientNet) |
|--------|---------|------------------|------------------------------|
| Encoder | Streaming EfficientNet-B4 (CNN) | Streaming EfficientNet-B4 (CNN) | SwinUnet (Transformer) |
| Decoder | SwinUnet (Transformer) | SwinUnet (Transformer) + Deep Supervision | Enhanced EfficientNet-style (CNN) |
| Parameters | ~38M | ~38M (+0.5%) | ~45M |
| Memory Usage | Low (streaming) | Low (streaming) | Moderate |
| Training Speed | Fast | Fast | Moderate |
| Inference Speed | Fast | Fast | Fast |
| Special Features | Streaming processing, immediate adaptation | Streaming + Deep Supervision + Multi-scale Aggregation | CBAM Attention, Smart Skip Connections |
| Best For | Memory-efficient training, good performance | Memory-efficient training with auxiliary outputs | Enhanced feature extraction, attention mechanisms |
| Performance (DIVAHISDB) | Baseline | IoU: 67.96%, F1: 79.17% | Enhanced feature extraction |

## Recent Updates & Improvements

### Training Standardization (SwinUnet-aligned)
- **Class Weights**: Dynamic computation from pixel frequency using inverse frequency weighting (replaces manual weights)
- **Loss Function**: Unified 0.5 * CE + 0.0 * Focal + 0.5 * Dice with proper class weighting
- **Optimizer**: AdamW with differential learning rates
- **Scheduler**: CosineAnnealingWarmRestarts (T_0=50, T_mult=2) for improved transformer convergence
- **Early Stopping**: Increased patience to 30 epochs for better convergence

### Hybrid1 Enhancements
- **Enhanced Version**: Deep Supervision + Multi-scale Aggregation (use `--use_enhanced`)
- **Deep Supervision**: 3 auxiliary outputs at different scales for better training
- **Multi-Scale Aggregation**: Enhanced feature fusion in bottleneck
- **Streaming Encoder**: Immediate channel adaptation and tokenization after each stage
- **Memory Efficiency**: No intermediate storage of raw features, reduced memory footprint
- **Ready Skip Connections**: Skip connections are immediately ready for decoder

### Hybrid2 Enhancements
- **CBAM Attention**: Channel and spatial attention mechanisms for better feature focus
- **Smart Skip Connections**: Attention-based feature fusion instead of simple concatenation
- **Deep Decoder Blocks**: Multi-layer convolutions with attention for better reconstruction
- **Feature Refinement**: Gradual channel reduction with residual connections

### Testing Improvements
- **Test-Time Augmentation (TTA)**: Enabled by default for all run scripts (+2-4% mIoU improvement)
- **Enhanced Inference**: Better handling of edge cases and rare classes
- **Consistent Evaluation**: All datasets now use the same TTA pipeline

## Notes

- Both models use SwinUnet-aligned training approach with dynamic class weights computed using inverse frequency weighting
- Hybrid1 uses Streaming EfficientNet-B4 encoder with immediate adaptation and SwinUnet decoder
- Hybrid2 uses SwinUnet encoder with Enhanced EfficientNet-style decoder
- All models support both U-DIADS-Bib and DIVAHISDB datasets
- TTA is enabled by default in all run scripts for improved accuracy (+2-4% mIoU improvement)
- Results are saved with model-specific naming to avoid conflicts
- Trainer architecture matches SwinUnet for consistency across models
- Actual loss function weights: 0.5×CE + 0.0×Focal + 0.5×Dice (Focal loss disabled for stability)
- Default parameters: batch_size=4, base_lr=0.0003, patience=30 epochs
