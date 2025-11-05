# CNN-Transformer Model for Historical Document Segmentation

This directory contains the CNN-Transformer model implementation for historical document segmentation, based on the EfficientNet-Swin Transformer U-Net architecture.

## Model Architecture

The CNN-Transformer model combines:
- **EfficientNet Encoder**: CNN-based feature extraction with pretrained weights
- **Swin Transformer Decoder**: Transformer-based decoding with window attention
- **Feature Adaptation**: Channel adaptation layers to bridge CNN and Transformer features
- **Skip Connections**: Native skip connections between encoder and decoder

## Key Features

- **Hybrid Architecture**: Combines CNN efficiency with Transformer attention
- **Pretrained Encoder**: Uses EfficientNet-B0/B4 pretrained weights
- **Progressive Training**: Supports layer-by-layer unfreezing
- **Multi-scale Processing**: Handles features at different resolutions
- **Flexible Configuration**: Easy to modify model parameters

## Files Structure

```
models/network/
├── cnn_transformer.py              # Core CNN-Transformer model implementation
├── vision_transformer_cnn.py       # Model wrapper for training/testing pipeline
├── train.py                        # Training script
├── test.py                         # Testing script
├── trainer.py                      # Training utilities
├── run_cnn_transformer.sh          # Training execution script
├── test_cnn_transformer.sh         # Testing execution script
└── README.md                       # This file
```

## Quick Start

### 1. Training

```bash
# Train CNN-Transformer model
cd models/network
./run_cnn_transformer.sh
```

### 2. Testing

```bash
# Test trained CNN-Transformer model
cd models/network
./test_cnn_transformer.sh
```

### 3. Manual Training

```bash
python3 train.py \
    --cfg "../../common/configs/swin_tiny_patch4_window7_224_lite.yaml" \
    --output_dir "./results/cnn_transformer_latin2_224" \
    --dataset UDIADS_BIB \
    --udiadsbib_root "../../U-DIADS-Bib-MS_patched" \
    --manuscript Latin2 \
    --use_patched_data \
    --batch_size 16 \
    --max_epochs 300 \
    --base_lr 0.0002 \
    --img_size 224 \
    --num_classes 6
```

### 4. Manual Testing

```bash
python3 test.py \
    --cfg "../../common/configs/swin_tiny_patch4_window7_224_lite.yaml" \
    --output_dir "./results/cnn_transformer_latin2_224" \
    --dataset UDIADS_BIB \
    --udiadsbib_root "../../U-DIADS-Bib-MS_patched" \
    --manuscript Latin2 \
    --use_patched_data \
    --img_size 224 \
    --num_classes 6 \
    --is_savenii
```

## Model Configuration

The CNN-Transformer model can be configured with the following parameters:

- **efficientnet_model**: 'efficientnet-b0' (lightweight) or 'efficientnet-b4' (balanced)
- **embed_dim**: Embedding dimension (default: 96)
- **depths_decoder**: Decoder layer depths (default: [2, 2, 2, 2])
- **num_heads**: Number of attention heads (default: [3, 6, 12, 24])
- **window_size**: Window size for attention (default: 7)
- **drop_rate**: Dropout rate (default: 0.0)
- **drop_path_rate**: Drop path rate (default: 0.1)

## Supported Datasets

- **U-DIADS-Bib**: Historical manuscript segmentation (6 classes)
- **DIVAHISDB**: Historical document analysis (4 classes)

## Training Features

- **Early Stopping**: Prevents overfitting with patience=30
- **Class Weights**: Automatic class weight computation for imbalanced datasets
- **Mixed Precision**: Optional mixed precision training
- **TensorBoard Logging**: Training progress visualization
- **Checkpoint Saving**: Automatic best model saving

## Performance

The CNN-Transformer model provides:
- **Efficient Training**: Faster than pure Transformer models
- **Good Accuracy**: Competitive performance on historical documents
- **Memory Efficient**: Lower memory usage than large Transformer models
- **Flexible**: Easy to modify and extend

## Dependencies

- PyTorch
- timm (for EfficientNet)
- efficientnet-pytorch
- einops
- torchvision
- numpy
- PIL

## Notes

- The model automatically handles single-channel input by replicating to 3 channels
- ImageNet normalization is applied for EfficientNet encoder
- The model supports both patched and original data formats
- Progressive training strategy is available for fine-tuning
