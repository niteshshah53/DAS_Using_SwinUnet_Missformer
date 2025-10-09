# Historical Document Segmentation - Advanced Deep Learning Models

This repository contains state-of-the-art deep learning models for historical document segmentation, featuring transformer-based architectures and hybrid CNN-transformer combinations optimized for manuscript analysis.

## 📁 Directory Structure

```
Swin-Unet-main/
├── models/                          # Model-specific implementations
│   ├── sstrans/                     # Smart Swin Transformer
│   │   ├── train.py                 # SSTrans training script
│   │   ├── test.py                  # SSTrans testing script
│   │   ├── trainer.py               # SSTrans-specific trainer
│   │   ├── run.sh                   # SSTrans execution script
│   │   ├── Only_Smart.py            # Smart attention mechanism
│   │   ├── vision_transformer.py    # SSTrans model implementation
│   │   └── ...                      # Other SSTrans-specific files
│   ├── swinunet/                    # Swin Transformer U-Net
│   │   ├── train.py                 # SwinUnet training script
│   │   ├── test.py                  # SwinUnet testing script
│   │   ├── trainer.py               # SwinUnet-specific trainer
│   │   ├── run.sh                   # SwinUnet execution script
│   │   ├── swin_transformer_unet_skip_expand_decoder_sys.py  # Main model
│   │   └── ...                      # Other SwinUnet-specific files
│   ├── missformer/                  # MissFormer (Multi-scale Transformer)
│   │   ├── train.py                 # MissFormer training script
│   │   ├── test.py                  # MissFormer testing script
│   │   ├── trainer.py               # MissFormer-specific trainer
│   │   ├── run.sh                   # MissFormer execution script
│   │   ├── MISSFormer.py            # MissFormer model implementation
│   │   ├── segformer.py             # SegFormer backbone
│   │   └── ...                      # Other MissFormer-specific files
│   └── hybrid/                      # Hybrid CNN-Transformer Models
│       ├── hybrid1/                 # EfficientNet-Swin Hybrid
│       │   ├── hybrid_model.py      # Main hybrid model
│       │   ├── efficientnet_encoder.py  # EfficientNet-B4 encoder
│       │   └── swin_decoder.py      # SwinUnet decoder
│       ├── hybrid2/                 # Swin-EfficientNet Hybrid (Enhanced)
│       │   ├── hybrid_model.py      # Main hybrid model
│       │   ├── swin_encoder.py      # SwinUnet encoder
│       │   └── efficientnet_decoder.py  # Enhanced EfficientNet decoder
│       ├── train.py                 # Unified training script
│       ├── test.py                  # Unified testing script
│       ├── trainer.py               # Hybrid-specific trainer
│       ├── augmentation.py          # Advanced data augmentation
│       ├── run.sh                   # Hybrid1 execution script
│       ├── run_hybrid2.sh           # Hybrid2 execution script
│       └── README.md                # Hybrid models documentation
├── common/                          # Shared components
│   ├── datasets/                    # Dataset implementations
│   │   ├── dataset_udiadsbib.py     # U-DIADS-Bib dataset loader
│   │   ├── dataset_divahisdb.py     # DivaHisDB dataset loader
│   │   ├── dataset_synapse.py       # Synapse dataset loader
│   │   └── sstrans_transforms.py    # SSTrans-specific transforms
│   ├── utils/                       # Utility functions
│   │   └── utils.py                 # Common utilities (losses, metrics)
│   └── configs/                     # Configuration files
│       ├── config.py                # Configuration management
│       └── swin_tiny_patch4_window7_224_lite.yaml
├── run_all_models.sh               # Script to run all models
└── requirements.txt                # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Load Python environment (if using module system)
module load python/pytorch2.6py3.12
```

### Run Individual Models

Each model has its own execution script:

```bash
# SSTrans (Smart Swin Transformer with attention mechanisms)
cd models/sstrans
./run.sh

# SwinUnet (Standard Swin Transformer U-Net)
cd models/swinunet
./run.sh

# MissFormer (Multi-scale Transformer with SegFormer backbone)
cd models/missformer
./run.sh

# Hybrid1 (EfficientNet-B4 encoder + SwinUnet decoder)
cd models/hybrid
./run.sh

# Hybrid2 (SwinUnet encoder + Enhanced EfficientNet decoder)
cd models/hybrid
./run_hybrid2.sh
```

### Run All Models

```bash
# Run all models sequentially
./run_all_models.sh
```

### Custom Training

```bash
# Train Hybrid2 with custom parameters
cd models/hybrid
python3 train.py \
    --model hybrid2 \
    --efficientnet_variant b4 \
    --dataset UDIADS_BIB \
    --udiadsbib_root "../../U-DIADS-Bib-MS_patched" \
    --manuscript Latin2 \
    --use_patched_data \
    --batch_size 16 \
    --max_epochs 300 \
    --base_lr 0.0002 \
    --patience 30 \
    --output_dir "./results/hybrid2_latin2"
```

## 🔧 Model Architectures & Configurations

### SSTrans (Smart Swin Transformer)
- **Architecture**: Enhanced Swin Transformer with smart attention mechanisms
- **Key Features**: 
  - Smart attention masks for improved focus
  - Heavy data augmentation pipeline
  - Advanced normalization strategies
- **Training**: Standardized with validation and early stopping
- **Loss Function**: 0.4 * CE + 0.2 * Focal + 0.4 * Dice (no class weights)
- **Optimizer**: AdamW with weight_decay=0.01
- **Validation**: Sliding window on full images

### SwinUnet (Swin Transformer U-Net)
- **Architecture**: Standard Swin Transformer with U-Net decoder
- **Key Features**:
  - Skip connections between encoder and decoder
  - Patch merging and expanding operations
  - Window-based self-attention
- **Training**: Standardized with validation and early stopping
- **Loss Function**: 0.4 * CE + 0.0 * Focal + 0.6 * Dice (no class weights)
- **Optimizer**: AdamW with weight_decay=0.01
- **Validation**: Sliding window on full images

### MissFormer (Multi-scale Transformer)
- **Architecture**: SegFormer backbone with multi-scale feature fusion
- **Key Features**:
  - Efficient self-attention mechanisms
  - Multi-scale feature aggregation
  - Bridge layers for feature fusion
- **Training**: Advanced with class weights and sliding window validation
- **Loss Function**: 0.4 * CE + 0.0 * Focal + 0.6 * Dice (with class weights)
- **Optimizer**: AdamW with weight_decay=1e-4
- **Validation**: Advanced sliding window with mask conversion

### Hybrid1 (EfficientNet-Swin)
- **Architecture**: EfficientNet-B4 encoder + SwinUnet decoder
- **Key Features**:
  - CNN-based feature extraction
  - Transformer-based decoding
  - Channel adaptation layers
- **Training**: Standardized with conditional focal loss
- **Loss Function**: 0.4 * CE + 0.2 * Focal + 0.4 * Dice (no class weights)
- **Optimizer**: AdamW with weight_decay=0.01
- **Validation**: Standard DataLoader validation

### Hybrid2 (Swin-EfficientNet Enhanced)
- **Architecture**: SwinUnet encoder + Enhanced EfficientNet decoder
- **Key Features**:
  - **CBAM Attention**: Channel and spatial attention mechanisms
  - **Feature Refinement**: Gradual channel reduction with residual connections
  - **Smart Skip Connections**: Attention-based feature fusion
  - **Deep Decoder Blocks**: Multi-layer convolutions with attention
  - **Enhanced Augmentation**: Advanced data augmentation pipeline
- **Training**: Standardized with conditional focal loss
- **Loss Function**: 0.4 * CE + 0.2 * Focal + 0.4 * Dice (no class weights)
- **Optimizer**: AdamW with weight_decay=0.01
- **Validation**: Standard DataLoader validation
- **Variants**: B0 (lightweight), B4 (balanced), B5 (heavy)

## 📊 Supported Datasets

### U-DIADS-Bib
- **Description**: Historical manuscript segmentation dataset
- **Classes**: 6 classes (5 for Syriaque341 manuscripts)
- **Classes**: Background, Paratext, Decoration, Main Text, Title, Chapter Headings
- **Note**: Syriaque341 manuscripts don't have Chapter Headings (5 classes)
- **Format**: RGB color-coded masks
- **Usage**: `--dataset UDIADS_BIB --use_patched_data`

### DIVAHISDB
- **Description**: Historical document analysis dataset
- **Classes**: 4 classes
- **Classes**: Background, Comment, Decoration, Main Text
- **Format**: Bitmask-encoded masks
- **Usage**: `--dataset DIVAHISDB --use_patched_data`

### Synapse
- **Description**: Medical image segmentation dataset
- **Classes**: Variable (typically 9 classes)
- **Format**: HDF5 format
- **Usage**: `--dataset Synapse`

## 🔧 Key Benefits of Repository Structure

1. **Modularity**: Each model is self-contained with its own implementation
2. **Flexibility**: Easy to experiment with different architectures
3. **Maintainability**: Clear separation between model-specific and shared code
4. **Extensibility**: Simple to add new models or modify existing ones
5. **Reproducibility**: Consistent training and evaluation pipelines

## 📝 Adding New Models

To add a new model:

1. Create a new folder in `models/`
2. Copy the structure from an existing model (recommend starting with `hybrid/`)
3. Implement your model architecture
4. Modify the training/testing scripts
5. Update the common trainer if needed
6. Add the model to `run_all_models.sh`

## 🐛 Troubleshooting

### Common Issues

#### Import Errors
- Ensure you're running scripts from the correct directory
- Check that the common directory is in the Python path
- Verify Python environment is properly loaded (`module load python/pytorch2.6py3.12`)

#### Model-Specific Issues
- **SSTrans**: Requires config file (`--cfg` parameter)
- **Hybrid Models**: Check `--model` parameter (hybrid1 vs hybrid2)
- **MissFormer**: Verify SegFormer dependencies
- Check individual `trainer.py` files for model-specific logic

#### Training Issues
- **CUDA Memory**: Reduce batch size if encountering OOM errors
- **Data Loading**: Ensure dataset paths are correct and accessible
- **Checkpoints**: Verify checkpoint paths and model compatibility

#### Performance Issues
- **Slow Training**: Consider reducing image size or using mixed precision
- **Poor Convergence**: Adjust learning rate or try different optimizers
- **Overfitting**: Increase regularization or use more data augmentation

## 📈 Model Performance Comparison

| Model | Architecture | Parameters | Memory | Speed | Best For |
|-------|-------------|-----------|--------|-------|----------|
| SSTrans | Smart Swin Transformer | ~28M | Moderate | Fast | Attention-focused tasks |
| SwinUnet | Standard Swin U-Net | ~27M | Moderate | Fast | General segmentation |
| MissFormer | Multi-scale Transformer | ~30M | High | Moderate | Multi-scale features |
| Hybrid1 | EfficientNet-Swin | ~50M | Moderate | Fast | CNN-Transformer fusion |
| Hybrid2 | Swin-EfficientNet | ~45M | Moderate | Moderate | Enhanced feature extraction |

## 🔄 Recent Updates & Improvements

### Hybrid2 Enhancements (Latest)
- **CBAM Attention**: Channel and spatial attention mechanisms for better feature focus
- **Feature Refinement**: Gradual channel reduction with residual connections
- **Smart Skip Connections**: Attention-based feature fusion instead of simple concatenation
- **Deep Decoder Blocks**: Multi-layer convolutions with attention for better reconstruction
- **Enhanced Augmentation**: Advanced data augmentation pipeline with MixUp/CutMix

### Training Standardization
- **All Models**: AdamW optimizer with ReduceLROnPlateau scheduler
- **Early Stopping**: Consistent patience=30 epochs across all models
- **Validation**: Proper validation during training with sliding window for transformer models
- **Logging**: Improved TensorBoard logging and progress tracking
- **Checkpointing**: Automatic best model saving and cleanup

### Key Technical Improvements
1. **Attention Mechanisms**: CBAM and smart attention for better feature focus
2. **Skip Connections**: Intelligent fusion instead of simple concatenation
3. **Residual Learning**: Better gradient flow and training stability
4. **Multi-scale Processing**: Enhanced feature extraction at different scales
5. **Advanced Augmentation**: MixUp, CutMix, and sophisticated transforms

### Future Enhancements
- **Deep Supervision**: Optional auxiliary outputs for better training
- **Model Ensembling**: Combining multiple models for improved performance
- **Efficient Variants**: Lightweight versions for deployment
- **Cross-dataset Training**: Multi-dataset learning capabilities

This repository provides a comprehensive framework for historical document segmentation with state-of-the-art models and best practices.
