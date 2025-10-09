# Hybrid Models for Historical Document Segmentation

This directory contains two hybrid models that combine different encoder-decoder architectures:

## Models

### Hybrid1: EfficientNet-Swin
- **Encoder**: EfficientNet-B4 (CNN-based)
- **Decoder**: SwinUnet (Transformer-based)
- **Architecture**: CNN Encoder + Transformer Decoder

### Hybrid2: Swin-EfficientNet  
- **Encoder**: SwinUnet (Transformer-based)
- **Decoder**: EfficientNet-style (CNN-based)
- **Architecture**: Transformer Encoder + CNN Decoder

## Directory Structure

```
hybrid/
├── hybrid1/                    # EfficientNet-Swin model
│   ├── hybrid_model.py         # Main model implementation
│   ├── efficientnet_encoder.py # EfficientNet-B4 encoder
│   └── swin_decoder.py         # SwinUnet decoder
├── hybrid2/                    # Swin-EfficientNet model
│   ├── hybrid_model.py         # Main model implementation
│   ├── swin_encoder.py         # SwinUnet encoder
│   └── efficientnet_decoder.py # EfficientNet-style decoder
├── train.py                    # Training script (supports both models)
├── test.py                     # Testing script (supports both models)
├── trainer.py                  # Training logic (shared)
├── run.sh                      # Run script for hybrid1
├── run_hybrid2.sh              # Run script for hybrid2
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
    --batch_size 16 \
    --max_epochs 300 \
    --base_lr 0.0002 \
    --patience 30 \
    --output_dir "./results/hybrid1_latin2"
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
    --batch_size 16 \
    --max_epochs 300 \
    --base_lr 0.0002 \
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

### Hybrid2 Specific Arguments
- `--efficientnet_variant`: EfficientNet variant for decoder (`b0`, `b4`, `b5`)

## Model Configurations

### Hybrid1 (EfficientNet-Swin)
- **Loss Function**: 0.4 * CE + 0.2 * Focal + 0.4 * Dice
- **Optimizer**: AdamW with weight_decay=0.01
- **Scheduler**: ReduceLROnPlateau
- **Early Stopping**: Yes (patience=30 epochs)

### Hybrid2 (Swin-EfficientNet)
- **Loss Function**: 0.4 * CE + 0.2 * Focal + 0.4 * Dice
- **Optimizer**: AdamW with weight_decay=0.01
- **Scheduler**: ReduceLROnPlateau
- **Early Stopping**: Yes (patience=30 epochs)
- **EfficientNet Variants**: B0 (lightweight), B4 (balanced), B5 (heavy)

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
# Run Hybrid1 on U-DIADS-Bib MS dataset
./run.sh

# Run Hybrid2 on U-DIADS-Bib MS dataset
./run_hybrid2.sh
```

### Custom Runs
```bash
# Train Hybrid1 with custom parameters
python3 train.py --model hybrid1 --manuscript Latin2 --batch_size 24

# Train Hybrid2 with EfficientNet-B5 decoder
python3 train.py --model hybrid2 --efficientnet_variant b5 --manuscript Latin2
```

## Model Comparison

| Aspect | Hybrid1 (EfficientNet-Swin) | Hybrid2 (Swin-EfficientNet) |
|--------|------------------------------|------------------------------|
| Encoder | EfficientNet-B4 (CNN) | SwinUnet (Transformer) |
| Decoder | SwinUnet (Transformer) | EfficientNet-style (CNN) |
| Parameters | ~50M | ~45M |
| Memory Usage | Moderate | Moderate |
| Training Speed | Fast | Moderate |
| Inference Speed | Fast | Fast |
| Best For | Quick training, good performance | Detailed feature extraction |

## Notes

- Both models use the same training approach with early stopping and class weights
- Hybrid2 supports different EfficientNet variants (B0, B4, B5) for the decoder
- All models support both U-DIADS-Bib and DIVAHISDB datasets
- Results are saved with model-specific naming to avoid conflicts
