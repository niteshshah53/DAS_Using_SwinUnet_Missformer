#!/bin/bash -l
#SBATCH --job-name=1st      
#SBATCH --output=./Result/a1/test_baseline_%j.out
#SBATCH --error=./Result/a1/test_baseline_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=22:00:00
#SBATCH --gres=gpu:1

#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

# Load modules
module purge
module load python/pytorch2.6py3.12
module load cuda/11.8
module load cudnn

conda activate pytorch2.6-py3.12

# Add user site-packages to PYTHONPATH to find user-installed packages like pydensecrf2
export PYTHONPATH="${HOME}/.local/lib/python3.12/site-packages:${PYTHONPATH}"

# Memory optimization: Reduce CUDA memory fragmentation
# This helps prevent OOM errors during TTA (Test-Time Augmentation) which processes 4 augmentations at once
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============================================================================
# BASE NETWORK MODEL CONFIGURATION (BEST RESULT: F1=0.7266)
# ============================================================================
# Base Model Configuration (minimal components, no extra enhancements):
#   ✓ EfficientNet-B4 Encoder
#   ✓ Bottleneck: 2 Swin Transformer blocks (enabled)
#   ✓ Swin Transformer Decoder
#   ✓ Fusion Method: simple (concatenation)
#   ✓ Adapter mode: streaming (integrated adapters)
#   ✓ GroupNorm: enabled
#   ✓ Loss functions: CE (weighted) + Focal (γ=2.0) + Dice
#   ✓ Differential LR: Encoder (0.05x), Bottleneck (1.0x), Decoder (1.0x)
#   ✗ Balanced Sampler: DISABLED (was causing performance degradation)
#   ✗ Class-Aware Augmentation: DISABLED (was causing performance degradation)
#
# Best Configuration (from baseline_1327314.out):
#   - Batch Size: 32 (NOT 12)
#   - NO balanced sampler
#   - NO class-aware augmentation
#   - Focal gamma: 2.0
#   - Result: F1=0.7266, Val Loss=0.3701
#
# Components Disabled (base model):
#   ✗ Deep Supervision
#   ✗ Fourier Feature Fusion
#   ✗ Smart Skip Connections
#   ✗ Multi-Scale Aggregation
# ============================================================================

echo "============================================================================"
echo "CNN-TRANSFORMER BASE NETWORK MODEL"
echo "============================================================================"
echo "Configuration: CNN-TRANSFORMER BASE MODEL (No Extra Components)"
echo ""
echo "Component Details:"
echo "  ✓ EfficientNet-B4 Encoder"
echo "  ✓ Bottleneck: 2 Swin Transformer blocks (enabled)"
echo "  ✓ Swin Transformer Decoder"
echo "  ✓ Fusion Method: simple (concatenation)"
echo "  ✓ Adapter mode: streaming (integrated)"
echo "  ✓ GroupNorm: enabled"
echo "  ✗ Balanced Sampler: DISABLED (causes performance degradation)"
echo "  ✗ Class-Aware Augmentation: DISABLED (causes performance degradation)"
echo "  ✓ Loss: CE (weighted) + Focal (γ=2.0) + Dice"
echo "  ✗ Deep Supervision: disabled (base model)"
echo "  ✗ Multi-Scale Aggregation: disabled (base model)"
echo "  ✗ Fourier Feature Fusion: disabled (using simple fusion)"
echo "  ✗ Smart Skip Connections: disabled (using simple fusion)"
echo ""
echo "Training Parameters:"
echo "  - Batch Size: 32 (best result configuration)"
echo "  - Max Epochs: 300"
echo "  - Learning Rate: 0.0001"
echo "  - Scheduler: CosineAnnealingWarmRestarts"
echo "  - Early Stopping: 150 epochs patience"
echo ""
echo "Best Result Configuration (F1=0.7266):"
echo "  ✓ Batch size: 32 (NOT 12)"
echo "  ✓ NO balanced sampler"
echo "  ✓ NO class-aware augmentation"
echo "  ✓ Focal gamma: 2.0"
echo "============================================================================"
echo ""

# Test all manuscripts one by one
MANUSCRIPTS=(Latin2 Latin14396 Latin16746 Syr341) 

for MANUSCRIPT in "${MANUSCRIPTS[@]}"; do
    echo ""
    echo "╔════════════════════════════════════════════════════════════════════════╗"
    echo "║  TESTING CNN-TRANSFORMER BASE MODEL: $MANUSCRIPT"
    echo "╚════════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Configuration: CNN-TRANSFORMER BASE MODEL"
    echo "Output Directory: ./Result/a1/${MANUSCRIPT}"
    echo ""
    echo "Test Configuration:"
    echo "  ✓ Test-Time Augmentation (TTA): ENABLED"
    echo "  ✗ CRF Post-processing: DISABLED"
    echo "  - Batch Size: 1 (reduced for TTA memory efficiency)"
    echo ""
    
    # Use batch_size=1 for testing to avoid OOM with TTA (4 augmentations per patch = 4x memory)
    # TTA processes 4 augmentations at once, so batch_size=1 means 4 patches in memory simultaneously
    python3 test.py \
        --dataset UDIADS_BIB \
        --udiadsbib_root "../../U-DIADS-Bib-MS_patched" \
        --manuscript ${MANUSCRIPT} \
        --use_patched_data \
        --is_savenii \
        --use_tta \
        --batch_size 1 \
        --bottleneck \
        --adapter_mode streaming \
        --fusion_method simple \
        --use_groupnorm \
        --output_dir "./Result/a1/${MANUSCRIPT}"
    
    TEST_EXIT_CODE=$?
    
    if [ $TEST_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "╔════════════════════════════════════════════════════════════════════════╗"
        echo "║  ✓ TESTING COMPLETED: $MANUSCRIPT"
        echo "╚════════════════════════════════════════════════════════════════════════╝"
        echo ""
    else
        echo ""
        echo "╔════════════════════════════════════════════════════════════════════════╗"
        echo "║  ✗ TESTING FAILED: $MANUSCRIPT (Exit Code: $TEST_EXIT_CODE)"
        echo "╚════════════════════════════════════════════════════════════════════════╝"
        echo ""
    fi
done

echo ""
echo "============================================================================"
echo "ALL MANUSCRIPTS TESTED"
echo "============================================================================"
echo "Configuration Used: CNN-TRANSFORMER BASE MODEL (No Extra Components)"
echo "Results Location: ./Result/a1/"
echo "============================================================================"