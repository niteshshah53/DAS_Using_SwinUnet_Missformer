#!/bin/bash -l
#SBATCH --job-name=3rd
#SBATCH --output=./Result/a3/baseline_smart_ds_%j.out
#SBATCH --error=./Result/a3/baseline_smart_ds_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=22:00:00
#SBATCH --gres=gpu:1

# Load modules
module purge
module load python/pytorch2.6py3.12
module load cuda/11.8
module load cudnn

conda activate pytorch2.6-py3.12

# Add user site-packages to PYTHONPATH to find user-installed packages like pydensecrf2
export PYTHONPATH="${HOME}/.local/lib/python3.12/site-packages:${PYTHONPATH}"

# Memory optimization: Reduce CUDA memory fragmentation
# This is critical for smart skip connections with TTA (attention mechanisms are memory-intensive)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


# ============================================================================
# CNN-TRANSFORMER BASE MODEL + SMART SKIP CONNECTIONS + DEEP SUPERVISION CONFIGURATION
# ============================================================================
# Base Model Configuration (minimal components, no extra enhancements):
#   ✓ EfficientNet-B4 Encoder
#   ✓ Bottleneck: 2 Swin Transformer blocks (enabled)
#   ✓ Swin Transformer Decoder
#   ✓ Fusion Method: simple (concatenation)
#   ✓ Adapter mode: streaming (integrated adapters)
#   ✓ GroupNorm: enabled
#   ✓ Deep Supervision: enabled
#   ✓ All three losses: CE + Dice + Focal
#   ✓ Differential LR: Encoder (0.05x), Bottleneck (1.0x), Decoder (1.0x)
#
# Components Disabled (base model):
#   ✗ Multi-Scale Aggregation
#   ✗ Fourier Feature Fusion
#   ✗ Simple Skip Connections
# ============================================================================

echo "============================================================================"
echo "CNN-TRANSFORMER BASE MODEL + SMART SKIP CONNECTIONS + DEEP SUPERVISION"
echo "============================================================================"
echo "Configuration: CNN-TRANSFORMER BASE MODEL + SMART SKIP CONNECTIONS + DEEP SUPERVISION"
echo ""
echo "Component Details:"
echo "  ✓ EfficientNet-B4 Encoder"
echo "  ✓ Bottleneck: 2 Swin Transformer blocks (enabled)"
echo "  ✓ Swin Transformer Decoder"
echo "  ✓ Fusion Method: smart (attention-based smart skip connections)"
echo "  ✓ Adapter mode: streaming (integrated)"
echo "  ✓ GroupNorm: enabled"
echo "  ✓ Deep Supervision: enabled"
echo "  ✗ Multi-Scale Aggregation: disabled (base model)"
echo "  ✗ Fourier Feature Fusion: disabled (using smart fusion)"
echo "  ✗ Simple Skip Connections: disabled (using smart fusion)"
echo ""
echo "Training Parameters:"
echo "  - Batch Size: 16"
echo "  - Max Epochs: 300"
echo "  - Learning Rate: 0.0001"
echo "  - Scheduler: CosineAnnealingWarmRestarts"
echo "  - Early Stopping: 150 epochs patience"
echo "============================================================================"
echo ""

# Train all manuscripts one by one
MANUSCRIPTS=(Latin2 Latin14396 Latin16746 Syr341) 

for MANUSCRIPT in "${MANUSCRIPTS[@]}"; do
    echo ""
    echo "╔════════════════════════════════════════════════════════════════════════╗"
    echo "║  TRAINING CNN-TRANSFORMER BASE MODEL + SMART SKIP CONNECTIONS + DEEP SUPERVISION: $MANUSCRIPT"
    echo "╚════════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Configuration: CNN-TRANSFORMER BASE MODEL + SMART SKIP CONNECTIONS + DEEP SUPERVISION"
    echo "Output Directory: ./Result/a3/${MANUSCRIPT}"
    echo ""
    
    python3 train.py \
        --dataset UDIADS_BIB \
        --udiadsbib_root "../../U-DIADS-Bib-MS_patched" \
        --manuscript ${MANUSCRIPT} \
        --use_patched_data \
        --scheduler_type CosineAnnealingWarmRestarts \
        --batch_size 16 \
        --max_epochs 300 \
        --base_lr 0.0001 \
        --patience 150 \
        --bottleneck \
        --adapter_mode streaming \
        --fusion_method smart \
        --use_groupnorm \
        --deep_supervision \
        --output_dir "./Result/a3/${MANUSCRIPT}"
    
    TRAIN_EXIT_CODE=$?
    
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "╔════════════════════════════════════════════════════════════════════════╗"
        echo "║  ✓ TRAINING COMPLETED: $MANUSCRIPT"
        echo "╚════════════════════════════════════════════════════════════════════════╝"
        echo ""
        echo "Proceeding to testing..."
        echo ""
        
        echo "╔════════════════════════════════════════════════════════════════════════╗"
        echo "║  TESTING CNN-TRANSFORMER BASE MODEL + SMART SKIP CONNECTIONS + DEEP SUPERVISION: $MANUSCRIPT"
        echo "╚════════════════════════════════════════════════════════════════════════╝"
        echo ""
        echo "Test Configuration:"
        echo "  ✓ Test-Time Augmentation (TTA): ENABLED"
        echo "  ✗ CRF Post-processing: DISABLED"
        echo "  - Batch Size: 1 (reduced for smart skip connections + TTA memory efficiency)"
        echo "    Note: Smart skip connections use attention mechanisms which are memory-intensive"
        echo ""
        
        # Use batch_size=1 for smart skip connections with TTA
        # Smart skip connections use attention (multi-head attention) which is very memory-intensive
        # With TTA (4 augmentations), batch_size=1 means 4 patches in memory simultaneously
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
            --fusion_method smart \
            --use_groupnorm \
            --deep_supervision \
            --output_dir "./Result/a3/${MANUSCRIPT}"
        
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
    else
        echo ""
        echo "╔════════════════════════════════════════════════════════════════════════╗"
        echo "║  ✗ TRAINING FAILED: $MANUSCRIPT (Exit Code: $TRAIN_EXIT_CODE)"
        echo "╚════════════════════════════════════════════════════════════════════════╝"
        echo ""
        echo "Skipping testing for $MANUSCRIPT due to training failure."
        echo ""
    fi
done

echo ""
echo "============================================================================"
echo "ALL MANUSCRIPTS PROCESSED"
echo "============================================================================"
echo "Configuration Used: CNN-TRANSFORMER BASE MODEL + SMART SKIP CONNECTIONS + DEEP SUPERVISION"
echo "Results Location: ./Result/a3/"
echo "============================================================================"