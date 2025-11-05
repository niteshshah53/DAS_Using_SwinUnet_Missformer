#!/bin/bash -l
#SBATCH --job-name=bl_ash_ds_bo_fl
#SBATCH --output=./a3/bl_ash_ds_bo_fl_%j.out
#SBATCH --error=./a3/bl_ash_ds_bo_fl_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --partition=rtx3080

#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

# Load modules
module purge
module load python/pytorch2.6py3.12
module load cuda/11.8
module load cudnn

conda activate pytorch2.6-py3.12

# ============================================================================
# ABLATION CONFIGURATION
# ============================================================================
# Components (based on ablation study):
# BL  = Baseline (Encoder-Decoder-SegmentationHead)
# ASH = Alternative Segmentation Head (Conv3x3-ReLU-Conv1x1)
# DS  = Deep Supervision (auxiliary outputs from decoder layers)
# Bo  = Bottleneck (2 Swin Transformer blocks)
# FL  = Focal Loss (in loss combination)
#
# CURRENT CONFIGURATION: BL + ASH + DS + Bo + FL
# ============================================================================

echo "============================================================================"
echo "CNN-TRANSFORMER ABLATION STUDY"
echo "============================================================================"
echo "Configuration: BL + ASH + DS + Bo + FL"
echo ""
echo "Component Details:"
echo "  ✓ BL  (Baseline)              : EfficientNet-B4 encoder + Swin-UNet decoder"
echo "  ✓ ASH (Alt Seg Head)          : Conv3x3-ReLU-Conv1x1 (richer features)"
echo "  ✓ DS  (Deep Supervision)      : 3 auxiliary outputs [384, 192, 96]"
echo "  ✓ Bo  (Bottleneck)            : 2 Swin blocks at encoder-decoder bridge"
echo "  ✓ FL  (Focal Loss)            : In combination (0.4*CE + 0.1*FL + 0.5*Dice)"
echo ""
echo "Training Parameters:"
echo "  - Batch Size: 4"
echo "  - Max Epochs: 300"
echo "  - Learning Rate: 0.0001"
echo "  - Scheduler: OneCycleLR (30% warmup, cosine annealing)"
echo "  - Early Stopping: 50 epochs patience"
echo "  - Adapter Mode: streaming"
echo "============================================================================"
echo ""

# Train all manuscripts one by one
MANUSCRIPTS=(Latin2 Latin14396 Latin16746 Syr341) 

for MANUSCRIPT in "${MANUSCRIPTS[@]}"; do
    echo ""
    echo "╔════════════════════════════════════════════════════════════════════════╗"
    echo "║  TRAINING: $MANUSCRIPT"
    echo "╚════════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Active Components: BL + ASH + DS + Bo + FL"
    echo "Output Directory: ./a3/${MANUSCRIPT}"
    echo ""
    
    python3 train.py \
        --dataset UDIADS_BIB \
        --udiadsbib_root "../../U-DIADS-Bib-MS_patched" \
        --manuscript ${MANUSCRIPT} \
        --use_patched_data \
        --deep_supervision \
        --adapter_mode streaming \
        --bottleneck \
        --scheduler_type OneCycleLR \
        --batch_size 4 \
        --max_epochs 300 \
        --base_lr 0.0001 \
        --patience 50 \
        --output_dir "./a3/${MANUSCRIPT}"
    
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
        echo "║  TESTING: $MANUSCRIPT"
        echo "╚════════════════════════════════════════════════════════════════════════╝"
        echo ""
        
        python3 test.py \
            --dataset UDIADS_BIB \
            --udiadsbib_root "../../U-DIADS-Bib-MS_patched" \
            --manuscript ${MANUSCRIPT} \
            --use_patched_data \
            --deep_supervision \
            --adapter_mode streaming \
            --bottleneck \
            --is_savenii \
            --output_dir "./a3/${MANUSCRIPT}"
        
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
echo "Configuration Used: BL + ASH + DS + Bo + FL"
echo "Results Location: ./a3/"
echo "============================================================================"