#!/bin/bash -l
#SBATCH --job-name=bl_ash_ds_fff_bo_fl_msa_freeze           
#SBATCH --output=./a4/bl_ash_ds_fff_bo_fl_msa_freeze_%j.out
#SBATCH --error=./a4/bl_ash_ds_fff_bo_fl_msa_freeze_%j.out
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

# ============================================================================
# ABLATION CONFIGURATION
# ============================================================================
# Components (based on ablation study):
# BL  = Baseline (Encoder-Decoder-SegmentationHead)
# ASH = Alternative Segmentation Head (Conv3x3-ReLU-Conv1x1)
# DS  = Deep Supervision (auxiliary outputs from decoder layers)
# FFF = Fourier Feature Fusion (frequency domain fusion)
# AFF = Attention Feature Fusion (attention-based fusion)
# Bo  = Bottleneck (2 Swin Transformer blocks)
# FL  = Focal Loss (in loss combination)
# MSA = Multi-Scale Aggregation (like hybrid1)
# freeze = Freeze encoder during training
# CURRENT CONFIGURATION: BL + ASH + DS + FFF + Bo + FL
# ============================================================================

echo "============================================================================"
echo "CNN-TRANSFORMER ABLATION STUDY"
echo "============================================================================"
echo "Configuration: BL + ASH + DS + FFF + Bo + FL + MSA + freeze"
echo ""
echo "Component Details:"
echo "  ✓ BL  (Baseline)              : EfficientNet-B4 encoder + Swin-UNet decoder"
echo "  ✓ ASH (Alt Seg Head)          : Conv3x3-ReLU-Conv1x1 (richer features)"
echo "  ✓ DS  (Deep Supervision)      : 3 auxiliary outputs [384, 192, 96]"
echo "  ✓ FFF (Fourier Feature Fusion): Frequency domain skip connections"
echo "  ✓ Bo  (Bottleneck)            : 2 Swin blocks at encoder-decoder bridge"
echo "  ✓ FL  (Focal Loss)            : In combination (0.3*CE + 0.2*FL + 0.5*Dice)"
echo "  ✓ MSA (Multi-Scale Aggregation): Like hybrid1"
echo "  ✓ freeze (Freeze encoder)      : Freeze encoder during training"
echo ""
echo "Training Parameters:"
echo "  - Batch Size: 4"
echo "  - Max Epochs: 400"
echo "  - Learning Rate: 0.0001"
echo "  - Scheduler: OneCycleLR"
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
    echo "Active Components: BL + ASH + DS + FFF + Bo + FL + MSA + freeze"
    echo "Output Directory: ./a4/${MANUSCRIPT}"
    echo ""
    
    python3 train.py \
        --dataset UDIADS_BIB \
        --udiadsbib_root "../../U-DIADS-Bib-MS_patched" \
        --manuscript ${MANUSCRIPT} \
        --use_patched_data \
        --fusion_method smart \
        --deep_supervision \
        --adapter_mode streaming \
        --use_multiscale_agg \
        --bottleneck \
        --freeze_encoder \
        --freeze_epochs 30 \
        --scheduler_type OneCycleLR \
        --batch_size 4 \
        --max_epochs 300 \
        --base_lr 0.0001 \
        --patience 50 \
        --output_dir "./a4/${MANUSCRIPT}"
    
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
            --fusion_method smart \
            --deep_supervision \
            --adapter_mode streaming \
            --bottleneck \
            --use_multiscale_agg \
            --freeze_encoder \
            --is_savenii \
            --output_dir "./a4/${MANUSCRIPT}"
        
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
echo "Configuration Used: BL + ASH + DS + FFF + Bo + FL + MSA + freeze"
echo "Results Location: ./a4/"
echo "============================================================================"