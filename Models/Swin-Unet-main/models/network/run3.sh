#!/bin/bash -l
#SBATCH --job-name=bl_ash_ds_fff_bo_fl_msa_freeze_test           
#SBATCH --output=./a4/bl_ash_ds_fff_bo_fl_msa_freeze_test_%j.out
#SBATCH --error=./a4/bl_ash_ds_fff_bo_fl_msa_freeze_test_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
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

# ============================================================================
# TESTING CONFIGURATION
# ============================================================================
# Components:
#   BL  = Baseline (EfficientNet-B4 encoder + Swin-UNet decoder)
#   ASH = Alternative Segmentation Head (Conv3x3-ReLU-Conv1x1)
#   DS  = Deep Supervision (3 auxiliary outputs [384, 192, 96])
#   FFF = Fourier Feature Fusion (frequency domain skip connections)
#   Bo  = Bottleneck (2 Swin Transformer blocks at encoder-decoder bridge)
#   FL  = Focal Loss (in combination: 0.3*CE + 0.2*FL + 0.5*Dice)
#   MSA = Multi-Scale Aggregation (like hybrid1)
#   freeze = Freeze encoder during testing
# 
# Current Configuration: BL + ASH + DS + FFF + Bo + FL + MSA + freeze
# ============================================================================

echo "============================================================================"
echo "CNN-TRANSFORMER TESTING"
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
echo "  ✓ freeze (Freeze encoder)      : Freeze encoder during testing"
echo ""
echo "Testing Parameters:"
echo "  - Test-Time Augmentation: Enabled"
echo "  - CRF Post-processing: Enabled"
echo "  - Adapter Mode: streaming"
echo "============================================================================"
echo ""

# Test all manuscripts sequentially
MANUSCRIPTS=(Latin2 Latin14396 Latin16746 Syr341) 

for MANUSCRIPT in "${MANUSCRIPTS[@]}"; do
    echo ""
    echo "╔════════════════════════════════════════════════════════════════════════╗"
    echo "║  TESTING: $MANUSCRIPT"
    echo "╚════════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Active Components: BL + ASH + DS + FFF + Bo + FL + MSA + freeze"
    echo "Output Directory: ./a4/${MANUSCRIPT}"
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
        --use_tta \
        --use_crf \
        --output_dir "./a4/${MANUSCRIPT}"
    
    TEST_EXIT_CODE=$?
    
    # Report testing status
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
echo "Configuration Used: BL + ASH + DS + FFF + Bo + FL + MSA + freeze"
echo "Results Location: ./a4/"
echo "============================================================================"