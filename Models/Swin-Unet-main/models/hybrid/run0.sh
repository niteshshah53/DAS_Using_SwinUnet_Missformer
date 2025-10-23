#!/bin/bash -l
#SBATCH --job-name=Hybrid1_enhanced_train_test
#SBATCH --output=./Results_Optimized_Hyperparameters/v3/hybrid12/UDIADS_BIB_MS/train_test_enhanced_%j.out
#SBATCH --error=./Results_Optimized_Hyperparameters/v3/hybrid12/UDIADS_BIB_MS/train_test_enhanced_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

# Load modules
module purge
module load python/pytorch2.6py3.12
module load cuda/11.8
module load cudnn

# Create logs directory 
mkdir -p ../../logs
mkdir -p ./Results_Optimized_Hyperparameters/v3/hybrid12/UDIADS_BIB_MS

conda activate base

# Set PyTorch CUDA memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

# Train all manuscripts one by one (Latin2 Latin14396 Latin16746 Syr341) (CB55, CSG18, CSG863)
MANUSCRIPTS=(Latin2 Latin14396 Latin16746 Syr341)

for MANUSCRIPT in "${MANUSCRIPTS[@]}"; do
    echo "=== Training hybrid12 Model $MANUSCRIPT ==="
    python3 train.py \
        --model hybrid1 \
        --use_enhanced \
        --dataset UDIADS_BIB \
        --udiadsbib_root "../../U-DIADS-Bib-MS_patched" \
        --manuscript ${MANUSCRIPT} \
        --use_patched_data \
        --batch_size 4 \
        --max_epochs 300 \
        --base_lr 0.0001 \
        --patience 50 \
        --scheduler_type OneCycleLR \
        --output_dir "./Results_Optimized_Hyperparameters/v3/hybrid12/UDIADS_BIB_MS/udiadsbib_Hybrid1_enhanced_${MANUSCRIPT}"

    echo "=== Testing hybrid12 Model $MANUSCRIPT ==="
    python3 test.py \
        --model hybrid1 \
        --use_enhanced \
        --dataset UDIADS_BIB \
        --udiadsbib_root "../../U-DIADS-Bib-MS_patched" \
        --manuscript ${MANUSCRIPT} \
        --use_patched_data \
        --is_savenii \
        --use_tta \
        --output_dir "./Results_Optimized_Hyperparameters/v3/hybrid12/UDIADS_BIB_MS/udiadsbib_Hybrid1_enhanced_${MANUSCRIPT}"
done

echo ""
echo "========================================================================"
echo "ALL MANUSCRIPTS COMPLETED - HYBRID12 Model!"
echo "========================================================================"
echo "Results saved in: ./Results_Optimized_Hyperparameters/v3/hybrid12/UDIADS_BIB_MS/"
echo ""
echo "Model: hybrid12 Model (EfficientNet-B4 + Swin-Unet with TransUNet Best Practices)"
echo "Architecture: EfficientNet-B4 Encoder → Swin-Unet Decoder"
echo "Features:"
echo "  ✓ Deep Supervision (auxiliary outputs)"
echo "  ✓ Multi-scale Aggregation (bottleneck)"
echo "  ✓ SwinUnet Training Approach (0.4*CE + 0.6*Dice)"
echo ""
