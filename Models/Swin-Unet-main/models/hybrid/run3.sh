#!/bin/bash -l
#SBATCH --job-name=hybrid1_divahisdb
#SBATCH --output=./Results_Optimized_Hyperparameters/v2/hybrid1/DIVAHISDB/train_test_optuna_%j.out
#SBATCH --error=./Results_Optimized_Hyperparameters/v2/hybrid1/DIVAHISDB/train_test_optuna_%j.out
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

# Create logs directory 
mkdir -p ../../logs
mkdir -p ./Results_Optimized_Hyperparameters/v2/hybrid1/DIVAHISDB

# Training configuration for Hybrid1 on DIVAHISDB:
# - model: hybrid1 (EfficientNet-Swin Encoder + Swin-Unet Decoder)
# - dataset: DIVAHISDB (4 classes: Background, Comment, Decoration, Main Text)
# - base_lr: 0.0002 (optimal learning rate)
# - patience: Early stopping patience

conda activate base

# Set PyTorch CUDA memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

# Train all manuscripts one by one for DIVAHISDB (CB55, CSG18, CSG863)
MANUSCRIPTS=(CB55 CSG18 CSG863) 

for MANUSCRIPT in "${MANUSCRIPTS[@]}"; do
    echo "=== Training Hybrid1-Enhanced EfficientNet $MANUSCRIPT ==="
    python3 train.py \
        --model hybrid1 \
        --use_enhanced \
        --dataset DIVAHISDB \
        --divahisdb_root "../../DivaHisDB_patched" \
        --manuscript ${MANUSCRIPT} \
        --use_patched_data \
        --num_classes 4 \
        --batch_size 4 \
        --max_epochs 300 \
        --base_lr 0.0002 \
        --patience 50 \
        --output_dir "./Results_Optimized_Hyperparameters/v2/hybrid1/DIVAHISDB/Hybrid1_enhanced_${MANUSCRIPT}"

    echo "=== Testing Hybrid1-Enhanced EfficientNet $MANUSCRIPT ==="
    python3 test.py \
        --model hybrid1 \
        --use_enhanced \
        --dataset DIVAHISDB \
        --divahisdb_root "../../DivaHisDB_patched" \
        --manuscript ${MANUSCRIPT} \
        --use_patched_data \
        --num_classes 4 \
        --is_savenii \
        --use_tta \
        --output_dir "./Results_Optimized_Hyperparameters/v2/hybrid1/DIVAHISDB/Hybrid1_enhanced_${MANUSCRIPT}"
done
