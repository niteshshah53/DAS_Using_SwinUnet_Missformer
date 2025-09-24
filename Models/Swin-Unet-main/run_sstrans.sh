#!/bin/bash -l
#SBATCH --job-name=das_train_test_sstrans
#SBATCH --output=All_Results_with_FocalLoss_DiceLoss/sstrans/DIVAHISDB/train_test_all_%j.out
#SBATCH --error=All_Results_with_FocalLoss_DiceLoss/sstrans/DIVAHISDB/train_test_all_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1

#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

# Load modules
module purge
module load python/pytorch2.6py3.12
module load cuda/11.8
module load cudnn

# Create logs directory 
mkdir -p logs

# Training configuration for SSTrans on U-DIADS-Bib:
# - model: sstrans (requires config file)
# - dataset: UDIADS_BIB (6 classes: background, paratext, decoration, main_text, title, chapter_headings)
# - base_lr: Initial learning rate
# - patience: Early stopping patience (stop if no improvement for N epochs)
# - lr_factor: Factor to reduce learning rate by when plateauing
# - lr_patience: Patience for learning rate reduction
# - lr_min: Minimum learning rate
# - lr_threshold: Threshold for considering improvement

conda activate pytorch2.6-py3.12

MANUSCRIPTS=(CB55 CSG18 CSG863) #CB55 CSG18 CSG863

for MANUSCRIPT in "${MANUSCRIPTS[@]}"; do
    echo "=== Training $MANUSCRIPT ==="
    python3 train.py \
        --model sstrans \
        --dataset DIVAHISDB \
        --divahisdb_root "DivaHisDB_patched" \
        --manuscript ${MANUSCRIPT} \
        --cfg configs/swin_tiny_patch4_window7_224_lite.yaml \
        --use_patched_data \
        --num_classes 4 \
        --batch_size 32 \
        --max_epochs 300 \
        --base_lr 0.0001 \
        --patience 30 \
        --output_dir "./All_Results_with_FocalLoss_DiceLoss/sstrans/DIVAHISDB/sstrans_patch224_${MANUSCRIPT}"

    echo "=== Testing $MANUSCRIPT ==="
    python3 test.py \
        --model sstrans \
        --dataset DIVAHISDB \
        --divahisdb_root "DivaHisDB_patched" \
        --manuscript ${MANUSCRIPT} \
        --cfg configs/swin_tiny_patch4_window7_224_lite.yaml \
        --use_patched_data \
        --is_savenii \
        --num_classes 4 \
        --output_dir "./All_Results_with_FocalLoss_DiceLoss/sstrans/DIVAHISDB/sstrans_patch224_${MANUSCRIPT}"
done
