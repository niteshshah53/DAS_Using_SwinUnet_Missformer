#!/bin/bash -l
#SBATCH --job-name=das_train_test_all
#SBATCH --output=logs/train_test_all_%j.out
#SBATCH --error=logs/train_test_all_%j.out
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
mkdir -p logs

conda activate pytorch2.6-py3.12

MANUSCRIPTS=(Latin2FS Latin14396FS Latin16746FS Syr341FS)

for MANUSCRIPT in "${MANUSCRIPTS[@]}"; do
    echo "=== Training $MANUSCRIPT ==="
    python3 train.py \
        --cfg configs/swin_tiny_patch4_window7_224_lite.yaml \
        --model swinunet \
        --dataset UDIADS_BIB \
        --udiadsbib_root "U-DIADS-Bib-FS_patched/${MANUSCRIPT}" \
        --use_patched_data \
        --img_size 224 \
        --num_classes 6 \
        --batch_size 32 \
        --max_epochs 300 \
        --output_dir "./model_out/udiadsbib_patch224_swinunet_${MANUSCRIPT}"

    echo "=== Testing $MANUSCRIPT ==="
    python3 test.py \
        --cfg configs/swin_tiny_patch4_window7_224_lite.yaml \
        --model swinunet \
        --dataset UDIADS_BIB \
        --udiadsbib_root "U-DIADS-Bib-FS_patched" \
        --manuscript ${MANUSCRIPT} \
        --use_patched_data \
        --is_savenii \
        --num_classes 6 \
        --output_dir "./model_out/udiadsbib_patch224_swinunet_${MANUSCRIPT}"

done
