#!/bin/bash -l
#SBATCH --job-name=das_test
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.out
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

# Create logs directory  "Latin14396" "Latin16746" "Syr341"
mkdir -p logs

conda activate pytorch2.6-py3.12

# --- Run testing for multiple manuscripts ---
manuscripts=("Latin2")

for m in "${manuscripts[@]}"; do
    echo "=== Testing $m ==="
    python3 test.py \
        --cfg configs/swin_tiny_patch4_window7_224_lite.yaml \
        --model swinunet \
        --dataset UDIADS_BIB \
        --udiadsbib_root "U-DIADS-Bib-MS_patched" \
        --manuscript $m \
        --use_patched_data \
        --is_savenii \
        --img_size 2016 \
        --num_classes 6 \
        --output_dir "./model_out/udiadsbib_patch224_swinunet_${m}"
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "test.py failed for $m with exit code $rc"
        break
    fi
done
