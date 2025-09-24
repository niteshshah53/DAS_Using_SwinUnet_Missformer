#!/bin/bash -l
#SBATCH --job-name=das_train_test
#SBATCH --output=All_Results_with_No_FocalLoss/swinunet/FewShot_UDIADS_BIB/testing_all_%j.out
#SBATCH --error=All_Results_with_No_FocalLoss/swinunet/FewShot_UDIADS_BIB/testing_all_%j.out
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

# Create logs directory  "Latin14396" "Latin16746" "Syr341"
mkdir -p logs

conda activate pytorch2.6-py3.12

# --- Run testing for multiple manuscripts --- "Latin14396FS" "Latin16746FS" "Syr341FS"
manuscripts=("Latin2FS" "Latin14396FS" "Latin16746FS" "Syr341FS")

for m in "${manuscripts[@]}"; do
    echo "=== Testing $m ==="
    python3 test.py \
        --model swinunet \
        --dataset UDIADS_BIB \
        --udiadsbib_root "U-DIADS-Bib-FS_patched" \
        --cfg configs/swin_tiny_patch4_window7_224_lite.yaml \
        --manuscript $m \
        --use_patched_data \
        --is_savenii \
        --num_classes 6 \
        --output_dir "./All_Results_with_No_FocalLoss/swinunet/FewShot_UDIADS_BIB/udiadsbib_patch224_swinunet_${m}"
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "testing.py failed for $m with exit code $rc"
        break
    fi
done
