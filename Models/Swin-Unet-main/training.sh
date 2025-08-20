#!/bin/bash
#SBATCH --job-name=das_train_test
#SBATCH --output=logs/train_test_%j.out
#SBATCH --error=logs/train_test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kshahnitesh@gmail.com

# Load modules
module purge
module load python/pytorch2.6py3.12
module load cuda/11.8
module load cudnn

# Activate virtual environment
source ~/venv/bin/activate  # adjust path if needed

# Create logs directory
mkdir -p logs

# --- Determine available GPU partition ---
PARTITION=$(sinfo -h -o "%P %a %D %C" | awk '$2=="up"{print $1}' | grep -E "a100|v100|rtx3080|TinyGPU" | head -n1)

if [ -z "$PARTITION" ]; then
    echo "No GPU partition available. Exiting."
    exit 1
fi

echo "Using partition: $PARTITION"

# GPU request (all types: 1 GPU)
GRES="gpu:1"

# --- Submit training job ---
TRAIN_JOB=$(sbatch --partition=$PARTITION --gres=$GRES <<EOT
#!/bin/bash
#SBATCH --job-name=das_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00

module purge
module load python/3.10
module load cuda/11.8
module load cudnn
source ~/venv/bin/activate

# --- Run training ---
python3 train.py \
    --dataset UDIADS_BIB \
    --udiadsbib_root "U-DIADS-Bib-MS" \
    --udiadsbib_split training \
    --img_size 512 \
    --num_classes 6 \
    --output_dir ./model_out/udiadsbib_patch512 \
    --max_epochs 300 \
    --batch_size 32 \
    --cfg configs/swin_tiny_patch4_window7_224_lite.yaml
EOT
)

# Extract Job ID
TRAIN_JOB_ID=$(echo $TRAIN_JOB | awk '{print $4}')
echo "Submitted training job with Job ID: $TRAIN_JOB_ID"

# --- Submit testing job dependent on training ---
sbatch --dependency=afterok:$TRAIN_JOB_ID --partition=$PARTITION --gres=$GRES <<EOT
#!/bin/bash
#SBATCH --job-name=das_test
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00

module purge
module load python/3.10
module load cuda/11.8
module load cudnn
source ~/venv/bin/activate

# --- Run testing ---
python3 test.py \
    --dataset UDIADS_BIB \
    --udiadsbib_root "U-DIADS-Bib-MS" \
    --udiadsbib_split test \
    --img_size 2016 \
    --cfg configs/swin_tiny_patch4_window7_224_lite.yaml \
    --num_classes 6 \
    --output_dir ./model_out/udiadsbib_patch512 \
    --is_savenii
EOT

echo "Submitted testing job to run after training completes."
