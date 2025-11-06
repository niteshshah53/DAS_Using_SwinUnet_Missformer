#!/bin/bash -l
#SBATCH --job-name=calculate_avg           
#SBATCH --output=./Results/a2/baseline_smart_skip_ds_groupnorm_1307088_%j.out
#SBATCH --error=./Results/a2/baseline_smart_skip_ds_groupnorm_1307088_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=22:00:00
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

python3 calculate_avg.py 