# DAS_Using_SwinUnet_Missformer

This repository contains two main projects for document analysis and segmentation using transformer-based architectures:

- **Swin-Unet-main**: Swin-Unet for medical/document image segmentation, adapted for HPC/SLURM usage.
- **MISSFormer-main**: MISSFormer for segmentation tasks.

## Directory Structure

```
Models/
  Swin-Unet-main/
  MISSFormer-main/
```

## Quick Start (HPC/SLURM)

1. Enter the desired project directory (e.g., `Swin-Unet-main`).
2. Edit the SLURM script (e.g., `training.sh`) to match your environment.
3. Submit your job:
   ```bash
   sbatch training.sh
   ```
4. Monitor logs in the `logs/` directory.

## Datasets

Datasets should be placed in the appropriate subdirectories (see each project's README for details). Large datasets and outputs are excluded from git.

## Requirements
- Python 3.12 (module: `python/pytorch2.6py3.12`)
- PyTorch 2.6
- CUDA 11.8
- cuDNN
- Other dependencies: see each project's `requirements.txt`

## Contact
For questions or issues, contact:
- Nitesh Kumar Shah (nitesh.shah@fau.de)

See each subproject's README for more details and usage instructions.
