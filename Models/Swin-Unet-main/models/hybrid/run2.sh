#!/bin/bash -l
#SBATCH --job-name=Syr341FS_baseline_smart_skip_groupnorm
#SBATCH --output=./Results/a1/Syr341FS_baseline_smart_skip_groupnorm_%j.out
#SBATCH --error=./Results/a1/Syr341FS_baseline_smart_skip_groupnorm_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --partition=rtx3080

#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

# ============================================================================
# HYBRID2 BASELINE TRAINING SCRIPT
# ============================================================================
# Model: Hybrid2 Baseline (Swin Transformer Encoder + EfficientNet Decoder)
# Dataset: U-DIADS-Bib-FS_patched (Full-Size patched dataset)
# Manuscripts: Latin2FS, Latin14396FS, Latin16746FS, Syr341FS
# ============================================================================

# Load modules
module purge
module load python/pytorch2.6py3.12
module load cuda/11.8
module load cudnn

# Create logs directory 
mkdir -p ../../logs
mkdir -p ./Results/a1

conda activate base

# Set PyTorch CUDA memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

# ============================================================================
# MODEL ARCHITECTURE COMPONENTS
# ============================================================================
# Hybrid2 Baseline consists of:
#
# ENCODER:
#   ✓ Swin Transformer Encoder (4 stages)
#     - Stage 1: 96 dim, 3 heads, 2 blocks, resolution: H/4 × W/4
#     - Stage 2: 192 dim, 6 heads, 2 blocks, resolution: H/8 × W/8
#     - Stage 3: 384 dim, 12 heads, 2 blocks, resolution: H/16 × W/16
#     - Stage 4: 768 dim, 24 heads, 2 blocks, resolution: H/32 × W/32
#     - Patch Embedding: 4×4 patches, 3 → 96 channels
#     - Patch Merging: 2×2 downsampling between stages
#     - Window Attention: 7×7 windows with relative position bias
#
# BOTTLENECK:
#   ✓ 2 Swin Transformer Blocks (768 dim, 24 heads)
#     - Resolution: H/32 × W/32
#     - Window size: 7×7
#     - Drop path rate: 0.1
#
# DECODER:
#   ✓ EfficientNet-B4 Style CNN Decoder
#     - Decoder channels: [256, 128, 64, 32]
#     - Upsampling: Bilinear interpolation + Conv layers
#     - Normalization: BatchNorm (baseline)
#     - Activation: ReLU
#
# SKIP CONNECTIONS:
#   ✓ Smart Skip Connections (token → CNN conversion)
#     - Converts encoder tokens to CNN features via projection
#     - Concatenates with decoder features
#     - No attention-based fusion (baseline)
#
# POSITIONAL EMBEDDINGS:
#   ✓ 2D Learnable Positional Embeddings (ENABLED by default)
#     - Matches SwinUnet pattern (relative position bias in Swin blocks)
#     - Added to bottleneck features before decoder
#     - Can be disabled with --no_pos_embed flag
#
# OPTIONAL FEATURES (DISABLED in baseline):
#   ✗ Deep Supervision (--use_deep_supervision)
#   ✗ CBAM Attention (--use_cbam)
#   ✗ Simple Skip Connections (remove --use_smart_skip flag to use simple skip connections)
#   ✗ Cross-Attention Bottleneck (--use_cross_attn)
#   ✗ Multi-Scale Aggregation (--use_multiscale_agg)
#   ✓ GroupNorm (--use_groupnorm) - uses BatchNorm instead
# ============================================================================

# Train all manuscripts one by one (Latin2FS Latin14396FS Latin16746FS Syr341FS)
MANUSCRIPTS=(Syr341FS)

for MANUSCRIPT in "${MANUSCRIPTS[@]}"; do
    echo ""
    echo "========================================================================"
    echo "Training Hybrid2 Baseline Model: $MANUSCRIPT"
    echo "========================================================================"
    echo "Dataset: U-DIADS-Bib-FS_patched"
    echo "Architecture: Swin Transformer Encoder → 2 Swin Blocks Bottleneck → EfficientNet-B4 Decoder"
    echo "Components Enabled:"
    echo "  ✓ Swin Encoder (4 stages)"
    echo "  ✓ Bottleneck: 2 Swin Transformer blocks"
    echo "  ✓ EfficientNet-B4 Decoder"
    echo "  ✓ Smart Skip Connections"
    echo "  ✓ Positional Embeddings (default: True)"
    echo "  ✓ GroupNorm (baseline normalization: uses BatchNorm instead)"
    echo "Components Disabled (baseline):"
    echo "  ✗ Deep Supervision"
    echo "  ✗ CBAM Attention"
    echo "  ✗ Simple Skip Connections (remove --use_smart_skip flag to use simple skip connections)"
    echo "  ✗ Cross-Attention Bottleneck"
    echo "  ✗ Multi-Scale Aggregation"
    echo "  ✗ GroupNorm (baseline normalization: uses BatchNorm instead)"
    echo "========================================================================"
    
    python3 train.py \
        --model hybrid2 \
        --use_baseline \
        --efficientnet_variant b4 \
        --dataset UDIADS_BIB \
        --udiadsbib_root "../../U-DIADS-Bib-FS_patched" \
        --manuscript ${MANUSCRIPT} \
        --use_patched_data \
        --batch_size 4 \
        --max_epochs 300 \
        --base_lr 0.0001 \
        --patience 150 \
        --use_smart_skip \
        --use_groupnorm \
        --scheduler_type ReduceLROnPlateau \
        --output_dir "./Results/a1/${MANUSCRIPT}"

    echo ""
    echo "========================================================================"
    echo "Testing Hybrid2 Baseline Model: $MANUSCRIPT"
    echo "========================================================================"
    
    python3 test.py \
        --model hybrid2 \
        --use_baseline \
        --efficientnet_variant b4 \
        --dataset UDIADS_BIB \
        --udiadsbib_root "../../U-DIADS-Bib-FS_patched" \
        --manuscript ${MANUSCRIPT} \
        --use_patched_data \
        --is_savenii \
        --use_tta \
        --use_crf \
        --use_smart_skip \
        --use_groupnorm \
        --output_dir "./Results/a1/${MANUSCRIPT}"
done

echo ""
echo "========================================================================"
echo "ALL MANUSCRIPTS COMPLETED - HYBRID2 BASELINE MODEL"
echo "========================================================================"
echo "Results saved in: ./Results/a1/"
echo ""
echo "Model Configuration:"
echo "  • Architecture: Hybrid2 Baseline (Swin-Encoder + EfficientNet-Decoder)"
echo "  • Encoder: Swin Transformer (4 stages, 96→768 dim)"
echo "  • Bottleneck: 2 Swin Transformer blocks (768 dim, 24 heads)"
echo "  • Decoder: EfficientNet-B4 style CNN decoder"
echo "  • Skip Connections: Smart token→CNN conversion"
echo "  • Positional Embeddings: Enabled (default, matching SwinUnet pattern)"
echo "  • Normalization: GroupNorm (baseline normalization: uses BatchNorm instead)"
echo ""
echo "Training Configuration:"
echo "  • Dataset: U-DIADS-Bib-FS_patched"
echo "  • Manuscripts: Latin2FS, Latin14396FS, Latin16746FS, Syr341FS"
echo "  • Batch Size: 4"
echo "  • Max Epochs: 300"
echo "  • Base Learning Rate: 0.0001"
echo "  • Scheduler: ReduceLROnPlateau (factor=0.5, patience=15)"
echo "  • Early Stopping: 150 epochs patience"
echo ""
echo "Testing Configuration:"
echo "  • Test-Time Augmentation: Enabled"
echo "  • CRF post-processing: Enabled"
echo "  • Save Predictions: Enabled (NIfTI format)"
echo "  • GroupNorm: Enabled (baseline normalization: uses BatchNorm instead)"
echo ""
