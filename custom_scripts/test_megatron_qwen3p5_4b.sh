#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ==============================================================================
# Qwen3.5 VL Full Supervised Fine-Tuning (SFT)
#
# Supports all Qwen3.5 VL models (dense and MoE).
# For smaller setups, use LoRA/DoRA instead (see slurm_peft.sh).
#
# Usage:
#   sbatch slurm_sft.sh <model>
#
#   model: 0.8B | 2B | 4B | 9B | 27B | 35B-A3B | 122B-A10B | 397B-A17B
#
# Recommended parallelism (recipe defaults for full SFT):
#   0.8B (dense):    TP=1, PP=1         (1 node)
#   2B (dense):      TP=1, PP=1         (1 node)
#   4B (dense):      TP=2, PP=1         (1 node)
#   9B (dense):      TP=4, PP=1         (1 node)
#   27B (dense):     TP=4, PP=4         (2 nodes)
#   35B-A3B (MoE):   TP=2, PP=1, EP=16  (2 nodes)
#   122B-A10B (MoE): TP=2, PP=6, EP=8   (6 nodes)
#   397B-A17B (MoE): TP=2, PP=4, EP=32  (16 nodes)
#
# Examples:
#   sbatch slurm_sft.sh 4B
#   sbatch --nodes=2 slurm_sft.sh 27B
#   sbatch --nodes=6 slurm_sft.sh 122B-A10B
#   sbatch --nodes=16 slurm_sft.sh 397B-A17B
# ==============================================================================

#SBATCH --job-name=qwen35vl-sft
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --account=my_account
#SBATCH --output=qwen35vl_sft_%j.out
#SBATCH --error=qwen35vl_sft_%j.err
#SBATCH --exclusive

set -euo pipefail

# ==============================================================================
# Parse arguments
# ==============================================================================

MODEL_SIZE="${1:?Usage: sbatch $0 <model>  (model: 0.8B|2B|4B|9B|27B|35B-A3B|122B-A10B|397B-A17B)}"
MODEL_SIZE="4B"

# Map model size to HF name and recipe
case "$MODEL_SIZE" in
    0.8B)
        HF_MODEL_NAME="Qwen3.5-0.8B"
        RECIPE="qwen35_vl_800m_sft_config"
        ;;
    2B)
        HF_MODEL_NAME="Qwen3.5-2B"
        RECIPE="qwen35_vl_2b_sft_config"
        ;;
    4B)
        HF_MODEL_NAME="Qwen3.5-4B"
        RECIPE="qwen35_vl_4b_sft_config"
        ;;
    9B)
        HF_MODEL_NAME="Qwen3.5-9B"
        RECIPE="qwen35_vl_9b_sft_config"
        ;;
    27B)
        HF_MODEL_NAME="Qwen3.5-27B"
        RECIPE="qwen35_vl_27b_sft_config"
        ;;
    35B-A3B)
        HF_MODEL_NAME="Qwen3.5-35B-A3B"
        RECIPE="qwen35_vl_35b_a3b_sft_config"
        ;;
    122B-A10B)
        HF_MODEL_NAME="Qwen3.5-122B-A10B"
        RECIPE="qwen35_vl_122b_a10b_sft_config"
        ;;
    397B-A17B)
        HF_MODEL_NAME="Qwen3.5-397B-A17B"
        RECIPE="qwen35_vl_397b_a17b_sft_config"
        ;;
    *)
        echo "ERROR: Unknown model '$MODEL_SIZE'. Must be one of: 0.8B, 2B, 4B, 9B, 27B, 35B-A3B, 122B-A10B, 397B-A17B"
        exit 1
        ;;
esac

# ==============================================================================
# CONFIGURATION
# ==============================================================================

WORKSPACE=${WORKSPACE:-/workspace}

export HF_HOME="/mnt/weka/all/.cache/huggingface"
export BASE_DIR="/mnt/weka/aisg/users/karthik/model_training_team/slime_test"
# export UV_CACHE_DIR="/path/to/shared/uv_cache"
# export HF_HOME="/path/to/shared/HF_HOME"
# export HF_TOKEN="hf_your_token_here"
# export WANDB_API_KEY="your_wandb_key_here"
# export WANDB_MODE=disabled

PRETRAINED_CHECKPOINT=${HF_HOME}/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
DATASET_NAME=cord_v2
SEQ_LENGTH=32768
TRAIN_ITERS=50
GLOBAL_BATCH_SIZE=32
MICRO_BATCH_SIZE=2
CONTEXT_PARALLEL_SIZE=1
LOG_INTERVAL=1
WANDB_PROJECT=megatron-sft-qwen3.5-${DATASET_NAME}

# Container image (required)
CONTAINER_IMAGE=""
# CONTAINER_IMAGE="/path/to/container.sqsh"

# Container mounts (optional, space-separated)
CONTAINER_MOUNTS=""
# CONTAINER_MOUNTS="/data:/data /workspace:/workspace"

# ==============================================================================
# Environment Setup
# ==============================================================================

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=1
export HTTPX_LOG_LEVEL=WARNING
export PYTHONWARNINGS="ignore::FutureWarning:torch.cuda,ignore::UserWarning:modelopt.torch"


# ==============================================================================
# Job Execution
# ==============================================================================

echo "======================================"
echo "Qwen3.5-VL Full SFT Training Job"
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs per node: $SLURM_GPUS_ON_NODE"
echo "Total GPUs: $((SLURM_JOB_NUM_NODES * SLURM_GPUS_ON_NODE))"
echo "Model: $HF_MODEL_NAME"
echo "Recipe: $RECIPE"
echo "Checkpoint: $PRETRAINED_CHECKPOINT"
echo "======================================"

CLI_OVERRIDES="\
    checkpoint.pretrained_checkpoint=$PRETRAINED_CHECKPOINT \
    model.context_parallel_size=${CONTEXT_PARALLEL_SIZE:-1} \
    model.tensor_model_parallel_size=1 \
    ddp.average_in_collective=false \
    model.seq_length=$SEQ_LENGTH \
    model.mtp_num_layers=0 \
    train.train_iters=$TRAIN_ITERS \
    train.eval_iters=2 \
    train.global_batch_size=$GLOBAL_BATCH_SIZE \
    train.micro_batch_size=$MICRO_BATCH_SIZE \
    logger.log_interval=$LOG_INTERVAL \
    logger.wandb_project=$WANDB_PROJECT \
    logger.wandb_exp_name=${MODEL_SIZE}_mbs_${MICRO_BATCH_SIZE}_seq_len_${SEQ_LENGTH}_cp_${CONTEXT_PARALLEL_SIZE} \
    dataset.maker_name=make_${DATASET_NAME}_dataset \
    dataset.pack_sequences_in_batch=true \
    dataset.seq_length=$SEQ_LENGTH"

# For multinode runs, the recipe's online HF path can be unstable. Pass --hf_path
# with a local model directory for more reliable config loading, e.g.:
#   --hf_path ${WORKSPACE}/models/Qwen/${HF_MODEL_NAME}
CMD="cd ${BASE_DIR}/Megatron-Bridge && uv run --no-sync torchrun --nproc-per-node $SLURM_GPUS_ON_NODE scripts/training/run_recipe.py \
    --recipe $RECIPE \
    --step_func qwen3_vl_step \
    $CLI_OVERRIDES"

echo "Executing command..."
echo "======================================"


SRUN_CMD="srun --mpi=pmix --container-image=$CONTAINER_IMAGE"

if [ -n "$CONTAINER_MOUNTS" ]; then
    for mount in $CONTAINER_MOUNTS; do
        SRUN_CMD="$SRUN_CMD --container-mounts=$mount"
    done
fi

# $SRUN_CMD bash -c "$CMD"
export PYTHONPATH=${BASE_DIR}/Megatron-Bridge/src:${BASE_DIR}/Megatron-LM
# export PYTHONPATH=/mnt/weka/aisg/users/karthik/model_training_team/aspire2b_test/repos/Megatron-Bridge/src:/mnt/weka/aisg/users/karthik/model_training_team/aspire2b_test/repos/Megatron-Bridge/3rdparty/Megatron-LM
export TRITON_CACHE_DIR="/tmp/triton_cache"
bash -c "$CMD"

rm -rf ${BASE_DIR}/Megatron-Bridge/nemo_experiments/default/checkpoints

echo "======================================"
echo "Job completed"
echo "======================================"
