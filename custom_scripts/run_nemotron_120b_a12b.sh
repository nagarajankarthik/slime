#!/bin/bash

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

ray stop --force
rm -rf /tmp/ray

redis-cli -h 127.0.0.1 -p 6379 FLUSHALL
redis-cli -h 127.0.0.1 -p 6379 SHUTDOWN NOSAVE

set -ex

export GPUS_PER_NODE=$(nvidia-smi -L | wc -l)
export WORLD_SIZE=${2:-1}
export NUM_NODES=${3:-1}
export MASTER_ADDR=${4:-"127.0.0.1"}
export MASTER_PORT=${5:-"12355"}
export NODE_RANK=${6:-0}
export TRITON_CACHE_DIR="/tmp/triton_cache"
export UV_CACHE_DIR="/tmp/uv_cache"
export NUMEXPR_MAX_THREADS=128
export NUMEXPR_NUM_THREADS=64
export OMP_NUM_THREADS=64


echo "NODE_RANK: ${NODE_RANK}"

# if base folder not set raise error
if [ -z "${BASE_FOLDER}" ]; then
  echo "BASE_FOLDER is not set. Please set it to the base directory of your checkpoints."
  exit 1
fi

if [ -z "${MASTER_ADDR}" ]; then
  echo "MASTER_ADDR is not set. Please set it to the master node address."
  exit 1
fi

# will prevent ray from buffering stdout/stderr
export PYTHONUNBUFFERED=1

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SCRIPT_DIR="${BASE_FOLDER}/slime/scripts"
source ${SCRIPT_DIR}/models/nemotron-3-120b-a12b.sh

cd ${BASE_FOLDER}/slime

# Install dependencies
uv pip install --no-deps sglang-router torch-memory-saver
uv pip install --upgrade --no-build-isolation transformer-engine[pytorch]


CKPT_ARGS=(
   --hf-checkpoint ${HF_HOME}/hub/models--nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-BF16/snapshots/d51eab0d1f979ebc26b546e634a04f450d99158e
   --load ${BASE_FOLDER}/megatron_ckpt/Nemotron_3_120B_A12B
   --save ${BASE_FOLDER}/megatron_ckpt/Nemotron_3_120B_A12B
   --save-interval 2000000
   --no-load-optim
   --megatron-to-hf-mode bridge
)

SFT_ARGS=(
   --rollout-function-path slime.rollout.sft_rollout.generate_rollout
   --prompt-data ${BASE_FOLDER}/train_data/openhermes2_5.parquet
   --input-key messages
   --rollout-shuffle
   --num-epoch 1
   --rollout-batch-size 64
   --global-batch-size 64
   --loss-type sft_loss
   --loss-mask-type qwen3_5
   --calculate-per-token-loss
   --disable-compute-advantages-and-returns
   --debug-train-only
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 2
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 8

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 2
   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-5
   --lr-decay-style cosine
   --min-lr 1e-6
   --lr-warmup-fraction 0.1
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   
   --use-distributed-optimizer
   # --optimizer-cpu-offload
   # --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=(
   # --use-wandb
   --wandb-project slime-sft-nemotron-3-super
   --wandb-group CP_1_EP_8_PP_1_max_tokens_8192
   --wandb-key ${WANDB_API_KEY}
   --disable-wandb-random-suffix
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)


# launch the master node of ray in container
export no_proxy="127.0.0.1,${MASTER_ADDR}"
export CURRENT_HOST=$(hostname)

export CURRENT_IP_ADDR=$(hostname -I | tr ' ' '\n' | grep '^192\.' | head -n1)
export PYTHONPATH="${BASE_FOLDER}/Megatron-LM:${BASE_FOLDER}/Megatron-Bridge/src"
export CPUS_PER_NODE=$((16 * GPUS_PER_NODE))
export CUDA_DEVICE_MAX_CONNECTIONS=1

ray symmetric-run --address "${MASTER_ADDR}:6379" \
    --min-nodes "${NUM_NODES}" \
    --num-cpus "${CPUS_PER_NODE}" \
    --num-gpus "${GPUS_PER_NODE}" \
   -- python3 train_async.py \
   --actor-num-nodes ${NUM_NODES} \
   --actor-num-gpus-per-node ${GPUS_PER_NODE} \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${SFT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${SPEC_ARGS[@]}

# ray status
# ray list placement-groups
# exit 0
#
# if [[ ${MASTER_ADDR} == ${CURRENT_IP_ADDR} ]]; then
#   echo "Starting Ray master"
#   ray start --head --node-ip-address ${MASTER_ADDR} --port=6379 --num-gpus ${GPUS_PER_NODE} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
# else
#   echo "Starting Ray worker on ${MASTER_ADDR}"
#   ray start --address=${MASTER_ADDR}:6379 --num-gpus ${GPUS_PER_NODE} --node-ip-address ${CURRENT_IP_ADDR} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265 --block
# fi
#
#
#
# # Build the runtime environment JSON with proper variable substitution
# RUNTIME_ENV_JSON="{
#   \"env_vars\": {
#     \"PYTHONPATH\": \"${BASE_FOLDER}/Megatron-LM/:${BASE_FOLDER}/Megatron-Bridge/src\",
#     \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
#     \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
#     \"no_proxy\": \"${no_proxy}\",
#     \"MASTER_ADDR\": \"${MASTER_ADDR}\",
#     \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\"
#   }
# }"
#
# if [[ ${MASTER_ADDR} == ${CURRENT_IP_ADDR} ]]; then
#     echo "Starting Ray job on ${MASTER_ADDR}"
#     ray status
#     ray list nodes
#     ray job submit --address="http://${MASTER_ADDR}:8265" \
#        --runtime-env-json="${RUNTIME_ENV_JSON}" \
#        -- python3 train_async.py \
#        --actor-num-nodes ${NUM_NODES} \
#        --actor-num-gpus-per-node ${GPUS_PER_NODE} \
#        ${MODEL_ARGS[@]} \
#        ${CKPT_ARGS[@]} \
#        ${SFT_ARGS[@]} \
#        ${OPTIMIZER_ARGS[@]} \
#        ${WANDB_ARGS[@]} \
#        ${PERF_ARGS[@]} \
#        ${EVAL_ARGS[@]} \
#        ${MISC_ARGS[@]} \
#        ${SPEC_ARGS[@]}
# fi
#
# rm -rf /tmp/* 2>/dev/null
