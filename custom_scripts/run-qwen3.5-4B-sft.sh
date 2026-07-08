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

set -ex

export GPUS_PER_NODE=$(nvidia-smi -L | wc -l)
export WORLD_SIZE=${2:-1}
export NUM_NODES=${3:-1}
export MASTER_ADDR=${4:-"127.0.0.1"}
export MASTER_PORT=${5:-"12355"}
export TRITON_CACHE_DIR="/tmp/triton_cache"


# if base folder not set raise error
if [ -z "${BASE_FOLDER}" ]; then
  echo "BASE_FOLDER is not set. Please set it to the base directory of your checkpoints."
  exit 1
fi

if [ -z "${MASTER_ADDR}" ]; then
  echo "MASTER_ADDR is not set. Please set it to the master node address."
  exit 1
fi
# export MASTER_ADDR="127.0.0.1"

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
source "${SCRIPT_DIR}/models/qwen3.5-4B.sh"

cd ${BASE_FOLDER}/slime

CKPT_ARGS=(
   --hf-checkpoint ${HF_HOME}/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/
   --load ${BASE_FOLDER}/megatron_ckpt/Qwen3.5-4B_test/
   --save ${BASE_FOLDER}/megatron_ckpt/Qwen3.5-4B_test/
   --save-interval 2000000
   --no-load-optim
)

SFT_ARGS=(
   --rollout-function-path slime.rollout.sft_rollout.generate_rollout
   --prompt-data ${BASE_FOLDER}/train_data/openhermes2_5.parquet
   --input-key messages
   --rollout-shuffle
   --num-epoch 1
   --rollout-batch-size 128
   --global-batch-size 128

   --loss-type sft_loss
   --loss-mask-type qwen3_5
   --calculate-per-token-loss
   --disable-compute-advantages-and-returns
   --debug-train-only
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 32768
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
   --wandb-project slime-sft-qwen3.5
   --wandb-group 4B_TP1_CP2_fla_max_tokens_32768_megatron
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

if [[ ${CURRENT_HOST} == ${MASTER_ADDR} ]]; then
  echo "Starting Ray master on ${MASTER_ADDR}"
  ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${GPUS_PER_NODE} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
else
  echo "Starting Ray worker on ${MASTER_ADDR}"
  ray start --address=${MASTER_ADDR}:6379 --num-gpus ${GPUS_PER_NODE} --node-ip-address ${CURRENT_HOST} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
fi

wait



# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${BASE_FOLDER}/Megatron-LM/:${BASE_FOLDER}/Megatron-Bridge/src\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"no_proxy\": \"${no_proxy}\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\"
  }
}"

if [[ ${CURRENT_HOST} == ${MASTER_ADDR} ]]; then
    ray job submit --address="http://127.0.0.1:8265" \
       --runtime-env-json="${RUNTIME_ENV_JSON}" \
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
fi
