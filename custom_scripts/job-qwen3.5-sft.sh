#!/bin/bash
#SBATCH --job-name=qwen3.5-rl
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=16
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/weka/aisg/users/karthik/model_training_team/slime_test/slurm_logs/%j.out


if [ -n "${SLURM_JOB_NODELIST}" ]; then
    export JOB_WORK_DIR=${SLURM_SUBMIT_DIR}
    export JOB_ID=${SLURM_JOB_ID}
    export JOB_NAME=${SLURM_JOB_NAME}
    hosts=$(scontrol show hostnames ${SLURM_JOB_NODELIST})
    export MASTER_ADDR=$(scontrol show hostnames ${SLURM_JOB_NODELIST} | head -n 1)
    export NUM_NODES=${SLURM_JOB_NUM_NODES}
    echo "NUM_NODES: ${NUM_NODES}"
elif [ -n "${PBS_JOBID}" ]; then
    export MASTER_ADDR=$(cat ${PBS_NODEFILE} | head -n 1)
fi

readarray -t host_array <<< "$hosts"
host_list=$(IFS=,; echo "${host_array[*]}")


export MASTER_ADDR=$(hostname)
export GPUS_PER_NODE=$(nvidia-smi -L | wc -l)
export WORLD_SIZE=$((GPUS_PER_NODE * NUM_NODES))
export CONTAINER_NAME="slime_test"
export CLUSTER_NAME="smc"

# ---- Cluster specific section ----
if [ ${CLUSTER_NAME} == "gcp" ]; then
    export BASE_FOLDER="/mnt/lustre/gcp640426-lustre1/aisg/users/karthik/model_training_team/slime_test"
    export MOUNT_DIR="/mnt/lustre/gcp640426-lustre1/aisg/users/karthik"
    export SQSH_FILE="${BASE_FOLDER}/slime_latest.sqsh"
    module load openmpi/v4.1.x
elif [ ${CLUSTER_NAME} == "smc" ]; then
    export BASE_FOLDER="/mnt/weka/aisg/users/karthik/model_training_team/slime_test"
    export MOUNT_DIR="/mnt/weka/aisg"
    export SQSH_FILE="${MOUNT_DIR}/sqsh/slime_10_june.sqsh"
    export SQSH_FILE="${MOUNT_DIR}/sqsh/slime_flash_linear_attn_context_parallel.sqsh"
fi
# ---- Cluster specific section end ----
export CREATE_ENROOT_SCRIPT="${BASE_FOLDER}/slime/custom_scripts/create_enroot.sh"
export LOG_DIR="${BASE_FOLDER}/logs/${JOB_ID}"
mkdir -p ${LOG_DIR}
export BASH_SCRIPT="${BASE_FOLDER}/slime/custom_scripts/run-qwen3.5-122B-A10B-sft.sh"
export BASH_SCRIPT="${BASE_FOLDER}/slime/custom_scripts/run-qwen3.5-4B-sft.sh"
cp ${BASH_SCRIPT} ${LOG_DIR}
export MASTER_PORT=$((10000 + $RANDOM % 9000))


container_mounts=("${BASE_FOLDER}")
container_mounts_str=$(IFS=,; echo "${container_mounts[*]}")
HOST_VARS=$(sed 's/ \{1,\}/,/g' <<<"${!HF*} WANDB_API_KEY BASE_FOLDER")

srun_args=" \
    --nodes=${NUM_NODES} \
    --ntasks-per-node=1 \
    --overlap \
    --cpu-bind=none \
    --container-image=${SQSH_FILE} \
    --container-mounts=${container_mounts_str} \
    --container-env=${HOST_VARS} \
    --container-writable \
    --container-workdir=${BASE_FOLDER} \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

srun $srun_args \
    --jobid ${SLURM_JOB_ID} \
    bash -c "${BASH_SCRIPT} \
        ${GPUS_PER_NODE} \
        ${WORLD_SIZE} \
        ${NUM_NODES} \
        ${MASTER_ADDR} \
        ${MASTER_PORT} \
        \${SLURM_PROCID} | tee ${LOG_DIR}/node_\${SLURM_PROCID}.log"

# Do not use mpirun. It degrades throughput in 26.xx versions of Nemo containers
# mpirun -np $NUM_NODES --host $host_list bash ${CREATE_ENROOT_SCRIPT} "${SQSH_FILE}" "${CONTAINER_NAME}"
#
# mpirun -np ${NUM_NODES} \
#     -x JOB_ID -x JOB_WORK_DIR -x LOG_DIR -x BASE_FOLDER -x WANDB_API_KEY -x HF_HOME \
#     --host $host_list \
# enroot start --rw \
#     -e JOB_ID -e JOB_WORK_DIR -e LOG_DIR -e BASE_FOLDER -e WANDB_API_KEY -e HF_HOME \
#     -e OMPI_COMM_WORLD_RANK \
#     --mount ${MOUNT_DIR} \
#     ${CONTAINER_NAME} \
#     bash -c "bash ${BASH_SCRIPT} \
#         ${GPUS_PER_NODE} \
#         ${WORLD_SIZE} \
#         ${NUM_NODES} \
#         ${MASTER_ADDR} \
#         ${MASTER_PORT} | tee ${LOG_DIR}/node_\${OMPI_COMM_WORLD_RANK}.log"
#
