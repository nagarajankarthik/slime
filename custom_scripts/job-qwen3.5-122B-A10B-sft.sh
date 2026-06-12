#!/bin/bash
#SBATCH --job-name=qwen3.5-122B-A10B-sft
#SBATCH --nodes=2
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
export SQSH_FILE="/mnt/weka/aisg/sqsh/slime_10_june.sqsh"
export CONTAINER_NAME="slime_test"
export BASE_FOLDER="/mnt/weka/aisg/users/karthik/model_training_team/slime_test"
export LOG_DIR="${BASE_FOLDER}/logs/${JOB_ID}"
mkdir -p ${LOG_DIR}
export BASH_SCRIPT="${BASE_FOLDER}/slime/custom_scripts/run-qwen3.5-122B-A10B-sft.sh"
export MASTER_PORT=$((10000 + $RANDOM % 9000))
mpirun -np $NUM_NODES --host $host_list bash custom_scripts/create_enroot.sh "${SQSH_FILE}" "${CONTAINER_NAME}"


mpirun -np ${NUM_NODES} \
    -x JOB_ID -x JOB_WORK_DIR -x LOG_DIR -x BASE_FOLDER \
    --host $host_list \
enroot start --root --rw \
    -e JOB_ID -e JOB_WORK_DIR -e LOG_DIR -e BASE_FOLDER \
    -e OMPI_COMM_WORLD_RANK \
    --mount ${JOB_WORK_DIR} \
    ${CONTAINER_NAME} \
    bash -c "bash ${BASH_SCRIPT} \
        ${GPUS_PER_NODE} \
        ${WORLD_SIZE} \
        ${NUM_NODES} \
        ${MASTER_ADDR} \
        ${MASTER_PORT} | tee ${LOG_DIR}/node_\${OMPI_COMM_WORLD_RANK}.log"

