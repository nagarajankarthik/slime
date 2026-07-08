"""
This script exports only the langugage sub-modeel of a Qwen 3.5 Huggingface checkpoint 
the Megatron torch distributed format.
Must use the if __name__ == "__main__": block to run the code.
Otherwise, python multiprocessing will throw an error.
"""


import torch
import torch.distributed as dist
from megatron.bridge import AutoBridge
import os
import shutil
from slime.backends.megatron_utils.arguments import set_default_megatron_args
from megatron.training import initialize_megatron, get_args
from megatron.training.checkpointing import save_checkpoint
from megatron.training.arguments import parse_and_validate_args, parse_args

# Checkpoint paths
MEGATRON_PATH="/mnt/weka/aisg/users/karthik/model_training_team/slime_test/megatron_ckpt/Qwen3.5-4B_language"
HF_PATH="/mnt/weka/all/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/"
MODEL_NAME="Qwen/Qwen3.5-4B"

def load_checkpoint_test(model_name, megatron_path):
    dist.init_process_group(backend="nccl", init_method="env://")
    bridge = AutoBridge.from_hf_pretrained(model_name)
    megatron_model = bridge.load_megatron_model(megatron_path)


def save_checkpoint_test(model_name, megatron_path, hf_path):

    args = parse_and_validate_args(
            args_defaults={
                "micro_batch_size": 1,
                "global_batch_size": 1,
                "save_interval": 1,
                "max_position_embeddings": 262144
                }
            )
    initialize_megatron()
    args = get_args()

    # Must set these to pass megatron validation
    args.micro_batch_size = 1
    args.global_batch_size = 1
    args.save_interval = 1
# Initialize torch
    dist.init_process_group(backend="nccl", init_method="env://")

# Initialize the bridge for Qwen 3.5
    bridge = AutoBridge.from_hf_pretrained(model_name)

# Provide the model and configure parallelism
    provider = bridge.to_megatron_provider()
    provider.tensor_model_parallel_size = 1
    provider.pipeline_model_parallel_size = 1
    provider.finalize()


# Provide the distributed model. This must be called first to ensure that certain distributed process groups, such as the pipeline and tensor model parallel groups are Initialized.
    model = provider.provide_distributed_model(wrap_with_ddp=False)
    language_model = provider.provide_language_model()

# Delete pre-existing checkpoints
    if os.path.exists(megatron_path):
        shutil.rmtree(megatron_path)

# Save the model
    save_checkpoint(0, language_model, None, None, 0)
    # bridge.save_megatron_model([language_model], megatron_path)


# Export the checkpoint
# bridge.export_ckpt(
#     megatron_path="/mnt/weka/aisg/users/karthik/model_training_team/slime_test/megatron_ckpt/Qwen3.5-4B_megatron/",
#     hf_path="/mnt/weka/all/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/",
# )


if __name__ == "__main__":
    # save_checkpoint(MODEL_NAME, MEGATRON_PATH, HF_PATH)
    # load_checkpoint(MODEL_NAME, MEGATRON_PATH)
    save_checkpoint_test(MODEL_NAME, MEGATRON_PATH, HF_PATH)
