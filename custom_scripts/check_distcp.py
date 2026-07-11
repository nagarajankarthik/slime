from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import Metadata
import torch.distributed.checkpoint as dcp

distcp_path = "/mnt/weka/aisg/users/karthik/model_training_team/slime_test/megatron_ckpt/Qwen3.5-4B_slime/iter_0000000"
distcp_path = "/mnt/weka/aisg/users/karthik/model_training_team/slime_test/megatron_ckpt/Qwen3.5-4B_test/release"

def check_parameters():
    """
    Inspect parameters in the checkpoint.
    """
    reader = FileSystemReader(distcp_path)
    metadata: Metadata = reader.read_metadata()
    print(metadata.state_dict_metadata.keys())

def check_tensors():
    """
    Inspect tensors in the checkpoint.
    """
    state_dict = {}  # empty dict — distcp will populate it
    dcp.load_state_dict(
        state_dict=state_dict,
        storage_reader=FileSystemReader(distcp_path),
        no_dist=True,  # load without distributed process group
    )
    for k, v in state_dict.items():
        print(k, v)


if __name__ == "__main__":
    # check_parameters()
    check_tensors()
