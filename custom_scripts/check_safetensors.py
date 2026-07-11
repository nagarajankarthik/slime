from safetensors import safe_open
import os
from transformers import AutoModelForCausalLM


SAFETENSORS_PATH = os.path.join( os.environ["HF_HOME"], "hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/" ) 

def safe_open(safetensors_path, **kwargs):
    tensors = {}
    with safe_open(safetensors_path, framework="pt") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)

    print(tensors)

def load_model_from_network_storage(checkpoint_path):
    """
    Load a safetensors model from network storage by first copying to memory
    
    Args:
        checkpoint_path: Path to the safetensors file
    """
    print(f"Loading model from: {checkpoint_path}")
    
    # Read the entire file into memory
    print("Reading file into memory...")
    with open(checkpoint_path, 'rb') as f:
        file_content = f.read()
    print(f"Read {len(file_content) / (1024*1024*1024):.2f}GB into memory")
    
    # Load using safetensors.torch.load
    try:
        print("Loading tensors...")
        tensors = safetensors.torch.load(file_content, device="cpu")
        print(f"Successfully loaded {len(tensors)} tensors")
        return tensors
    except Exception as e:
        print(f"Error loading tensors: {str(e)}")
        raise

def load_hf_checkpoint(checkpoint_path):
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
    state_dict = model.state_dict()
    print(state_dict["model.layers.27.self_attn.q_norm.weight"])

if __name__ == "__main__":
    load_hf_checkpoint(SAFETENSORS_PATH)
