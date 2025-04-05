import copy
import torch

from transformers import AutoModelForCausalLM


def load_llama_embedding_layer(model_path):
    """
    Load only the embedding layer of a Llama model.
    
    Args:
        model_name: The name or path of the Llama model to load
        
    Returns:
        The embedding layer of the model
    """
    # Load the full model's state dict
    full_model = AutoModelForCausalLM.from_pretrained(model_path)
    
    # Make a deep copy of the embedding layer to later remove the full model
    embeddings = copy.deepcopy(full_model.model.embed_tokens)
    
    # Delete the full model to save memory
    del full_model
    torch.cuda.empty_cache()
    
    return embeddings