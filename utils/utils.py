from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    AutoTokenizer,
    AutoModelForCausalLM,
)
import torch


def load_model_and_tokenizer(
    model_dir: str,
    device: torch.device | None = None,
    load_just_tokenizer: bool = False,
) -> tuple[PreTrainedModel, PreTrainedTokenizer | None]:
    """
    Load a model and tokenizer from the specified directory.

    Args:
    model_dir (str): Path to the model directory.

    Returns:
    Optional[model]: The loaded model.
    tokenizer: The loaded tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, padding_side="left", cache_dir="./.cache"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if load_just_tokenizer:
        return None, tokenizer

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        cache_dir="./.cache",
    ).to(device)

    return model, tokenizer
