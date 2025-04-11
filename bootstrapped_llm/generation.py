from transformers import PreTrainedTokenizerBase
import torch

from bootstrapped_llm.bootstrapped_model import RNNBootstrappedLlamaModel
from neural_tokenizer.neural_tokenizer import NeuralTokenizer


def generate_with_neural_tokenizer(
    bootstrapped_model: RNNBootstrappedLlamaModel,
    neural_tokenizer: NeuralTokenizer,
    llm_tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    search_substring=None,
    max_new_tokens=10,
):
    input_text = prompt
    for _ in range(max_new_tokens):
        dynamic_boundaries, char_embeddings = neural_tokenizer.tokenize(
            input_text,
            search_substring=search_substring,
        )

        generated_text = llm_tokenizer.batch_decode(
            bootstrapped_model.forward_neural_tokenizer(
                dynamic_boundaries,
                char_embeddings,
            ),
            skip_special_tokens=True,
        )[0]

        input_text += generated_text
    return input_text
