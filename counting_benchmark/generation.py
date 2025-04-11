import re
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from tqdm import tqdm

from neural_tokenizer.neural_tokenizer import NeuralTokenizer
from bootstrapped_llm.generation import (
    generate_with_neural_tokenizer,
)
from bootstrapped_llm.bootstrapped_model import RNNBootstrappedLlamaModel
from counting_benchmark.dataset import BenchmarkDataset


def generate_text(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: list[str],
    max_new_tokens: int = 50,
) -> str:
    """
    Generate text based on a prompt using the provided model and tokenizer.

    Args:
        model: The model to use for text generation.
        tokenizer: The tokenizer to use for encoding/decoding.
        prompt (str): The input prompt to generate from.
        max_new_tokens (int, optional): Maximum length of the generated text. Defaults to 50.

    Returns:
        str: The generated text.
    """

    inputs = tokenizer(prompt, padding=True, return_tensors="pt").to(model.device)

    # print(prompt)
    # decode the input IDs one by one to check the tokenization
    # print([tokenizer.decode(x, skip_special_tokens=True) for x in inputs["input_ids"][0]])

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        # do_sample=True,
        # top_k=50,
        # top_p=0.95,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
    )
    generated_tokens = outputs[:, inputs["input_ids"].shape[-1] :]
    return tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True,
    )


def process_llm_outputs(output: list[str]) -> list[int]:
    """
    Process a batch of outputa of a language model to extract the number of occurrences of a substring.

    Args:
        output (list[str]): The batch od outputs of the language model.

    Returns:
        list[int]: The number of occurrences of the substring.
    """
    # Extract the number of occurrences from the generated text
    if not isinstance(output, list):
        output = [output]

    numbers = []
    for output_element in output:
        num_occurrences = re.search(r"\d+", output_element)
        numbers.append(
            int(num_occurrences.group()) if num_occurrences is not None else 0
        )

    assert len(numbers) == len(
        output
    ), "The number of occurrences does not match the number of outputs. Oops..."
    return numbers


def evaluate_llm_batch(
    model: PreTrainedModel | RNNBootstrappedLlamaModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: BenchmarkDataset,
    using_bootstrapped_model: bool = False,
    neural_tokenizer: NeuralTokenizer | None = None,
    conditioned: bool = False,
    verbose_evalation: bool = False,
    batch_size: int = 1,
) -> list[dict[str, int | str]]:

    if using_bootstrapped_model:
        assert (
            batch_size == 1
        ), "Batch size must be 1 for bootstrapped model evaluation."
        assert neural_tokenizer is not None, "Neural tokenizer must be provided."

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    results = []
    for prompts, search_substrings, counts in tqdm(dataloader):

        if using_bootstrapped_model:
            generated_text_batch = generate_with_neural_tokenizer(
                model,
                neural_tokenizer,
                tokenizer,
                prompt=prompts[0],
                search_substring=search_substrings[0] if conditioned else None,
                max_new_tokens=1,
            )
            generated_text_batch = [generated_text_batch]
        else:
            generated_text_batch = generate_text(
                model,
                tokenizer,
                prompts,
                max_new_tokens=1,
            )

        output_batch = process_llm_outputs(generated_text_batch)

        for i in range(len(prompts)):
            if verbose_evalation:
                print(f"Prompt:\n{prompts[i]}")
                print(f"Generated text:\n{generated_text_batch[i]}\n")
                print(f"Expected count: {counts[i]}, Provided count: {output_batch[i]}")
            results.append(
                {
                    "true_count": counts[i].item(),
                    "predicted_count": output_batch[i],
                    "prompt": prompts[i],
                    "generated_output": generated_text_batch[i],
                }
            )

    correct = sum(
        [1 for entry in results if entry["true_count"] == entry["predicted_count"]]
    )
    accuracy = correct / len(results)
    print(f"Accuracy: {accuracy:.4f}")
    print(len(results))

    return results
