import re
import os
import json

from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizerBase
from tqdm import tqdm
from torch.utils.data import Dataset


# detokenizer from https://github.com/EleutherAI/lm-evaluation-harness/blob/6d62a69cb5db963f998c486af6efee43fca63dd3/lm_eval/tasks/wikitext/preprocess_wikitext.py#L4
def wikitext_detokenizer(doc: dict) -> dict:
    """
    Detokenizes text from WikiText format by fixing spacing and punctuation.

    This function processes text from the WikiText dataset to improve readability
    by fixing contractions, number separators, punctuation spacing, brackets,
    and other formatting issues.

    Args:
        doc (dict): A dictionary containing a 'text' key with the tokenized text.

    Returns:
        dict: A dictionary with the detokenized text under the 'text' key.
    """
    string = doc["text"]
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")

    return {"text": string}


def load_wiki_data(cache_dir="./.cache") -> Dataset:
    """
    Load and preprocess the WikiText-2 dataset.

    This function loads the WikiText-2 dataset (raw version), applies detokenization,
    and filters out empty entries.

    Returns:
        Dataset: The processed WikiText dataset.
    """
    data_wikitext = load_dataset(
        "wikitext", "wikitext-2-raw-v1", split="test", cache_dir=cache_dir
    )
    data_wikitext = data_wikitext.map(wikitext_detokenizer)
    data_wikitext = data_wikitext.filter(lambda x: len(x["text"]) > 100)

    return data_wikitext


def sorted_counter(tokens_lst: list[str]) -> dict[str, int]:
    """
    Count the occurrences of each token in a text.

    Args:
        text (str): The input text to count token in.

    Returns:
        dict[str, int]: A dictionary with tokens as keys and their counts as values.
    """
    token_counts = {}
    for word in tokens_lst:
        token_counts[word] = token_counts.get(word, 0) + 1

    # Sort the token counts in decreasing order of frequency
    token_counts = dict(
        sorted(token_counts.items(), key=lambda item: item[1], reverse=True)
    )
    return token_counts


def is_bpe_token_final(
    token_id: int, non_final_set: set[str], tokenizer: PreTrainedTokenizerBase
) -> bool:
    """
    Check if a token is the final token of a BPE merge.

    Args:
        token_id (int): The token ID to check.
        non_finals (set[str]): A set of non-final tokens.

    Returns:
        bool: True if the token is the final token of a BPE merge, False otherwise.
    """
    token = tokenizer.convert_ids_to_tokens(token_id)
    if token in non_final_set:
        return False
    return True


def create_set_of_non_final_tokens(tokenizer_dir: str) -> set[str]:
    """Create a set of non-final tokens from a list of BPE merges.

    Args:
        tokenizer_dir (str): The directory containing the tokenizer file.

    Returns:
        set[str]: A set of non-final tokens.
    """
    tokenizer_file = None
    with open(os.path.join(tokenizer_dir, "tokenizer.json"), "r") as f:
        tokenizer_file = json.load(f)
    assert tokenizer_file is not None, "The tokenizer file hasn't been loaded properly!"

    merges_set = set()
    for merge_rule in tokenizer_file["model"]["merges"]:
        for token in merge_rule.split(" "):
            merges_set.add(token)

    return merges_set


def load_prefix_suffix_set(path: str) -> set[str]:
    prefix_suffix_lst = None
    with open(path, "r") as f:
        prefix_suffix_lst = f.readlines()
        assert (
            prefix_suffix_lst is not None
        ), "The list of prefixes and suffixes hasn't been loaded properly!"
    return set([x.strip() for x in prefix_suffix_lst])


def prepare_counts_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset: Dataset,
    non_final_set: set[str],
    prefix_suffix_set: set[str],
    sample_length: int = -1,
) -> list[dict[str, int | str]]:
    # Iterate over the dataset and tokenize each text
    new_dataset = []
    for item in tqdm(dataset):
        item_text = item["text"][:sample_length]
        if len(item_text) < sample_length:
            continue
        # Tokenize the text into individual tokens (not token IDs)
        tokenized = tokenizer.tokenize(item_text)
        sc = sorted_counter(tokenized)

        # convert the token counts to token ID counts
        id_sc = {}
        for k, v in sc.items():
            id_sc[tokenizer.convert_tokens_to_ids(k)] = v

        final_tokens = []
        non_final_tokens = []
        prefix_suffix_tokens = []

        for token_id, count in id_sc.items():
            token = tokenizer.convert_ids_to_tokens(token_id)
            stripped_token = token[1:] if token[0] == "Ġ" else token

            token_info = {
                "token": token,
                "token_id": token_id,
                "count": count,
            }

            if count > 1:
                if is_bpe_token_final(token_id, non_final_set, tokenizer):
                    final_tokens.append(token_info)
                else:
                    non_final_tokens.append(token_info)

            if stripped_token in prefix_suffix_set and count > 1:
                prefix_suffix_tokens.append(token_info)

        new_dataset_entry = {
            "text": item_text,
            "tokenized_text": tokenized,
            "final_tokens": final_tokens,
            "non_final_tokens": non_final_tokens,
            "prefix_suffix_tokens": prefix_suffix_tokens,
        }
        new_dataset.append(new_dataset_entry)
    return new_dataset


def create_benchmark_datasets_wiki(
    counts_dataset: list[dict[str, int | str]],
    num_prompts_per_group: int,
) -> tuple[list[tuple[str, str, int]], list[tuple[str, str, int]]]:
    # schema of the benchmark dataset: (text, substring, count)
    benchmark_dataset_final = []
    benchmark_dataset_non_final = []
    num_final_rejected = 0
    total_final = 0
    num_non_final_rejected = 0
    total_non_final = 0

    for item in counts_dataset:
        if len(item["final_tokens"]) > 0:
            for token_info in item["final_tokens"][:num_prompts_per_group]:
                # Count how many time the final token occurs in the text vs the count in the tokenized text
                # if it is different, than it means that there exists a derived token is split into multiple tokens.
                # In this case, we should not include this token in this particular context in the benchmark dataset.
                num_counts = len(
                    re.findall(
                        re.escape(token_info["token"].replace("Ġ", " ")), item["text"]
                    )
                )
                total_final += 1
                if num_counts != token_info["count"]:
                    num_final_rejected += 1
                    continue

                benchmark_dataset_final.append(
                    (
                        item["text"],
                        token_info["token"].replace(
                            "Ġ", " "
                        ),  # important, so that the substring will get tokenized as it occurs in the text
                        token_info["count"],
                    )
                )
        if len(item["non_final_tokens"]) > 0:
            for token_info in item["non_final_tokens"][:num_prompts_per_group]:
                # Count how many time the final token occurs in the text vs the count in the tokenized text
                # if it is different, than it means that there exists a derived token is split into multiple tokens.
                # In this case, we want to include this token in this particular context in the benchmark dataset.
                num_counts = len(
                    re.findall(
                        re.escape(token_info["token"].replace("Ġ", " ")),
                        item["text"],
                    )
                )

                total_non_final += 1
                if num_counts == token_info["count"]:
                    num_non_final_rejected += 1
                    continue

                benchmark_dataset_non_final.append(
                    (
                        item["text"],
                        token_info["token"].replace(
                            "Ġ", " "
                        ),  # important, so that the substring will get tokenized as it occurs in the text
                        num_counts,
                    )
                )
    print(
        f"Stats: final rejected ratio: {num_final_rejected}/{total_final} ({(num_final_rejected / total_final):.2f}), non-final rejected ratio: {num_non_final_rejected}/{total_non_final} ({(num_non_final_rejected / total_non_final):.2f})",
    )
    print(
        "Final dataset size:",
        len(benchmark_dataset_final),
        "non-final dataset size:",
        len(benchmark_dataset_non_final),
    )

    return benchmark_dataset_final, benchmark_dataset_non_final
