import copy

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


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


class EmbeddingsDataset(Dataset):
    """
    A PyTorch Dataset for loading embeddings.
    """

    def __init__(self, embeddings_path, tokenizer_path, split, split_files_path):
        """
        Initialize the dataset.

        Args:
            embeddings_path (str): Path to the embeddings file
            tokenizer_path (str): Path to the tokenizer file
            split (str): Split type (train, valid, test)
        """
        # Load embeddings
        self.embeddings = load_llama_embedding_layer(embeddings_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, cache_dir="./.cache"
        )
        self.split = split
        self.split_files_path = split_files_path

        self.vocab_set = set(self.tokenizer.get_vocab().keys())
        self.features, self.labels = self._create_dataset()

    def split_embedding(self, idx: int) -> torch.Tensor:
        decoded_token = self.tokenizer.decode([idx])

        # IMPORTANT! If a character is represented by multiple tokens, we currently do not consider it.
        # This is an important limitation, since it currently does not support chinese, korean, and other languages. Nor emojis :(
        try:
            token_ids = self.tokenizer(
                list(decoded_token), return_tensors="pt", add_special_tokens=False
            ).input_ids
        except ValueError as e:
            # print(f"Error decoding token {idx} ({decoded_token}): {e}. ")
            return None

        for token_id, char in zip(token_ids, list(decoded_token)):
            assert (
                self.tokenizer.convert_ids_to_tokens([token_id[0]])[0] in self.vocab_set
            ), f"Character '{char}' (idx {idx}) from token '{decoded_token}' not in tokenizer vocab!"

        # note that we are splitting a word into a sequence with a "whitespace" token as well!
        return self.embeddings(token_ids).squeeze(1)

    def _create_dataset(self):
        """
        Create the dataset by splitting the embeddings into individual tokens.
        """

        # Load the split indices
        split_indices = []
        with open(f"{self.split_files_path}/{self.split}_indices.txt", "r") as f:
            for line in f:
                split_indices.append(int(line.strip()))

        features = []
        labels = []
        for idx in tqdm(split_indices, desc="Creating dataset"):
            split = self.split_embedding(idx)
            if split is not None:
                features.append(split.detach())
                labels.append(self.embeddings.weight[idx].detach())

        return features, labels

    def __len__(self):
        """Returns the size of the dataset."""
        return len(self.features)

    def __getitem__(self, idx):
        """
        Returns a sample from the dataset.

        Args:
            idx (int): Index of the sample to return

        Returns:
            tuple: (features, labels) where features are the embeddings and labels are the corresponding token embeddings
        """
        return self.features[idx], self.labels[idx]
