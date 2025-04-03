import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.nn import GCNConv
import numpy as np

from transformers import GPT2TokenizerFast, GPT2Model

def pseudo_bpe_tokenizer_boundaries(text: str) -> list:
    """
    Returns a list of 0/1 boundaries for each character.
    This is a stub simulating boundary detection from a BPE tokenizer.
    In practice, you would do something like:

        tokens = bert_tokenizer.tokenize(text)
        # Convert tokens back to positions/characters ...
        # Then mark boundary positions with 1.
    """
    boundaries = [0]*len(text)
    # For simplicity: insert a 'boundary' after punctuation, spaces, or at the end.
    # This is just a dummy rule for demonstration:
    for i, ch in enumerate(text):
        if ch.isspace() or ch in {'.', ',', ';', '!', '?'}:
            boundaries[i] = 1

    if len(text) > 0:
        boundaries[-1] = 1  # end of the string is always a boundary
    return boundaries


def substring_boundaries(text: str, substring: str) -> list:
    """
    Return a list of 0/1 boundaries for each character,
    ensuring that occurrences of `substring` are tokenized as standalone pieces.
    """
    boundaries = [0] * len(text)
    start_index = 0

    # For each occurrence, set boundary on start-1 (if valid) and end
    while True:
        idx = text.find(substring, start_index)
        if idx == -1:
            break
        # substring found at [idx, idx + len(substring))
        # Mark the boundary right before it if not beginning:
        if idx - 1 >= 0:
            boundaries[idx - 1] = 1
        # Mark boundary at the end of substring
        end_pos = idx + len(substring) - 1
        boundaries[end_pos] = 1

        start_index = end_pos + 1

    # Also mark the end of the entire string as a boundary if not already
    if len(text) > 0:
        boundaries[-1] = 1
    return boundaries


class ConditionalTokenizationDataset(Dataset):
    def __init__(self,
                 texts,
                 conditions,
                 substrings=None,
                 char2idx=None,
                 embedding_dim=32,
                 transform=None,
                 pre_transform=None,
                 device='cuda'):
        """
        texts: list of strings
        conditions: list of strings (e.g. ['normal', 'counting', ...])
        substrings: list of substrings (same length as texts, used only if condition='counting')
        char2idx: dict to map characters to integer embeddings
        embedding_dim: dimension for random embedding initialization if char not in char2idx
        """
        super().__init__(transform, pre_transform)
        self.texts = texts
        self.conditions = conditions
        self.substrings = substrings if substrings is not None else ["" for _ in texts]
        self.char2idx = char2idx if char2idx is not None else {}
        self.embedding_dim = embedding_dim
        self.device = device

        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.model = GPT2Model.from_pretrained("gpt2")
        self.model.eval()  # Set to eval mode
        self.embedding_layer = self.model.get_input_embeddings()


        # If needed, create random embeddings for known chars
        self.char_embeddings = {}
        for ch in self.char2idx:
            # random vector for each character
            self.char_embeddings[ch] = torch.randn(embedding_dim)

    def __len__(self):
        return len(self.texts)

    def get_character_embeddings(self, text):
        chars = list(text)

        char_input_ids = self.tokenizer(chars, return_tensors="pt")['input_ids'].view(1, -1)

        with torch.no_grad():
            raw_embeddings = self.embedding_layer(char_input_ids)  # Shape: (1, seq_len, hidden_size)
        return raw_embeddings.squeeze(0)


    def get_boundaries(self,text):
        encoded = self.tokenizer(text, return_offsets_mapping=True, return_tensors='pt', add_special_tokens=False)
        offsets = encoded['offset_mapping'][0].tolist()
        input_ids = encoded['input_ids'][0].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        # Create boundary array
        boundaries = [0] * len(text)
        for i, (start, end) in enumerate(offsets):
            for j in range(start, end):
                boundaries[j] = i  # or 1 if you want just 0/1 marking token edges

        mask = [0] * len(boundaries)
        prev = boundaries[0]
        for i in range(1, len(boundaries)):
            if boundaries[i] != prev:
                mask[i] = 1
                prev = boundaries[i]
        return tokens, mask

        # return tokens, boundaries

    def __getitem__(self, idx):
        text = self.texts[idx]
        condition = self.conditions[idx]
        substring = self.substrings[idx]

        x = self.get_character_embeddings(text)
        # 1) Build node features (one node per character)
        #    We will have x.shape = (num_chars, embedding_dim)
        # char_embs = []
        # for ch in text:
        #     char_embs.append(self.get_character_embedding(ch).unsqueeze(0))
        # x = torch.cat(char_embs, dim=0) if len(char_embs) > 0 else torch.zeros((0, self.embedding_dim))

        # 2) Build adjacency (chain edges) for each consecutive character
        num_nodes = len(text)
        if num_nodes > 1:
            # edges: (0->1,1->0,1->2,2->1,...)
            src = list(range(num_nodes - 1))
            dst = list(range(1, num_nodes))
            edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)  # no edges if only one character


        # 3) Build the label y = boundary(0/1) for each character
        if condition == "normal":
            # boundaries = pseudo_bpe_tokenizer_boundaries(thext)
            tokens, boundaries = self.get_boundaries(text)
        else:  # counting mode
            boundaries = substring_boundaries(text, substring)

        y = torch.tensor(boundaries, dtype=torch.float)

        # 4) Optionally, store the condition in a numeric format
        #    For simplicity: 0 = normal, 1 = counting
        cond_val = 0 if condition == "normal" else 1
        cond_tensor = torch.tensor([cond_val], dtype=torch.float)

        data = Data(
            x=x,
            edge_index=edge_index,
            y=y
        )

        # We can store the condition + substring if needed
        data.condition = cond_tensor
        data.substring_embed = self.embedding_layer(self.tokenizer(substring, return_tensors="pt")['input_ids'].view(1, -1)).view(-1)

        return data.to(self.device)
