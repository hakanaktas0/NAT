import torch
from torch_geometric.data import Data, Dataset, DataLoader


from transformers import GPT2TokenizerFast, GPT2Model


def substring_boundaries(text: str, substring: str,boundaries : list) -> list:
    """
    After BPE, if the substring is tokenized into multiple tokens, make it one token again.
    """
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
        for i in range(idx,end_pos):
            boundaries[i] = 0
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
                 transform=None,
                 pre_transform=None,
                 used_llm='GPT2',
                 device='cuda'):
        """
        texts: list of strings
        conditions: list of strings (e.g. ['normal', 'counting', ...])
        substrings: list of substrings (used only if condition='counting')
        """
        super().__init__(transform, pre_transform)
        self.texts = texts
        self.conditions = conditions
        self.substrings = substrings if substrings is not None else ["" for _ in texts]
        self.device = device
        if used_llm == 'GPT2':
            self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
            self.model = GPT2Model.from_pretrained("gpt2")
            self.model.eval()  # Set to eval mode
            self.embedding_layer = self.model.get_input_embeddings()
        # TODO add support for other LLMs



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


    def __getitem__(self, idx):
        text = self.texts[idx]
        condition = self.conditions[idx]
        substring = self.substrings[idx]
        # 1) Build node features (one node per character)
        x = self.get_character_embeddings(text)

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
            tokens, boundaries = self.get_boundaries(text)
        else:  # counting mode
            tokens, boundaries = self.get_boundaries(text) # Get boundaries
            boundaries = substring_boundaries(text, substring,boundaries) # make sure the substring is tokenized as 1 token

        y = torch.tensor(boundaries, dtype=torch.float)


        cond_val = 0 if condition == "normal" else 1
        cond_tensor = torch.tensor([cond_val], dtype=torch.float)

        data = Data(
            x=x,
            edge_index=edge_index,
            y=y
        )

        # multiply the substring embed with condition, if condition is normal, the embedding will be all 0s, if coundting, the embedding will be the substring embedding
        data.substring_embed = self.embedding_layer(self.tokenizer(substring, return_tensors="pt")['input_ids'].view(1, -1)).view(1,-1) * cond_tensor

        return data.to(self.device)
