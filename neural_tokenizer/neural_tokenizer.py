import copy

import torch
from transformers import LlamaModel, PreTrainedTokenizerFast
from torch_geometric.data import Data

from neural_tokenizer.models import ConditionalGNN
from neural_tokenizer.dataset import substring_boundaries
from torch_geometric.loader import DataLoader


class NeuralTokenizer:
    def __init__(self, gnn_model_path: str, language_model_path: str, device):
        self.device = device
        self.language_model_path = language_model_path
        self.gnn, self.connection_distance = self.initialize_neural_tokenizer(
            gnn_model_path
        )
        lm, self.static_tokenizer = self.initialize_static_tokenizer()
        self.embed_tokens = copy.deepcopy(lm.embed_tokens)
        del lm

    def initialize_static_tokenizer(self):
        model = LlamaModel.from_pretrained(
            self.language_model_path, cache_dir="./.cache"
        )

        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            self.language_model_path, cache_dir="./.cache"
        )
        tokenizer.pad_token = tokenizer.eos_token
        model.eval()
        return model, tokenizer

    def initialize_neural_tokenizer(self, model_path: str):
        neural_tokenizer_gnn = ConditionalGNN(
            in_channels=2048,
            condition_emb_dim=2048,
            hidden_dim=256,
            num_layers=2,
        ).to(self.device)

        neural_tokenizer_gnn.load_state_dict(
            torch.load(
                model_path,
                # weights_only=False,
                map_location=self.device,
            )
        )
        connection_distance = 1
        return neural_tokenizer_gnn, connection_distance

    def get_character_embeddings(self, text):
        chars = list(text)
        char_input_ids = self.static_tokenizer(
            chars,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )["input_ids"][:, 1]
        with torch.no_grad():
            raw_embeddings = self.embed_tokens(char_input_ids)
        return raw_embeddings.squeeze(0)

    def get_boundaries(self, text):
        encoded = self.static_tokenizer(
            text,
            return_offsets_mapping=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        offsets = encoded["offset_mapping"][0].tolist()
        input_ids = encoded["input_ids"][0].tolist()
        tokens = self.static_tokenizer.convert_ids_to_tokens(input_ids)
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

    def prepare_inputs(self, text, substring=None):
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
            edge_index = torch.zeros(
                (2, 0), dtype=torch.long
            )  # no edges if only one character

        src, dst = [], []
        for i in range(num_nodes):
            for offset in range(1, self.connection_distance + 1):
                if i + offset < num_nodes:
                    src += [i, i + offset]
                    dst += [i + offset, i]

        edge_index = (
            torch.tensor([src, dst], dtype=torch.long)
            if src
            else torch.zeros((2, 0), dtype=torch.long)
        )

        # 3) Build the label y = boundary(0/1) for each character
        if substring is None:
            substring = text[0]
            _, boundaries = self.get_boundaries(text)
            _, borders = substring_boundaries(
                text, substring, copy.deepcopy(boundaries)
            )
            cond_val = 0
        else:  # counting mode
            _, boundaries = self.get_boundaries(text)  # Get boundaries
            boundaries, borders = substring_boundaries(
                text, substring, boundaries
            )  # make sure the substring is tokenized as 1 token
            cond_val = 1

        cond_tensor = torch.tensor([cond_val], dtype=torch.float)

        data = Data(x=x, edge_index=edge_index)
        data.condition = cond_val

        data.borders = borders

        # multiply the substring embed with condition, if condition is normal, the embedding will be all 0s, if coundting, the embedding will be the substring embedding

        data.substring_embed = (
            self.embed_tokens(
                self.static_tokenizer(substring, return_tensors="pt")["input_ids"][:, 1]
            )
            * cond_tensor
        )
        return data

    def tokenize(self, text: str, search_substring=None):
        sample = self.prepare_inputs(text, search_substring)

        with torch.no_grad():
            return (
                (
                    self.gnn(
                        sample.x.to(self.device),
                        sample.edge_index.to(self.device),
                        sample.substring_embed.to(self.device),
                        torch.zeros(sample.x.shape[0]).int().to(self.device),
                    )
                    >= 0
                )
                .int()
                .detach()
                .cpu(),
                sample.x,
            )
