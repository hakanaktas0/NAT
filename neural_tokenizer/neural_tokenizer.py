import torch

from neural_tokenizer.models import ConditionalGNN
from neural_tokenizer.dataset import ConditionalTokenizationDataset
from torch_geometric.loader import DataLoader


class NeuralTokenizer:
    def __init__(self, model_path: str, device):
        self.device = device
        self.gnn = self.initialise_tokenizer(model_path)

    def initialise_tokenizer(self, model_path: str):
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
        return neural_tokenizer_gnn

    def tokenize(self, text: str, search_substring=None):
        if search_substring is None:
            condition = "normal"
            search_substring = text[0]
        else:
            condition = "counting"

        # TODO! This is a pretty damn bad way of doing it but I needed a quick hack.
        # TODO! It reloads the entire llama model every time.
        # TODO! Needs changes in the Dataset class or diverge from using the dataset class.
        dataset = ConditionalTokenizationDataset(
            [text],
            [condition],
            [search_substring],
            used_llm="Llama-3.2-1B",
            connection_distance=1,
        )
        val_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        sample = next(iter(val_dataloader))

        self.gnn.eval()
        with torch.no_grad():
            return (
                (
                    self.gnn(
                        sample.x.to(self.device),
                        sample.edge_index.to(self.device),
                        sample.substring_embed.to(self.device),
                        sample.batch.to(self.device),
                    )
                    >= 0
                )
                .int()
                .detach()
                .cpu(),
                sample.x,
            )
