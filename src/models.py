import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv



class ConditionalGNN(nn.Module):
    def __init__(self,
                 in_channels=768,
                 condition_emb_dim=768,
                 hidden_dim=128,
                 num_layers=2):
        super().__init__()


        # GCN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels + condition_emb_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # Output layer (binary boundary classification: 1 or 0)
        self.linear_out = nn.Linear(hidden_dim, 1)

        self.relu = nn.ReLU()

    def forward(self, x, edge_index,substring_embed,batch):
        """
        x: [num_nodes, in_channels]
        condition: [1]-shaped tensor with condition index
        """
        # condition_emb: [1, condition_emb_dim]
        # cond_emb = self.condition_embedding(condition)  # shape [1, emb_dim]

        # We want to broadcast cond_emb to each node in x
        # x: [num_nodes, in_channels], so replicate cond_emb for each node:
        # num_nodes = x.size(0)
        # substring_embed = substring_embed[batch]
        # substring_embed = substring_embed.repeat(num_nodes, 1)  # [num_nodes, condition_emb_dim]

        substring_embed_batched = substring_embed[batch]

        # Concat
        x_cond = torch.cat([x, substring_embed_batched], dim=-1)  # [num_nodes, in_channels + condition_emb_dim]

        # Pass through GCN layers
        for conv in self.convs:
            x_cond = conv(x_cond, edge_index)
            x_cond = self.relu(x_cond)

        # Output layer
        logits = self.linear_out(x_cond).squeeze(-1)  # shape [num_nodes]
        return logits
