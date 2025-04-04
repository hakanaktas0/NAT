import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, MessagePassing



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




class ConditionalGAT(nn.Module):
    def __init__(self,
                 in_channels=768,
                 condition_emb_dim=768,
                 hidden_dim=128,
                 num_layers=2,
                 heads=4):
        super().__init__()

        self.layers = nn.ModuleList()
        self.layers.append(
            GATConv(in_channels + condition_emb_dim, hidden_dim, heads=heads, concat=True)
        )
        for _ in range(num_layers - 2):
            self.layers.append(
                GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True)
            )
        self.layers.append(
            GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=True)
        )

        self.output_layer = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, substring_embed, batch):
        """
        x: [num_nodes, in_channels] - character embeddings
        substring_embed: [batch_size, condition_emb_dim] - condition vector per graph
        batch: [num_nodes] - indicates graph ID per node
        """
        # Broadcast substring_embed to each node in the batch
        cond_expanded = substring_embed[batch]  # [num_nodes, condition_emb_dim]

        x = torch.cat([x, cond_expanded], dim=-1)  # [num_nodes, in_channels + condition_emb_dim]

        for layer in self.layers:
            x = self.relu(layer(x, edge_index))

        out = self.output_layer(x).squeeze(-1)  # [num_nodes]
        return out



class ConditionalMPNN(nn.Module):
    def __init__(self,
                 in_channels=768,
                 condition_emb_dim=768,
                 hidden_dim=128,
                 num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(CustomMPNNLayer(in_channels + condition_emb_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(CustomMPNNLayer(hidden_dim, hidden_dim))

        self.output_layer = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, substring_embed, batch):
        cond_expanded = substring_embed[batch]  # [num_nodes, cond_dim]
        x = torch.cat([x, cond_expanded], dim=-1)

        for layer in self.layers:
            x = self.relu(layer(x, edge_index))

        return self.output_layer(x).squeeze(-1)


class CustomMPNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='mean')
        self.lin_msg = nn.Linear(in_channels, out_channels)
        self.lin_update = nn.Linear(in_channels + out_channels, out_channels)

    def forward(self, x, edge_index):
        x_trans = self.lin_msg(x)
        return self.propagate(edge_index=edge_index, x=x, x_trans=x_trans)

    def message(self, x_trans_j):
        return x_trans_j

    def update(self, aggr_out, x):
        return self.lin_update(torch.cat([x, aggr_out], dim=1))
