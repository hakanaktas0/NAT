import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.nn import GCNConv
import numpy as np
from dataset import ConditionalTokenizationDataset
from models import ConditionalGNN
from utils import evaluate, generate_data
import nltk
nltk.download('words')
from nltk.corpus import brown
nltk.download('brown')

# Sample data
# texts = [
#     "Hello world!",
#     "Counting test for substring",
#     "Another normal example"
# ]
# conditions = [
#     "normal",
#     "normal",
#     "normal"
# ]
# substrings = [
#     "test",
#     "test",
#     "test"
# ]

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

texts, conditions, substrings = generate_data(500)

# Create dataset
dataset = ConditionalTokenizationDataset(
    texts, conditions, substrings,
    char2idx=None,  # or supply a dictionary
    embedding_dim=32
)

val_texts, val_conditions, val_substrings = generate_data(100)

val_dataset = ConditionalTokenizationDataset(
    val_texts, val_conditions, val_substrings,
    char2idx=None, embedding_dim=32
)

# Wrap in dataloader
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Instantiate model
model = ConditionalGNN(in_channels=768, condition_emb_dim=768, hidden_dim=128, num_layers=2).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()  # since we’re predicting boundary 0 or 1

# Training loop
for epoch in range(1000000):
    total_loss = 0
    for batch in dataloader:
        # In PyG, if batch_size>1, we get a merged graph, but here it’s 1 for illustration
        x, edge_index, y, substring_embed = batch.x, batch.edge_index, batch.y, batch.substring_embed
        condition = batch.condition  # shape [batch_size=1, 1]

        optimizer.zero_grad()
        logits = model(x, edge_index, condition, substring_embed)

        # We have a boundary label for each node
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch}, Loss: {total_loss:.4f}")
    if epoch % 10 == 0:
        accuracy = evaluate(model, val_dataloader)
        print(f"Accuracy: {accuracy:.4f}")
