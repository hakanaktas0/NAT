import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from dataset import ConditionalTokenizationDataset
from models import ConditionalGNN
from utils import evaluate, generate_data
import nltk
nltk.download('words')
nltk.download('brown')

use_wandb = False


layer_count = 2
model_spec = 'GCN'
train_data_size = 1000
val_data_size = 100
balanced_train_data = True
balanced_val_data = True
hidden_dim = 128
vocabulary_word_size = 5000
used_llm = 'GPT2'

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('USING CUDA')
else:
    device = torch.device('cpu')
    print('USING CPU')




if use_wandb:
    import wandb
    run = wandb.init(
        project="L65-NAT",
        config={
            'layer_count': layer_count,
            'model_spec': model_spec,
            'train_data_size': train_data_size,
            'val_data_size': val_data_size,
            'balanced_train_data': balanced_train_data,
            'balanced_val_data': balanced_val_data,
            'hidde_dim': hidden_dim,
            'vocabulary_word_size': vocabulary_word_size,
            'used_llm': used_llm,
        }
    )



texts, conditions, substrings = generate_data(num_samples=train_data_size,balanced=balanced_train_data,vocabulary_word_size=vocabulary_word_size)

# Create dataset
dataset = ConditionalTokenizationDataset(
    texts, conditions, substrings,
    device=device
)

val_texts, val_conditions, val_substrings = generate_data(num_samples=val_data_size,balanced=balanced_val_data,vocabulary_word_size=vocabulary_word_size)

val_dataset = ConditionalTokenizationDataset(
    val_texts, val_conditions, val_substrings,
    device=device
)

# Wrap in dataloader
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Instantiate model
model = ConditionalGNN(in_channels=768, condition_emb_dim=768, hidden_dim=hidden_dim, num_layers=layer_count).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()  # since we’re predicting boundary 0 or 1

# Training loop
for epoch in range(1000000):
    total_loss = 0
    # if epoch % 10 == 0:
    results = evaluate(model, val_dataloader)
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"F1: {results['f1']:.4f}")
    if use_wandb:
        wandb.log(results)

    model.train()
    for batch in dataloader:
        # In PyG, if batch_size>1, we get a merged graph, but here it’s 1 for illustration
        x, edge_index, y, substring_embed, batch_vector = batch.x, batch.edge_index, batch.y, batch.substring_embed, batch.batch
        # condition = batch.condition  # shape [batch_size=1, 1]

        optimizer.zero_grad()
        logits = model(x, edge_index, substring_embed,batch_vector)

        # We have a boundary label for each node
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    if use_wandb:
        wandb.log({'loss': total_loss})

    print(f"Epoch {epoch}, Loss: {total_loss:.4f}")
