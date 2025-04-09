import os
import time
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import nltk
import numpy as np
from tqdm import tqdm

from dataset import ConditionalTokenizationDataset
from models import ConditionalGNN, ConditionalGAT, ConditionalMPNN
from utils import evaluate, generate_data, generate_wiki_data

# from dotenv import load_dotenv
# import os

# load_dotenv()  # Loads from .env by default


def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed(157)
print("All seeds set.")


nltk.download("words")
nltk.download("brown")

use_wandb = False


layer_count = 2
model_spec = "GCN"
train_data_size = 250000
val_data_size = 100
connection_distance = 1
balanced_train_data = True
balanced_val_data = True
hidden_dim = 256
vocabulary_word_size = 1000
sentence_length = 30
used_llm = "Llama-3.2-1B"
# used_llm = 'GPT2'
timestamp = time.strftime("%Y%m%d_%H%M%S")
save_dir = f"./model_save/checkpoints-{timestamp}"
# Create save directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("USING CUDA")
else:
    device = torch.device("cpu")
    print("USING CPU")

# TODO WORDS ONLY TESTED USING GPT2 TOKENIZER - SHOULD BE TESTED FOR DIFFERENT LLMs
string_to_add = [
    "bedroom",
    "mailbox",
    "football",
    "breakdown",
    "fingerprint",
    "underdog",
    "smartphone",
    "website",
    "headphone",
    "database",
    "keyboard",
]
string_to_search = [
    "room",
    "mail",
    "ball",
    "down",
    "print",
    "dog",
    "phone",
    "site",
    "head",
    "data",
    "board",
]

texts, conditions, substrings = generate_data(
    num_samples=train_data_size,
    balanced=balanced_train_data,
    vocabulary_word_size=vocabulary_word_size,
    string_to_add=string_to_add,
    string_to_search=string_to_search,
)


val_texts, val_conditions, val_substrings = generate_data(
    num_samples=val_data_size,
    balanced=balanced_val_data,
    vocabulary_word_size=vocabulary_word_size,
    string_to_add=string_to_add,
    string_to_search=string_to_search,
)

# val_texts, val_conditions, val_substrings = generate_wiki_data(
#     num_samples=val_data_size
# )

# Create dataset
dataset = ConditionalTokenizationDataset(
    texts,
    conditions,
    substrings,
    used_llm=used_llm,
    connection_distance=connection_distance,
)


val_dataset = ConditionalTokenizationDataset(
    val_texts,
    val_conditions,
    val_substrings,
    used_llm=used_llm,
    connection_distance=connection_distance,
)

batch_size = 1
# Wrap in dataloader
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

val_dataloader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True
)

if used_llm == "GPT2":
    embedding_size = 768
if used_llm == "Llama-3.2-1B":
    embedding_size = 2048

# Instantiate model
if model_spec == "GCN":
    model = ConditionalGNN(
        in_channels=embedding_size,
        condition_emb_dim=embedding_size,
        hidden_dim=hidden_dim,
        num_layers=layer_count,
    ).to(device)
if model_spec == "GAT":
    model = ConditionalGAT(
        in_channels=embedding_size,
        condition_emb_dim=embedding_size,
        hidden_dim=hidden_dim,
        num_layers=layer_count,
    ).to(device)
if model_spec == "MPNN":
    model = ConditionalMPNN(
        in_channels=embedding_size,
        condition_emb_dim=embedding_size,
        hidden_dim=hidden_dim,
        num_layers=layer_count,
    ).to(device)


lr = 1e-3
epoch_num = 1
if use_wandb:
    import wandb

    run = wandb.init(
        project="embeddings-gnn",
        name=f"run-training-{timestamp}",
        config={
            "layer_count": layer_count,
            "model_spec": model_spec,
            "train_data_size": train_data_size,
            "val_data_size": val_data_size,
            "balanced_train_data": balanced_train_data,
            "balanced_val_data": balanced_val_data,
            "hidde_dim": hidden_dim,
            "vocabulary_word_size": vocabulary_word_size,
            "used_llm": used_llm,
            "batch_size": batch_size,
            "lr": lr,
            "epochs": epoch_num,
        },
    )


optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCEWithLogitsLoss()  # since we’re predicting boundary 0 or 1

# Training loop
for epoch in range(epoch_num):
    total_loss = 0
    step = 0

    model.train()
    for batch in tqdm(dataloader, desc=f"Training epoch {epoch}"):
        # In PyG, if batch_size>1, we get a merged graph, but here it’s 1 for illustration
        x, edge_index, y, substring_embed, batch_vector = (
            batch.x.to(device),
            batch.edge_index.to(device),
            batch.y.to(device),
            batch.substring_embed.to(device),
            batch.batch.to(device),
        )
        # condition = batch.condition  # shape [batch_size=1, 1]

        optimizer.zero_grad()
        logits = model(x, edge_index, substring_embed, batch_vector)

        # We have a boundary label for each node
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        step += 1

        if step % 1000 == 0:
            results = evaluate(model, val_dataloader, device)
            results["loss"] = total_loss / step
            if use_wandb:
                wandb.log(results)
            print(f"Accuracy: {results['accuracy']:.4f}")
            print(f"Recall: {results['recall']:.4f}")
            print(f"Precision: {results['precision']:.4f}")
            print(f"F1: {results['f1']:.4f}")
            print(f"Balanced Accuracy : {results['balanced_accuracy']:.4f}")
            print(f"Counting Accuracy : {results['counting_accuracy']:.4f}")
            print(f"Non Counting Accuracy : {results['non_counting_accuracy']:.4f}")
            print(f"Correct splits : {results['groupped_accuracy']:.4f}")

        if step % 20_000 == 0:
            torch.save(model.state_dict(), f"./{save_dir}/save_{epoch}.pth")
        print(f"Epoch {epoch}, Loss: {(total_loss/step):.4f}")

torch.save(model.state_dict(), f"./{save_dir}/save_final_model.pth")
