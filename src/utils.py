import torch
import random
from nltk.corpus import words
import pandas as pd
from nltk.corpus import brown
from collections import Counter

def evaluate(model, dataloader):
    """
    Evaluates the model on a given dataloader and returns accuracy.
    Accuracy = #correct boundary predictions / total boundaries
    """
    model.eval()
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for batch in dataloader:
            # batch.x: [num_nodes, embedding_dim]
            # batch.edge_index: [2, num_edges]
            # batch.y: [num_nodes] with 0/1 boundaries
            # batch.condition: [1] specifying 0/1 condition
            logits = model(batch.x, batch.edge_index,batch.substring_embed,batch.batch)

            # Convert logits to predictions (threshold=0 for BCEWithLogits)
            preds = (logits >= 0).long()
            # Count how many are correct
            correct = (preds == batch.y.long()).sum().item()
            total_correct += correct
            total_count += batch.y.numel()

    accuracy = total_correct / total_count if total_count > 0 else 0.0
    return accuracy




def generate_data(num_samples = 1000):
    # word_list = words.words()
    def generate_sentence(word_count):
        return ' '.join(random.choices(word_list, k=word_count))

    word_freq = Counter(w.lower() for w in brown.words())

    word_list = [word for word, freq in word_freq.most_common(1500)][500:]
    substring_pool = ["test", "count", "token", "data", "check"]
    data = []
    texts = []
    conditions = []
    substrings = []
    for _ in range(num_samples):
        condition = random.choice(["normal", "counting"])
        condition = 'normal'
        substring = random.choice(substring_pool)
        if condition == "normal":
            text = generate_sentence(random.randint(4, 10))
        else:
            text = generate_sentence(random.randint(4, 10))
            # Ensure the substring is in the text
            text += ' ' + substring
        texts.append(text)
        conditions.append(condition)
        substrings.append(substring)
    return texts, conditions, substrings




