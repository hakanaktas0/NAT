import torch
import random
from nltk.corpus import words
import pandas as pd
from nltk.corpus import brown
from collections import Counter
from sklearn.metrics import precision_recall_fscore_support


def evaluate(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            logits = model(batch.x, batch.edge_index, batch.substring_embed, batch.batch)
            preds = (logits >= 0).long()
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch.y.long().cpu().tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', pos_label=1
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": sum([p == l for p, l in zip(all_preds, all_labels)]) / len(all_labels)
    }


def generate_data(num_samples=1000, balanced=True,vocabulary_word_size=5000):
    # word_list = words.words()
    def generate_sentence(word_count):
        return ' '.join(random.choices(word_list, k=word_count))

    word_freq = Counter(w.lower() for w in brown.words())

    word_list = [word for word, freq in word_freq.most_common(vocabulary_word_size)]
    substring_pool = ["test", "count", "token", "data", "check"]
    data = []
    texts = []
    conditions = []
    substrings = []
    if balanced:
        # each iteration adds 2 samples when balanced
        num_samples = num_samples // 2
    for _ in range(num_samples):
        if balanced:
            text = generate_sentence(random.randint(20, 40))
            substring = random.choice(substring_pool)
            text += ' ' + substring
            # APPEND TWICE TO USE THE SAME SENTENCE BOTH FOR NORMAL AND COUNTING
            texts.append(text)
            texts.append(text)
            conditions.append('normal')
            conditions.append('counting')
            substrings.append(substring)
            substrings.append(substring)
        else:
            condition = random.choice(["normal", "counting"])
            condition = 'normal'
            substring = random.choice(substring_pool)
            if condition == "normal":
                text = generate_sentence(random.randint(20, 40))
            else:
                text = generate_sentence(random.randint(20, 40))
                # Ensure the substring is in the text
                text += ' ' + substring
            texts.append(text)
            conditions.append(condition)
            substrings.append(substring)

    return texts, conditions, substrings




