import torch
import random
from nltk.corpus import words
from nltk.corpus import brown
from collections import Counter
from sklearn.metrics import precision_recall_fscore_support, balanced_accuracy_score
from torch_geometric.data import Data, Dataset, DataLoader
import os
from dataset import substring_boundaries

from transformers import (
    GPT2TokenizerFast,
    GPT2Model,
    LlamaModel,
    PreTrainedTokenizerFast,
)


def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    counting_acc = []
    non_counting_acc = []
    all_preds_grouped = []
    all_labels_grouped = []
    with torch.no_grad():
        for batch in dataloader:
            logits = model(
                batch.x.to(device),
                batch.edge_index.to(device),
                batch.substring_embed.to(device),
                batch.batch.to(device),
            )
            preds = (logits >= 0).long().cpu()
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch.y.long().tolist())
            all_preds_grouped.append(preds.cpu())
            all_labels_grouped.append(batch.y.long())
            if bool(batch.condition == 1):
                counting_acc.extend(
                    (preds[batch.borders[0]] == batch.y[batch.borders[0]]).tolist()
                )
            else:
                non_counting_acc.extend(
                    (preds[batch.borders[0]] == batch.y[batch.borders[0]]).tolist()
                )
    counting_accuracy = 0
    non_counting_accuracy = 0
    if len(counting_acc) != 0:
        counting_accuracy = sum(counting_acc) / len(counting_acc)
    if len(non_counting_acc) != 0:
        non_counting_accuracy = sum(non_counting_acc) / len(non_counting_acc)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", pos_label=1
    )
    balanced_accuracy = balanced_accuracy_score(all_labels, all_preds)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": sum([p == l for p, l in zip(all_preds, all_labels)])
        / len(all_labels),
        "balanced_accuracy": balanced_accuracy,
        "counting_accuracy": counting_accuracy,
        "non_counting_accuracy": non_counting_accuracy,
        "grouped_accuracy": sum(
            [torch.equal(p, l) for p, l in zip(all_preds_grouped, all_labels_grouped)]
        )
        / len(all_labels_grouped),
    }


def generate_data(
    num_samples=1000,
    balanced=True,
    vocabulary_word_size=5000,
    sentence_lenght=30,
    string_to_add=None,
    string_to_search=None,
):
    # word_list = words.words()
    def generate_sentence(word_count, substring=None, num_sub=1):
        words = random.choices(word_list, k=word_count)
        if substring is not None:
            for _ in range(num_sub):
                words.append(substring)
            random.shuffle(words)
        return " ".join(words)

    word_freq = Counter(w.lower() for w in brown.words())

    word_list = [
        word for word, freq in word_freq.most_common(vocabulary_word_size + 250)
    ][
        250:
    ]  # remove the 250 to get rid of too short words
    texts = []
    conditions = []
    substrings = []
    if balanced:
        # each iteration adds 2 samples when balanced
        num_samples = num_samples // 2
    for _ in range(num_samples):
        idx = random.randint(0, 10)
        substring = string_to_add[idx]
        search_substring = string_to_search[idx]
        num_sub = random.randint(1, 5)
        text = generate_sentence(sentence_lenght, substring=substring, num_sub=num_sub)

        if balanced:
            # text = generate_sentence(random.randint(20, 40))
            # substring = random.choice(substring_pool)
            # num_sub = random.randint(1, 5)
            # text = generate_sentence(sentence_lenght,substring=substring,num_sub=num_sub)
            # APPEND TWICE TO USE THE SAME SENTENCE BOTH FOR NORMAL AND COUNTING
            texts.append(text)
            texts.append(text)
            conditions.append("normal")
            conditions.append("counting")
            substrings.append(search_substring)
            substrings.append(search_substring)
        else:
            condition = random.choice(["normal", "counting"])
            # condition = 'normal'
            # substring = random.choice(substring_pool)
            # num_sub = random.randint(1, 5)
            # text = generate_sentence(sentence_lenght,substring=substring,num_sub=num_sub)
            texts.append(text)
            conditions.append(condition)
            substrings.append(search_substring)

    return texts, conditions, substrings


def generate_wiki_data(num_samples=100):
    from datasets import load_dataset

    ds = load_dataset("wikimedia/wikipedia", "20231101.en", cache_dir="./.cache")
    all_sentences = []
    i = 0
    while True:
        sentences = ds["train"][i]["text"].split("\n")
        for j in range(len(sentences)):
            if len(sentences[j]) < 250:
                sentences[j] = ""
        while "" in sentences:
            sentences.remove("")
        short_sentences = []
        for sentence in sentences:
            short_sentences.extend(sentence.split("."))
        for j in range(len(short_sentences)):
            if len(short_sentences[j]) < 20:
                short_sentences[j] = ""
        while "" in short_sentences:
            short_sentences.remove("")
        all_sentences.extend(short_sentences)
        if len(all_sentences) >= num_samples:
            break
        i += 1

    substrings = []
    conditions = []
    for i in range(num_samples):
        substrings.append("room")
        conditions.append("normal")
    return all_sentences[:num_samples], conditions, substrings


# A BAD IDEA
# class SyntheticDataGenerator:
#     def __init__(self,
#                  num_samples=1000,
#                  balanced=True,
#                  vocabulary_word_size=5000,
#                  sentence_lenght=30,
#                  string_to_add=None,string_to_search=None,
#                  used_llm='GPT2',
#                  device='cuda'):
#         super().__init__()
#         self.num_samples = num_samples
#         self.balanced = balanced
#         self.vocabulary_word_size = vocabulary_word_size
#         self.sentence_lenght = sentence_lenght
#         self.string_to_add = string_to_add
#         self.string_to_search = string_to_search
#         self.device = device
#         self.used_llm = used_llm
#         if used_llm == 'GPT2':
#             self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
#             self.model = GPT2Model.from_pretrained("gpt2")
#             self.model.eval()  # Set to eval mode
#             self.embedding_layer = self.model.get_input_embeddings()
#         if used_llm == 'Llama-3.2-1B':
#             self.model = LlamaModel.from_pretrained("meta-llama/Llama-3.2-1B", token=os.getenv('HG_TOKEN'))
#             self.tokenizer = PreTrainedTokenizerFast.from_pretrained("meta-llama/Llama-3.2-1B",
#                                                                 token=os.getenv('HG_TOKEN'))
#             self.model.eval()
#
#     def get_character_embeddings(self, text):
#         if self.used_llm == 'GPT2':
#             chars = list(text)
#             char_input_ids = self.tokenizer(chars, return_tensors="pt")['input_ids'].view(1, -1)
#             with torch.no_grad():
#                 raw_embeddings = self.embedding_layer(char_input_ids)  # Shape: (1, seq_len, hidden_size)
#             return raw_embeddings.squeeze(0)
#         elif self.used_llm == 'Llama-3.2-1B':
#             chars = list(text)
#             char_input_ids = self.tokenizer(chars, return_tensors="pt")['input_ids'][:,1]
#             with torch.no_grad():
#                 raw_embeddings = self.model.embed_tokens(char_input_ids)
#             return raw_embeddings.squeeze(0)
#
#
#
#     def get_boundaries(self,text):
#         encoded = self.tokenizer(text, return_offsets_mapping=True, return_tensors='pt', add_special_tokens=False)
#         offsets = encoded['offset_mapping'][0].tolist()
#         input_ids = encoded['input_ids'][0].tolist()
#         tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
#
#         # Create boundary array
#         boundaries = [0] * len(text)
#         for i, (start, end) in enumerate(offsets):
#             for j in range(start, end):
#                 boundaries[j] = i  # or 1 if you want just 0/1 marking token edges
#
#         mask = [0] * len(boundaries)
#         prev = boundaries[0]
#         for i in range(1, len(boundaries)):
#             if boundaries[i] != prev:
#                 mask[i] = 1
#                 prev = boundaries[i]
#         return tokens, mask
#
#     def get_data(self):
#         texts, conditions, substrings = generate_data(self.num_samples,balanced=self.balanced,vocabulary_word_size=self.vocabulary_word_size,sentence_lenght=self.sentence_lenght,string_to_add=self.string_to_add,string_to_search=self.string_to_search)
#         datas = []
#         for idx in range(len(texts)):
#             text = texts[idx]
#             condition = conditions[idx]
#             substring = substrings[idx]
#             # 1) Build node features (one node per character)
#             x = self.get_character_embeddings(text)
#
#             # 2) Build adjacency (chain edges) for each consecutive character
#             num_nodes = len(text)
#             if num_nodes > 1:
#                 # edges: (0->1,1->0,1->2,2->1,...)
#                 src = list(range(num_nodes - 1))
#                 dst = list(range(1, num_nodes))
#                 edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
#             else:
#                 edge_index = torch.zeros((2, 0), dtype=torch.long)  # no edges if only one character
#
#             # 3) Build the label y = boundary(0/1) for each character
#             if condition == "normal":
#                 tokens, boundaries = self.get_boundaries(text)
#             else:  # counting mode
#                 tokens, boundaries = self.get_boundaries(text)  # Get boundaries
#                 boundaries = substring_boundaries(text, substring,
#                                                   boundaries)  # make sure the substring is tokenized as 1 token
#
#             y = torch.tensor(boundaries, dtype=torch.float)
#
#             cond_val = 0 if condition == "normal" else 1
#             cond_tensor = torch.tensor([cond_val], dtype=torch.float)
#
#             data = Data(
#                 x=x,
#                 edge_index=edge_index,
#                 y=y
#             )
#
#             # multiply the substring embed with condition, if condition is normal, the embedding will be all 0s, if coundting, the embedding will be the substring embedding
#             if self.used_llm == 'GPT2':
#                 data.substring_embed = self.embedding_layer(
#                     self.tokenizer(substring, return_tensors="pt")['input_ids'].view(1, -1)).view(1, -1) * cond_tensor
#             if self.used_llm == 'Llama-3.2-1B':
#                 data.substring_embed = self.model.embed_tokens(
#                     self.tokenizer(substring, return_tensors="pt")['input_ids'][:, 1]) * cond_tensor
#             datas.append(data.to(self.device))
#         return datas
#
