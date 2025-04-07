import torch

from transformers import AutoTokenizer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import numpy as np

from embedding_hypernetwork.embeddings import (
    load_llama_embedding_layer,
    split_embedding_idx,
)
from embedding_hypernetwork.rnn_model import DynamicRNNModel


def experiment_digits(embedding_layer, tokenizer, model):
    """Just plot the numbers 0-99."""
    # Encode a sequence of digits, each separately
    digits = [str(i) for i in range(100)] + [str(555), str(756), str(986)]

    # Tokenize the digits separately and skip special tokens
    input_ids = tokenizer(
        digits, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(
        embedding_layer.weight.device
    )  # Move to the same device as the embedding layer

    # Get the embeddings
    embeddings = embedding_layer(input_ids.squeeze(1))

    vocab_set = set(tokenizer.get_vocab().keys())
    splits = []
    for idx in input_ids:
        splits.append(
            split_embedding_idx(idx[0], tokenizer, vocab_set, embedding_layer)
        )

    lengths = [len(s) for s in splits]
    padded_sequences = torch.nn.utils.rnn.pad_sequence(splits, batch_first=True)

    model.eval()
    with torch.no_grad():
        predicted_embeddings = model(padded_sequences, lengths)

    combined_embeddings = torch.cat(
        [embeddings, predicted_embeddings], dim=0
    )  # Concatenate along the last dimension

    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=0)
    reduced_embeddings = tsne.fit_transform(combined_embeddings.detach().cpu().numpy())

    # Plotting
    labels = list(map(lambda x: int(x), digits))
    plt.figure(figsize=(12, 10), dpi=300)
    scatter = plt.scatter(
        reduced_embeddings[: embeddings.shape[0], 0],
        reduced_embeddings[: embeddings.shape[0], 1],
        c="blue",
        alpha=0.5,
        label="Original",
    )

    scatter = plt.scatter(
        reduced_embeddings[embeddings.shape[0] :, 0],
        reduced_embeddings[embeddings.shape[0] :, 1],
        c="red",
        alpha=0.5,
        label="Predicted",
    )

    # add a text label for each point
    for i, label in enumerate(labels):
        plt.annotate(
            label, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]), fontsize=8
        )
        plt.annotate(
            label,
            (
                reduced_embeddings[embeddings.shape[0] + i, 0],
                reduced_embeddings[embeddings.shape[0] + i, 1],
            ),
            fontsize=8,
        )
        # draw a line between the two points
        plt.plot(
            [reduced_embeddings[i, 0], reduced_embeddings[embeddings.shape[0] + i, 0]],
            [
                reduced_embeddings[i, 1],
                reduced_embeddings[embeddings.shape[0] + i, 1],
            ],
            c="gray",
            alpha=0.5,
            linewidth=0.5,
        )

    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend()
    plt.title("t-SNE visualization of embeddings")
    plt.savefig("tsne_compare_predicted_embeddings_small_trained.png")


def experiment_embedding_vocab(embedding_layer, tokenizer, model, dataset_split_file):
    "Plot a selection of the vocabulary."

    # TODO Ngl, we could just modify the dataset to have an additional
    # TODO output of the embedding id to match it with a textual representation.

    # Load the test set indices
    split_indices = []
    with open(dataset_split_file, "r") as f:
        for line in f:
            split_indices.append(int(line.strip()))

    # Randomly select 100 indices from the test set
    input_ids = (
        torch.tensor(np.random.choice(split_indices[1000:5000], 100))
        .unsqueeze(-1)
        .to(embedding_layer.weight.device)
    )

    # Get labels for plotting
    words = [tokenizer.convert_ids_to_tokens(idx[0].item()) for idx in input_ids]

    vocab_set = set(tokenizer.get_vocab().keys())

    # Select the embeddings
    embeddings = embedding_layer(input_ids.squeeze(1))

    splits = []
    for i, idx in enumerate(input_ids):
        if (
            split := split_embedding_idx(idx[0], tokenizer, vocab_set, embedding_layer)
        ) is not None:
            splits.append(split)
        else:
            # remove ith element from embeddings
            embeddings = torch.cat(
                [
                    embeddings[:i],
                    embeddings[i + 1 :],
                ],
                dim=0,
            )

    lengths = [len(s) for s in splits]
    padded_sequences = torch.nn.utils.rnn.pad_sequence(splits, batch_first=True)

    model.eval()
    with torch.no_grad():
        predicted_embeddings = model(padded_sequences, lengths)

    combined_embeddings = torch.cat(
        [embeddings, predicted_embeddings], dim=0
    )  # Concatenate along the last dimension

    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=0)
    reduced_embeddings = tsne.fit_transform(combined_embeddings.detach().cpu().numpy())

    # Plotting
    plt.figure(figsize=(12, 10), dpi=300)
    scatter = plt.scatter(
        reduced_embeddings[: embeddings.shape[0], 0],
        reduced_embeddings[: embeddings.shape[0], 1],
        c="blue",
        alpha=0.5,
        label="Original",
    )

    scatter = plt.scatter(
        reduced_embeddings[embeddings.shape[0] :, 0],
        reduced_embeddings[embeddings.shape[0] :, 1],
        c="red",
        alpha=0.5,
        label="Predicted",
    )

    # add a text label for each point
    for i, label in enumerate(words):
        plt.annotate(
            label, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]), fontsize=6
        )
        plt.annotate(
            label,
            (
                reduced_embeddings[embeddings.shape[0] + i, 0],
                reduced_embeddings[embeddings.shape[0] + i, 1],
            ),
            fontsize=6,
        )
        # draw a line between the two points
        plt.plot(
            [reduced_embeddings[i, 0], reduced_embeddings[embeddings.shape[0] + i, 0]],
            [
                reduced_embeddings[i, 1],
                reduced_embeddings[embeddings.shape[0] + i, 1],
            ],
            c="gray",
            alpha=0.5,
            linewidth=0.5,
        )

    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend()

    plt.title("t-SNE visualization of embeddings")
    plt.tight_layout()
    plt.savefig("tsne_compare_predicted_embeddings_small_words_trained.png")


def main():
    # Load the embedding layer
    model_dir = "/nfs-share/as3623/models/Llama-3.2-1B/"
    rnn_model_dir = "/nfs-share/as3623/projects/L65-nat/NAT/checkpoints-20250407_145814/final_model.pt"
    dataset_split_file = "./test_indices.txt"

    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embedding_layer = load_llama_embedding_layer(model_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, cache_dir="./.cache")

    rnn_model = DynamicRNNModel(
        input_dim=2048,
        hidden_dim=1024,
        output_dim=2048,
        num_layers=4,
        rnn_type="lstm",
        dropout=0.1,
    ).to(device)

    # load the model state dict
    rnn_model.load_state_dict(
        torch.load(
            rnn_model_dir,
            weights_only=False,
            map_location=device,
        )["model_state_dict"]
    )

    # experiment_digits(embedding_layer, tokenizer, rnn_model)
    experiment_embedding_vocab(
        embedding_layer, tokenizer, rnn_model, dataset_split_file
    )


if __name__ == "__main__":
    main()
