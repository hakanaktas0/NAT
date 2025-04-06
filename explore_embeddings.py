import torch

from transformers import AutoTokenizer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from embedding_hypernetwork.embeddings import load_llama_embedding_layer


def experiment_digits(embedding_layer, tokenizer):
    """Just plot the numbers 0-99."""
    # Encode a sequence of digits, each separately
    digits = [str(i) for i in range(100)]

    # Tokenize the digits separately and skip special tokens
    input_ids = tokenizer(
        digits, add_special_tokens=False, return_tensors="pt"
    ).input_ids

    # Get the embeddings
    embeddings = embedding_layer(input_ids.squeeze(1))

    # Perform t-SNE
    tsne = TSNE(n_components=2, perplexity=9, random_state=0)
    reduced_embeddings = tsne.fit_transform(embeddings.detach().numpy())

    # Plotting
    labels = list(map(lambda x: int(x), digits))
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        reduced_embeddings[:, 0],
        reduced_embeddings[:, 1],
        c=labels,
        cmap="viridis",
        alpha=0.5,
    )

    # add a text label for each point
    for i, label in enumerate(labels):
        plt.annotate(
            label, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]), fontsize=8
        )

    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.title("t-SNE visualization of embeddings")
    plt.colorbar(scatter)
    plt.title("t-SNE visualization of embeddings")
    plt.savefig("tsne_embeddings_normal.png")


def experiment_digits_additive(embedding_layer, tokenizer):
    """Show digits 0-99 but also try to see if the embeddings space has an additive structure."""
    # Encode a sequence of digits, each separately
    digits = [str(i) for i in range(100)]

    # Tokenize the digits separately and skip special tokens
    input_ids = tokenizer(
        digits, add_special_tokens=False, return_tensors="pt"
    ).input_ids

    # Get the embeddings
    embeddings = embedding_layer(input_ids.squeeze(1))

    additive_embeddings = [
        (embeddings[i] + (embeddings[i + 2] - embeddings[i + 1])).unsqueeze(0)
        for i in range(10)
    ]
    embeddings = torch.cat([embeddings, *additive_embeddings])

    # Perform t-SNE
    tsne = TSNE(n_components=2, perplexity=9, random_state=0)
    reduced_embeddings = tsne.fit_transform(embeddings.detach().numpy())

    # Plotting
    labels = list(map(lambda x: int(x), digits))
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        reduced_embeddings[: input_ids.shape[0], 0],
        reduced_embeddings[: input_ids.shape[0], 1],
        c=labels,
        cmap="viridis",
        alpha=0.5,
    )
    # add a text label for each point
    for i, label in enumerate(labels):
        plt.annotate(
            label, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]), fontsize=8
        )

    plt.scatter(
        reduced_embeddings[input_ids.shape[0] :, 0],
        reduced_embeddings[input_ids.shape[0] :, 1],
        color="red",
        alpha=0.5,
    )
    for i, label in enumerate(
        [f"{i}+({i+2} - {i+1})" for i in range(10)], start=input_ids.shape[0]
    ):
        plt.annotate(
            label, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]), fontsize=8
        )

    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.title("t-SNE visualization of embeddings")
    plt.colorbar(scatter)
    plt.title("t-SNE visualization of embeddings")
    plt.savefig("tsne_embeddings_additive.png")


def show_all_vocab(embedding_layer, tokenizer):
    """Project all (or a considerable subset of) the vocabulary into 2D space.
    Show the digits separately.
    """
    # Encode a sequence of digits, each separately
    digits_10 = [str(i) for i in range(10)]
    digits_100 = [str(i) for i in range(10, 100)]
    digits_1000 = [str(i) for i in range(100, 1000)]

    # Tokenize the digits separately and skip special tokens
    input_ids_digits = tokenizer(
        digits_10 + digits_100 + digits_1000,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids

    # Get the embeddings
    embeddings_digits = embedding_layer(input_ids_digits.squeeze(1))
    embeddings_all = embedding_layer.weight[5000:15000, :]

    embeddings = torch.cat([embeddings_digits, embeddings_all], dim=0)

    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=0)
    reduced_embeddings = tsne.fit_transform(embeddings.detach().numpy())

    # Plotting
    # labels = list(map(lambda x: int(x), digits))
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        reduced_embeddings[embeddings_digits.shape[0] :, 0],
        reduced_embeddings[embeddings_digits.shape[0] :, 1],
        c="blue",
        alpha=0.2,
        label="All Vocab",
    )

    plt.scatter(
        reduced_embeddings[
            embeddings_digits.shape[0] : embeddings_digits.shape[0] + 10, 0
        ],
        reduced_embeddings[
            embeddings_digits.shape[0] : embeddings_digits.shape[0] + 10, 1
        ],
        c="red",
        alpha=0.5,
        label="0-9",
    )
    plt.scatter(
        reduced_embeddings[
            embeddings_digits.shape[0] + 10 : embeddings_digits.shape[0] + 100, 0
        ],
        reduced_embeddings[
            embeddings_digits.shape[0] + 10 : embeddings_digits.shape[0] + 100, 1
        ],
        c="orange",
        alpha=0.5,
        label="10-99",
    )
    plt.scatter(
        reduced_embeddings[
            embeddings_digits.shape[0] + 100 : embeddings_digits.shape[0] + 1000, 0
        ],
        reduced_embeddings[
            embeddings_digits.shape[0] + 100 : embeddings_digits.shape[0] + 1000, 1
        ],
        c="green",
        alpha=0.5,
        label="100-999",
    )

    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.title("t-SNE visualization of embeddings")
    # plt.colorbar(scatter)
    plt.legend()
    plt.title("t-SNE visualization of embeddings")
    plt.savefig("tsne_embeddings_all.png")


if __name__ == "__main__":
    # Load the embedding layer
    model_dir = "/nfs-share/as3623/models/Llama-3.2-1B/"

    embedding_layer = load_llama_embedding_layer(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, cache_dir="./.cache")

    experiment_digits(embedding_layer, tokenizer)
    experiment_digits_additive(embedding_layer, tokenizer)
    show_all_vocab(embedding_layer, tokenizer)
