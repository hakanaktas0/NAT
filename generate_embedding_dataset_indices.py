from transformers import AutoTokenizer
import numpy as np

if __name__ == "__main__":
    # Example usage
    model_dir = "/nfs-share/as3623/models/Llama-3.2-1B/"

    # Set the random seed for reproducibility
    seed = 42
    np.random.seed(seed)

    # set proportions
    train_proportion = 0.8
    val_proportion = 0.1
    test_proportion = 0.1
    assert (
        train_proportion + val_proportion + test_proportion == 1.0
    ), "Proportions must sum to 1"

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, cache_dir="./.cache")
    vocab = tokenizer.get_vocab()
    vocab_set = set(vocab.values())

    # Split the vocabulary into train, validation, and test sets
    # Randomly select indices for train, validation, and test sets

    train_idx = np.random.choice(
        list(vocab_set), size=int(len(vocab_set) * train_proportion), replace=False
    )
    remaining_idx = list(set(vocab_set) - set(train_idx))
    val_idx = np.random.choice(
        remaining_idx, size=int(len(vocab_set) * val_proportion), replace=False
    )
    test_idx = list(set(remaining_idx) - set(val_idx))
    # Convert indices to lists
    train_idx = sorted(list(train_idx))
    val_idx = sorted(list(val_idx))
    test_idx = sorted(list(test_idx))

    print(f"Proportions: {train_proportion}, {val_proportion}, {test_proportion}")
    print(
        f"Actual proportions: {len(train_idx) / len(vocab_set)}, {len(val_idx) / len(vocab_set)}, {len(test_idx) / len(vocab_set)}"
    )

    # Save the indices to a file
    with open("train_indices.txt", "w") as f:
        for idx in train_idx:
            f.write(f"{idx}\n")
    with open("val_indices.txt", "w") as f:
        for idx in val_idx:
            f.write(f"{idx}\n")
    with open("test_indices.txt", "w") as f:
        for idx in test_idx:
            f.write(f"{idx}\n")
