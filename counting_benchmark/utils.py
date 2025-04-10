from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    AutoTokenizer,
    AutoModelForCausalLM,
)
import torch
import matplotlib.pyplot as plt

from tabulate import tabulate
import pandas as pd
import numpy as np


def load_model_and_tokenizer(
    model_dir: str,
    device: torch.device | None = None,
    load_just_tokenizer: bool = False,
) -> tuple[PreTrainedModel, PreTrainedTokenizer | None]:
    """
    Load a model and tokenizer from the specified directory.

    Args:
    model_dir (str): Path to the model directory.

    Returns:
    Optional[model]: The loaded model.
    tokenizer: The loaded tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, padding_side="left", cache_dir="./.cache"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if load_just_tokenizer:
        return None, tokenizer

    model = AutoModelForCausalLM.from_pretrained(
        model_dir, cache_dir="./.cache", torch_dtype=torch.float16
    ).to(device)

    return model, tokenizer


def pretty_please_print_results(
    results: list[dict[str, int | str]], filename: str | None = None
) -> None:
    summary_table = []

    for run in results["runs"]:
        row = [*run["meta"].values()]
        row.append(
            sum(
                [
                    1
                    for entry in run["results"]
                    if entry["true_count"] == entry["predicted_count"]
                ]
            )
            / len(run["results"])
        )
        row.append(
            np.mean(
                [
                    abs(entry["true_count"] - entry["predicted_count"])
                    for entry in run["results"]
                ]
            )
        )
        row.append(np.mean([entry["true_count"] for entry in run["results"]]))
        row.append(np.mean([entry["predicted_count"] for entry in run["results"]]))
        row.append(
            np.mean(
                [
                    (
                        abs(entry["true_count"] - entry["predicted_count"])
                        / entry["true_count"]
                        if entry["true_count"] != 0
                        else 0
                    )
                    for entry in run["results"]
                ]
            )
        )

        summary_table.append(row)

    headers = [
        *run["meta"].keys(),
        "acc",
        "avg Δ",
        "avg real count",
        "avg pred count",
        "rel err",
    ]

    print(
        tabulate(
            summary_table,
            headers=headers,
            floatfmt=".4f",
        )
    )
    if filename is not None:
        pd.DataFrame(summary_table, columns=headers).to_csv(filename)


def visualize_results(results: list[tuple[int, int]], filename: str) -> None:
    true_values = np.array([x["true_count"] for x in results])
    predictions = np.array([x["predicted_count"] for x in results])
    indices = np.arange(len(true_values))

    # Identify matching and mismatching indices
    mismatches = true_values != predictions
    matches = ~mismatches

    plt.figure(figsize=(10, 6))

    # Plot all true values and predictions as points (no lines)
    plt.scatter(
        indices[matches],
        true_values[matches],
        label="Correct Predictions",
        marker="o",
        color="green",
        s=100,
        zorder=2,
    )

    plt.scatter(
        indices[mismatches],
        predictions[mismatches],
        color="red",
        marker="x",
        s=100,
        zorder=2,
        label="Mismatch (Predicted)",
    )

    plt.scatter(
        indices[mismatches],
        true_values[mismatches],
        color="green",
        marker="o",
        facecolors="none",
        s=100,
        zorder=2,
        label="Mismatch (True)",
    )

    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title("Comparison of True Values and Predictions (Points Only)")
    plt.ylim(-0.1, min(10, max(*predictions, *true_values)) + 1)
    # ticks every 1
    plt.yticks(np.arange(0, min(10, max(*predictions, *true_values)) + 1, 1))

    plt.legend()
    plt.grid(alpha=0.6, zorder=1)

    plt.savefig(filename)
    plt.close()
