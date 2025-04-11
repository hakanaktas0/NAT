import matplotlib.pyplot as plt

from tabulate import tabulate
import pandas as pd
import numpy as np


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
        s=50,
        zorder=3,
    )

    plt.scatter(
        indices[mismatches],
        predictions[mismatches],
        color="red",
        marker="x",
        s=50,
        zorder=3,
        label="Mismatch (Predicted)",
    )

    plt.axhline(
        y=5,
        linestyle="--",
        label="True Number of Occurences",
        color="green",
        linewidth=1,
        zorder=2,
    )

    # plt.scatter(
    #     indices[mismatches],
    #     true_values[mismatches],
    #     color="green",
    #     marker="o",
    #     facecolors="none",
    #     s=100,
    #     zorder=2,
    #     label="Mismatch (True)",
    # )

    plt.xlabel("Index")
    plt.ylabel("Predicted Number of Occurrences")
    plt.title("Comparison of True and Predicted Number of Occurrences")
    plt.ylim(-0.1, min(10, max(*predictions, *true_values)) + 1)
    # ticks every 1
    plt.yticks(np.arange(0, min(10, max(*predictions, *true_values)) + 1, 1))

    plt.legend()
    plt.grid(alpha=0.6, zorder=1)

    plt.savefig(filename)
    plt.close()
    plt.cla()

    # Calculate max value for bins
    max_val = min(15, max(*predictions, *true_values)) + 1

    # Create bins with explicit edges, offset by 0.5 to center bars on integers
    bin_edges = np.arange(-0.5, max_val + 0.5, 1)

    plt.hist(
        true_values,
        bins=bin_edges,
        alpha=0.5,
        label="True Values",
        color="blue",
        align="mid",  # Ensure alignment is at the middle
        zorder=2,
    )
    plt.hist(
        predictions,
        bins=bin_edges,
        alpha=0.5,
        label="Predictions",
        color="orange",
        align="mid",  # Ensure alignment is at the middle
        zorder=2,
    )

    # Set x-ticks at integer positions
    plt.xticks(range(0, max_val))
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Histogram of the True and Predicted Number of Occurences")
    plt.legend()
    plt.grid(alpha=0.6)
    plt.savefig(filename.replace(".png", "_hist.png"))
    plt.close()
