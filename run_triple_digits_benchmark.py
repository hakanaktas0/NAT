import os
import datetime
import json
import random

import torch

from counting_benchmark.utils import (
    load_model_and_tokenizer,
    pretty_please_print_results,
    visualize_results,
)
from counting_benchmark.triple_digits_dataset_generation import generate_subset
from counting_benchmark.dataset import BenchmarkDataset
from counting_benchmark.generation import evaluate_llm_batch


def main():
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)

    # model_dir = "/Users/aszab/repos/models/Llama-3.1-8B/"
    model_dir = "/nfs-share/as3623/models/Llama-3.2-1B"
    # model_dir = "/nfs-share/as3623/models/Llama-3.1-8B"
    # model_dir = "/nfs-share/as3623/models/gpt2"
    results_save_dir = "./results_triple_digits_benchmark"
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
    )

    model, tokenizer = load_model_and_tokenizer(model_dir, device)
    # Generate a directory name using current date and time

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"{results_save_dir}/{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")

    num_samples_per_eval = 1000
    results_log = {
        "model": model_dir,
        "num_samples_per_eval": num_samples_per_eval,
        "runs": [],
    }

    for num_different_digits in [3, 5, 10]:
        for idx, (fsps1, fsps2) in enumerate(
            [(True, 0), (True, 1), (True, 2), (False, 0)]
        ):
            print(
                f"Evaluating with num_different_digits={num_different_digits}, {'' if fsps1 else 'NOT '}forcing starting substrings with offset {fsps2 if fsps1 else ''}"
            )

            meta = {
                "num_samples": num_samples_per_eval,
                "len_text": 100,
                "len_substring": 3,
                "num_different_digits": num_different_digits,
                "min_num_substring_occurrences": 5,
                "max_num_substring_occurrences": 5,
                "force_substring_position_start": fsps1,
                "force_substring_position_start_offset": fsps2,
            }

            dataset = generate_subset(**meta)
            if len(dataset) == 0:
                print(
                    f"Could not generate enough samples for the current configuration: {meta}"
                )
                continue

            dataset = BenchmarkDataset(dataset)
            results = evaluate_llm_batch(
                model,
                tokenizer,
                dataset,
                # verbose_evalation=True,
                batch_size=128,
            )
            results_log["runs"].append(
                {
                    "meta": meta,
                    "results": results,
                }
            )

            visualize_results(
                results,
                os.path.join(
                    results_dir, f"scatter_synthetic_{num_different_digits}_{idx}.png"
                ),
            )

    if len(results_log["runs"]) > 0:
        pretty_please_print_results(
            results_log, os.path.join(results_dir, "summary_table.csv")
        )

    json.dump(results_log, open(os.path.join(results_dir, "results.json"), "w"))


if __name__ == "__main__":
    main()
