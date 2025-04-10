import datetime
import json
import os

import torch

from counting_benchmark.dataset import BenchmarkDataset
from counting_benchmark.wiki_dataset_generation import (
    create_set_of_non_final_tokens,
    prepare_counts_dataset,
    create_benchmark_datasets_wiki,
    load_prefix_suffix_set,
    load_wiki_data,
)
from counting_benchmark.utils import (
    load_model_and_tokenizer,
    visualize_results,
    pretty_please_print_results,
)
from counting_benchmark.generation import evaluate_llm_batch


def initialize_datasets(
    tokenizer,
    dataset,
    non_final_set,
    prefix_suffix_set,
    max_sample_length: int = -1,
):
    pre_dataset = prepare_counts_dataset(
        tokenizer, dataset, non_final_set, prefix_suffix_set, max_sample_length
    )
    return create_benchmark_datasets_wiki(pre_dataset, 3)


def main():
    model_dir = "/nfs-share/as3623/models/Llama-3.2-1B"
    results_save_dir = "./results_wiki_benchmark"

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
    )

    non_final_set = create_set_of_non_final_tokens(model_dir)
    prefix_suffix_set = load_prefix_suffix_set("./prefix_suffix_list.txt")
    wiki_raw_dataset = load_wiki_data()
    # _, tokenizer = load_model_and_tokenizer(model_dir, load_just_tokenizer=True)
    model, tokenizer = load_model_and_tokenizer(model_dir, device)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"{results_save_dir}/{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")

    num_samples_per_eval = 500

    results_log = {
        "model": model_dir,
        "num_samples_per_eval": num_samples_per_eval,
        "runs": [],
    }

    for sample_length in [200]:
        meta = {
            "num_samples": num_samples_per_eval,
            "len_text": sample_length,
        }

        processed_dataset_final, processed_dataset_non_final = initialize_datasets(
            tokenizer,
            wiki_raw_dataset,
            non_final_set,
            prefix_suffix_set,
            max_sample_length=sample_length,
        )

        wiki_dataset_final = BenchmarkDataset(processed_dataset_final)
        wiki_dataset_non_final = BenchmarkDataset(processed_dataset_non_final)

        res_final = evaluate_llm_batch(
            model,
            tokenizer,
            wiki_dataset_final,
            # verbose_evalation=True,
            batch_size=128,
        )

        res_non_final = evaluate_llm_batch(
            model,
            tokenizer,
            wiki_dataset_non_final,
            # verbose_evalation=True,
            batch_size=128,
        )

        meta["mode"] = "final"
        results_log["runs"].append(
            {
                "meta": meta.copy(),
                "results": res_final,
            }
        )
        meta["mode"] = "non-final"
        results_log["runs"].append(
            {
                "meta": meta,
                "results": res_non_final,
            }
        )

        visualize_results(res_final, f"{results_dir}/scatter_final_{sample_length}.png")
        visualize_results(
            res_non_final, f"{results_dir}/scatter_non_final_{sample_length}.png"
        )

    pretty_please_print_results(
        results_log, os.path.join(results_dir, "summary_table.csv")
    )

    with open(f"{results_dir}/results_log.json", "w") as f:
        json.dump(results_log, f)


if __name__ == "__main__":
    main()
