import os
import datetime
import json
import random

import torch

from counting_benchmark.utils import (
    pretty_please_print_results,
    visualize_results,
)
from utils.utils import load_model_and_tokenizer
from counting_benchmark.triple_digits_dataset_generation import generate_subset
from counting_benchmark.dataset import BenchmarkDataset
from counting_benchmark.generation import (
    evaluate_llm_batch,
)
from bootstrapped_llm.bootstrapped_model import RNNBootstrappedLlamaModel
from neural_tokenizer.neural_tokenizer import NeuralTokenizer
from embedding_hypernetwork.rnn_model import get_loaded_rnn


def main():
    model_dir = "/nfs-share/as3623/models/Llama-3.2-1B"
    rnn_model_dir = "/nfs-share/as3623/projects/L65-nat/NAT/rnn_checkpoints/checkpoints-20250407_145814/final_model.pt"
    neural_tokenizer_path = "/nfs-share/as3623/projects/L65-nat/NAT/neural_tokenizer/model_save/checkpoints-20250409_020804/save_18.pth"

    results_save_dir = "./results_triple_digits_benchmark"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model, tokenizer = load_model_and_tokenizer(model_dir, device)
    neural_tokenizer = NeuralTokenizer(neural_tokenizer_path, model_dir, device)
    rnn_model = get_loaded_rnn(rnn_model_dir, device)
    bootstrapped_model = RNNBootstrappedLlamaModel(
        tokenizer,
        model,
        rnn_model,
    )

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

    for seed in range(10):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.random.manual_seed(seed)
        random.seed(seed)

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
                    "seed": seed,
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

                meta["tag"] = "baseline"
                results_log["runs"].append(
                    {
                        "meta": meta.copy(),
                        "results": results,
                    }
                )

                results_neural = evaluate_llm_batch(
                    bootstrapped_model,
                    tokenizer,
                    dataset,
                    using_bootstrapped_model=True,
                    neural_tokenizer=neural_tokenizer,
                    # verbose_evalation=True,
                    batch_size=1,
                )
                meta["tag"] = "neural_tok"
                results_log["runs"].append(
                    {
                        "meta": meta.copy(),
                        "results": results_neural,
                    }
                )

                results_neural_conditioned = evaluate_llm_batch(
                    bootstrapped_model,
                    tokenizer,
                    dataset,
                    using_bootstrapped_model=True,
                    neural_tokenizer=neural_tokenizer,
                    conditioned=True,
                    # verbose_evalation=True,
                    batch_size=1,
                )
                meta["tag"] = "neural_tok_conditioned"
                results_log["runs"].append(
                    {
                        "meta": meta,
                        "results": results_neural_conditioned,
                    }
                )

                visualize_results(
                    results,
                    os.path.join(
                        results_dir,
                        f"scatter_synthetic_{num_different_digits}_{idx}.png",
                    ),
                )
                visualize_results(
                    results_neural,
                    os.path.join(
                        results_dir,
                        f"scatter_synthetic_neural_{num_different_digits}_{idx}.png",
                    ),
                )
                visualize_results(
                    results_neural_conditioned,
                    os.path.join(
                        results_dir,
                        f"scatter_synthetic_neural_conditioned_{num_different_digits}_{idx}.png",
                    ),
                )

    if len(results_log["runs"]) > 0:
        pretty_please_print_results(
            results_log, os.path.join(results_dir, "summary_table.csv")
        )

    json.dump(results_log, open(os.path.join(results_dir, "results.json"), "w"))


if __name__ == "__main__":
    main()
