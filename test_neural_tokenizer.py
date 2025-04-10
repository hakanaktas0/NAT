import torch
from transformers import AutoTokenizer

from bootstrapped_llm.neural_tokenizer import NeuralTokenizer
from bootstrapped_llm.utils import split_by_boundaries


def main():
    model_dir = "/nfs-share/as3623/models/Llama-3.2-1B/"
    cache_dir = "./.cache"
    gnn_model_dir = "/nfs-share/as3623/projects/L65-nat/NAT/src/model_save/checkpoints-20250409_020804/save_final_model.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    static_tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        cache_dir=cache_dir,
    )

    neural_tokenizer = NeuralTokenizer(gnn_model_dir, device)

    prompt = "Frogs are interestinganimals that canjump and swim in open water."
    substring = "canjump"

    print("Prompt: ", prompt)

    print(
        "Static Tokenizer: ",
        static_tokenizer.tokenize(prompt),
    )

    boundaries, _ = neural_tokenizer.tokenize(prompt, search_substring=None)
    print("Neural Tokenizer (n/c): ", split_by_boundaries(prompt, boundaries))

    boundaries, _ = neural_tokenizer.tokenize(prompt, search_substring=substring)
    print("Neural Tokenizer (c): ", split_by_boundaries(prompt, boundaries))


if __name__ == "__main__":
    main()
