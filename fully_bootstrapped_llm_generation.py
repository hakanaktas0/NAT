import torch

from embedding_hypernetwork.rnn_model import get_loaded_rnn
from bootstrapped_llm.bootstrapped_model import RNNBootstrappedLlamaModel
from utils.utils import load_model_and_tokenizer
from neural_tokenizer.neural_tokenizer import NeuralTokenizer
from bootstrapped_llm.generation import generate_with_neural_tokenizer


def main():
    model_dir = "/nfs-share/as3623/models/Llama-3.2-1B/"
    cache_dir = "./.cache"
    rnn_model_dir = "/nfs-share/as3623/projects/L65-nat/NAT/rnn_checkpoints/checkpoints-20250407_145814/final_model.pt"
    neural_tokenizer_path = "/nfs-share/as3623/projects/L65-nat/NAT/neural_tokenizer/model_save/checkpoints-20250409_020804/save_18.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    neural_tokenizer = NeuralTokenizer(neural_tokenizer_path, model_dir, device)

    language_model, tokenizer = load_model_and_tokenizer(model_dir, device)

    rnn_model = get_loaded_rnn(rnn_model_dir, device)

    bootstrapped_model = RNNBootstrappedLlamaModel(
        tokenizer,
        language_model,
        rnn_model,
    )

    prompt = "Transformer is a deep learning modal that uses self-attention and"
    max_new_tokens = 30
    static_model_inputs = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=True
    ).to(device)

    print(
        tokenizer.batch_decode(
            language_model.generate(
                **static_model_inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            ),
            skip_special_tokens=True,
        )[0],
        "\n",
    )

    static_model_inputs = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=False
    ).to(device)

    print(
        tokenizer.batch_decode(
            bootstrapped_model.generate(
                static_model_inputs,
                max_new_tokens=max_new_tokens,
                tokenize_generated_tokens=True,
            ),
            skip_special_tokens=True,
        )[0],
        "\n",
    )

    print(
        generate_with_neural_tokenizer(
            bootstrapped_model,
            neural_tokenizer,
            tokenizer,
            prompt,
            search_substring="capita",  # intentionally looking for capita
            max_new_tokens=10,
        ),
        "\n",
    )


if __name__ == "__main__":
    main()
