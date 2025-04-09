import torch
from transformers import AutoTokenizer, LlamaForCausalLM

from embedding_hypernetwork.rnn_model import DynamicRNNModel
from bootstrapped_llm.bootstrapped_model import RNNBootstrappedLlamaModel
from bootstrapped_llm.neural_tokenizer import NeuralTokenizer


def generate_with_neural_tokenizer(
    bootstrapped_model,
    neural_tokenizer,
    llm_tokenizer,
    prompt,
    search_substring=None,
    max_new_tokens=10,
):
    input_text = prompt
    for _ in range(max_new_tokens):

        dynamic_boundaries, char_embeddings = neural_tokenizer.tokenize(
            input_text,
            search_substring=search_substring,  # intentionally looking for capita
        )
        generated_text = llm_tokenizer.batch_decode(
            bootstrapped_model.forward_neural_tokenizer(
                dynamic_boundaries,
                char_embeddings,
            )
            .logits[:, -1:]
            .argmax(-1),
            skip_special_tokens=True,
        )[0]

        input_text += generated_text
    return input_text


def main():
    model_dir = "/nfs-share/as3623/models/Llama-3.2-1B/"
    cache_dir = "./.cache"
    rnn_model_dir = "/nfs-share/as3623/projects/L65-nat/NAT/checkpoints-20250407_145814/final_model.pt"
    neural_tokenizer_path = "/nfs-share/as3623/projects/L65-nat/NAT/src/model_save/checkpoints-20250409_020804/save_18.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    neural_tokenizer = NeuralTokenizer(neural_tokenizer_path, device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        cache_dir=cache_dir,
    )

    language_model = LlamaForCausalLM.from_pretrained(
        model_dir, cache_dir=cache_dir
    ).to(device)

    rnn_model = DynamicRNNModel(
        input_dim=2048,
        hidden_dim=1024,
        output_dim=2048,
        num_layers=4,
        rnn_type="lstm",
    ).to(device)

    # load the model state dict
    rnn_model.load_state_dict(
        torch.load(
            rnn_model_dir,
            weights_only=False,
            map_location=device,
        )["model_state_dict"]
    )

    bootstrapped_model = RNNBootstrappedLlamaModel(
        tokenizer,
        language_model,
        rnn_model,
    )

    prompt = "Transformer is a deep learning modal that uses self-attention and"
    max_new_tokens = 30
    static_model_inputs = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=False
    ).to(device)

    print(
        tokenizer.batch_decode(
            language_model.generate(
                **static_model_inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id
            ),
            skip_special_tokens=True,
        )[0],
        "\n",
    )

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
