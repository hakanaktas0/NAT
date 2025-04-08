import torch
from transformers import AutoTokenizer, LlamaForCausalLM

from embedding_hypernetwork.rnn_model import DynamicRNNModel
from bootstrapped_llm.bootstrapped_model import BootstrappedLlamaModel


def main():
    model_dir = "/nfs-share/as3623/models/Llama-3.2-1B/"
    cache_dir = "./.cache"
    rnn_model_dir = "/nfs-share/as3623/projects/L65-nat/NAT/checkpoints-20250407_145814/final_model.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    bootstrapped_model = BootstrappedLlamaModel(
        tokenizer,
        language_model,
        rnn_model,
    )

    prompt = "What is the capital of France? The capital of France is"
    max_new_tokens = 30
    model_inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(
        device
    )

    print(
        tokenizer.batch_decode(
            language_model.generate(
                **model_inputs,
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
                model_inputs,
                max_new_tokens=max_new_tokens,
                tokenize_generated_tokens=True,
            ),
            skip_special_tokens=True,
        )[0]
    )


if __name__ == "__main__":
    main()
