from utils.utils import load_model_and_tokenizer

from transformers import PreTrainedTokenizer


def test_all_numbers(tokenizer: PreTrainedTokenizer, num_digits: int) -> None:
    # Generate all possible digit combinations
    all_combinations = []
    # Iterate from 0 to 10^num_digits - 1 to generate all combinations
    for i in range(10**num_digits):
        # Format the number with leading zeros to ensure it has num_digits
        digit_str = str(i).zfill(num_digits)
        all_combinations.append(digit_str)

    # Print some statistics
    print(f"Generated {len(all_combinations)} combinations of {num_digits} digits")

    # # You can also analyze token lengths for these digit combinations
    tokens = [
        tokenizer.encode(digits, add_special_tokens=False)
        for digits in all_combinations
    ]
    token_lengths = [len(token) for token in tokens]

    for target, tokens in zip(all_combinations, tokens):
        assert (
            len(tokens) == 1
        ), f"Target: {target}, Target length: {len(target)}, Tokenization: {[tokenizer.decode(token) for token in tokens]}"

    avg_token_length = sum(token_lengths) / len(token_lengths)
    print(f"Average token length: {avg_token_length}")


def test_tokenization(tokenizer: PreTrainedTokenizer, strings: list[str]) -> None:
    print(f"String -> Tokens")
    for string in strings:
        tokens = tokenizer.encode(string, add_special_tokens=False)
        print(f"{string} -> {[tokenizer.decode(token) for token in tokens]}")


if __name__ == "__main__":
    model_dir = "/nfs-share/as3623/models/Llama-3.2-1B"
    # model_dir = "/nfs-share/as3623/models/Llama-3.1-8B"
    # model_dir = "/nfs-share/as3623/models/gpt2"

    _, tokenizer = load_model_and_tokenizer(model_dir, load_just_tokenizer=True)

    test_all_numbers(tokenizer, 1)
    test_all_numbers(tokenizer, 2)
    test_all_numbers(tokenizer, 3)
    # test_all_numbers(tokenizer, 4)

    test_tokenization(
        tokenizer,
        [
            "1",
            "01",
            "001",
            "0001",
            "00001",
            "000001",
            "0000001",
            "48726212398471298109",
            "111101000111101000011001100101",
            "799755783751163232251147051196",
        ],
    )
