import random

from torch.utils.data import Dataset


def find_overlapping(string, substring):
    start = 0
    count = 0
    while True:
        start = string.find(substring, start)
        if start == -1:
            break
        count += 1
        start += 1

    return count


def generate_subset(
    num_samples: int,
    len_text: int,
    len_substring: int,
    num_different_digits: int,
    min_num_substring_occurrences: int,
    max_num_substring_occurrences: int,
    force_substring_position_start: bool = False,
    force_substring_position_start_offset: int | None = None,
) -> list[tuple[str, str, int]]:
    assert (
        0 < num_different_digits <= 10
    ), "num_different_digits must be between 1 and 10"

    assert (
        min_num_substring_occurrences > 0
        and min_num_substring_occurrences <= max_num_substring_occurrences
    ), "min_num_substring_occurrences must be greater than 0 and less than or equal to max_num_substring_occurrences"

    substring_position_start_indices = [
        i
        for i in range(len_text - len_substring + 1)
        if i % 3 == force_substring_position_start_offset
    ]

    choices = list(range(num_different_digits))
    samples = []
    i = 0
    while len(samples) < num_samples:
        num_occurrences = random.randint(
            min_num_substring_occurrences, max_num_substring_occurrences
        )
        text_lst = [random.choice(choices) for _ in range(len_text)]
        substring_lst = [random.choice(choices) for _ in range(len_substring)]
        for _ in range(num_occurrences):
            if force_substring_position_start:
                start = random.choice(substring_position_start_indices)
            else:
                start = random.randint(0, len_text - len_substring)

            text_lst[start : start + len_substring] = substring_lst

        text = "".join(map(str, text_lst))
        substring = "".join(map(str, substring_lst))

        true_num = find_overlapping(text, substring)

        if true_num == num_occurrences:
            assert len(text) == len_text
            samples.append((text, substring, true_num))
        i += 1

        if i >= 1000 * num_samples:
            break

    if len(samples) != num_samples:
        print("Could not generate enough samples")
        return []

    return samples


if __name__ == "__main__":

    a = generate_subset(
        10,
        30,
        3,
        10,
        1,
        5,
        force_substring_position_start=True,
        force_substring_position_start_offset=1,
    )
    print(a)
