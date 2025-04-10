from torch.utils.data import Dataset


class BenchmarkDataset(Dataset):
    def __init__(self, data: list[tuple[str, str, int]]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, substring, true_num = self.data[idx]
        prompt = (
            "How many times does the substring "
            + substring.replace("Ġ", " ")
            + f" occur in the following text:\n{text}\nNumber of occurrences (provide only a number): "
        )
        return prompt, true_num
