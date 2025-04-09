import torch
from transformers import AutoTokenizer

from src.models import ConditionalGNN
from src.dataset import ConditionalTokenizationDataset
from torch_geometric.loader import DataLoader


def main():
    model_dir = "/nfs-share/as3623/models/Llama-3.2-1B/"
    cache_dir = "./.cache"
    gnn_model_dir = "/nfs-share/as3623/projects/L65-nat/NAT/src/model_save/checkpoints-20250409_020804/save_18.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        cache_dir=cache_dir,
    )

    neural_tokenizer_gnn = ConditionalGNN(
        in_channels=2048,
        condition_emb_dim=2048,
        hidden_dim=256,
        num_layers=2,
    ).to(device)

    neural_tokenizer_gnn.load_state_dict(
        torch.load(
            gnn_model_dir,
            # weights_only=False,
            map_location=device,
        )
    )

    prompt = "Frogs are interesting animals that canjump and swim in open water."
    substring = "jump"

    dataset = ConditionalTokenizationDataset(
        [prompt, prompt],
        ["normal", "counting"],
        [substring, substring],
        used_llm="Llama-3.2-1B",
        connection_distance=1,
    )

    val_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    ds = iter(val_dataloader)
    a = next(ds)
    b = next(ds)
    print(a.x.shape, a.edge_index.shape, a.substring_embed.shape)
    print(b.x.shape, b.edge_index.shape, b.substring_embed.shape)

    neural_tokenizer_gnn.eval()
    with torch.no_grad():
        a_pred = (
            neural_tokenizer_gnn(
                a.x.to(device),
                a.edge_index.to(device),
                a.substring_embed.to(device),
                a.batch.to(device),
            )
            .detach()
            .cpu()
        )
        b_pred = (
            neural_tokenizer_gnn(
                b.x.to(device),
                b.edge_index.to(device),
                b.substring_embed.to(device),
                b.batch.to(device),
            )
            .detach()
            .cpu()
        )
    print((a_pred >= 0).int())
    print((b_pred >= 0).int())
    print(a.y.int())

    print((a_pred >= 0).int() == a.y.int())
    print((b_pred >= 0).int() == b.y.int())


if __name__ == "__main__":
    main()
