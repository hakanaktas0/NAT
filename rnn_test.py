import torch

from embedding_hypernetwork.rnn_model import create_model


if __name__ == "__main__":
    # Parameters
    batch_size = 4
    max_seq_len = 16  # Maximum sequence length
    embedding_dim = 50  # k-dimensional embeddings
    hidden_dim = 8
    output_dim = 10
    
    # Create variable length sequences
    seq_lengths = torch.randint(1, max_seq_len + 1, (batch_size,)).sort(descending=True)[0]
    
    # Create dummy input batch with padded sequences
    x = torch.randn(batch_size, max_seq_len, embedding_dim)
    
    # Create model
    model = create_model(
        input_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim
    )
    
    # Forward pass
    output = model(x, seq_lengths)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")