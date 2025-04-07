import torch

import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence


class DynamicRNNModel(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers=1,
        rnn_type="lstm",
        dropout=0.0,
    ):
        """
        Initialize the RNN model.

        Args:
            input_dim (int): Dimension of input embeddings (k)
            hidden_dim (int): Dimension of hidden state
            output_dim (int): Dimension of output prediction
            num_layers (int): Number of RNN layers
            rnn_type (str): Type of RNN ('lstm', 'gru', or 'rnn')
            dropout (float): Dropout probability
        """
        super(DynamicRNNModel, self).__init__()

        # Set model parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # Create RNN layer

        if rnn_type.lower() == "lstm":
            cell_class = nn.LSTM
        elif rnn_type.lower() == "gru":
            cell_class = nn.GRU
        else:
            cell_class = nn.RNN

        self.rnn = cell_class(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, lengths=None):
        """
        Alternative forward that returns all outputs, not just the last hidden state.
        Useful for attention mechanisms or when all time steps are needed.

        Args:
            x: Input sequence, can be a tensor or PackedSequence
            lengths: Original sequence lengths (before padding)

        Returns:
            All outputs from the RNN, unpacked
        """
        is_packed = isinstance(x, PackedSequence)

        # If input is not already packed, pack it
        if not is_packed:
            if lengths is None:
                # Assume all sequences are same length if lengths not provided
                lengths = [x.size(1)] * x.size(0)
            # Pack the sequence
            x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # Process with RNN
        output, _ = self.rnn(x)

        # Unpack the sequence
        output, output_lengths = pad_packed_sequence(output, batch_first=True)

        # Extract the last actual output for each sequence
        batch_size = output.size(0)

        # Create a tensor of indices for the last element of each sequence
        idx = torch.zeros(batch_size, dtype=torch.long, device=output.device)
        for i, length in enumerate(output_lengths):
            idx[i] = length - 1

        # Select the last actual output from each sequence
        # Output shape is [batch_size, seq_len, hidden_size]
        # We want to gather along dimension 1 (sequence length)
        last_outputs = output[torch.arange(batch_size), idx]

        # Apply final transformation
        result = self.output_layer(last_outputs)
        return result
