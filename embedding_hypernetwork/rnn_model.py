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
            self.rnn = nn.LSTM(
                input_dim,
                hidden_dim,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )
        elif rnn_type.lower() == "gru":
            self.rnn = nn.GRU(
                input_dim,
                hidden_dim,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )
        else:
            self.rnn = nn.RNN(
                input_dim,
                hidden_dim,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    # def forward(self, x, seq_lengths=None):
    #     """
    #     Forward pass through the network.

    #     Args:
    #         x (Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
    #                     where seq_len can be variable
    #         seq_lengths (Tensor): Actual sequence lengths for each batch item

    #     Returns:
    #         Tensor: Output predictions of shape (batch_size, output_dim)
    #     """
    #     batch_size, _, _ = x.size()

    #     # Handle dynamic sequence lengths if provided
    #     if seq_lengths is not None:
    #         # Pack padded sequence for efficient computation
    #         x_packed = nn.utils.rnn.pack_padded_sequence(
    #             x, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
    #         )

    #         # Process through RNN
    #         output_packed, hidden = self.rnn(x_packed)

    #         # Unpack the sequence
    #         output, _ = nn.utils.rnn.pad_packed_sequence(
    #             output_packed, batch_first=True
    #         )

    #         # Get the output for the last actual element of each sequence
    #         idx = (seq_lengths - 1).view(-1, 1).expand(batch_size, self.hidden_dim)
    #         time_dimension = 1
    #         idx = idx.unsqueeze(time_dimension)
    #         last_output = output.gather(time_dimension, idx).squeeze(time_dimension)
    #     else:
    #         # Process through RNN normally, taking the output from the last step
    #         output, _ = self.rnn(x)
    #         last_output = output[:, -1]

    #     # Get final prediction
    #     predictions = self.output_layer(last_output)

    #     return predictions

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
