import torch
from transformers import (
    PreTrainedTokenizerBase,
    LlamaForCausalLM,
)

from embedding_hypernetwork.rnn_model import DynamicRNNModel
from embedding_hypernetwork.embeddings import split_embedding_idx


class RNNBootstrappedLlamaModel(torch.nn.Module):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        language_model: LlamaForCausalLM,
        rnn_model: DynamicRNNModel,
    ):
        super(RNNBootstrappedLlamaModel, self).__init__()
        self.tokenizer = tokenizer
        self.language_model = language_model
        self.rnn_model = rnn_model
        self.embeddings = language_model.model.embed_tokens

    def predict_embeddings(
        self,
        input_ids,
        add_bos_token=True,
    ):
        vocab_set = set(self.tokenizer.get_vocab().keys())

        splits = []
        for idx in input_ids.squeeze(0):
            assert (
                split := split_embedding_idx(
                    idx,
                    self.tokenizer,
                    vocab_set,
                    self.embeddings,
                )
            ) is not None, f"No split found for the token idx {idx} (token {self.tokenizer.convert_ids_to_tokens(idx.item())})!"
            splits.append(split)

        lengths = [len(s) for s in splits]
        padded_sequences = torch.nn.utils.rnn.pad_sequence(splits, batch_first=True)

        self.rnn_model.eval()
        with torch.no_grad():
            predicted_embeddings = self.rnn_model(padded_sequences, lengths)

        if add_bos_token:
            bos_token = self.tokenizer.bos_token_id
            bos_embeddings = self.embeddings(
                torch.tensor([bos_token]).to(input_ids.device)
            )
            predicted_embeddings = torch.cat(
                [bos_embeddings, predicted_embeddings], dim=0
            )

        return predicted_embeddings

    def predict_embeddings_boundaries(
        self,
        boundaries,
        input_embeddings,
        add_bos_token=True,
    ):
        # Convert input_string and boundaries to list of substrings
        splits = []
        start_idx = 0
        for i, boundary in enumerate(boundaries):
            if boundary.item() == 1:
                splits.append(input_embeddings[start_idx : i + 1])
                start_idx = i + 1

        lengths = [len(s) for s in splits]
        padded_sequences = torch.nn.utils.rnn.pad_sequence(splits, batch_first=True)

        self.rnn_model.eval()
        with torch.no_grad():
            predicted_embeddings = self.rnn_model(
                padded_sequences.to(self.language_model.device),
                lengths,
            )

        if add_bos_token:
            bos_token = self.tokenizer.bos_token_id
            bos_embeddings = self.embeddings(
                torch.tensor([bos_token]).to(self.language_model.device)
            )
            predicted_embeddings = torch.cat(
                [bos_embeddings, predicted_embeddings], dim=0
            )

        return predicted_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        additional_inputs_no_predict: list[torch.Tensor] = None,
    ):
        predicted_embeddings = self.predict_embeddings(input_ids)
        if (
            additional_inputs_no_predict is not None
            and len(additional_inputs_no_predict) > 0
        ):

            predicted_embeddings = torch.cat(
                [
                    predicted_embeddings,
                    *[
                        self.embeddings.weight[idx.squeeze(0)]
                        for idx in additional_inputs_no_predict
                    ],
                ],
                dim=0,
            )
        self.language_model.eval()
        with torch.no_grad():
            return self.language_model(inputs_embeds=predicted_embeddings.unsqueeze(0))

    def forward_neural_tokenizer(
        self,
        boundaries,
        input_embeddings,
    ):
        predicted_embeddings = self.predict_embeddings_boundaries(
            boundaries, input_embeddings
        )

        self.language_model.eval()
        with torch.no_grad():
            return self.language_model(inputs_embeds=predicted_embeddings.unsqueeze(0))

    def generate(
        self,
        model_inputs,
        max_new_tokens=10,
        tokenize_generated_tokens=True,
    ):
        generated_token_ids = []
        for _ in range(max_new_tokens):
            if tokenize_generated_tokens:
                outputs = self(
                    torch.cat([model_inputs.input_ids, *generated_token_ids], dim=1)
                )
            else:
                outputs = self(model_inputs.input_ids, generated_token_ids)

            next_token_ids = outputs.logits[:, -1:].argmax(-1)
            generated_token_ids.append(next_token_ids)

        return torch.cat([model_inputs.input_ids, *generated_token_ids], dim=1)
