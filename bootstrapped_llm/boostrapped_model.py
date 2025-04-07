import torch
from transformers import (
    PreTrainedTokenizerBase,
    LlamaForCausalLM,
)

from embedding_hypernetwork.rnn_model import DynamicRNNModel
from embedding_hypernetwork.embeddings import split_embedding_idx


class BootstrappedLlamaModel(torch.nn.Module):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        language_model: LlamaForCausalLM,
        rnn_model: DynamicRNNModel,
    ):
        super(BootstrappedLlamaModel, self).__init__()
        self.tokenizer = tokenizer
        self.langauge_model = language_model
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

    def forward(self, input_ids):
        predicted_embeddings = self.predict_embeddings(input_ids)
        self.langauge_model.eval()
        with torch.no_grad():
            return self.langauge_model(inputs_embeds=predicted_embeddings.unsqueeze(0))

    def generate(
        self,
        model_inputs,
        max_new_tokens=10,
    ):
        generated_token_ids = []
        for _ in range(max_new_tokens):
            outputs = self(
                torch.cat([model_inputs.input_ids, *generated_token_ids], dim=1)
            )
            next_token_ids = outputs.logits[:, -1:].argmax(-1)
            generated_token_ids.append(next_token_ids)

        return torch.cat([model_inputs.input_ids, *generated_token_ids], dim=1)
