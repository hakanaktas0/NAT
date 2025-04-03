from transformers import GPT2Tokenizer, GPT2Model
import torch

# Load tokenizer and GPT-2 model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")
model.eval()  # Set to eval mode

# Your input text
text = "Token embeddings from GPT-2"

chars = list(text)

char_input_ids = tokenizer(chars, return_tensors="pt")['input_ids'].view(1,-1)

embedding_layer = model.get_input_embeddings()


with torch.no_grad():
    raw_embeddings = embedding_layer(char_input_ids)  # Shape: (1, seq_len, hidden_size)

# Inspect
print("Input tokens:", tokenizer.convert_ids_to_tokens(char_input_ids[0]))
print("Raw embeddings shape:", raw_embeddings.shape)
#
#
# # Tokenize the input
# inputs = tokenizer(text, return_tensors="pt")
# input_ids = inputs["input_ids"]  # Shape: (1, seq_len)
#
# # Get the embedding layer
# embedding_layer = model.get_input_embeddings()
#
# # Get raw embeddings (before any transformer layers)
# with torch.no_grad():
#     raw_embeddings = embedding_layer(input_ids)  # Shape: (1, seq_len, hidden_size)
#
# # Inspect
# print("Input tokens:", tokenizer.convert_ids_to_tokens(input_ids[0]))
# print("Raw embeddings shape:", raw_embeddings.shape)
