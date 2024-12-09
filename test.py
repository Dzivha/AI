import torch
import tiktoken
from torch.nn import Embedding, TransformerEncoder, TransformerEncoderLayer, Linear
from torch.nn.functional import softmax

class SimpleTransformerModel(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers):
        super(SimpleTransformerModel, self).__init__()
        self.embedding = Embedding(vocab_size, embed_dim)
        transformer_layer = TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True) # added batch_first = true
        self.transformer = TransformerEncoder(transformer_layer, num_layers=num_layers)
        self.fc_out = Linear(embed_dim, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.fc_out(x)
        return x

def load_model(model_path, vocab_size, embed_dim, num_heads, num_layers):
    model = SimpleTransformerModel(vocab_size, embed_dim, num_heads, num_layers)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict(model, tokenizer, initial_text, max_length=50):
    token_ids = tokenizer.encode(initial_text, allowed_special={"<|endoftext|>"})
    for _ in range(max_length):
        input_tensor = torch.tensor([token_ids])
        with torch.no_grad():
            output = model(input_tensor)
        predicted_token_id = output[0, -1].argmax().item()
        token_ids.append(predicted_token_id)
        if predicted_token_id == tokenizer.encode('<|endoftext|>', allowed_special={"<|endoftext|>"})[0]:
            break
    return tokenizer.decode(token_ids)


# Constants
VOCAB_SIZE = 50257
EMBED_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 2
MODEL_PATH = 'simple_transformer_model.pth'

# Load the model and tokenizer
tokenizer = tiktoken.get_encoding("gpt2")
model = load_model(MODEL_PATH, VOCAB_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS)

# Predict and generate text
initial_text = "my name is"
generated_text = predict(model, tokenizer, initial_text)
print("Generated text:", generated_text)

