import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt) #A
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    def __len__(self): #C
        return len(self.input_ids)
    def __getitem__(self, idx): #D
        return self.input_ids[idx], self.target_ids[idx]
    def create_dataloader(txt, batch_size=40, max_length=256,stride=128):
        tokenizer = tiktoken.get_encoding("gpt2") #A
        dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
        #B
        dataloader = DataLoader(dataset, batch_size=batch_size) #C
        return dataloader
    
    def create_dataloaders(txt, batch_size=40, max_length=256, stride=128, train_ratio=0.8):
        tokenizer = tiktoken.get_encoding("gpt2")
        dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
        
        # Splitting the dataset into training and validation sets
        train_size = int(len(dataset) * train_ratio)
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader

tokenizer = tiktoken.get_encoding("gpt2")

text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace."
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
#print(integers)

strings = tokenizer.decode(integers)
#print(strings)

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
enc_text = tokenizer.encode(raw_text)
#print(len(enc_text))

enc_sample = enc_text[50:]

context_size = 4 #A
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]
# print(f"x: {x}")
# print(f"y: {y}")

# for i in range(1, context_size+1):
#     context = enc_sample[:i]
#     desired = enc_sample[i]
#     print(context, "---->", desired)

# for i in range(1, context_size+1):
#     context = enc_sample[:i]
#     desired = enc_sample[i]
#     print(tokenizer.decode(context), "---->",
#     tokenizer.decode([desired]))

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
# dataloader = GPTDatasetV1.create_dataloader(raw_text, batch_size=1,max_length=4, stride=1)
# data_iter = iter(dataloader) #A
# first_batch = next(data_iter)
# print(first_batch)

dataloader = GPTDatasetV1.create_dataloader(raw_text, batch_size=8,max_length=4, stride=5)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)

output_dim = 256
vocab_size = 50257
token_embedding_layer = torch.nn.Embedding(vocab_size,output_dim)

token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)

class SimpleTransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers):
        super(SimpleTransformerModel, self).__init__()
        self.vocab_size = vocab_size 
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        transformer_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads) 
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.fc_out(x)
        return x


# Parameters
vocab_size = 50257  # Size of your vocabulary
embed_dim = 256     # Dimension of the token embeddings
num_heads = 8       # Number of attention heads
num_layers = 2      # Number of transformer layers

# Model, Loss, and Optimizer
model = SimpleTransformerModel(vocab_size, embed_dim, num_heads, num_layers)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
def train(dataloader, model, loss_fn, optimizer):
    model.train()
    for batch, (inputs, targets) in enumerate(dataloader):
        # Forward pass
        outputs = model(inputs)
        targets = targets.reshape(-1)  # Flatten targets to match output shape
        loss = loss_fn(outputs.view(-1, outputs.size(-1)), targets)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            print(f"Batch {batch}, Loss: {loss.item()}")

# Assuming 'dataloader' is already created and available
train(dataloader, model, loss_fn, optimizer)

def train_and_validate(train_loader, val_loader, model, loss_fn, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        for batch, (inputs, targets) in enumerate(train_loader):
            outputs = model(inputs)
            loss = loss_fn(outputs.view(-1, model.vocab_size), targets.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch}, Train Loss: {loss.item()}")

        model.eval()
        with torch.no_grad():
            total_val_loss = 0
            for inputs, targets in val_loader:
                outputs = model(inputs)
                val_loss = loss_fn(outputs.view(-1, model.vocab_size), targets.view(-1))
                total_val_loss += val_loss.item()
            avg_val_loss = total_val_loss / len(val_loader)
            print(f"Epoch {epoch}, Avg Val Loss: {avg_val_loss}")

# Create your dataloaders for training and validation
train_loader, val_loader = GPTDatasetV1.create_dataloaders(raw_text, batch_size=8, max_length=4, stride=5, train_ratio=0.8)
train_and_validate(train_loader, val_loader, model, loss_fn, optimizer, epochs=12)


# Save the model
torch.save(model.state_dict(), 'simple_transformer_model.pth')
