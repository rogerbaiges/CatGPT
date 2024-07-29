from CatGPT_model import GPT, GPTConfig

import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken


# Create Data Loadet class

class DataLoaderLite:
    def __init__(self, file, B, T):
        self.file = file
        self.B = B
        self.T = T

        # at init load tokens from disk and store them in memory
        with open(file, 'r') as f:
            text = f.read()
            enc = tiktoken.get_encoding('gpt2')
            tokens = enc.encode(text)
            self.tokens = torch.tensor(tokens)
            print(f"loaded {len(self.tokens)} tokens")
            print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets
        # advance the position in the tensor
        self.current_position += B * T
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y


# Attempt to autodetect the device
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"

device = "cpu" # Force CPU as on mac book pro m2 mps is slower

train_loader = DataLoaderLite("data/tiny_corpus.txt", B=4, T=32)

model = GPT(GPTConfig())
model.to(device)
 


optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    print(f"Step {i}, Loss: {loss.item()}")
