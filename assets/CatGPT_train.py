from CatGPT_model import GPT, GPTConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
from time import time
from dataclasses import dataclass
from math import cos, pi


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

@dataclass
class CatGPT_training_config:
    B = 2
    T = 1024
    float_matmul_precision = 'medium'
    vocab_size = 50304
    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = 10
    steps = 50
    betas = (0.9, 0.95)
    eps = 1e-8
    compile_model = False
    use_gpu = False

CatGPT_basic_config = CatGPT_training_config()

device = "cpu"

if CatGPT_training_config.use_gpu:
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")


# Create DataLoader
train_loader = DataLoaderLite("data/tiny_corpus.txt", B=CatGPT_training_config.B, T=CatGPT_training_config.T)

# Set matmul precision to lower

torch.set_float32_matmul_precision(CatGPT_training_config.float_matmul_precision)

# Create model and optimizer
model = GPT(GPTConfig(vocab_size=CatGPT_training_config.vocab_size))
model.to(device)

if CatGPT_training_config.compile_model:
    model = torch.compile(model)


# Warmup + cosine decay learning rate schedule

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < CatGPT_basic_config.warmup_steps:
        return CatGPT_basic_config.max_lr * (it + 1) / CatGPT_basic_config.warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > CatGPT_basic_config.steps:
        return CatGPT_basic_config.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - CatGPT_basic_config.warmup_steps) / (CatGPT_basic_config.steps - CatGPT_basic_config.warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + cos(pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return CatGPT_basic_config.min_lr + coeff * (CatGPT_basic_config.max_lr - CatGPT_basic_config.min_lr)


optimizer = torch.optim.AdamW(model.parameters(), lr=CatGPT_training_config.max_lr, betas=CatGPT_training_config.betas, eps=CatGPT_basic_config.eps)

for i in range(CatGPT_basic_config.steps):
    initial_time = time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # Update the learning rate
    lr = get_lr(i)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    dt = time() - initial_time
    tokens_processed = train_loader.B * train_loader.T
    tokens_per_second = tokens_processed / dt
    print(f"Step {i} | Loss: {loss.item()} | Time: {dt} | Tokens/s: {tokens_per_second} | LR: {lr}")




















num_return_sequences = 5
max_length = 35

import tiktoken

# prefix tokens
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Durant la primera guerra mundial, ")
tokens = torch.tensor(tokens, dtype=torch.long)  # (8,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # (B, 8)
x = tokens.to(device)


torch.manual_seed(42)
while x.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad():
        logits, _ = model(x)  # (B, T, vocab_size)
        # take the logits at the last position
        logits = logits[:, -1, :]  # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (huggingface pipeline default)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probabilities
        ix = torch.multinomial(topk_probs, 1)  # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
        # append to the sequence
        x = torch.cat((x, xcol), dim=1)

# print the generated text

for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(f"Sample {i+1}: {decoded}")