import torch
import tiktoken

def prepare_dataset(data_path, split=0.9):
  with open(data_path, 'r', encoding='utf-8') as f:
    text = f.read()

  enc = tiktoken.get_encoding("cl100k_base")
  tokens = torch.tensor(enc.encode(text), dtype=torch.long)

  n = int(split * len(tokens))
  return tokens[:n], tokens[n:]

def get_batch(data, batch_size, block_size, device):
  ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  return x.to(device), y.to(device)