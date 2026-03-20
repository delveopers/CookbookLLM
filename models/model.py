import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class RMSNorm(nn.Module):
  def __init__(self, dim, eps=1e-6):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim))

  def forward(self, x):
    norm = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(norm + self.eps)
    return self.weight * x


class RoPE(nn.Module):
  def __init__(self, head_dim, base=10000):
    super().__init__()
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    self.register_buffer("inv_freq", inv_freq)

  def get_embed(self, seq_len, device):
    t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
    freqs = torch.outer(t, self.inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]

  def rotate_half(self, x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.cat((-x2, x1), dim=-1)

  def apply_rotary(self, q, k, cos, sin):
    q = (q * cos) + (self.rotate_half(q) * sin)
    k = (k * cos) + (self.rotate_half(k) * sin)
    return q, k

class Attention(nn.Module):
  def __init__(self, d_model, n_heads, block_size):
    super().__init__()
    self.n_heads = n_heads
    self.head_dim = d_model // n_heads
    self.scale = self.head_dim ** -0.5
    self.block_size = block_size
    self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
    self.proj = nn.Linear(d_model, d_model, bias=False)
    self.rope = RoPE(self.head_dim)

    self.register_buffer(
      "mask",
      torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
      persistent=False
    )

  def forward(self, x, kv_cache=None):
    B, T, C = x.shape
    qkv = self.qkv(x)
    q, k, v = qkv.chunk(3, dim=-1)

    q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
    k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
    v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

    cos, sin = self.rope.get_embed(T, x.device)
    q, k = self.rope.apply_rotary(q, k, cos, sin)

    if kv_cache is not None:
      k = torch.cat([kv_cache[0], k], dim=2)
      v = torch.cat([kv_cache[1], v], dim=2)

    new_cache = (k, v)
    att = torch.matmul(q, k.transpose(-2, -1)) * self.scale
    att = att.masked_fill(self.mask[:, :, :T, :k.size(2)] == 0, -1e9)
    att = F.softmax(att, dim=-1)

    out = torch.matmul(att, v)
    out = out.transpose(1, 2).contiguous().view(B, T, C)
    out = self.proj(out)

    return out, new_cache

class SwiGLU(nn.Module):
  def __init__(self, d_model):
    super().__init__()
    hidden = int(2 * d_model * 4 / 3)
    self.w1 = nn.Linear(d_model, hidden, bias=False)
    self.w2 = nn.Linear(hidden, d_model, bias=False)
    self.w3 = nn.Linear(d_model, hidden, bias=False)

  def forward(self, x):
    return self.w2(F.silu(self.w1(x)) * self.w3(x))

class DecoderBlock(nn.Module):
  def __init__(self, d_model, n_heads, block_size):
    super().__init__()
    self.norm1 = RMSNorm(d_model)
    self.norm2 = RMSNorm(d_model)
    self.attn = Attention(d_model, n_heads, block_size)
    self.ff = SwiGLU(d_model)

  def forward(self, x, kv_cache=None):
    attn_out, cache = self.attn(self.norm1(x), kv_cache)
    x = x + attn_out
    x = x + self.ff(self.norm2(x))
    return x, cache

class GPT(nn.Module):
  def __init__(self, vocab_size, params):
    super().__init__()
    self.vocab_size = vocab_size
    self.d_model = params.d_model
    self.block_size = params.block_size
    self.embed = nn.Embedding(vocab_size, params.d_model)
    self.layers = nn.ModuleList([ DecoderBlock(params.d_model, params.n_heads, params.block_size) for _ in range(params.n_layers) ])
    self.norm = RMSNorm(params.d_model)
    self.lm_head = nn.Linear(params.d_model, vocab_size, bias=False)
    self.lm_head.weight = self.embed.weight
    self.apply(self._init_weights)

  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      nn.init.normal_(module.weight, mean=0.0, std=0.02)
      if module.bias is not None:
        nn.init.zeros_(module.bias)

    if isinstance(module, nn.Embedding):
      nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def forward(self, idx, targets=None):

    B, T = idx.shape
    tok = self.embed(idx) * math.sqrt(self.d_model)
    x = tok
    caches = []
    for layer in self.layers:
      x, cache = layer(x)
      caches.append(cache)

    x = self.norm(x)
    logits = self.lm_head(x)
    loss = None

    if targets is not None:
      loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1)
      )

    return logits, loss

  @torch.no_grad()
  def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):

    for _ in range(max_new_tokens):
      idx_cond = idx[:, -self.block_size:]
      logits, _ = self(idx_cond)
      logits = logits[:, -1, :] / temperature

      if top_k is not None:
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = -float("inf")

      if top_p is not None:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        mask = cumulative > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        sorted_logits[mask] = -float("inf")
        logits = torch.zeros_like(logits).scatter(1, sorted_indices, sorted_logits)

      probs = F.softmax(logits, dim=-1)
      next_token = torch.multinomial(probs, 1)
      idx = torch.cat((idx, next_token), dim=1)
    return idx