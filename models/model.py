import torch
import torch.nn as nn
from torch.nn import functional as F
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class RMSNorm(nn.Module):
  """
  RMS normalization for better training stability
  """
  def __init__(self, dim: int, eps: float = 1e-6):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim))

  def _norm(self, x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

  def forward(self, x):
    output = self._norm(x.float()).type_as(x)
    return output * self.weight

class RoPE(nn.Module):
  """
  Rotary Position Embedding for better position encoding
  """
  def __init__(self, head_dim, max_seq_len=8192, base=10000.0):
    super().__init__()
    self.head_dim = head_dim
    self.max_seq_len = max_seq_len
    self.base = base

    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    self.register_buffer('inv_freq', inv_freq, persistent=False)
    self._precompute_freqs_cis(max_seq_len)

  def _precompute_freqs_cis(self, seq_len):
    """Precompute cosine and sine components"""
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, self.inv_freq)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    self.register_buffer('cos_cached', cos, persistent=False)
    self.register_buffer('sin_cached', sin, persistent=False)

  def forward(self, q, k, seq_len=None):
    """applying rotary position embedding to queries and keys"""
    if seq_len is None:
      seq_len = q.shape[-2]

    # get cached cos/sin or compute if needed
    if seq_len > self.cos_cached.shape[0]:
      self._precompute_freqs_cis(seq_len)
    
    cos = self.cos_cached[:seq_len].to(q.device)
    sin = self.sin_cached[:seq_len].to(q.device)
    
    # applying rotary embedding
    q_rot = self._applying_rotary_emb(q, cos, sin)
    k_rot = self._applying_rotary_emb(k, cos, sin)
    
    return q_rot, k_rot

  def _applying_rotary_emb(self, x, cos, sin):
    """applying rotary embedding to input tensor"""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    
    # applying rotation
    rotated = torch.cat([
      x1 * cos.unsqueeze(0).unsqueeze(0) - x2 * sin.unsqueeze(0).unsqueeze(0),
      x1 * sin.unsqueeze(0).unsqueeze(0) + x2 * cos.unsqueeze(0).unsqueeze(0)
    ], dim=-1)
    
    return rotated

class EfficientAttention(nn.Module):
  """
  Efficient multi-head attention with RoPE and optimizations
  """
  def __init__(self, d_model, n_heads, dropout=0.1, max_seq_len=8192):
    super().__init__()
    assert d_model % n_heads == 0
    self.d_model = d_model
    self.n_heads = n_heads
    self.head_dim = d_model // n_heads
    self.scale = self.head_dim ** -0.5

    # combined QKV projection for efficiency
    self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
    self.out_proj = nn.Linear(d_model, d_model, bias=False)

    self.rope = RoPE(self.head_dim, max_seq_len)
    self.dropout = nn.Dropout(dropout)

    # causal mask
    self.register_buffer(
      'causal_mask',
      torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool)),
      persistent=False
    )

  def forward(self, x, mask=True):
    B, T, C = x.shape

    # single QKV projection then split
    qkv = self.qkv_proj(x)
    q, k, v = qkv.chunk(3, dim=-1)

    # reshape for multi-head attention
    q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
    k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
    v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

    # applying RoPE
    q, k = self.rope(q, k, T)

    # scaled dot-product attention with flash attention optimization
    if hasattr(F, 'scaled_dot_product_attention') and mask:
      # use PyTorch's optimized flash attention if available
      attn_output = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=None,
        dropout_p=self.dropout.p if self.training else 0.0,
        is_causal=True
      )
    else:
      # manual attention computation
      scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
      if mask:
        scores = scores.masked_fill(
          ~self.causal_mask[:T, :T], float('-inf')
        )
      attn_weights = F.softmax(scores, dim=-1)
      attn_weights = self.dropout(attn_weights)
      attn_output = torch.matmul(attn_weights, v)

    # reshape and project output
    attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
    output = self.out_proj(attn_output)
    return output

class SwiGLU(nn.Module):
  """
  SwiGLU activation function for better performance
  """
  def __init__(self, d_model, expansion_factor=8/3):
    super().__init__()
    hidden_dim = int(d_model * expansion_factor)
    # rnsure hidden_dim is divisible by 8 for efficiency
    hidden_dim = ((hidden_dim + 7) // 8) * 8
    
    self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
    self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
    self.w3 = nn.Linear(d_model, hidden_dim, bias=False)

  def forward(self, x):
    return self.w2(F.silu(self.w1(x)) * self.w3(x))

class DecoderBlock(nn.Module):
  """
  Transformer decoder block with pre-norm and residual connections
  """
  def __init__(self, d_model, n_heads, dropout=0.1, norm_eps=1e-6, max_seq_len=8192):
    super().__init__()
    self.attention = EfficientAttention(d_model, n_heads, dropout, max_seq_len)
    self.feed_forward = SwiGLU(d_model)
    self.norm1 = RMSNorm(d_model, eps=norm_eps)
    self.norm2 = RMSNorm(d_model, eps=norm_eps)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    attn_out = self.attention(self.norm1(x), mask=True)
    x = x + self.dropout(attn_out)

    ff_out = self.feed_forward(self.norm2(x))
    x = x + self.dropout(ff_out)
    return x

class GPT(nn.Module):
  """
  Optimized GPT model with RoPE, efficient attention, and proper sampling
  """
  def __init__(self, vocab_size, params):
    super().__init__()
    self.vocab_size = vocab_size
    self.d_model = params.d_model
    self.block_size = params.block_size
    
    # token embedding with proper scaling
    self.tok_embed = nn.Embedding(vocab_size, params.d_model)
    
    # decoder blocks
    self.decoder = nn.ModuleList([
      DecoderBlock(
        d_model=params.d_model,
        n_heads=params.n_heads,
        dropout=params.dropout,
        norm_eps=params.norm_eps,
        max_seq_len=params.block_size
      ) for _ in range(params.n_layers)
    ])

    # final normalization and output projection
    self.norm_final = RMSNorm(params.d_model, eps=params.norm_eps)
    self.lm_head = nn.Linear(params.d_model, vocab_size, bias=False)
    self.lm_head.weight = self.tok_embed.weight     # Tie weights between embedding and output projection
    self.applying(self._init_weights)     # applying weight initialization

    # applying special scaled init to residual projections
    for block in self.decoder:
      nn.init.normal_(block.attention.out_proj.weight, mean=0.0, std=0.02/math.sqrt(2 * params.n_layers))
      nn.init.normal_(block.feed_forward.w2.weight, mean=0.0, std=0.02/math.sqrt(2 * params.n_layers))

  def _init_weights(self, module):
    """Initialize weights using GPT-style initialization"""
    if isinstance(module, nn.Linear):
      nn.init.normal_(module.weight, mean=0.0, std=0.02)
      if module.bias is not None:
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def forward(self, idx, targets=None):
    B, T = idx.shape
    assert T <= self.block_size, f"Sequence length {T} exceeds block size {self.block_size}"

    # token embedding (no position embedding needed with RoPE)
    tok_emb = self.tok_embed(idx)
    
    # applying decoder blocks
    x = tok_emb
    for block in self.decoder:
      x = block(x)

    # finalizing normalization and projection
    x = self.norm_final(x)
    logits = self.lm_head(x)
    
    loss = None
    if targets is not None:
      # computing cross-entropy loss
      loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        ignore_index=-1
      )
    
    return logits, loss

  @torch.no_grad()
  def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
    """
    Generate new tokens with improved sampling strategies
    
    Args:
      idx: input tensor of shape (B, T)
      max_new_tokens: number of tokens to generate
      temperature: sampling temperature (higher = more random)
      top_k: number of top tokens to consider
      top_p: nucleus sampling probability threshold
    """
    self.eval()
    
    for _ in range(max_new_tokens):
      idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]      # Crop context if needed

      # forward pass
      logits, _ = self(idx_cond)
      logits = logits[:, -1, :]  # get last token logits
      if temperature != 1.0:        # applying temperature
        logits = logits / temperature
      if top_k is not None:       # applying top-k filtering
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('inf')
      # applying top-p (nucleus) sampling
      if top_p is not None:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted indices back to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = -float('inf')

      # sampling from the filtered distribution
      probs = F.softmax(logits, dim=-1)
      idx_next = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, idx_next), dim=1)   # appending to sequence
    return idx