import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
  def __init__(self):
    super(SwiGLU, self).__init__()
    self.silu = nn.SiLU()

  def forward(self, x):
    # x shape: (batch, seq_len, 2*hidden)
    hidden, gate = x.chunk(2, dim=-1)
    return hidden * self.silu(gate)

class MultiHeadSelfAttention(nn.Module):
  def __init__(self, hidden_size, num_heads, dropout_prob=0.1):
    super(MultiHeadSelfAttention, self).__init__()
    assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
    self.num_heads = num_heads
    self.head_dim = hidden_size // num_heads
    self.scale = math.sqrt(self.head_dim)

    self.query = nn.Linear(hidden_size, hidden_size)
    self.key = nn.Linear(hidden_size, hidden_size)
    self.value = nn.Linear(hidden_size, hidden_size)
    self.out = nn.Linear(hidden_size, hidden_size)
    self.dropout = nn.Dropout(dropout_prob)

  def forward(self, x, attention_mask=None):
    # x shape: (batch, seq_len, hidden_size)
    batch_size, seq_len, hidden_size = x.size()

    # Linear projections
    Q = self.query(x)   # (batch, seq_len, hidden_size)
    K = self.key(x)
    V = self.value(x)

    # Split into heads
    def split_heads(tensor):
      # tensor: (batch, seq_len, hidden_size)
      return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim) \
                   .transpose(1, 2)
      # returns (batch, num_heads, seq_len, head_dim)

    Qh = split_heads(Q)
    Kh = split_heads(K)
    Vh = split_heads(V)

    # Scaled dot-product attention
    scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / self.scale
    # scores: (batch, num_heads, seq_len, seq_len)

    if attention_mask is not None:
      # mask shape: (batch, 1, 1, seq_len) or (batch, 1, seq_len, seq_len)
      scores = scores + attention_mask

    attn_probs = F.softmax(scores, dim=-1)
    attn_probs = self.dropout(attn_probs)

    context = torch.matmul(attn_probs, Vh)
    # context: (batch, num_heads, seq_len, head_dim)

    # Merge heads
    context = context.transpose(1, 2).contiguous() \
             .view(batch_size, seq_len, hidden_size)
    # context: (batch, seq_len, hidden_size)

    out = self.out(context)
    return out

class TransformerLayer(nn.Module):
  def __init__(self, hidden_size, num_heads, intermediate_size, dropout_prob=0.1):
    super(TransformerLayer, self).__init__()
    self.attention = MultiHeadSelfAttention(hidden_size, num_heads, dropout_prob)
    self.attn_layernorm = nn.LayerNorm(hidden_size, eps=1e-12)
    self.attn_dropout = nn.Dropout(dropout_prob)

    # Feedforward with SwiGLU
    self.ffn_dense = nn.Linear(hidden_size, intermediate_size * 2)
    self.ffn_swiglu = SwiGLU()
    self.ffn_output = nn.Linear(intermediate_size, hidden_size)
    self.ffn_layernorm = nn.LayerNorm(hidden_size, eps=1e-12)
    self.ffn_dropout = nn.Dropout(dropout_prob)

  def forward(self, x, attention_mask=None):
    # Multi-Head Self-Attention
    attn_out = self.attention(x, attention_mask)
    attn_out = self.attn_dropout(attn_out)
    x = self.attn_layernorm(x + attn_out)

    # Feed-Forward
    ffn_in = self.ffn_dense(x)            # (batch, seq_len, 2*intermediate)
    ffn_act = self.ffn_swiglu(ffn_in)     # (batch, seq_len, intermediate)
    ffn_out = self.ffn_output(ffn_act)    # (batch, seq_len, hidden_size)
    ffn_out = self.ffn_dropout(ffn_out)
    x = self.ffn_layernorm(x + ffn_out)
    return x

class BertEmbeddings(nn.Module):
  def __init__(self, vocab_size, hidden_size, max_position_embeddings, type_vocab_size, dropout_prob=0.1):
    super(BertEmbeddings, self).__init__()
    self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
    self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
    self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)

    self.layernorm = nn.LayerNorm(hidden_size, eps=1e-12)
    self.dropout = nn.Dropout(dropout_prob)

  def forward(self, input_ids, token_type_ids=None):
    seq_len = input_ids.size(1)
    if token_type_ids is None:
      token_type_ids = torch.zeros_like(input_ids)

    position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

    words_emb = self.word_embeddings(input_ids)
    pos_emb = self.position_embeddings(position_ids)
    type_emb  = self.token_type_embeddings(token_type_ids)

    embeddings = words_emb + pos_emb + type_emb
    embeddings = self.layernorm(embeddings)
    embeddings = self.dropout(embeddings)
    return embeddings

class BertModel(nn.Module):
  def __init__(self, vocab_size, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, max_position_embeddings=512, type_vocab_size=2, dropout_prob=0.1):
    super(BertModel, self).__init__()
    self.embeddings = BertEmbeddings(
      vocab_size, hidden_size, max_position_embeddings, type_vocab_size, dropout_prob
    )
    self.encoder_layers = nn.ModuleList([
      TransformerLayer(hidden_size, num_attention_heads, intermediate_size, dropout_prob)
      for _ in range(num_hidden_layers)
    ])

  def forward(self, input_ids, token_type_ids=None, attention_mask=None):
    """
    input_ids: (batch, seq_len)
    token_type_ids: (batch, seq_len)
    attention_mask: (batch, seq_len) with 1 for real tokens and 0 for padding
    """
    if attention_mask is not None:
      # create 4D mask for attention: (batch, 1, 1, seq_len)
      extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)
      # convert to float with large negative for masked positions
      extended_mask = (1.0 - extended_mask) * -10000.0
    else:
      extended_mask = None

    x = self.embeddings(input_ids, token_type_ids)

    for layer in self.encoder_layers:
      x = layer(x, extended_mask)

    return x  # (batch, seq_len, hidden_size)