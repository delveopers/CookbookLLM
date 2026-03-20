from dataclasses import dataclass

# =========================
# Model Config
# =========================

@dataclass
class ModelConfig:
  n_layers: int
  n_heads: int
  d_model: int
  vocab_size: int = 50304
  max_seq_len: int = 2048
  ffn_multiplier: float = 2.67
  bias: bool = False
  dropout: float = 0.0
  norm_eps: float = 1e-5
  device: str = "cuda"

  @property
  def head_dim(self):
    assert self.d_model % self.n_heads == 0
    return self.d_model // self.n_heads

  @property
  def ffn_dim(self):
    return int(self.d_model * self.ffn_multiplier)


# =========================
# Training Config
# =========================

@dataclass
class TrainConfig:
  learning_rate: float
  min_lr: float
  weight_decay: float = 0.1
  beta1: float = 0.9
  beta2: float = 0.95
  grad_clip: float = 1.0

  warmup_steps: int = 2000
  lr_decay_steps: int = 600000

  batch_size: int = 32
  micro_batch_size: int = 4
  gradient_accumulation_steps: int = 8

  block_size: int = 1024

  eval_interval: int = 2000
  eval_iters: int = 200
  log_interval: int = 10

  dtype: str = "bfloat16"
  compile: bool = True
  device: str = "cuda"


# =========================
# LoRA Config (for finetuning)
# =========================

@dataclass
class LoRAConfig:
  r: int = 8
  alpha: int = 16
  dropout: float = 0.0
  target_modules: tuple = ("qkv", "proj")


# =========================
# Presets
# =========================

GPT_CONFIGS = {

  "GPT_500M": {
    "model": ModelConfig(
      n_layers=24,
      n_heads=16,
      d_model=1024
    ),
    "train": TrainConfig(
      learning_rate=3e-4,
      min_lr=3e-5,
      batch_size=64,
      micro_batch_size=4,
      gradient_accumulation_steps=16
    )
  },

  "GPT_750M": {
    "model": ModelConfig(
      n_layers=24,
      n_heads=20,
      d_model=1280
    ),
    "train": TrainConfig(
      learning_rate=2.5e-4,
      min_lr=2.5e-5,
      batch_size=48,
      micro_batch_size=3,
      gradient_accumulation_steps=16
    )
  },

  "GPT_1B": {
    "model": ModelConfig(
      n_layers=24,
      n_heads=24,
      d_model=1536
    ),
    "train": TrainConfig(
      learning_rate=2e-4,
      min_lr=2e-5,
      batch_size=32,
      micro_batch_size=2,
      gradient_accumulation_steps=16
    )
  },

  "GPT_3B": {
    "model": ModelConfig(
      n_layers=32,
      n_heads=32,
      d_model=2560
    ),
    "train": TrainConfig(
      learning_rate=1.5e-4,
      min_lr=1.5e-5,
      batch_size=16,
      micro_batch_size=2,
      gradient_accumulation_steps=16
    )
  },

  "GPT_7B": {
    "model": ModelConfig(
      n_layers=32,
      n_heads=32,
      d_model=4096,
      max_seq_len=4096
    ),
    "train": TrainConfig(
      learning_rate=1e-4,
      min_lr=1e-5,
      batch_size=8,
      micro_batch_size=1,
      gradient_accumulation_steps=32
    )
  }
}