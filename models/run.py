import time, math
from contextlib import nullcontext
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn

from model import GPT
from config import ModelConfig, TrainConfig
from dataset import prepare_dataset, get_batch

@torch.no_grad()
def estimate_loss(model: nn.Module, train_data: torch.Tensor, val_data: torch.Tensor, eval_iters: int, batch_size: int, block_size: int, device: str) -> Dict[str, float]:
  """Estimate loss on train and validation sets"""
  model.eval()
  losses = {}
  
  for split, data in [('train', train_data), ('val', val_data)]:
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    
    for _ in range(eval_iters):
      X, Y = get_batch(data, batch_size, block_size, device)
      logits, loss = model(X, Y)
      
      total_loss += loss.item()
      
      # Calculate accuracy
      predictions = torch.argmax(logits, dim=-1)
      correct = (predictions == Y).sum().item()
      total_correct += correct
      total_tokens += Y.numel()
    
    losses[split] = total_loss / eval_iters
    losses[f'{split}_acc'] = total_correct / total_tokens
  
  model.train()
  return losses

def get_lr(step: int, warmup_steps: int, lr_decay_steps: int, max_lr: float, min_lr: float) -> float:
  """Get learning rate with warmup and cosine decay"""
  # Linear warmup
  if step < warmup_steps:
    return max_lr * step / warmup_steps

  # Cosine decay
  if step > lr_decay_steps:
    return min_lr
  
  decay_ratio = (step - warmup_steps) / (lr_decay_steps - warmup_steps)
  coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
  return min_lr + coeff * (max_lr - min_lr)

def configure_optimizers(model: nn.Module, weight_decay: float, learning_rate: float, betas: Tuple[float, float]) -> torch.optim.Optimizer:
  """Configure optimizer with weight decay for specific parameters"""
  # Separate parameters that should and shouldn't be weight decayed
  decay = set()
  no_decay = set()

  whitelist_weight_modules = (torch.nn.Linear, )
  blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
  
  for mn, m in model.named_modules():
    for pn, p in m.named_parameters():
      fpn = '%s.%s' % (mn, pn) if mn else pn
      
      if pn.endswith('bias'):
        no_decay.add(fpn)
      elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
        decay.add(fpn)
      elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
        no_decay.add(fpn)
  
  # creating parameter groups
  param_dict = {pn: p for pn, p in model.named_parameters()}
  optim_groups = [
    {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
    {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
  ]

  optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
  return optimizer

def count_parameters(model: nn.Module) -> int:
  """Count total number of parameters in the model"""
  return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_info(model: nn.Module, model_config: Dict[str, Any]) -> None:
  """Print detailed model information"""
  print("\n" + "="*60)
  print("MODEL INFORMATION")
  print("="*60)

  total_params = count_parameters(model)
  print(f"Total Parameters: {total_params:,} ({total_params/1e6:.1f}M)")
  
  print(f"Architecture:")
  print(f"  - Layers: {model_config['n_layers']}")
  print(f"  - Heads: {model_config['n_heads']}")
  print(f"  - Model Dimension: {model_config['d_model']}")
  print(f"  - Head Dimension: {model_config['d_model'] // model_config['n_heads']}")
  print(f"  - FFN Multiplier: {model_config['ffn_multiplier']}")
  print(f"  - Max Sequence Length: {model_config['max_seq_len']}")
  print(f"  - Vocabulary Size: {model_config['vocab_size']}")
  print(f"  - Dropout: {model_config['dropout']}")
  
  # memory estimation
  param_size = total_params * 4 / (1024**3)  # 4 bytes per parameter
  print(f"  - Estimated Model Size: {param_size:.2f} GB")
  print("="*60)

def main():
  """Main training function"""  
  # google-drive dataset URL (replace with your actual URL)
  dataset_url = "https://drive.google.com/"
  dataset_path = "dataset.txt"

  # load configurations
  print("Loading configurations...")

  # set device
  device = TrainConfig.device
  if device == 'cuda' and not torch.cuda.is_available():
    device = 'cpu'
    print("CUDA not available, using CPU")
  
  print(f"Using device: {device}")

  # setting random seeds for reproducibility
  torch.manual_seed(1337)
  if device == 'cuda':
    torch.cuda.manual_seed(1337)

  # downloading and preparing dataset
  train_data, val_data = prepare_dataset(dataset_path)
  print("Initializing model...")

  # create a params object with the configuration
  model = GPT(
    vocab_size=ModelConfig.vocab_size,
    params=ModelConfig
  ).to(device)
  model = model.to(memory_format=torch.channels_last)

  torch.backends.cuda.matmul.allow_tf32 = True
  torch.backends.cudnn.allow_tf32 = True
  # print model information
  print_model_info(model, vars(ModelConfig))

  # configure optimizer
  optimizer = configure_optimizers(model,
    weight_decay=TrainConfig.weight_decay,
    learning_rate=TrainConfig.learning_rate,
    betas=(TrainConfig.beta1, TrainConfig.beta2)
  )

  # compiling model for faster training (PyTorch 2.0+)
  if getattr(TrainConfig, "compile", False):
    print("Compiling model...")
    try:
      model = torch.compile(model)
      print("Model compiled successfully!")
    except:
      print("Model compilation failed, continuing without compilation")

  # Training setup
  use_amp = TrainConfig.dtype == "float16"
  scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
  ctx = nullcontext() if device == "cpu" else torch.autocast(
    device_type=device,
    dtype=torch.bfloat16 if TrainConfig.dtype == "bfloat16" else torch.float16
  )
  
  # Training parameters
  batch_size = TrainConfig.batch_size
  block_size = TrainConfig.block_size
  gradient_accumulation_steps = TrainConfig.gradient_accumulation_steps
  grad_clip = TrainConfig.grad_clip
  eval_interval = TrainConfig.eval_interval
  eval_iters = TrainConfig.eval_iters
  log_interval = TrainConfig.log_interval

  warmup_steps = TrainConfig.warmup_steps
  lr_decay_steps = TrainConfig.lr_decay_steps
  learning_rate = TrainConfig.learning_rate
  min_lr = TrainConfig.min_lr
  max_iters = TrainConfig.max_iters

  print(f"\nStarting training for {max_iters:,} iterations...")
  print(f"Batch size: {batch_size}, Block size: {block_size}")
  print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
  print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")

  # Training loop
  model.train()
  step = 0
  start_time = time.time()

  losses = estimate_loss(model, train_data, val_data, eval_iters, batch_size, block_size, device)
  print(f"\nStep {step:6d} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f} | " f"Train Acc: {losses['train_acc']:.3f} | Val Acc: {losses['val_acc']:.3f}")

  while step < max_iters:
    lr = get_lr(step, warmup_steps, lr_decay_steps, learning_rate, min_lr)
    for param_group in optimizer.param_groups:
      param_group['lr'] = lr

    # Evaluate and log
    if step % eval_interval == 0 and step > 0:
      losses = estimate_loss(model, train_data, val_data, eval_iters, batch_size, block_size, device)
      elapsed_time = time.time() - start_time
      avg_time_per_step = elapsed_time / step if step > 0 else 0

      print(f"Step {step:6d} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f} | " f"Train Acc: {losses['train_acc']:.3f} | Val Acc: {losses['val_acc']:.3f} | " f"LR: {lr:.2e} | Time/Step: {avg_time_per_step:.2f}s")

    # Training step
    optimizer.zero_grad(set_to_none=True)
    loss_accum = 0.0

    for micro_step in range(gradient_accumulation_steps):
      X, Y = get_batch(train_data, batch_size, block_size, device)

      with ctx:
        logits, loss = model(X, Y)
        loss = loss / gradient_accumulation_steps
        loss_accum += loss.detach()
      scaler.scale(loss).backward()

    # Gradient clipping
    if grad_clip != 0.0:
      scaler.unscale_(optimizer)
      torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # Optimizer step
    scaler.step(optimizer)
    scaler.update()

    step += 1

    # Log training progress
    if step % log_interval == 0:
      elapsed_time = time.time() - start_time
      avg_time_per_step = elapsed_time / step
      print(f"Step {step:6d} | Loss: {loss_accum:.4f} | LR: {lr:.2e} | Time/Step: {avg_time_per_step:.2f}s")

  print(f"\nTraining completed! Total time: {(time.time() - start_time) / 3600:.2f} hours")

  # Save final model
  checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'model_config': vars(ModelConfig),
    'train_config': vars(TrainConfig),
    'step': step,
  }

  model_name = "consolidated_00"
  torch.save(checkpoint, f'{model_name}.pt')
  print(f"Model saved as {model_name}.pt")

  # Final evaluation
  losses = estimate_loss(model, train_data, val_data, eval_iters, batch_size, block_size, device)
  print(f"\nFinal Results:")
  print(f"Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f}")
  print(f"Train Accuracy: {losses['train_acc']:.3f} | Val Accuracy: {losses['val_acc']:.3f}")

if __name__ == "__main__":
  main()