import torch
import torch.nn as nn

from config import ModelConfig, TrainConfig, LoRAConfig
from model import GPT
from lora import inject_lora, freeze_base_model, get_lora_params
from dataset import prepare_dataset, get_batch

device = TrainConfig.device

# init model
model = GPT(
  vocab_size=ModelConfig.vocab_size,
  params=ModelConfig
).to(device)

# load pretrained weights
checkpoint = torch.load("base_model.pt", map_location=device)
state_dict = checkpoint.get("model_state_dict", checkpoint)
model.load_state_dict(state_dict)

# inject LoRA
model = inject_lora(
  model,
  r=LoRAConfig.r,
  alpha=LoRAConfig.alpha,
  dropout=LoRAConfig.dropout,
  target_modules=LoRAConfig.target_modules
)

# freeze base
freeze_base_model(model)

# optimizer ONLY LoRA
optimizer = torch.optim.AdamW(
  get_lora_params(model),
  lr=LoRAConfig.lr, weight_decay=0.0)

model.train()

train_data, _ = prepare_dataset("dataset.txt")

for step in range(LoRAConfig.max_steps):

  x, y = get_batch(train_data,
    LoRAConfig.batch_size,ModelConfig.max_seq_len, device)

  logits, loss = model(x, y)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  if step % 100 == 0:
    print(f"step {step} | loss {loss.item():.4f}")

# save ONLY LoRA weights
torch.save({
  "lora_state_dict": {
    k: v for k, v in model.state_dict().items() if "lora" in k
  }
}, "lora_adapter.pt")