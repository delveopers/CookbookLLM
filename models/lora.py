import torch
import torch.nn as nn

class LoRALinear(nn.Module):
  def __init__(self, linear, r=8, alpha=16):
    super().__init__()
    self.linear = linear
    self.r = r
    self.alpha = alpha
    self.scaling = alpha / r

    in_features = linear.in_features
    out_features = linear.out_features

    self.A = nn.Parameter(torch.randn(r, in_features) * 0.01)
    self.B = nn.Parameter(torch.zeros(out_features, r))
    self.merged = False

  def forward(self, x):
    if self.merged:
      return self.linear(x)

    base = self.linear(x)
    lora = (x @ self.A.t()) @ self.B.t()
    return base + lora * self.scaling

  def merge(self):
    if self.merged: return
    delta_w = (self.B @ self.A) * self.scaling
    self.linear.weight.data += delta_w
    self.merged = True

  def unmerge(self):
    if not self.merged: return
    delta_w = (self.B @ self.A) * self.scaling
    self.linear.weight.data -= delta_w
    self.merged = False

def inject_lora(model, r=8, alpha=16, target_modules=None):
  if target_modules is None:
    target_modules = ["qkv", "proj", "w1", "w2", "w3"]

  for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
      if any(t in name for t in target_modules):
        parent = model
        parts = name.split('.')

        for p in parts[:-1]:
          parent = getattr(parent, p)
        setattr(parent, parts[-1], LoRALinear(module, r, alpha))
  return model

def freeze_base_model(model):

  for param in model.parameters():
    param.requires_grad = False

  for module in model.modules():
    if isinstance(module, LoRALinear):
      module.A.requires_grad = True
      module.B.requires_grad = True

def get_lora_params(model):
  params = []
  for module in model.modules():
    if isinstance(module, LoRALinear):
      params.append(module.A)
      params.append(module.B)
  return params

def save_lora(model, path):
  state = {}
  for name, module in model.named_modules():
    if isinstance(module, LoRALinear):
      state[name + ".A"] = module.A.detach().cpu()
      state[name + ".B"] = module.B.detach().cpu()

  torch.save(state, path)

def load_lora(model, path):
  state = torch.load(path, map_location="cpu")
  for name, module in model.named_modules():
    if isinstance(module, LoRALinear):
      module.A.data = state[name + ".A"]
      module.B.data = state[name + ".B"]

def merge_lora(model):
  for module in model.modules():
    if isinstance(module, LoRALinear):
      module.merge()