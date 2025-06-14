# GPT Model Documentation

## Overview

This codebase implements a modern, optimized GPT (Generative Pre-trained Transformer) model with state-of-the-art improvements including Rotary Position Embedding (RoPE), RMS Normalization, SwiGLU activation, and efficient attention mechanisms. The implementation supports three model sizes: 500M, 750M, and 1B parameters.

## Model Architecture

### Core Components

#### 1. **RMS Normalization (RMSNorm)**
- **Purpose**: Provides better training stability compared to LayerNorm
- **Implementation**: Normalizes using root mean square instead of mean and variance
- **Benefits**: Faster computation, better gradient flow

#### 2. **Rotary Position Embedding (RoPE)**
- **Purpose**: Superior position encoding that maintains relative position information
- **Key Features**:
  - Precomputes cosine and sine components for efficiency
  - Applies rotation to query and key vectors
  - Better length generalization than absolute position embeddings
  - Supports sequences up to 8192 tokens by default

#### 3. **Efficient Attention Mechanism**
- **Features**:
  - Combined QKV projection for reduced memory access
  - Flash Attention optimization when available
  - Causal masking for autoregressive generation
  - Scaled dot-product attention with dropout

#### 4. **SwiGLU Feed-Forward Network**
- **Purpose**: Improved activation function for better performance
- **Architecture**: Uses SiLU (Swish) activation with gating mechanism
- **Expansion Factor**: 8/3 (2.67) for optimal parameter efficiency

#### 5. **Decoder Block**
- **Structure**: Pre-normalization with residual connections
- **Components**:
  - Multi-head self-attention with RoPE
  - SwiGLU feed-forward network
  - RMS normalization layers
  - Dropout for regularization

### Model Configurations

| Model Size | Parameters | Layers | Heads | d_model | FFN Multiplier | Context Length |
|------------|------------|--------|-------|---------|----------------|----------------|
| GPT-500M   | ~500M      | 24     | 16    | 1024    | 2.67           | 2048           |
| GPT-750M   | ~750M      | 24     | 20    | 1280    | 2.67           | 2048           |
| GPT-1B     | ~1B        | 24     | 16    | 1536    | 2.67           | 2048           |

## Installation and Setup

### Prerequisites
```bash
# Required packages
pip install torch torchvision torchaudio
pip install tiktoken
pip install numpy
```

### File Structure
```
project/
├── config.json          # Model and training configurations
├── model.py             # Model architecture implementation
├── run.py               # Training and evaluation script
├── __init__.py          # Package initialization
└── dataset.txt          # Training data (text file)
```

## Usage Guide

### 1. Model Initialization

```python
from model import GPT
import json

# Load configuration
with open('config.json', 'r') as f:
  configs = json.load(f)[0]  # Select first config set

model_config = configs['GPT_500M']['ModelConfig']
train_config = configs['GPT_500M']['TrainConfig']

# Create model parameters object
class ModelConfig:
  def __init__(self, config_dict):
    for key, value in config_dict.items():
      setattr(self, key, value)

params = ModelConfig(model_config)
model = GPT(vocab_size=50304, params=params)
```

### 2. Text Generation

```python
import torch
import tiktoken

# Initialize tokenizer
enc = tiktoken.get_encoding("cl100k_base")

# Prepare input
prompt = "Once upon a time"
tokens = enc.encode(prompt)
input_ids = torch.tensor([tokens], dtype=torch.long)

# Generate text
model.eval()
with torch.no_grad():
    generated = model.generate(
        input_ids,
        max_new_tokens=100,
        temperature=0.8,
        top_k=50,
        top_p=0.9
    )

# Decode output
generated_text = enc.decode(generated[0].tolist())
print(generated_text)
```

### 3. Training Configuration

The model supports three pre-configured sizes. To select a different model:

```python
# In run.py, modify the model_name variable
model_name = "GPT_500M"  # or "GPT_750M" or "GPT_1B"
```

### 4. Dataset Preparation

Prepare your training data as a single text file:

```python
# dataset.txt should contain your training text
# Example format:
"""
Your training text goes here.
Multiple paragraphs and documents can be concatenated.
The tokenizer will handle the preprocessing.
"""
```

## Training Guide

### 1. Basic Training

```bash
python run.py
```

### 2. Training Parameters

Key training hyperparameters for each model size:

#### GPT-500M
- **Learning Rate**: 6e-4
- **Batch Size**: 12 (effective: 36 with gradient accumulation)
- **Weight Decay**: 0.1
- **Warmup Steps**: 2000
- **Max Iterations**: 600,000

#### GPT-750M
- **Learning Rate**: 3e-4
- **Batch Size**: 8 (effective: 32 with gradient accumulation)
- **Weight Decay**: 0.1
- **Warmup Steps**: 2000
- **Max Iterations**: 600,000

#### GPT-1B
- **Learning Rate**: 3e-4
- **Batch Size**: 6 (effective: 18 with gradient accumulation)
- **Weight Decay**: 0.1
- **Warmup Steps**: 2000
- **Max Iterations**: 600,000

### 3. Training Features

- **Mixed Precision**: Uses bfloat16 for faster training
- **Gradient Accumulation**: Enables larger effective batch sizes
- **Model Compilation**: PyTorch 2.0+ optimization
- **Learning Rate Scheduling**: Warmup followed by cosine decay
- **Gradient Clipping**: Prevents gradient explosion

### 4. Monitoring Training

The training script provides detailed logging:

```
Step   2000 | Train Loss: 3.2451 | Val Loss: 3.2834 | 
Train Acc: 0.234 | Val Acc: 0.231 | LR: 6.00e-04 | Time/Step: 0.45s
```

### 5. Checkpointing

Models are automatically saved with:
- Model state dictionary
- Optimizer state dictionary
- Configuration parameters
- Training step information

## Advanced Usage

### 1. Custom Model Configuration

```python
# Create custom configuration
custom_config = {
    "n_heads": 16,
    "n_layers": 24,
    "d_model": 1024,
    "bias": False,
    "ffn_multiplier": 2.67,
    "dropout": 0.1,
    "norm_eps": 1e-5,
    "max_seq_len": 2048,
    "vocab_size": 50304,
    "device": "cuda"
}
```

### 2. Fine-tuning from Checkpoint

```python
# Load pre-trained model
checkpoint = torch.load('GPT_500M_final.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Continue training or fine-tune
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

### 3. Generation Parameters

| Parameter | Description | Recommended Range |
|-----------|-------------|-------------------|
| `temperature` | Controls randomness | 0.7-1.2 |
| `top_k` | Top-k sampling | 20-100 |
| `top_p` | Nucleus sampling | 0.8-0.95 |
| `max_new_tokens` | Generation length | 50-500 |

### 4. Memory Optimization

For limited GPU memory:
- Reduce `batch_size` and increase `gradient_accumulation_steps`
- Use smaller model variant (500M instead of 1B)
- Enable gradient checkpointing (requires code modification)

## Performance Optimization

### 1. Hardware Recommendations

| Model Size | Minimum VRAM | Recommended VRAM | Training Time (Estimate) |
|------------|--------------|------------------|--------------------------|
| GPT-500M   | 8GB          | 16GB             | 2-3 days                 |
| GPT-750M   | 12GB         | 24GB             | 3-4 days                 |
| GPT-1B     | 16GB         | 32GB             | 4-5 days                 |

### 2. Optimization Features

- **Flash Attention**: Automatic when available
- **Model Compilation**: PyTorch 2.0+ optimization
- **Efficient Data Loading**: Optimized batch generation
- **Weight Tying**: Shared embedding and output layer weights

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Increase gradient accumulation steps
   - Use smaller model variant

2. **Slow Training**
   - Enable model compilation
   - Use mixed precision training
   - Optimize data loading

3. **Poor Generation Quality**
   - Increase training steps
   - Adjust generation parameters
   - Use larger model variant

### Error Messages

- **"Sequence length exceeds block size"**: Input sequence too long
- **"Model compilation failed"**: PyTorch version incompatibility
- **"Dataset not found"**: Check dataset path

## Model Limitations

1. **Context Length**: Maximum 2048 tokens (can be extended)
2. **Vocabulary**: Fixed to GPT-4 tokenizer (50,304 tokens)
3. **Architecture**: Decoder-only (no encoder component)
4. **Training Data**: Requires large text corpus for good performance

## Contributing

When modifying the codebase:

1. **Model Architecture**: Changes in `model.py`
2. **Training Logic**: Modifications in `run.py`
3. **Configuration**: Updates in `config.json`
4. **Testing**: Ensure compatibility across model sizes

## License and Citations

If you use this implementation, please cite:

- **RoPE**: RoFormer: Enhanced Transformer with Rotary Position Embedding
- **SwiGLU**: GLU Variants Improve Transformer
- **RMSNorm**: Root Mean Square Layer Normalization

## Appendix

### Configuration Schema

```json
{
  "ModelConfig": {
    "n_heads": "Number of attention heads",
    "n_layers": "Number of transformer layers",
    "d_model": "Model dimension",
    "bias": "Use bias in linear layers",
    "ffn_multiplier": "Feed-forward expansion factor",
    "dropout": "Dropout rate",
    "norm_eps": "Normalization epsilon",
    "max_seq_len": "Maximum sequence length",
    "vocab_size": "Vocabulary size",
    "device": "Training device"
  },
  "TrainConfig": {
    "learning_rate": "Initial learning rate",
    "weight_decay": "L2 regularization",
    "beta1": "Adam beta1 parameter",
    "beta2": "Adam beta2 parameter",
    "grad_clip": "Gradient clipping threshold",
    "warmup_steps": "Learning rate warmup steps",
    "lr_decay_steps": "Learning rate decay steps",
    "min_lr": "Minimum learning rate",
    "batch_size": "Training batch size",
    "gradient_accumulation_steps": "Gradient accumulation",
    "block_size": "Sequence block size",
    "eval_interval": "Evaluation frequency",
    "eval_iters": "Evaluation iterations"
  }
}
```

### Memory Usage Estimation

```python
def estimate_memory_usage(model_config):
    """Estimate GPU memory usage for training"""
    params = calculate_parameters(model_config)
    
    # Model parameters (4 bytes per param)
    model_memory = params * 4
    
    # Gradients (same size as parameters)
    gradient_memory = params * 4
    
    # Optimizer state (2x parameters for Adam)
    optimizer_memory = params * 8
    
    # Activations (depends on batch size and sequence length)
    activation_memory = estimate_activations(model_config)
    
    total_gb = (model_memory + gradient_memory + optimizer_memory + activation_memory) / (1024**3)
    return total_gb
```

This documentation provides a comprehensive guide to understanding, using, and training the GPT model implementation. For additional questions or advanced use cases, refer to the source code comments and PyTorch documentation.
