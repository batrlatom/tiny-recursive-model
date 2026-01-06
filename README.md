# Tiny Recursive Model (TRM)

A concise, well-commented PyTorch implementation of **TRM** from:

> **Less is More: Recursive Reasoning with Tiny Networks**  
> [arXiv:2510.04871](https://arxiv.org/abs/2510.04871)

## Key Insight 🎯

A **TINY 2-layer network** with deep recursion beats large models!
- 87.4% on Sudoku-Extreme
- 45% on ARC-AGI-1  
- Only **~1.6M parameters**

## Installation

```bash
pip install torch einops
```

## Quick Start

```python
import torch
from trm import TinyRecursiveModel, train_step

# Create model
model = TinyRecursiveModel(
    dim=256,
    num_tokens=11,
    num_latent_refinements=6,  # n in paper
    num_refinement_blocks=3,   # T in paper
)

# Forward pass
x = torch.randint(0, 11, (2, 32))
out = model(x)
print(out['logits'].shape)  # [2, 32, 11]

# Training with deep supervision
labels = torch.randint(0, 11, (2, 32))
loss, metrics = train_step(model, x, labels, max_steps=16)
loss.backward()

# Inference with dynamic halting
preds, steps = model.predict(x, max_steps=16)
```

## Pluggable Networks

```python
from trm import TinyRecursiveModel, MLPMixer

# For Sudoku: MLP-Mixer on sequence dimension
mixer = MLPMixer(dim=256, seq_len=81, depth=2)
model = TinyRecursiveModel(dim=256, num_tokens=10, network=mixer)
```

## Architecture

```
Deep Supervision (K steps):
┌─────────────────────────────────────────────────────┐
│  For each step:                                     │
│    T refinement blocks (first T-1 no gradients):   │
│      n iterations: z ← network(y + z + x)          │
│      1 iteration:  y ← network(y + z)              │
│    loss = cross_entropy(y) + halt_loss             │
│    y, z = detach(y, z)                             │
│    if halt_prob > 0.5: break                       │
└─────────────────────────────────────────────────────┘
```

## File Structure

```
trm/
├── __init__.py     # Exports
└── trm.py          # Everything in one file! (~300 lines)
```

## Key Features

- ✅ **Pluggable network** - use any nn.Module
- ✅ **Register tokens** - extra scratch space for reasoning
- ✅ **Einops** - clean, readable tensor operations
- ✅ **Concise** - ~300 lines of well-commented code
- ✅ **Dynamic halting** - stop early when confident

## Citation

```bibtex
@article{jolicoeur2024trm,
  title={Less is More: Recursive Reasoning with Tiny Networks},
  author={Jolicoeur-Martineau, Alexia},
  journal={arXiv preprint arXiv:2510.04871},
  year={2024}
}
```
