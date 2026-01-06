"""
TRM Training Script - Sort Task

A proper training script to validate that TRM works on a simple reasoning task:
  Given a sequence of digits, predict the sorted sequence.

This is a good test for recursive reasoning because:
  - It's simple to verify (just check if output is sorted)
  - It requires comparing multiple elements (not just local patterns)
  - It benefits from iterative refinement (TRM's strength!)

Example:
  Input:  [3, 1, 4, 1, 5, 9, 2, 6]
  Output: [1, 1, 2, 3, 4, 5, 6, 9]
"""

import os
import time
import random
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Import our TRM
from trm import TinyRecursiveModel, MLPMixer


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TrainConfig:
    # Task
    seq_len: int = 16           # Length of sequences to sort
    num_tokens: int = 10        # Digits 0-9
    
    # Data
    train_size: int = 10000     # Training examples
    val_size: int = 1000        # Validation examples
    
    # Model
    dim: int = 128              # Hidden dimension (small for speed)
    network_depth: int = 2      # Number of layers
    num_latent_refinements: int = 4   # n in paper
    num_refinement_blocks: int = 2    # T in paper
    use_mlp_mixer: bool = True  # MLP-Mixer is faster for fixed seq_len
    
    # Training
    batch_size: int = 64
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 0.1
    max_supervision_steps: int = 8
    
    # Misc
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    log_every: int = 50
    eval_every: int = 1         # Evaluate every N epochs


# =============================================================================
# Dataset: Sort Task
# =============================================================================

class SortDataset(Dataset):
    """
    Dataset for the sorting task.
    
    Each example is a random sequence of digits.
    The target is the sorted sequence.
    """
    
    def __init__(self, size: int, seq_len: int, num_tokens: int, seed: int = 42):
        self.size = size
        self.seq_len = seq_len
        self.num_tokens = num_tokens
        
        # Generate all data upfront for reproducibility
        rng = random.Random(seed)
        self.data = []
        for _ in range(size):
            seq = [rng.randint(0, num_tokens - 1) for _ in range(seq_len)]
            target = sorted(seq)
            self.data.append((seq, target))
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        seq, target = self.data[idx]
        return torch.tensor(seq), torch.tensor(target)


# =============================================================================
# Training Functions
# =============================================================================

def train_one_step(
    model: TinyRecursiveModel,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    max_steps: int,
    halt_exploration: float = 0.1,
) -> tuple:
    """
    One training step with deep supervision.
    
    Returns (total_loss, metrics_dict)
    """
    y, z = None, None
    total_loss = torch.tensor(0.0, device=inputs.device)
    step_losses = []
    
    for step in range(max_steps):
        out = model(inputs, y, z, labels)
        y, z = out['y'], out['z']
        step_loss = out['loss'].mean()
        total_loss = total_loss + step_loss
        step_losses.append(step_loss.item())
        
        # ACT: Maybe halt early
        if step > 0:
            should_halt = out['halt_prob'] > 0.5
            explore = torch.rand_like(out['halt_prob']) < halt_exploration
            should_halt = should_halt & ~explore
            if should_halt.all():
                break
    
    num_steps = step + 1
    
    # Compute accuracy on final output
    preds = out['logits'].argmax(-1)
    correct = (preds == labels).all(-1).float().mean()
    token_acc = (preds == labels).float().mean()
    
    metrics = {
        'loss': (total_loss / num_steps).item(),
        'accuracy': correct.item(),
        'token_acc': token_acc.item(),
        'steps': num_steps,
    }
    
    return total_loss, metrics


@torch.no_grad()
def evaluate(
    model: TinyRecursiveModel,
    dataloader: DataLoader,
    max_steps: int,
    device: str,
) -> dict:
    """Evaluate the model on a dataset."""
    model.eval()
    
    total_correct = 0
    total_token_correct = 0
    total_tokens = 0
    total_samples = 0
    total_steps = 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Use predict() for proper inference with halting
        preds, steps = model.predict(inputs, max_steps=max_steps, halt_threshold=0.5)
        
        # Sequence-level accuracy (entire sequence correct)
        correct = (preds == labels).all(-1).sum().item()
        total_correct += correct
        
        # Token-level accuracy
        token_correct = (preds == labels).sum().item()
        total_token_correct += token_correct
        total_tokens += labels.numel()
        
        total_samples += inputs.size(0)
        total_steps += steps.float().mean().item() * inputs.size(0)
    
    return {
        'accuracy': total_correct / total_samples,
        'token_acc': total_token_correct / total_tokens,
        'avg_steps': total_steps / total_samples,
    }


# =============================================================================
# Main Training Loop
# =============================================================================

def train(config: TrainConfig):
    """Main training function."""
    
    print("=" * 60)
    print("TRM Training - Sort Task")
    print("=" * 60)
    print(f"\nConfig:")
    print(f"  seq_len={config.seq_len}, tokens=0-{config.num_tokens-1}")
    print(f"  dim={config.dim}, depth={config.network_depth}")
    print(f"  n={config.num_latent_refinements}, T={config.num_refinement_blocks}")
    print(f"  use_mlp_mixer={config.use_mlp_mixer}")
    print(f"  device={config.device}")
    
    # Set seed
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    
    # Create datasets
    print("\nCreating datasets...")
    train_ds = SortDataset(config.train_size, config.seq_len, config.num_tokens, seed=config.seed)
    val_ds = SortDataset(config.val_size, config.seq_len, config.num_tokens, seed=config.seed + 1)
    
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size)
    
    # Create model
    print("Creating model...")
    if config.use_mlp_mixer:
        network = MLPMixer(
            dim=config.dim,
            seq_len=config.seq_len,
            depth=config.network_depth,
        )
    else:
        network = None  # Use default transformer
    
    model = TinyRecursiveModel(
        dim=config.dim,
        num_tokens=config.num_tokens,
        network=network,
        num_latent_refinements=config.num_latent_refinements,
        num_refinement_blocks=config.num_refinement_blocks,
    ).to(config.device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs * len(train_loader),
    )
    
    # Training loop
    print("\nStarting training...")
    print("-" * 60)
    
    best_val_acc = 0.0
    global_step = 0
    
    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_steps = 0
        num_batches = 0
        
        epoch_start = time.time()
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(config.device), labels.to(config.device)
            
            optimizer.zero_grad()
            loss, metrics = train_one_step(
                model, inputs, labels,
                max_steps=config.max_supervision_steps,
            )
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            epoch_loss += metrics['loss']
            epoch_acc += metrics['accuracy']
            epoch_steps += metrics['steps']
            num_batches += 1
            global_step += 1
            
            # Log
            if global_step % config.log_every == 0:
                print(f"  [Step {global_step:5d}] loss={metrics['loss']:.4f}, "
                      f"acc={metrics['accuracy']:.2%}, steps={metrics['steps']}")
        
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / num_batches
        avg_acc = epoch_acc / num_batches
        avg_steps = epoch_steps / num_batches
        
        # Evaluate
        if epoch % config.eval_every == 0:
            val_metrics = evaluate(model, val_loader, config.max_supervision_steps, config.device)
            
            is_best = val_metrics['accuracy'] > best_val_acc
            if is_best:
                best_val_acc = val_metrics['accuracy']
            
            print(f"\nEpoch {epoch:3d}/{config.epochs} ({epoch_time:.1f}s)")
            print(f"  Train: loss={avg_loss:.4f}, acc={avg_acc:.2%}, steps={avg_steps:.1f}")
            print(f"  Val:   acc={val_metrics['accuracy']:.2%}, "
                  f"token_acc={val_metrics['token_acc']:.2%}, "
                  f"steps={val_metrics['avg_steps']:.1f}"
                  f"{' (best!)' if is_best else ''}")
            print("-" * 60)
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    
    final_metrics = evaluate(model, val_loader, config.max_supervision_steps, config.device)
    print(f"\nValidation Results:")
    print(f"  Sequence Accuracy: {final_metrics['accuracy']:.2%}")
    print(f"  Token Accuracy:    {final_metrics['token_acc']:.2%}")
    print(f"  Average Steps:     {final_metrics['avg_steps']:.1f}")
    
    # Show some examples
    print("\n" + "=" * 60)
    print("Examples")
    print("=" * 60)
    
    model.eval()
    test_inputs, test_labels = next(iter(val_loader))
    test_inputs, test_labels = test_inputs[:5].to(config.device), test_labels[:5].to(config.device)
    
    preds, steps = model.predict(test_inputs, max_steps=config.max_supervision_steps)
    
    for i in range(5):
        inp = test_inputs[i].tolist()
        tgt = test_labels[i].tolist()
        pred = preds[i].tolist()
        correct = "✓" if pred == tgt else "✗"
        print(f"\n  Input:  {inp}")
        print(f"  Target: {tgt}")
        print(f"  Pred:   {pred} {correct} (steps={steps[i].item()})")
    
    return model, final_metrics


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    config = TrainConfig(
        # Task
        seq_len=8,
        num_tokens=10,
        
        # Data
        train_size=5000,
        val_size=500,
        
        # Model - simpler to debug
        dim=64,
        network_depth=2,
        num_latent_refinements=2,   # Reduced
        num_refinement_blocks=1,    # No grad truncation initially  
        use_mlp_mixer=True,
        
        # Training
        batch_size=64,
        epochs=50,
        lr=1e-3,
        max_supervision_steps=4,
        
        # Misc
        log_every=50,
        eval_every=5,
    )
    
    model, metrics = train(config)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Final Accuracy: {metrics['accuracy']:.2%}")
    print("=" * 60)
    
    # Save model
    save_path = "trm_sort_model.pt"
    torch.save(model.state_dict(), save_path)
    print(f"\nSaved trained model to: {save_path}")
    
    # Export to ONNX
    print("\nExporting trained model to ONNX...")
    from export_onnx import export_to_onnx
    
    export_to_onnx(
        dim=config.dim,
        seq_len=config.seq_len,
        num_tokens=config.num_tokens,
        network_depth=config.network_depth,
        num_latent_refinements=config.num_latent_refinements,
        num_refinement_blocks=config.num_refinement_blocks,
        num_supervision_steps=config.max_supervision_steps,
        output_dir="./exports",
        trained_model=model,  # Pass the trained model!
    )
