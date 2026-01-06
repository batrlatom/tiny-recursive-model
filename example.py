"""
TRM Example - Demonstrates the concise implementation
"""

import torch
from trm import TinyRecursiveModel, MLPMixer, SimpleNetwork, train_step


def main():
    print("=" * 60)
    print("Tiny Recursive Model (TRM) - Concise Implementation Demo")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # =========================================================================
    # Example 1: Default configuration (2-layer transformer)
    # =========================================================================
    print("\n1. Default TRM (2-layer transformer)")
    
    model = TinyRecursiveModel(
        dim=256,
        num_tokens=11,
        # network=None uses default SimpleNetwork (2-layer transformer)
        num_latent_refinements=4,  # n in paper
        num_refinement_blocks=2,   # T in paper
    ).to(device)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {params:,} ({params/1e6:.2f}M)")
    
    # Forward pass
    x = torch.randint(0, 11, (2, 32), device=device)
    out = model(x)
    print(f"   Input:  {x.shape}")
    print(f"   Output: {out['logits'].shape}")
    print(f"   Halt prob: {out['halt_prob'].tolist()}")
    
    # =========================================================================
    # Example 2: Pluggable network - MLP-Mixer for Sudoku
    # =========================================================================
    print("\n2. Custom network: MLP-Mixer for Sudoku (81 tokens)")
    
    # For Sudoku: 9x9 = 81 fixed tokens, MLP-Mixer works great!
    mixer = MLPMixer(dim=256, seq_len=81, depth=2)
    
    model_sudoku = TinyRecursiveModel(
        dim=256,
        num_tokens=10,  # 0-9 for Sudoku
        network=mixer,  # Plug in the MLP-Mixer!
        num_latent_refinements=6,
        num_refinement_blocks=3,
    ).to(device)
    
    params = sum(p.numel() for p in model_sudoku.parameters())
    print(f"   Parameters: {params:,}")
    
    x_sudoku = torch.randint(0, 10, (4, 81), device=device)
    out = model_sudoku(x_sudoku)
    print(f"   Input:  {x_sudoku.shape}")
    print(f"   Output: {out['logits'].shape}")
    
    # =========================================================================
    # Example 3: Training with deep supervision
    # =========================================================================
    print("\n3. Training step with deep supervision")
    
    model.train()
    x = torch.randint(0, 11, (8, 32), device=device)
    labels = torch.roll(x, -1, dims=1)  # Dummy task: predict next token
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    for step in range(3):
        optimizer.zero_grad()
        loss, metrics = train_step(model, x, labels, max_steps=8)
        loss.backward()
        optimizer.step()
        print(f"   Step {step+1}: loss={metrics['loss']:.4f}, "
              f"acc={metrics['accuracy']:.4f}, steps={metrics['steps']}")
    
    # =========================================================================
    # Example 4: Inference with dynamic halting
    # =========================================================================
    print("\n4. Inference with dynamic halting")
    
    model.eval()
    x = torch.randint(0, 11, (4, 32), device=device)
    
    preds, steps = model.predict(x, max_steps=16, halt_threshold=0.5)
    print(f"   Predictions: {preds.shape}")
    print(f"   Steps taken: {steps.tolist()}")
    
    # =========================================================================
    # Example 5: Register tokens (scratch space)
    # =========================================================================
    print("\n5. With register tokens (scratch space)")
    
    model_regs = TinyRecursiveModel(
        dim=256,
        num_tokens=11,
        num_registers=4,  # 4 extra "thinking" tokens
    ).to(device)
    
    x = torch.randint(0, 11, (2, 32), device=device)
    out = model_regs(x)
    print(f"   Input (32 tokens) + 4 registers")
    print(f"   Output: {out['logits'].shape} (only input positions)")
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
