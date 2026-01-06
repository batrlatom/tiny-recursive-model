"""
Tiny Recursive Model (TRM) - Concise Implementation

A minimal, well-documented implementation of TRM from:
  "Less is More: Recursive Reasoning with Tiny Networks" (arXiv:2510.04871)

Key insight: A TINY 2-layer network with deep recursion beats large models!
"""

from __future__ import annotations
from typing import Callable, Optional, Tuple
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange, Reduce


# =============================================================================
# Helpers
# =============================================================================

def exists(v): return v is not None
def default(v, d): return v if exists(v) else d


# =============================================================================
# Default Networks (pluggable - you can pass your own!)
# =============================================================================

class MLPMixer(nn.Module):
    """
    MLP-Mixer: Applies MLP on sequence dimension instead of attention.
    
    Great for small fixed-length inputs (e.g., Sudoku 9x9 = 81 tokens).
    Paper showed this beats attention on Sudoku-Extreme (87.4% vs 74.7%)!
    
    Uses post-norm (norm after residual) like the official TRM.
    """
    def __init__(self, dim: int, seq_len: int, depth: int = 2, expansion: float = 4.0):
        super().__init__()
        self.depth = depth
        self.scale = dim ** -0.5  # Scale factor for stability
        
        # Token mixing layers (MLP across sequence dimension)
        self.token_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(seq_len, int(seq_len * expansion)),
                nn.GELU(),
                nn.Linear(int(seq_len * expansion), seq_len),
            ) for _ in range(depth)
        ])
        
        # Channel mixing layers (MLP across hidden dimension)  
        self.channel_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, int(dim * expansion)),
                nn.GELU(),
                nn.Linear(int(dim * expansion), dim),
            ) for _ in range(depth)
        ])
        
        # Post-norm (applied after residual, like official TRM)
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(depth * 2)])
        
        # Initialize to small values for stability
        for mlp in self.token_mlps:
            nn.init.zeros_(mlp[-1].weight)
            nn.init.zeros_(mlp[-1].bias)
        for mlp in self.channel_mlps:
            nn.init.zeros_(mlp[-1].weight)
            nn.init.zeros_(mlp[-1].bias)
    
    def forward(self, x: Tensor) -> Tensor:
        for i in range(self.depth):
            # Token mixing with post-norm residual
            mixed = self.token_mlps[i](x.transpose(1, 2)).transpose(1, 2)
            x = self.norms[i * 2](x + mixed)
            
            # Channel mixing with post-norm residual
            x = self.norms[i * 2 + 1](x + self.channel_mlps[i](x))
        
        return x


class TransformerBlock(nn.Module):
    """
    Simple transformer block with bidirectional attention + SwiGLU FFN.
    Uses post-norm (RMSNorm after residual) as in the paper.
    """
    def __init__(self, dim: int, num_heads: int = 8, expansion: float = 4.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Attention
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        
        # FFN (SwiGLU)
        hidden = int(dim * expansion * 2 / 3)
        self.w1 = nn.Linear(dim, hidden * 2, bias=False)  # Gate + Up
        self.w2 = nn.Linear(hidden, dim, bias=False)      # Down
    
    def forward(self, x: Tensor) -> Tensor:
        # Bidirectional self-attention (non-causal for reasoning)
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, 'b n (three h d) -> three b h n d', three=3, h=self.num_heads)
        attn = F.scaled_dot_product_attention(q, k, v)  # PyTorch 2.0+
        attn = rearrange(attn, 'b h n d -> b n (h d)')
        x = self._rms_norm(x + self.proj(attn))
        
        # SwiGLU FFN
        gate, up = self.w1(x).chunk(2, dim=-1)
        x = self._rms_norm(x + self.w2(F.silu(gate) * up))
        return x
    
    def _rms_norm(self, x: Tensor, eps: float = 1e-5) -> Tensor:
        """RMSNorm - faster than LayerNorm, no mean computation."""
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps).to(x.dtype)


class SimpleNetwork(nn.Module):
    """
    Default 2-layer transformer network for TRM.
    
    Why 2 layers? The paper found that "less is more" - 
    2 layers prevents overfitting on small datasets!
    """
    def __init__(self, dim: int, num_heads: int = 8, depth: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock(dim, num_heads) for _ in range(depth)])
    
    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# =============================================================================
# TRM Core
# =============================================================================

@dataclass
class TRMCarry:
    """
    Recurrent state carried between refinement steps.
    
    y: Current solution embedding (will be decoded to output)
    z: Latent reasoning state (the "thinking" happens here)
    """
    y: Tensor
    z: Tensor


class TinyRecursiveModel(nn.Module):
    """
    Tiny Recursive Model (TRM)
    
    The key idea:
    1. Start with initial y (solution) and z (latent reasoning)
    2. Recursively refine: z learns from (y + input), then y learns from z
    3. Deep supervision: train at each refinement step, detach and repeat
    4. Simplified ACT: predict when to halt
    
    Args:
        dim: Hidden dimension (default: 512)
        num_tokens: Vocabulary size
        network: The neural network to use for refinement (default: 2-layer transformer)
                 Must accept and return tensor of shape [batch, seq_len, dim]
        num_latent_refinements: Inner loop iterations (n in paper, default: 6)
        num_refinement_blocks: Outer loop iterations with gradient truncation (T in paper, default: 3)
        num_registers: Extra learnable tokens for "scratch space" (default: 0)
    """
    
    def __init__(
        self,
        dim: int = 512,
        num_tokens: int = 11,
        network: Optional[nn.Module] = None,
        num_latent_refinements: int = 6,    # n in paper
        num_refinement_blocks: int = 3,     # T in paper
        num_registers: int = 0,
    ):
        super().__init__()
        
        # Config
        self.dim = dim
        self.num_latent_refinements = num_latent_refinements
        self.num_refinement_blocks = num_refinement_blocks
        
        # Embeddings
        self.embed = nn.Embedding(num_tokens, dim)
        
        # Initial states for y and z (learned)
        self.y_init = nn.Parameter(torch.randn(dim) * 0.01)
        self.z_init = nn.Parameter(torch.randn(dim) * 0.01)
        
        # Optional register tokens (extra "scratch" tokens)
        self.registers = nn.Parameter(torch.randn(num_registers, dim) * 0.01) if num_registers > 0 else None
        
        # The network! Default: 2-layer transformer (tiny but mighty)
        self.network = default(network, SimpleNetwork(dim))
        
        # Output head: hidden -> vocab logits
        self.to_logits = nn.Linear(dim, num_tokens, bias=False)
        
        # Halt prediction: mean pool -> scalar logit
        # Predicts "should I stop refining?" (simplified ACT)
        self.to_halt = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),  # Pool all positions
            nn.Linear(dim, 1, bias=False),
            Rearrange('b 1 -> b'),
        )
        nn.init.zeros_(self.to_halt[1].weight)  # Start with "don't halt"
    
    # -------------------------------------------------------------------------
    # Core recursion logic
    # -------------------------------------------------------------------------
    
    def _expand_initial_state(self, init: Tensor, batch: int, seq_len: int) -> Tensor:
        """Expand scalar initial state to full sequence."""
        return repeat(init, 'd -> b n d', b=batch, n=seq_len)
    
    def _add_registers(self, x: Tensor) -> Tuple[Tensor, list]:
        """Prepend register tokens if configured."""
        if self.registers is None:
            return x, []
        regs = repeat(self.registers, 'r d -> b r d', b=x.shape[0])
        packed, ps = pack([regs, x], 'b * d')
        return packed, ps
    
    def _remove_registers(self, x: Tensor, ps: list) -> Tensor:
        """Remove register tokens to get original sequence."""
        if not ps:
            return x
        _, x = unpack(x, ps, 'b * d')
        return x
    
    def _refine_once(self, x: Tensor, y: Tensor, z: Tensor) -> Tuple[Tensor, Tensor]:
        """
        One complete refinement block:
        1. Update z (latent) n times, injecting (y + x) as context
        2. Update y (solution) once, injecting z as context
        
        This is the heart of TRM's recursive reasoning!
        
        Key insight from official implementation:
        - We ADD the injection to z/y, then pass through the network
        - The network should output a DELTA (residual)
        - This prevents explosion because we're not compounding additions
        """
        # n iterations: Refine latent z given current solution y and input x
        injection = y + x  # The context for z updates
        for _ in range(self.num_latent_refinements):
            # z receives context injection, network refines it
            z = self.network(z + injection)
        
        # 1 iteration: Update solution y, using z as context
        y = self.network(y + z)
        
        return y, z
    
    def _deep_refinement(self, x: Tensor, y: Tensor, z: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Apply T refinement blocks with gradient truncation (TBPTL).
        
        Only the LAST block receives gradients - this prevents gradient 
        instability from deep recursion while still allowing the model
        to reason over many iterations.
        """
        for step in range(1, self.num_refinement_blocks + 1):
            is_last = (step == self.num_refinement_blocks)
            
            # First T-1 blocks: forward only, no gradients
            # Last block: full gradients for training
            ctx = nullcontext if is_last else torch.no_grad
            with ctx():
                y, z = self._refine_once(x, y, z)
        
        return y, z
    
    # -------------------------------------------------------------------------
    # Forward pass
    # -------------------------------------------------------------------------
    
    def forward(
        self,
        seq: Tensor,
        y: Optional[Tensor] = None,
        z: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> dict:
        """
        Forward pass for one deep supervision step.
        
        Args:
            seq: Input token IDs [batch, seq_len]
            y: Previous solution state (None = use initial)
            z: Previous latent state (None = use initial)
            labels: Target tokens for loss computation
        
        Returns:
            Dictionary with: y, z (for next step), logits, halt_prob, loss (if labels)
        """
        batch, seq_len = seq.shape
        
        # Embed input tokens
        x = self.embed(seq)
        
        # Add registers if configured
        x, ps = self._add_registers(x)
        full_seq_len = x.shape[1]
        
        # Initialize y and z if not provided (first supervision step)
        if y is None:
            y = self._expand_initial_state(self.y_init, batch, full_seq_len)
        if z is None:
            z = self._expand_initial_state(self.z_init, batch, full_seq_len)
        
        # Deep refinement with gradient truncation
        y, z = self._deep_refinement(x, y, z)
        
        # Remove registers for output
        y_out = self._remove_registers(y, ps)
        
        # Predictions
        logits = self.to_logits(y_out)
        halt_logits = self.to_halt(y)  # Uses full y including registers
        halt_prob = halt_logits.sigmoid()
        
        # Detach for next supervision step (this is key to deep supervision!)
        y_next, z_next = y.detach(), z.detach()
        
        # Build output
        out = {
            'y': y_next,
            'z': z_next,
            'logits': logits,
            'halt_logits': halt_logits,
            'halt_prob': halt_prob,
        }
        
        # Compute loss if labels provided
        if labels is not None:
            # Cross-entropy loss
            ce_loss = F.cross_entropy(
                rearrange(logits, 'b n c -> b c n'),
                labels,
                reduction='none'
            )
            ce_loss = reduce(ce_loss, 'b n -> b', 'mean')
            
            # Halt loss: predict if answer is correct
            is_correct = (logits.argmax(-1) == labels).all(-1).float()
            halt_loss = F.binary_cross_entropy_with_logits(halt_logits, is_correct, reduction='none')
            
            out['loss'] = ce_loss + halt_loss
            out['ce_loss'] = ce_loss
            out['halt_loss'] = halt_loss
            out['accuracy'] = is_correct.mean()
        
        return out
    
    # -------------------------------------------------------------------------
    # Inference with dynamic halting
    # -------------------------------------------------------------------------
    
    @torch.no_grad()
    def predict(
        self,
        seq: Tensor,
        max_steps: int = 16,
        halt_threshold: float = 0.5,
    ) -> Tuple[Tensor, Tensor]:
        """
        Inference with dynamic halting.
        
        Args:
            seq: Input tokens [batch, seq_len]
            max_steps: Maximum refinement steps
            halt_threshold: Probability threshold for early stopping
        
        Returns:
            predictions: Predicted tokens [batch, seq_len]
            steps_taken: Number of steps each sample took [batch]
        """
        batch = seq.shape[0]
        device = seq.device
        
        y, z = None, None  # Start fresh
        steps = torch.zeros(batch, dtype=torch.long, device=device)
        
        for step in range(1, max_steps + 1):
            out = self(seq, y, z)
            y, z = out['y'], out['z']
            
            # Update step counts for samples still running
            still_running = out['halt_prob'] < halt_threshold
            steps = torch.where(still_running, torch.tensor(step, device=device), steps)
            
            # Early exit if all samples halted
            if not still_running.any() and step > 1:
                break
        
        # Final step count for samples that never halted
        steps = torch.where(steps == 0, torch.tensor(max_steps, device=device), steps)
        
        return out['logits'].argmax(-1), steps


# =============================================================================
# Training helper
# =============================================================================

def train_step(
    model: TinyRecursiveModel,
    seq: Tensor,
    labels: Tensor,
    max_steps: int = 16,
    halt_exploration: float = 0.1,
) -> Tuple[Tensor, dict]:
    """
    One training step with deep supervision.
    
    Runs multiple refinement steps, computing loss at each and stopping
    when the model predicts it should halt (with exploration).
    
    Returns:
        total_loss: Sum of losses across all steps
        metrics: Dictionary with training metrics
    """
    y, z = None, None
    total_loss = 0.0
    num_steps = 0
    
    for step in range(max_steps):
        out = model(seq, y, z, labels)
        y, z = out['y'], out['z']
        total_loss = total_loss + out['loss'].sum()
        num_steps += 1
        
        # ACT: Maybe halt early (with exploration during training)
        if step > 0:
            should_halt = out['halt_prob'] > 0.5
            # Exploration: sometimes continue even if model wants to halt
            explore = torch.rand_like(out['halt_prob']) < halt_exploration
            should_halt = should_halt & ~explore
            if should_halt.all():
                break
    
    metrics = {
        'loss': total_loss.item() / num_steps,
        'accuracy': out['accuracy'].item(),
        'steps': num_steps,
    }
    
    return total_loss, metrics
