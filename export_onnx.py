"""
TRM ONNX Export for Hailo Deployment

This script exports the TRM model to ONNX format for edge deployment.

IMPORTANT NOTES FOR HAILO:
1. TRM uses recursion (loops) which is NOT directly supported in static ONNX
2. We export a "flattened" version that unrolls the recursion to a fixed depth
3. This trades flexibility for hardware compatibility

The export creates:
  - trm_single_step.onnx: One refinement step (for iterative execution on host)
  - trm_full.onnx: Full model with unrolled recursion (for standalone inference)
"""

import os
import torch
import torch.nn as nn
from typing import Tuple
from dataclasses import dataclass

# We'll create a simpler export-friendly version of TRM


class MLPMixerBlockONNX(nn.Module):
    """Export-friendly MLPMixer block."""
    
    def __init__(self, dim: int, seq_len: int, expansion: float = 4.0):
        super().__init__()
        # Token mixing
        self.token_mlp = nn.Sequential(
            nn.Linear(seq_len, int(seq_len * expansion)),
            nn.GELU(),
            nn.Linear(int(seq_len * expansion), seq_len),
        )
        # Channel mixing
        self.channel_mlp = nn.Sequential(
            nn.Linear(dim, int(dim * expansion)),
            nn.GELU(),
            nn.Linear(int(dim * expansion), dim),
        )
        # Post-norm (applied after residual)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Token mixing with post-norm
        # x is [batch, seq_len, dim]
        # Transpose for token mixing: [batch, dim, seq_len]
        mixed = self.token_mlp(x.transpose(1, 2)).transpose(1, 2)
        x = self.norm1(x + mixed)
        
        # Channel mixing with post-norm
        mixed = self.channel_mlp(x)
        x = self.norm2(x + mixed)
        return x


class TRMNetworkONNX(nn.Module):
    """Export-friendly network (n layers)."""
    
    def __init__(self, dim: int, seq_len: int, depth: int = 2, expansion: float = 4.0, use_mlp_mixer: bool = True):
        super().__init__()
        # Use MLPMixer blocks by default since that's what we trained
        self.layers = nn.ModuleList([
            MLPMixerBlockONNX(dim, seq_len, expansion) for _ in range(depth)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x



class TRMSingleStepONNX(nn.Module):
    """
    Single refinement step - can be called iteratively by host.
    
    Takes (y, z, x_embedded) and returns updated (y, z).
    This allows the recursion to happen on the host CPU while
    the heavy compute runs on Hailo.
    """
    
    def __init__(
        self,
        dim: int = 64,
        seq_len: int = 8,
        num_tokens: int = 10,
        network_depth: int = 2,
        num_latent_refinements: int = 2,
    ):
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.num_latent_refinements = num_latent_refinements
        
        # Embedding
        self.embed = nn.Embedding(num_tokens, dim)
        
        # Network
        self.network = TRMNetworkONNX(dim, seq_len, network_depth)
        
        # Output head
        self.to_logits = nn.Linear(dim, num_tokens, bias=False)
        
        # Halt prediction (mean pool -> scalar)
        self.to_halt = nn.Linear(dim, 1, bias=False)
    
    def forward(
        self, 
        input_ids: torch.Tensor,  # [batch, seq_len]
        y: torch.Tensor,          # [batch, seq_len, dim]
        z: torch.Tensor,          # [batch, seq_len, dim]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        One refinement step.
        
        Returns:
            y_new: Updated solution [batch, seq_len, dim]
            z_new: Updated latent [batch, seq_len, dim]
            logits: Output logits [batch, seq_len, num_tokens]
            halt_prob: Halt probability [batch]
        """
        # Embed inputs
        x = self.embed(input_ids)
        
        # Latent refinement (unrolled loop)
        injection = y + x
        for _ in range(self.num_latent_refinements):
            z = self.network(z + injection)
        
        # Solution update
        y = self.network(y + z)
        
        # Output
        logits = self.to_logits(y)
        halt_logits = self.to_halt(y.mean(dim=1))  # Mean pool
        halt_prob = torch.sigmoid(halt_logits)
        
        return y, z, logits, halt_prob.squeeze(-1)


class TRMFullONNX(nn.Module):
    """
    Full TRM with unrolled recursion for standalone inference.
    
    All loops are fully unrolled for ONNX compatibility.
    Fixed number of supervision steps.
    """
    
    def __init__(
        self,
        dim: int = 64,
        seq_len: int = 8,
        num_tokens: int = 10,
        network_depth: int = 2,
        num_latent_refinements: int = 2,
        num_refinement_blocks: int = 1,
        num_supervision_steps: int = 4,
    ):
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.num_latent_refinements = num_latent_refinements
        self.num_refinement_blocks = num_refinement_blocks
        self.num_supervision_steps = num_supervision_steps
        
        # Embedding
        self.embed = nn.Embedding(num_tokens, dim)
        
        # Initial states (fixed, not learned for export simplicity)
        self.register_buffer('y_init', torch.zeros(1, seq_len, dim))
        self.register_buffer('z_init', torch.zeros(1, seq_len, dim))
        
        # Network
        self.network = TRMNetworkONNX(dim, seq_len, network_depth)
        
        # Output head
        self.to_logits = nn.Linear(dim, num_tokens, bias=False)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass with all loops unrolled.
        
        Args:
            input_ids: [batch, seq_len]
            
        Returns:
            logits: [batch, seq_len, num_tokens]
        """
        batch = input_ids.shape[0]
        
        # Embed inputs
        x = self.embed(input_ids)
        
        # Initialize y and z
        y = self.y_init.expand(batch, -1, -1).clone()
        z = self.z_init.expand(batch, -1, -1).clone()
        
        # Unrolled supervision steps
        for _ in range(self.num_supervision_steps):
            # Unrolled refinement blocks
            for _ in range(self.num_refinement_blocks):
                # Unrolled latent refinement
                injection = y + x
                for _ in range(self.num_latent_refinements):
                    z = self.network(z + injection)
                # Solution update
                y = self.network(y + z)
            
            # Detach for next step (no-op in inference, but matches training)
            y = y.detach()
            z = z.detach()
        
        return self.to_logits(y)


def copy_weights_from_trained(trained_model, onnx_model):
    """Copy weights from a trained TRM to the ONNX-compatible version."""
    print("  Copying weights from trained model...")
    with torch.no_grad():
        # Embedding
        if hasattr(trained_model, 'embed'):
            onnx_model.embed.weight.copy_(trained_model.embed.weight)
        
        # Output head
        if hasattr(trained_model, 'to_logits'):
            onnx_model.to_logits.weight.copy_(trained_model.to_logits.weight)
            
        # Halt prediction
        if hasattr(trained_model, 'to_halt') and hasattr(onnx_model, 'to_halt'):
            # The trained model uses [Reduce, Linear, Rearrange]
            # ONNX model uses [Linear] (assuming input is pre-pooled)
            # The weights should be directly copyable from the Linear layer (index 1)
            onnx_model.to_halt.weight.copy_(trained_model.to_halt[1].weight)
        
        # Network weights (MLP-Mixer)
        if hasattr(trained_model, 'network') and hasattr(onnx_model, 'network'):
            trained_net = trained_model.network
            # Check if it's the MLPMixer we expect
            if hasattr(trained_net, 'token_mlps') and hasattr(trained_net, 'channel_mlps'):
                print("  backend: MLPMixer")
                for i, layer in enumerate(onnx_model.network.layers):
                    # Token MLP
                    if i < len(trained_net.token_mlps):
                        layer.token_mlp.load_state_dict(trained_net.token_mlps[i].state_dict())
                    
                    # Channel MLP
                    if i < len(trained_net.channel_mlps):
                        layer.channel_mlp.load_state_dict(trained_net.channel_mlps[i].state_dict())
                    
                    # Norms
                    # Trained has flat list: [token_norm_0, channel_norm_0, token_norm_1, ...]
                    if (2*i + 1) < len(trained_net.norms):
                        layer.norm1.load_state_dict(trained_net.norms[2*i].state_dict())
                        layer.norm2.load_state_dict(trained_net.norms[2*i+1].state_dict())
                print("  ✓ Network weights copied successfully")
            else:
                 print("  [!] Unknown network architecture, weights not copied")


def export_to_onnx(
    dim: int = 64,
    seq_len: int = 8,
    num_tokens: int = 10,
    network_depth: int = 2,
    num_latent_refinements: int = 2,
    num_refinement_blocks: int = 1,
    num_supervision_steps: int = 4,
    output_dir: str = ".",
    trained_model=None,
):
    """Export TRM to ONNX format."""
    
    print("=" * 60)
    print("TRM ONNX Export")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # =========================================================================
    # Export single-step model (for iterative host execution)
    # =========================================================================
    print("\n1. Exporting single-step model...")
    
    single_step = TRMSingleStepONNX(
        dim=dim,
        seq_len=seq_len,
        num_tokens=num_tokens,
        network_depth=network_depth,
        num_latent_refinements=num_latent_refinements,
    )
    single_step.eval()
    
    if trained_model:
        copy_weights_from_trained(trained_model, single_step)
    
    # Dummy inputs
    batch = 1
    input_ids = torch.randint(0, num_tokens, (batch, seq_len))
    y = torch.randn(batch, seq_len, dim)
    z = torch.randn(batch, seq_len, dim)
    
    single_step_path = os.path.join(output_dir, "trm_single_step.onnx")
    
    torch.onnx.export(
        single_step,
        (input_ids, y, z),
        single_step_path,
        input_names=["input_ids", "y_in", "z_in"],
        output_names=["y_out", "z_out", "logits", "halt_prob"],
        dynamic_axes={
            "input_ids": {0: "batch"},
            "y_in": {0: "batch"},
            "z_in": {0: "batch"},
            "y_out": {0: "batch"},
            "z_out": {0: "batch"},
            "logits": {0: "batch"},
            "halt_prob": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    
    print(f"   Saved: {single_step_path}")
    
    # =========================================================================
    # Export full model (with unrolled recursion)
    # =========================================================================
    print("\n2. Exporting full model (unrolled recursion)...")
    
    full_model = TRMFullONNX(
        dim=dim,
        seq_len=seq_len,
        num_tokens=num_tokens,
        network_depth=network_depth,
        num_latent_refinements=num_latent_refinements,
        num_refinement_blocks=num_refinement_blocks,
        num_supervision_steps=num_supervision_steps,
    )
    full_model.eval()
    
    if trained_model:
        copy_weights_from_trained(trained_model, full_model)
    
    input_ids = torch.randint(0, num_tokens, (batch, seq_len))
    
    full_path = os.path.join(output_dir, "trm_full.onnx")
    
    torch.onnx.export(
        full_model,
        (input_ids,),
        full_path,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    
    print(f"   Saved: {full_path}")
    
    # =========================================================================
    # Verify with ONNX
    # =========================================================================
    print("\n3. Verifying ONNX models...")
    
    try:
        import onnx
        
        model = onnx.load(single_step_path)
        onnx.checker.check_model(model)
        print(f"   ✓ {single_step_path} is valid")
        
        model = onnx.load(full_path)
        onnx.checker.check_model(model)
        print(f"   ✓ {full_path} is valid")
        
    except ImportError:
        print("   [!] onnx package not installed, skipping verification")
    
    # =========================================================================
    # Simplify for Hailo (recommended)
    # =========================================================================
    print("\n4. Simplifying for Hailo...")
    
    try:
        import onnxsim
        
        for path in [single_step_path, full_path]:
            model = onnx.load(path)
            simplified, check = onnxsim.simplify(model)
            if check:
                simplified_path = path.replace(".onnx", "_simplified.onnx")
                onnx.save(simplified, simplified_path)
                print(f"   ✓ Saved: {simplified_path}")
            else:
                print(f"   [!] Could not simplify {path}")
                
    except ImportError:
        print("   [!] onnx-simplifier not installed")
        print("   Install with: pip install onnx-simplifier")
    
    print("\n" + "=" * 60)
    print("Export Complete!")
    print("=" * 60)
    
    return single_step_path, full_path


def create_hailo_compilation_script(onnx_path: str, output_dir: str = "."):
    """Generate a Hailo compilation script."""
    
    script = f'''#!/usr/bin/env python3
"""
Hailo Compilation Script for TRM

Requirements:
- Hailo AI Software Suite (hailo_sdk)
- Download from: https://hailo.ai/developer-zone/

Run this script after installing the Hailo SDK.
"""

import os
import numpy as np

# Check for Hailo SDK
try:
    from hailo_sdk_client import ClientRunner
except ImportError:
    print("Hailo SDK not found!")
    print("Please install from: https://hailo.ai/developer-zone/")
    exit(1)


def compile_for_hailo(onnx_path: str, calibration_data=None):
    """
    Compile ONNX model for Hailo accelerator.
    
    Steps:
    1. Parse (translate) the ONNX model
    2. Optimize (quantize to int8)
    3. Compile to HEF format
    """
    
    print("=" * 60)
    print("Hailo Compilation for TRM")
    print("=" * 60)
    
    # Model name (without extension)
    model_name = os.path.splitext(os.path.basename(onnx_path))[0]
    
    # Create runner
    runner = ClientRunner(hw_arch="hailo8")  # or "hailo8l" for lite
    
    # =========================================================================
    # Step 1: Parse (Translate) ONNX to Hailo format
    # =========================================================================
    print("\\n1. Parsing ONNX model...")
    
    hn, npz = runner.translate_onnx_model(
        onnx_path,
        model_name,
        start_node_names=None,  # Start from input
        end_node_names=None,    # End at output
        net_input_shapes={{
            "input_ids": [1, 8],  # Fixed batch=1, seq_len=8
        }},
    )
    
    print(f"   Parsed model saved to: {{model_name}}.hn")
    
    # =========================================================================
    # Step 2: Optimize (Quantize)
    # =========================================================================
    print("\\n2. Optimizing (quantizing to int8)...")
    
    # Calibration data - random data if not provided
    if calibration_data is None:
        print("   [!] Using random calibration data")
        print("   For best results, use real input data!")
        calibration_data = {{
            "input_ids": np.random.randint(0, 10, (100, 8)).astype(np.int64)
        }}
    
    runner.optimize(
        hn,
        calib_data=calibration_data,
    )
    
    print("   Optimization complete")
    
    # =========================================================================
    # Step 3: Compile to HEF
    # =========================================================================
    print("\\n3. Compiling to HEF format...")
    
    hef_path = f"{{model_name}}.hef"
    runner.compile(hef_path)
    
    print(f"   ✓ Compiled: {{hef_path}}")
    
    print("\\n" + "=" * 60)
    print("Compilation Complete!")
    print(f"Deploy {{hef_path}} to your Hailo device")
    print("=" * 60)
    
    return hef_path


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python compile_hailo.py <onnx_model.onnx>")
        exit(1)
    
    compile_for_hailo(sys.argv[1])
'''
    
    script_path = os.path.join(output_dir, "compile_hailo.py")
    with open(script_path, "w") as f:
        f.write(script)
    
    print(f"Created Hailo compilation script: {script_path}")
    return script_path


if __name__ == "__main__":
    # Export with default settings matching our trained model
    export_to_onnx(
        dim=64,
        seq_len=8,
        num_tokens=10,
        network_depth=2,
        num_latent_refinements=2,
        num_refinement_blocks=1,
        num_supervision_steps=4,
        output_dir="./exports",
    )
    
    # Create Hailo compilation script
    create_hailo_compilation_script(
        "./exports/trm_full_simplified.onnx",
        output_dir="./exports",
    )
