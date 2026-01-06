"""
Tiny Recursive Model (TRM) - Concise Implementation

From: "Less is More: Recursive Reasoning with Tiny Networks" (arXiv:2510.04871)
"""

from .trm import (
    TinyRecursiveModel,
    TRMCarry,
    SimpleNetwork,
    TransformerBlock,
    MLPMixer,
    train_step,
)

__version__ = "0.2.0"
__all__ = [
    "TinyRecursiveModel",
    "TRMCarry", 
    "SimpleNetwork",
    "TransformerBlock",
    "MLPMixer",
    "train_step",
]
