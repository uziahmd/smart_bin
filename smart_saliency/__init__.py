"""
Smart Saliency: Activation-aware binarization saliency detection.

This module implements error-aware + activation-aware saliency scoring
for intelligent weight selection in partially-binarized LLMs.

Key formula:
    S_ij = (ΔW_ij)² × a_j
    
where:
    ΔW_ij = W_ij - α_j × sign(W_ij)   (binarization error)
    α_j = mean(|W_:,j|)                (column-wise scaling factor)  
    a_j = E[x_j²]                      (activation energy from calibration)
"""

from .activation_collector import ActivationCollector
from .saliency_scorer import compute_layer_saliency, compute_all_saliency
from .budget_allocator import allocate_budgets
from .mask_generator import generate_masks, generate_all_masks

__all__ = [
    "ActivationCollector",
    "compute_layer_saliency",
    "compute_all_saliency", 
    "allocate_budgets",
    "generate_masks",
    "generate_all_masks",
]
