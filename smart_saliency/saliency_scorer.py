"""
Saliency Scorer: Compute activation-aware saliency scores per weight.

For each weight W_ij, the saliency score is:
    S_ij = (ΔW_ij)² × a_j

where:
    ΔW_ij = W_ij - α_j × sign(W_ij)   (binarization error)
    α_j = mean(|W_:,j|)                (column-wise scaling factor)
    a_j = E[x_j²]                      (activation energy)
"""

import torch
from typing import Dict, Tuple, Optional
import torch.nn as nn


def compute_layer_saliency(
    W: torch.Tensor,
    a: torch.Tensor,
    return_components: bool = False
) -> Tuple[torch.Tensor, float, torch.Tensor]:
    """
    Compute activation-aware saliency for a single layer.
    
    Args:
        W: Weight matrix of shape [d_out, d_in]
        a: Activation energy of shape [d_in]
        return_components: Whether to return intermediate values
        
    Returns:
        S: Saliency matrix of shape [d_out, d_in]
        L: Scalar layer need score (sum of S)
        alpha: Column-wise scaling factors [d_in]
    """
    assert W.dim() == 2, f"Expected 2D weight, got {W.dim()}D"
    assert a.dim() == 1, f"Expected 1D activation energy, got {a.dim()}D"
    assert W.shape[1] == a.shape[0], f"Dimension mismatch: W={W.shape}, a={a.shape}"
    
    # Ensure float for computation
    W = W.float()
    a = a.float()
    
    # Column-wise scaling factor: α_j = mean(|W_:,j|)
    alpha = W.abs().mean(dim=0)  # [d_in]
    
    # Binarized weights: α_j × sign(W_:,j)
    W_bin = alpha.unsqueeze(0) * torch.sign(W)  # [d_out, d_in]
    
    # Binarization error: ΔW = W - W_bin
    delta_W = W - W_bin  # [d_out, d_in]
    
    # Saliency matrix: S_ij = (ΔW_ij)² × a_j
    S = (delta_W ** 2) * a.unsqueeze(0)  # [d_out, d_in]
    
    # Layer need score: L = sum(S)
    L = S.sum().item()
    
    return S, L, alpha


def compute_all_saliency(
    model: nn.Module,
    activation_energies: Dict[str, torch.Tensor],
    skip_patterns: Optional[list] = None
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float], Dict[str, torch.Tensor]]:
    """
    Compute saliency matrices for all layers in the model.
    
    Args:
        model: The model with linear layers
        activation_energies: Dict mapping layer names to activation energies
        skip_patterns: Patterns to skip (e.g., ['embed', 'lm_head'])
        
    Returns:
        S_dict: Dict mapping layer names to saliency matrices
        L_dict: Dict mapping layer names to layer need scores
        alpha_dict: Dict mapping layer names to scaling factors
    """
    skip_patterns = skip_patterns or ['embed', 'lm_head', 'head', 'norm']
    
    S_dict = {}
    L_dict = {}
    alpha_dict = {}
    
    for name, module in model.named_modules():
        # Skip non-linear layers
        if not isinstance(module, nn.Linear):
            continue
            
        # Skip based on patterns
        skip = False
        for pattern in skip_patterns:
            if pattern.lower() in name.lower():
                skip = True
                break
        if skip:
            continue
            
        # Check if we have activation energy for this layer
        if name not in activation_energies:
            print(f"Warning: No activation energy for layer {name}, skipping")
            continue
            
        W = module.weight.data
        a = activation_energies[name]
        
        # Handle device mismatch
        if a.device != W.device:
            a = a.to(W.device)
            
        # Compute saliency
        S, L, alpha = compute_layer_saliency(W, a)
        
        S_dict[name] = S
        L_dict[name] = L
        alpha_dict[name] = alpha
        
    return S_dict, L_dict, alpha_dict


def compare_saliency_methods(
    W: torch.Tensor,
    a: Optional[torch.Tensor] = None,
    H_diag: Optional[torch.Tensor] = None
) -> Dict[str, torch.Tensor]:
    """
    Compare different saliency methods for analysis.
    
    Args:
        W: Weight matrix [d_out, d_in]
        a: Activation energy [d_in] (for smart method)
        H_diag: Hessian diagonal [d_in] (for hessian method)
        
    Returns:
        Dict with saliency matrices for each method
    """
    W = W.float()
    results = {}
    
    # 1. Magnitude-based (current baseline)
    results['magnitude'] = W.abs()
    
    # 2. Smart (activation-aware) - if activation energy provided
    if a is not None:
        a = a.float()
        alpha = W.abs().mean(dim=0)
        W_bin = alpha.unsqueeze(0) * torch.sign(W)
        delta_W = W - W_bin
        results['smart'] = (delta_W ** 2) * a.unsqueeze(0)
    
    # 3. Hessian-based - if Hessian diagonal provided
    if H_diag is not None:
        H_diag = H_diag.float()
        # Hessian saliency: W² / H_diag²
        results['hessian'] = (W ** 2) / (H_diag.unsqueeze(0) ** 2 + 1e-10)
    
    # 4. Combined: Smart + Hessian (experimental)
    if a is not None and H_diag is not None:
        # Combine activation-aware error with Hessian importance
        alpha = W.abs().mean(dim=0)
        W_bin = alpha.unsqueeze(0) * torch.sign(W)
        delta_W = W - W_bin
        hessian_weight = 1.0 / (H_diag.unsqueeze(0) ** 2 + 1e-10)
        results['smart_hessian'] = (delta_W ** 2) * a.unsqueeze(0) * hessian_weight
    
    return results


def analyze_saliency_distribution(
    S_dict: Dict[str, torch.Tensor],
    L_dict: Dict[str, float]
) -> Dict[str, Dict]:
    """
    Analyze the distribution of saliency across layers.
    
    Returns statistics useful for debugging and understanding.
    """
    total_L = sum(L_dict.values())
    
    analysis = {}
    for name in S_dict:
        S = S_dict[name]
        L = L_dict[name]
        
        analysis[name] = {
            "shape": tuple(S.shape),
            "num_weights": S.numel(),
            "layer_need": L,
            "fraction_of_total": L / total_L if total_L > 0 else 0,
            "min": S.min().item(),
            "max": S.max().item(),
            "mean": S.mean().item(),
            "std": S.std().item(),
            "median": S.median().item(),
        }
        
    return analysis
