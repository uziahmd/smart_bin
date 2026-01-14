"""
Mask Generator: Generate boolean masks selecting top-K salient weights per layer.

For each layer, flatten the saliency matrix, select top-K indices,
and create a boolean mask (True = salient, keep high precision).
"""

import torch
from typing import Dict, Optional
import torch.nn as nn


def generate_masks(
    S_dict: Dict[str, torch.Tensor],
    K_dict: Dict[str, int]
) -> Dict[str, torch.Tensor]:
    """
    Generate boolean masks selecting top-K salient weights per layer.
    
    Args:
        S_dict: Saliency matrices per layer
        K_dict: Budget (number of salient weights) per layer
        
    Returns:
        mask_dict: Boolean tensors (True = salient) per layer
    """
    mask_dict = {}
    
    for name in S_dict:
        if name not in K_dict:
            print(f"Warning: No budget for layer {name}, skipping mask generation")
            continue
            
        S = S_dict[name]
        K = K_dict[name]
        
        # Flatten saliency matrix
        flat_S = S.flatten()
        
        # Clamp K to valid range
        K = max(0, min(K, flat_S.numel()))
        
        if K == 0:
            # All weights are non-salient (binarized)
            mask = torch.zeros(S.shape, dtype=torch.bool, device=S.device)
        elif K >= flat_S.numel():
            # All weights are salient (no binarization)
            mask = torch.ones(S.shape, dtype=torch.bool, device=S.device)
        else:
            # Select top-K indices
            _, top_indices = torch.topk(flat_S, K)
            
            # Create mask
            flat_mask = torch.zeros(flat_S.shape, dtype=torch.bool, device=S.device)
            flat_mask[top_indices] = True
            
            # Reshape to original
            mask = flat_mask.view(S.shape)
        
        mask_dict[name] = mask
        
    return mask_dict


def generate_all_masks(
    model: nn.Module,
    activation_energies: Dict[str, torch.Tensor],
    p_global: float = 0.1,
    skip_patterns: Optional[list] = None,
    min_frac: float = 0.0001,
    max_frac: float = 0.9,
    return_stats: bool = False
) -> Dict[str, torch.Tensor]:
    """
    End-to-end mask generation for all layers.
    
    Combines saliency scoring, budget allocation, and mask generation.
    
    Args:
        model: The model to generate masks for
        activation_energies: Dict of activation energies per layer
        p_global: Global salient fraction (e.g., 0.1 = 10% salient)
        skip_patterns: Layer patterns to skip
        min_frac: Minimum salient fraction per layer
        max_frac: Maximum salient fraction per layer
        return_stats: Whether to return additional statistics
        
    Returns:
        mask_dict: Boolean masks per layer
        (optional) stats: Allocation and saliency statistics
    """
    from .saliency_scorer import compute_all_saliency
    from .budget_allocator import allocate_budgets
    
    skip_patterns = skip_patterns or ['embed', 'lm_head', 'head', 'norm']
    
    # Step 1: Compute saliency for all layers
    S_dict, L_dict, alpha_dict = compute_all_saliency(
        model, activation_energies, skip_patterns
    )
    
    if not S_dict:
        print("Warning: No layers to generate masks for")
        return {} if not return_stats else ({}, {})
    
    # Step 2: Get layer sizes
    sizes_dict = {}
    for name, module in model.named_modules():
        if name in S_dict:
            sizes_dict[name] = module.weight.numel()
    
    # Step 3: Allocate budgets
    K_dict, K_total = allocate_budgets(
        L_dict, sizes_dict, p_global, min_frac, max_frac
    )
    
    print(f"Allocated {K_total:,} salient weights ({p_global*100:.1f}% of total)")
    
    # Step 4: Generate masks
    mask_dict = generate_masks(S_dict, K_dict)
    
    if return_stats:
        stats = {
            "L_dict": L_dict,
            "K_dict": K_dict,
            "alpha_dict": alpha_dict,
            "K_total": K_total,
            "N_total": sum(sizes_dict.values()),
            "actual_p": K_total / sum(sizes_dict.values()) if sizes_dict else 0,
        }
        return mask_dict, stats
    
    return mask_dict


def verify_masks(
    mask_dict: Dict[str, torch.Tensor],
    K_dict: Dict[str, int]
) -> Dict[str, bool]:
    """
    Verify that masks are correctly generated.
    
    Returns dict of verification results per layer.
    """
    results = {}
    
    for name in mask_dict:
        mask = mask_dict[name]
        expected_K = K_dict.get(name, -1)
        actual_K = mask.sum().item()
        
        results[name] = {
            "expected": expected_K,
            "actual": actual_K,
            "match": expected_K == actual_K,
            "shape_valid": mask.dim() == 2,
        }
        
        if expected_K != actual_K:
            print(f"Warning: Mask mismatch for {name}: expected {expected_K}, got {actual_K}")
    
    return results


def save_masks(mask_dict: Dict[str, torch.Tensor], path: str):
    """Save masks to file."""
    torch.save(mask_dict, path)
    print(f"Saved masks to {path}")


def load_masks(path: str) -> Dict[str, torch.Tensor]:
    """Load masks from file."""
    mask_dict = torch.load(path)
    print(f"Loaded masks from {path} ({len(mask_dict)} layers)")
    return mask_dict
