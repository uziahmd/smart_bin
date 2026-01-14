"""
Budget Allocator: Allocate salient weight budget per layer proportionally.

Given layer need scores L_l and total budget K_total, allocate:
    K_l = K_total × (L_l / sum(L))

With constraints:
    - min_per_layer: minimum budget per layer
    - max_per_layer: maximum budget per layer
    - Rounding correction to ensure sum(K_l) == K_total
"""

import torch
from typing import Dict, Optional, Tuple
import math


def allocate_budgets(
    L_dict: Dict[str, float],
    sizes_dict: Dict[str, int],
    p_global: float,
    min_frac: float = 0.01,
    max_frac: float = 0.95,
    blend_uniform: float = 0.3
) -> Tuple[Dict[str, int], int]:
    """
    Allocate salient weight budget to each layer with hybrid proportional+uniform.
    
    Uses a blend of proportional (based on L) and uniform allocation to ensure
    we can actually hit the target p_global even with imbalanced L values.
    
    Args:
        L_dict: Layer need scores (sum of saliency per layer)
        sizes_dict: Number of weights per layer
        p_global: Global fraction of weights to keep salient (e.g., 0.1 for 10%)
        min_frac: Minimum fraction of layer weights to keep salient
        max_frac: Maximum fraction of layer weights to keep salient
        blend_uniform: How much to blend uniform allocation (0=pure proportional, 1=uniform)
        
    Returns:
        K_dict: Allocated budget (number of salient weights) per layer
        K_total: Total budget allocated
    """
    # Ensure consistent layer ordering
    layer_names = list(L_dict.keys())
    
    # Total weights and budget
    N_total = sum(sizes_dict[name] for name in layer_names)
    K_target = int(round(p_global * N_total))
    
    # Normalize L to get proportional weights
    L_total = sum(L_dict[name] for name in layer_names)
    
    if L_total <= 0:
        # Pure uniform allocation
        K_dict = {name: int(round(p_global * sizes_dict[name])) for name in layer_names}
        return K_dict, sum(K_dict.values())
    
    # Compute blended allocation weights
    # Proportional component: L_l / L_total  
    # Uniform component: size_l / N_total
    # Blend: (1 - blend_uniform) * proportional + blend_uniform * uniform
    
    allocation_weights = {}
    for name in layer_names:
        prop_weight = L_dict[name] / L_total
        uniform_weight = sizes_dict[name] / N_total
        allocation_weights[name] = (1 - blend_uniform) * prop_weight + blend_uniform * uniform_weight
    
    # Normalize weights
    weight_sum = sum(allocation_weights.values())
    allocation_weights = {k: v / weight_sum for k, v in allocation_weights.items()}
    
    # Initial allocation
    K_dict = {}
    for name in layer_names:
        size = sizes_dict[name]
        raw_k = K_target * allocation_weights[name]
        
        # Apply per-layer constraints
        min_k = max(1, int(min_frac * size))
        max_k = int(max_frac * size)
        
        K_dict[name] = max(min_k, min(max_k, int(round(raw_k))))
    
    # Iteratively adjust to hit target
    for iteration in range(100):
        current_total = sum(K_dict.values())
        diff = K_target - current_total
        
        if abs(diff) <= len(layer_names):  # Close enough
            break
        
        # Sort layers by how much room they have to adjust
        if diff > 0:
            # Need to add more - sort by room to grow
            adjustable = [(name, int(max_frac * sizes_dict[name]) - K_dict[name]) 
                         for name in layer_names]
            adjustable = [(n, room) for n, room in adjustable if room > 0]
            adjustable.sort(key=lambda x: -x[1])  # Most room first
            
            # Distribute proportionally to room
            if adjustable:
                total_room = sum(room for _, room in adjustable)
                for name, room in adjustable:
                    add = min(room, int(round(diff * room / total_room)) + 1)
                    K_dict[name] += add
        else:
            # Need to remove - sort by room to shrink
            adjustable = [(name, K_dict[name] - max(1, int(min_frac * sizes_dict[name])))
                         for name in layer_names]
            adjustable = [(n, room) for n, room in adjustable if room > 0]
            adjustable.sort(key=lambda x: -x[1])  # Most room first
            
            if adjustable:
                total_room = sum(room for _, room in adjustable)
                for name, room in adjustable:
                    remove = min(room, int(round(abs(diff) * room / total_room)) + 1)
                    K_dict[name] -= remove
    
    # Final clamp pass
    for name in layer_names:
        size = sizes_dict[name]
        min_k = max(1, int(min_frac * size))
        max_k = int(max_frac * size)
        K_dict[name] = max(min_k, min(max_k, K_dict[name]))
    
    final_total = sum(K_dict.values())
    
    return K_dict, final_total


def allocate_budgets_grouped(
    L_dict: Dict[str, float],
    sizes_dict: Dict[str, int],
    groups: Dict[str, list],  # e.g., {"attention": [...], "mlp": [...]}
    p_global: float,
    group_weights: Optional[Dict[str, float]] = None,
    min_frac: float = 0.0001,
    max_frac: float = 0.9
) -> Tuple[Dict[str, int], int]:
    """
    Allocate budgets with separate allocation per group (e.g., attention vs MLP).
    
    Args:
        L_dict: Layer need scores
        sizes_dict: Number of weights per layer
        groups: Dict mapping group name to list of layer names
        p_global: Global salient fraction
        group_weights: Optional weights per group (default: proportional to size)
        min_frac: Minimum fraction per layer
        max_frac: Maximum fraction per layer
        
    Returns:
        K_dict: Allocated budget per layer
        K_total: Total budget allocated
    """
    # Calculate total sizes per group
    group_sizes = {}
    for group_name, layer_names in groups.items():
        group_sizes[group_name] = sum(sizes_dict[name] for name in layer_names if name in sizes_dict)
    
    total_size = sum(group_sizes.values())
    
    # Calculate budget per group
    if group_weights is None:
        # Proportional to group size
        group_budgets = {
            group_name: int(round(p_global * group_sizes[group_name]))
            for group_name in groups
        }
    else:
        # Use provided weights
        total_weight = sum(group_weights.values())
        total_budget = int(round(p_global * total_size))
        group_budgets = {
            group_name: int(round(total_budget * group_weights[group_name] / total_weight))
            for group_name in groups
        }
    
    # Allocate within each group
    K_dict = {}
    for group_name, layer_names in groups.items():
        group_L = {name: L_dict[name] for name in layer_names if name in L_dict}
        group_sizes_local = {name: sizes_dict[name] for name in layer_names if name in sizes_dict}
        
        if not group_L:
            continue
            
        # Calculate group-local p to achieve group budget
        group_size = sum(group_sizes_local.values())
        group_p = group_budgets[group_name] / group_size if group_size > 0 else p_global
        
        group_K, _ = allocate_budgets(group_L, group_sizes_local, group_p, min_frac, max_frac)
        K_dict.update(group_K)
    
    return K_dict, sum(K_dict.values())


def analyze_allocation(
    K_dict: Dict[str, int],
    sizes_dict: Dict[str, int],
    L_dict: Optional[Dict[str, float]] = None
) -> Dict[str, Dict]:
    """
    Analyze budget allocation for debugging.
    
    Returns per-layer and global statistics.
    """
    total_k = sum(K_dict.values())
    total_size = sum(sizes_dict.values())
    
    analysis = {
        "_global": {
            "total_salient": total_k,
            "total_weights": total_size,
            "global_salient_frac": total_k / total_size if total_size > 0 else 0,
        }
    }
    
    for name in K_dict:
        k = K_dict[name]
        size = sizes_dict[name]
        
        analysis[name] = {
            "budget": k,
            "size": size,
            "salient_frac": k / size if size > 0 else 0,
            "frac_of_total_budget": k / total_k if total_k > 0 else 0,
        }
        
        if L_dict and name in L_dict:
            analysis[name]["layer_need"] = L_dict[name]
            analysis[name]["need_frac"] = L_dict[name] / sum(L_dict.values()) if sum(L_dict.values()) > 0 else 0
    
    return analysis
