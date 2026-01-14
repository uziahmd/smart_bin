"""
Smart Quantizer: Linear layer with activation-aware partial binarization.

Implements the SmartBinaryLinear class that:
- Keeps salient weights at high precision (fp16/bf16)
- Binarizes non-salient weights using column-wise scaling: α × sign(W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
from torch.utils.checkpoint import checkpoint

import sys
sys.path.insert(0, '..')
from quant.quantizer import BinaryInterface, STEBinary


class SmartBinaryLinear(nn.Module, BinaryInterface):
    """
    Linear layer with activation-aware partial binarization.
    
    Salient weights (determined by mask) are kept at original precision.
    Non-salient weights are binarized: α_j × sign(W_ij)
    """
    
    def __init__(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        salient_mask: torch.Tensor,
        alpha: Optional[torch.Tensor] = None
    ):
        """
        Args:
            weight: Original weight tensor [d_out, d_in]
            bias: Optional bias tensor [d_out]
            salient_mask: Boolean mask [d_out, d_in] (True = salient)
            alpha: Column-wise scaling factors [d_in]. If None, computed from weight.
        """
        super().__init__()
        
        # Store original weight in float32 for training stability
        self.weight = nn.Parameter(weight.float().clone())
        
        # Bias
        if bias is not None:
            self.bias = nn.Parameter(bias.float().clone())
        else:
            self.bias = None
        
        # Register mask as buffer (not trained)
        self.register_buffer('salient_mask', salient_mask.bool())
        
        # Compute or store alpha
        if alpha is None:
            alpha = weight.abs().mean(dim=0)
        self.register_buffer('alpha', alpha.float())
        
        # Statistics
        self.num_salient = salient_mask.sum().item()
        self.num_total = salient_mask.numel()
        self.salient_ratio = self.num_salient / self.num_total
        
    def get_quantized_weight(self) -> torch.Tensor:
        """
        Get the mixed-precision weight tensor.
        
        Returns:
            Weight tensor with salient weights at original precision
            and non-salient weights binarized.
        """
        # Binarized weights: α × sign(W)
        W_bin = self.alpha.unsqueeze(0) * torch.sign(self.weight)
        
        # Mixed weights: salient = original, non-salient = binarized
        W_mixed = torch.where(self.salient_mask, self.weight, W_bin)
        
        return W_mixed
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with mixed-precision weights."""
        W = self.get_quantized_weight()
        # Ensure dtype matches input
        if W.dtype != x.dtype:
            W = W.to(x.dtype)
        bias = self.bias
        if bias is not None and bias.dtype != x.dtype:
            bias = bias.to(x.dtype)
        return F.linear(x, W, bias)
    
    def get_save_weight_dict(self) -> Dict:
        """For saving checkpoints compatible with BinaryInterface."""
        return {
            "weight": self.weight.data.half().cpu(),
            "bias": self.bias.data.half().cpu() if self.bias is not None else None,
            "salient_mask": self.salient_mask.cpu(),
            "alpha": self.alpha.cpu(),
        }
    
    def extra_repr(self) -> str:
        return (
            f"in_features={self.weight.shape[1]}, "
            f"out_features={self.weight.shape[0]}, "
            f"salient={self.salient_ratio*100:.1f}%"
        )


class SmartBinaryLinearSTE(SmartBinaryLinear):
    """
    SmartBinaryLinear with Straight-Through Estimator for training.
    
    Uses STE for the sign operation to allow gradients to flow
    through binarized weights during QAT.
    """
    
    def get_quantized_weight(self) -> torch.Tensor:
        """Get mixed weight with STE for binarization."""
        # Use STE for sign operation
        W_sign = STEBinary.apply(self.weight)
        W_bin = self.alpha.unsqueeze(0) * W_sign
        
        # Mixed weights
        W_mixed = torch.where(self.salient_mask, self.weight, W_bin)
        
        return W_mixed


def convert_to_smart_binary(
    module: nn.Linear,
    salient_mask: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    use_ste: bool = False
) -> SmartBinaryLinear:
    """
    Convert a nn.Linear module to SmartBinaryLinear.
    
    Args:
        module: Original Linear layer
        salient_mask: Boolean mask for salient weights
        alpha: Optional pre-computed scaling factors
        use_ste: Whether to use STE version for training
        
    Returns:
        SmartBinaryLinear (or SmartBinaryLinearSTE) module
    """
    cls = SmartBinaryLinearSTE if use_ste else SmartBinaryLinear
    
    return cls(
        weight=module.weight.data,
        bias=module.bias.data if module.bias is not None else None,
        salient_mask=salient_mask,
        alpha=alpha
    )


def apply_smart_binarization(
    model: nn.Module,
    mask_dict: Dict[str, torch.Tensor],
    alpha_dict: Optional[Dict[str, torch.Tensor]] = None,
    use_ste: bool = False,
    inplace: bool = True
) -> nn.Module:
    """
    Apply smart binarization to all layers in the model.
    
    Args:
        model: Model to binarize
        mask_dict: Dict of salient masks per layer name
        alpha_dict: Optional dict of scaling factors
        use_ste: Whether to use STE for training
        inplace: Whether to modify model in place
        
    Returns:
        Model with binarized layers
    """
    if not inplace:
        import copy
        model = copy.deepcopy(model)
    
    alpha_dict = alpha_dict or {}
    
    # Track replacements
    replacements = []
    
    for name, module in model.named_modules():
        if name in mask_dict and isinstance(module, nn.Linear):
            mask = mask_dict[name]
            alpha = alpha_dict.get(name, None)
            
            # Create replacement
            new_module = convert_to_smart_binary(module, mask, alpha, use_ste)
            
            # Move to same device
            new_module = new_module.to(module.weight.device)
            
            replacements.append((name, new_module))
    
    # Apply replacements
    for name, new_module in replacements:
        # Navigate to parent and replace
        parts = name.rsplit('.', 1)
        if len(parts) == 1:
            setattr(model, name, new_module)
        else:
            parent_name, child_name = parts
            parent = dict(model.named_modules())[parent_name]
            setattr(parent, child_name, new_module)
    
    print(f"Applied smart binarization to {len(replacements)} layers")
    
    return model


def get_binarization_stats(model: nn.Module) -> Dict:
    """
    Get statistics about binarized layers in the model.
    """
    stats = {
        "layers": {},
        "total_salient": 0,
        "total_weights": 0,
    }
    
    for name, module in model.named_modules():
        if isinstance(module, SmartBinaryLinear):
            stats["layers"][name] = {
                "salient": module.num_salient,
                "total": module.num_total,
                "ratio": module.salient_ratio,
            }
            stats["total_salient"] += module.num_salient
            stats["total_weights"] += module.num_total
    
    if stats["total_weights"] > 0:
        stats["global_salient_ratio"] = stats["total_salient"] / stats["total_weights"]
    else:
        stats["global_salient_ratio"] = 0
        
    return stats
