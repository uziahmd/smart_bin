"""
Activation Collector: Collect activation energy a_j = E[x_j²] per linear layer.

During calibration, we hook linear layers to capture their input activations
and accumulate running statistics (sum of squares, count) to compute
the expected squared activation per input channel.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable
from tqdm import tqdm


class ActivationCollector:
    """
    Collects activation energy statistics from linear layers during calibration.
    
    For each target linear layer, computes:
        a_j = E[x_j²] = sum(x²) / count
    
    where x is the input activation tensor and j indexes input channels.
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        target_layers: Optional[List[str]] = None,
        skip_patterns: Optional[List[str]] = None
    ):
        """
        Args:
            model: The model to collect activations from
            target_layers: List of layer names to hook. If None, hooks all Linear layers.
            skip_patterns: Patterns to skip (e.g., ['embed', 'lm_head', 'norm'])
        """
        self.model = model
        self.target_layers = target_layers
        self.skip_patterns = skip_patterns or ['embed', 'lm_head', 'head', 'norm']
        
        # Statistics accumulators
        self.sumsq: Dict[str, torch.Tensor] = {}  # layer_name -> [d_in] sum of squares
        self.count: Dict[str, int] = {}           # layer_name -> count of activations
        
        # Hook handles for cleanup
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        
        # Cached inputs during forward
        self._cached_inputs: Dict[str, torch.Tensor] = {}
        
    def _should_hook(self, name: str, module: nn.Module) -> bool:
        """Check if a layer should be hooked based on name and type."""
        # Must be a Linear layer
        if not isinstance(module, nn.Linear):
            return False
            
        # Check skip patterns
        for pattern in self.skip_patterns:
            if pattern.lower() in name.lower():
                return False
                
        # If target_layers specified, must be in list
        if self.target_layers is not None:
            return name in self.target_layers
            
        return True
    
    def _create_hook(self, name: str) -> Callable:
        """Create a forward hook that captures input activations."""
        def hook(module: nn.Module, input: tuple, output: torch.Tensor):
            if len(input) == 0:
                return
            x = input[0]  # [B, T, d_in] or [B, d_in]
            
            # Flatten to 2D: [N, d_in]
            if x.dim() == 3:
                x = x.reshape(-1, x.shape[-1])
            elif x.dim() == 1:
                x = x.unsqueeze(0)
            
            # Move to float for accumulation
            x = x.float()
            
            # Accumulate statistics
            if name not in self.sumsq:
                self.sumsq[name] = torch.zeros(x.shape[-1], device=x.device, dtype=torch.float64)
                self.count[name] = 0
            
            # Sum of squares per channel
            self.sumsq[name] += (x ** 2).sum(dim=0).double()
            self.count[name] += x.shape[0]
            
        return hook
    
    def register_hooks(self) -> int:
        """Register forward hooks on all target linear layers."""
        self.clear()
        
        hooked_count = 0
        for name, module in self.model.named_modules():
            if self._should_hook(name, module):
                hook = module.register_forward_hook(self._create_hook(name))
                self.hooks.append(hook)
                hooked_count += 1
                
        return hooked_count
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def clear(self):
        """Clear all accumulated statistics and hooks."""
        self.remove_hooks()
        self.sumsq = {}
        self.count = {}
        self._cached_inputs = {}
    
    @torch.no_grad()
    def collect(
        self, 
        dataloader, 
        device: str = "cuda:0",
        max_batches: Optional[int] = None,
        show_progress: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Run calibration and collect activation energies.
        
        Args:
            dataloader: Calibration data loader (list of (input_ids, labels) tuples)
            device: Device to run calibration on
            max_batches: Maximum number of batches to process (None = all)
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary mapping layer names to activation energy tensors [d_in]
        """
        # Register hooks
        n_hooks = self.register_hooks()
        print(f"Registered {n_hooks} hooks for activation collection")
        
        # Move model to device
        self.model.to(device)
        self.model.eval()
        
        # Process batches
        iterator = enumerate(dataloader)
        if show_progress:
            iterator = tqdm(iterator, total=len(dataloader) if max_batches is None else max_batches,
                          desc="Collecting activations")
        
        try:
            for batch_idx, batch in iterator:
                if max_batches is not None and batch_idx >= max_batches:
                    break
                    
                # Handle different batch formats
                if isinstance(batch, (list, tuple)):
                    input_ids = batch[0]
                elif isinstance(batch, dict):
                    input_ids = batch.get('input_ids', batch.get('inputs'))
                else:
                    input_ids = batch
                
                # Forward pass (hooks will capture activations)
                try:
                    self.model(input_ids.to(device))
                except Exception as e:
                    # Some models may error but hooks still fire
                    pass
                    
        finally:
            # Always remove hooks
            self.remove_hooks()
        
        # Compute activation energies: a_j = sumsq / count
        activation_energies = {}
        for name in self.sumsq:
            if self.count[name] > 0:
                a = self.sumsq[name] / self.count[name]
                activation_energies[name] = a.float()  # Convert back to float32
                
        print(f"Collected activation energies for {len(activation_energies)} layers")
        return activation_energies
    
    def get_layer_info(self) -> Dict[str, Dict]:
        """Get information about hookable layers."""
        info = {}
        for name, module in self.model.named_modules():
            if self._should_hook(name, module):
                info[name] = {
                    "type": type(module).__name__,
                    "in_features": module.in_features,
                    "out_features": module.out_features,
                    "num_params": module.weight.numel()
                }
        return info


def collect_activation_energy(
    model: nn.Module,
    dataloader,
    device: str = "cuda:0",
    skip_patterns: Optional[List[str]] = None,
    max_batches: Optional[int] = None
) -> Dict[str, torch.Tensor]:
    """
    Convenience function to collect activation energies in one call.
    
    Args:
        model: Model to collect from
        dataloader: Calibration data
        device: Device to use
        skip_patterns: Layer name patterns to skip
        max_batches: Max batches to process
        
    Returns:
        Dict mapping layer names to activation energy tensors
    """
    collector = ActivationCollector(model, skip_patterns=skip_patterns)
    return collector.collect(dataloader, device=device, max_batches=max_batches)
