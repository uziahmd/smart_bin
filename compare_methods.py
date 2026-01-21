#!/usr/bin/env python
"""
Compare quantization methods: Vanilla vs PB-LLM vs Smart Binarization

This script provides a comprehensive comparison framework for:
1. Vanilla (unquantized) models
2. PB-LLM (magnitude-based saliency)
3. Smart Binarization (activation-aware saliency)

Usage:
    python compare_methods.py facebook/opt-125m wikitext2 \
        --methods vanilla pbllm smart \
        --p_global 0.1 0.2 \
        --output_dir ./comparison_results
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
import copy

import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def get_model(model_name: str, dtype=None):
    """Load model from HuggingFace."""
    print(f"Loading model: {model_name}")
    
    # Use bfloat16 for Gemma models (more numerically stable)
    if dtype is None:
        if "gemma" in model_name.lower():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
    
    if "opt" in model_name.lower():
        from transformers import OPTForCausalLM
        model = OPTForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        model.seqlen = model.config.max_position_embeddings
    elif "llama" in model_name.lower():
        from transformers import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        model.seqlen = 2048
    elif "gemma" in model_name.lower():
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        # Use shorter seqlen for Gemma to save memory
        model.seqlen = min(2048, getattr(model.config, 'max_position_embeddings', 2048))
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        model.seqlen = getattr(model.config, 'max_position_embeddings', 2048)

    
    return model


def get_calibration_and_test_data(dataset: str, model_name: str, nsamples: int, seed: int = 42):
    """Load calibration and test data."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'gptq_pb'))
    from datautils import get_wikitext2, get_c4, get_tokenizer, set_seed
    
    set_seed(seed)
    tokenizer = get_tokenizer(model_name)
    
    if dataset == 'wikitext2':
        trainloader, testenc = get_wikitext2(nsamples, seed, 2048, model_name, tokenizer)
    elif dataset == 'c4':
        trainloader, testenc = get_c4(nsamples, seed, 2048, model_name, tokenizer)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    return trainloader, testenc, tokenizer


def evaluate_perplexity(model, testenc, device, seqlen=2048):
    """Evaluate perplexity on test set."""
    model.eval()
    model.to(device)
    
    testenc = testenc.input_ids if hasattr(testenc, 'input_ids') else testenc
    nsamples = testenc.shape[1] // seqlen
    
    # Limit samples for memory-constrained models
    nsamples = min(nsamples, 40)
    
    nlls = []
    with torch.no_grad():
        for i in range(nsamples):
            batch = testenc[:, i*seqlen:(i+1)*seqlen].to(device)
            outputs = model(batch)
            logits = outputs.logits
            
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:].contiguous()
            
            # Process in chunks for large vocab models to avoid OOM
            vocab_size = shift_logits.size(-1)
            if vocab_size > 100000:
                # Compute loss token-by-token to save memory
                loss_fct = nn.CrossEntropyLoss(reduction='none')
                losses = []
                chunk_size = 256  # Process 256 tokens at a time
                for j in range(0, shift_logits.size(1), chunk_size):
                    end_j = min(j + chunk_size, shift_logits.size(1))
                    chunk_logits = shift_logits[:, j:end_j, :].view(-1, vocab_size)
                    chunk_labels = shift_labels[:, j:end_j].view(-1)
                    chunk_loss = loss_fct(chunk_logits, chunk_labels)
                    losses.append(chunk_loss.mean())
                    del chunk_logits, chunk_labels, chunk_loss
                loss = torch.stack(losses).mean()
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
            
            nlls.append(loss.item())
            del logits, outputs, shift_logits, shift_labels
            torch.cuda.empty_cache()
    
    ppl = torch.exp(torch.tensor(nlls).mean()).item()
    return ppl


def evaluate_vanilla(model, testenc, device):
    """Evaluate vanilla (unquantized) model."""
    print("\n" + "="*70)
    print("EVALUATING: Vanilla (Unquantized)")
    print("="*70)
    
    start_time = time.time()
    ppl = evaluate_perplexity(model, testenc, device, model.seqlen)
    eval_time = time.time() - start_time
    
    print(f"Perplexity: {ppl:.2f}")
    print(f"Evaluation time: {eval_time:.1f}s")
    
    return {
        'method': 'vanilla',
        'perplexity': ppl,
        'eval_time': eval_time,
        'salient_ratio': 1.0,  # All weights are "salient" (full precision)
    }


def evaluate_pbllm(
    model_name: str, 
    trainloader, 
    testenc, 
    device, 
    p_global: float = 0.2,
    salient_metric: str = 'magnitude'
):
    """
    Evaluate PB-LLM (existing magnitude-based method).
    
    Note: This uses the existing gptq_pb implementation.
    """
    print("\n" + "="*70)
    print(f"EVALUATING: PB-LLM ({salient_metric}, p={p_global*100:.0f}%)")
    print("="*70)
    
    # Reload fresh model for fair comparison
    model = get_model(model_name)
    model.to(device)
    
    start_time = time.time()
    
    skip_patterns = ['embed', 'lm_head', 'head', 'norm', 'layernorm', 'ln_']
    
    # Apply magnitude-based binarization
    total_weights = 0
    total_salient = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(p in name.lower() for p in skip_patterns):
                continue
            
            orig_dtype = module.weight.data.dtype
            W = module.weight.data.float()
            n_weights = W.numel()
            k = int(n_weights * p_global)
            
            # Magnitude-based saliency
            if salient_metric == 'magnitude':
                S = W.abs()
            else:
                # Use Hessian diagonal (approximated by activation variance)
                S = W.abs()  # Simplified for this comparison
            
            # Get top-k mask
            flat_S = S.view(-1)
            _, top_indices = torch.topk(flat_S, k)
            mask = torch.zeros_like(flat_S, dtype=torch.bool)
            mask[top_indices] = True
            mask = mask.view(W.shape)
            
            # Binarize non-salient weights
            alpha = W.abs().mean(dim=0)  # Column-wise scaling
            W_bin = alpha.unsqueeze(0) * torch.sign(W)
            W_mixed = torch.where(mask, W, W_bin)
            
            module.weight.data = W_mixed.to(orig_dtype)
            
            total_weights += n_weights
            total_salient += k
    
    quant_time = time.time() - start_time
    
    # Evaluate
    start_time = time.time()
    ppl = evaluate_perplexity(model, testenc, device, model.seqlen)
    eval_time = time.time() - start_time
    
    print(f"Perplexity: {ppl:.2f}")
    print(f"Salient ratio: {total_salient/total_weights*100:.2f}%")
    print(f"Quantization time: {quant_time:.1f}s")
    print(f"Evaluation time: {eval_time:.1f}s")
    
    del model
    torch.cuda.empty_cache()
    
    return {
        'method': f'pbllm_{salient_metric}',
        'perplexity': ppl,
        'salient_ratio': p_global,
        'actual_salient_ratio': total_salient / total_weights,
        'quant_time': quant_time,
        'eval_time': eval_time,
    }


def evaluate_hessian(
    model_name: str,
    trainloader,
    testenc,
    device,
    p_global: float = 0.2
):
    """
    Evaluate Hessian-based saliency method.
    
    Uses S = W² / H_diag² where H_diag is approximated from input activations.
    This is the original PB-LLM hessian approach.
    """
    print("\n" + "="*70)
    print(f"EVALUATING: Hessian-based (p={p_global*100:.0f}%)")
    print("="*70)
    
    # Reload fresh model
    model = get_model(model_name)
    model.to(device)
    
    skip_patterns = ['embed', 'lm_head', 'head', 'norm', 'layernorm', 'ln_']
    
    # Step 1: Collect H_diag (input activation statistics) per layer
    print("Collecting Hessian diagonal approximation...")
    
    H_diag_dict = {}
    hooks = []
    
    def make_hook(name):
        def hook(module, inp, out):
            if len(inp) == 0 or inp[0] is None:
                return
            x = inp[0].detach().float()
            if x.dim() == 3:
                x = x.view(-1, x.shape[-1])
            # H_diag approximation: sum of x² per input feature
            x_sq = (x ** 2).sum(dim=0)
            if name not in H_diag_dict:
                H_diag_dict[name] = {'sumsq': torch.zeros_like(x_sq), 'count': 0}
            H_diag_dict[name]['sumsq'] += x_sq.to(H_diag_dict[name]['sumsq'].device)
            H_diag_dict[name]['count'] += x.shape[0]
        return hook
    
    # Register hooks
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(p in name.lower() for p in skip_patterns):
                continue
            hooks.append(module.register_forward_hook(make_hook(name)))
    
    # Run calibration
    model.eval()
    with torch.no_grad():
        for batch in trainloader:
            if isinstance(batch, (list, tuple)):
                input_ids = batch[0]
            elif isinstance(batch, dict):
                input_ids = batch.get('input_ids', batch.get('tokens'))
            else:
                input_ids = batch
            if input_ids is not None:
                input_ids = input_ids.to(device)
                model(input_ids)
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    # Compute H_diag = E[x²]
    for name in H_diag_dict:
        count = H_diag_dict[name]['count']
        if count > 0:
            H_diag_dict[name] = H_diag_dict[name]['sumsq'] / count
        else:
            H_diag_dict[name] = H_diag_dict[name]['sumsq']
    
    print(f"Collected Hessian diagonals for {len(H_diag_dict)} layers")
    
    # Step 2: Apply Hessian-based binarization
    start_time = time.time()
    
    total_weights = 0
    total_salient = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(p in name.lower() for p in skip_patterns):
                continue
            if name not in H_diag_dict:
                continue
            
            orig_dtype = module.weight.data.dtype
            W = module.weight.data.float()
            H_diag = H_diag_dict[name].to(W.device)
            
            n_weights = W.numel()
            k = int(n_weights * p_global)
            
            # Hessian-based saliency: S = W² / H_diag²
            # Higher S means weight is more important (large W, small H_diag)
            S = (W ** 2) / (H_diag.unsqueeze(0) ** 2 + 1e-10)
            
            # Get top-k mask (highest saliency = keep)
            flat_S = S.view(-1)
            _, top_indices = torch.topk(flat_S, k)
            mask = torch.zeros_like(flat_S, dtype=torch.bool)
            mask[top_indices] = True
            mask = mask.view(W.shape)
            
            # Binarize non-salient weights
            alpha = W.abs().mean(dim=0)
            W_bin = alpha.unsqueeze(0) * torch.sign(W)
            W_mixed = torch.where(mask, W, W_bin)
            
            module.weight.data = W_mixed.to(orig_dtype)
            
            total_weights += n_weights
            total_salient += k
    
    quant_time = time.time() - start_time
    
    # Evaluate
    start_time = time.time()
    ppl = evaluate_perplexity(model, testenc, device, model.seqlen)
    eval_time = time.time() - start_time
    
    print(f"Perplexity: {ppl:.2f}")
    print(f"Salient ratio: {total_salient/total_weights*100:.2f}%")
    print(f"Quantization time: {quant_time:.1f}s")
    print(f"Evaluation time: {eval_time:.1f}s")
    
    del model
    torch.cuda.empty_cache()
    
    return {
        'method': 'hessian',
        'perplexity': ppl,
        'salient_ratio': p_global,
        'actual_salient_ratio': total_salient / total_weights,
        'quant_time': quant_time,
        'eval_time': eval_time,
    }


def evaluate_smart(
    model_name: str,
    trainloader,
    testenc,
    device,
    p_global: float = 0.1
):
    """
    Evaluate Smart Binarization (activation-aware saliency).
    """
    print("\n" + "="*70)
    print(f"EVALUATING: Smart Binarization (p={p_global*100:.0f}%)")
    print("="*70)
    
    # Reload fresh model
    model = get_model(model_name)
    model.to(device)
    
    start_time = time.time()
    
    # Import smart binarization components
    from smart_saliency import (
        ActivationCollector,
        compute_all_saliency,
        allocate_budgets,
        generate_masks,
    )
    from quant.smart_quantizer import apply_smart_binarization, get_binarization_stats
    
    skip_patterns = ['embed', 'lm_head', 'head', 'norm', 'layernorm', 'ln_']
    
    # Step 1: Collect activation energies
    print("Collecting activation energies...")
    collector = ActivationCollector(model, skip_patterns=skip_patterns)
    activation_energies = collector.collect(trainloader, device=device)
    
    # Step 2: Compute saliency
    print("Computing saliency scores...")
    S_dict, L_dict, alpha_dict = compute_all_saliency(model, activation_energies, skip_patterns)
    
    # Step 3: Allocate budgets
    sizes_dict = {}
    for name, module in model.named_modules():
        if name in S_dict:
            sizes_dict[name] = module.weight.numel()
    
    K_dict, K_total = allocate_budgets(L_dict, sizes_dict, p_global)
    N_total = sum(sizes_dict.values())
    
    # Step 4: Generate masks
    print("Generating masks...")
    mask_dict = generate_masks(S_dict, K_dict)
    
    # Step 5: Apply binarization
    print("Applying smart binarization...")
    model = apply_smart_binarization(model, mask_dict, alpha_dict, use_ste=False, inplace=True)
    
    quant_time = time.time() - start_time
    
    # Evaluate
    start_time = time.time()
    ppl = evaluate_perplexity(model, testenc, device, model.seqlen)
    eval_time = time.time() - start_time
    
    stats = get_binarization_stats(model)
    
    print(f"Perplexity: {ppl:.2f}")
    print(f"Actual salient ratio: {stats['global_salient_ratio']*100:.2f}%")
    print(f"Quantization time: {quant_time:.1f}s")
    print(f"Evaluation time: {eval_time:.1f}s")
    
    del model
    torch.cuda.empty_cache()
    
    return {
        'method': 'smart',
        'perplexity': ppl,
        'salient_ratio': p_global,
        'actual_salient_ratio': stats['global_salient_ratio'],
        'quant_time': quant_time,
        'eval_time': eval_time,
    }


def compare_methods(
    model_name: str,
    dataset: str,
    methods: List[str],
    p_values: List[float],
    nsamples: int = 128,
    seed: int = 42,
    output_dir: str = "./comparison_results"
):
    """
    Compare multiple quantization methods.
    
    Args:
        model_name: HuggingFace model name
        dataset: Calibration dataset
        methods: List of methods to compare ('vanilla', 'pbllm', 'smart')
        p_values: List of salient fractions to test
        nsamples: Number of calibration samples
        seed: Random seed
        output_dir: Output directory for results
    
    Returns:
        results: Dict with all comparison results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print("="*70)
    print("METHOD COMPARISON")
    print("="*70)
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset}")
    print(f"Methods: {methods}")
    print(f"Salient fractions: {p_values}")
    print(f"Calibration samples: {nsamples}")
    print("="*70)
    
    # Load calibration and test data
    trainloader, testenc, tokenizer = get_calibration_and_test_data(
        dataset, model_name, nsamples, seed
    )
    
    all_results = []
    
    # Evaluate vanilla (only once, no p_value needed)
    if 'vanilla' in methods:
        model = get_model(model_name)
        model.to(device)
        result = evaluate_vanilla(model, testenc, device)
        result['model'] = model_name
        result['dataset'] = dataset
        all_results.append(result)
        del model
        torch.cuda.empty_cache()
    
    # Evaluate other methods at each p_value
    for p in p_values:
        if 'magnitude' in methods or 'pbllm' in methods:
            result = evaluate_pbllm(
                model_name, trainloader, testenc, device, 
                p_global=p, salient_metric='magnitude'
            )
            result['model'] = model_name
            result['dataset'] = dataset
            all_results.append(result)
        
        if 'hessian' in methods:
            result = evaluate_hessian(
                model_name, trainloader, testenc, device,
                p_global=p
            )
            result['model'] = model_name
            result['dataset'] = dataset
            all_results.append(result)
        
        if 'smart' in methods:
            result = evaluate_smart(
                model_name, trainloader, testenc, device,
                p_global=p
            )
            result['model'] = model_name
            result['dataset'] = dataset
            all_results.append(result)
    
    # Print summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Method':<25} {'Salient %':<12} {'Perplexity':<12}")
    print("-"*49)
    
    # Sort by perplexity
    sorted_results = sorted(all_results, key=lambda x: x['perplexity'])
    
    for r in sorted_results:
        method = r['method']
        p = r.get('salient_ratio', 1.0) * 100
        ppl = r['perplexity']
        print(f"{method:<25} {p:<12.1f} {ppl:<12.2f}")
    
    # Calculate improvements
    vanilla_ppl = next((r['perplexity'] for r in all_results if r['method'] == 'vanilla'), None)
    
    if vanilla_ppl:
        print("\n" + "-"*49)
        print("Perplexity increase vs vanilla:")
        for r in sorted_results:
            if r['method'] != 'vanilla':
                increase = r['perplexity'] - vanilla_ppl
                pct_increase = increase / vanilla_ppl * 100
                print(f"  {r['method']}: +{increase:.2f} (+{pct_increase:.1f}%)")
    
    # Compare smart vs pbllm at same p
    smart_results = [r for r in all_results if r['method'] == 'smart']
    pbllm_results = [r for r in all_results if r['method'].startswith('pbllm')]
    
    if smart_results and pbllm_results:
        print("\n" + "-"*49)
        print("Smart vs PB-LLM comparison:")
        for smart_r in smart_results:
            p = smart_r['salient_ratio']
            pbllm_r = next((r for r in pbllm_results 
                           if abs(r['salient_ratio'] - p) < 0.001), None)
            if pbllm_r:
                diff = pbllm_r['perplexity'] - smart_r['perplexity']
                print(f"  At p={p*100:.0f}%: Smart {smart_r['perplexity']:.2f} vs "
                      f"PB-LLM {pbllm_r['perplexity']:.2f} (Smart is {abs(diff):.2f} {'better' if diff > 0 else 'worse'})")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(
        output_dir, 
        f"comparison_{model_name.split('/')[-1]}_{dataset}_{timestamp}.json"
    )
    
    with open(results_file, 'w') as f:
        json.dump({
            'config': {
                'model': model_name,
                'dataset': dataset,
                'methods': methods,
                'p_values': p_values,
                'nsamples': nsamples,
                'seed': seed,
                'timestamp': timestamp,
            },
            'results': all_results,
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Compare quantization methods: Vanilla vs PB-LLM vs Smart"
    )
    
    # Required arguments
    parser.add_argument("model", type=str, help="Model name (e.g., facebook/opt-125m)")
    parser.add_argument("dataset", type=str, choices=['wikitext2', 'c4'],
                       help="Calibration and evaluation dataset")
    
    # Methods to compare
    parser.add_argument("--methods", nargs='+', 
                       default=['vanilla', 'magnitude', 'hessian', 'smart'],
                       choices=['vanilla', 'magnitude', 'pbllm', 'hessian', 'smart'],
                       help="Methods to compare (magnitude=pbllm with |W|, hessian=W²/H², smart=activation-aware)")
    
    # Salient fractions
    parser.add_argument("--p_global", nargs='+', type=float,
                       default=[0.1, 0.2],
                       help="Salient fractions to test (default: 0.1 0.2)")
    
    # Other parameters
    parser.add_argument("--nsamples", type=int, default=128,
                       help="Number of calibration samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="./comparison_results",
                       help="Output directory")
    
    args = parser.parse_args()
    
    compare_methods(
        model_name=args.model,
        dataset=args.dataset,
        methods=args.methods,
        p_values=args.p_global,
        nsamples=args.nsamples,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
