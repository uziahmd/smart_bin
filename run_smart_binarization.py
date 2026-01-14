#!/usr/bin/env python
"""
Smart Binarization: Main entry point for activation-aware quantization.

This script implements the full pipeline:
1. Load model and calibration data
2. Collect activation energies
3. Compute saliency and allocate budgets
4. Generate masks and apply binarization
5. Evaluate perplexity

Usage:
    python run_smart_binarization.py facebook/opt-125m wikitext2 \
        --p_global 0.1 --nsamples 128 --eval
"""

import os
import sys
import argparse
import time
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from smart_saliency import (
    ActivationCollector,
    compute_all_saliency,
    allocate_budgets,
    generate_masks,
)
from quant.smart_quantizer import apply_smart_binarization, get_binarization_stats


def get_model(model_name: str):
    """Load model from HuggingFace."""
    print(f"Loading model: {model_name}")
    
    if "opt" in model_name.lower():
        from transformers import OPTForCausalLM
        model = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        model.seqlen = model.config.max_position_embeddings
    elif "llama" in model_name.lower():
        from transformers import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        model.seqlen = 2048
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        model.seqlen = getattr(model.config, 'max_position_embeddings', 2048)
    
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    return model


def get_calibration_data(dataset: str, model_name: str, nsamples: int, seed: int = 42):
    """Load calibration data."""
    print(f"Loading calibration data: {dataset} ({nsamples} samples)")
    
    # Import from gptq_pb
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
    print("Evaluating perplexity...")
    
    model.eval()
    model.to(device)
    
    testenc = testenc.input_ids if hasattr(testenc, 'input_ids') else testenc
    nsamples = testenc.shape[1] // seqlen
    
    nlls = []
    with torch.no_grad():
        for i in range(nsamples):
            batch = testenc[:, i*seqlen:(i+1)*seqlen].to(device)
            outputs = model(batch)
            logits = outputs.logits
            
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            nlls.append(loss.item())
    
    ppl = torch.exp(torch.tensor(nlls).mean()).item()
    return ppl


def run_smart_binarization(
    model_name: str,
    dataset: str,
    p_global: float = 0.1,
    nsamples: int = 128,
    seed: int = 42,
    device: str = "cuda:0",
    skip_patterns: list = None,
    min_frac: float = 0.0001,
    max_frac: float = 0.9,
    save_masks: bool = True,
    eval_ppl: bool = True,
    output_dir: str = "./outputs/smart"
):
    """
    Run the complete smart binarization pipeline.
    
    Returns:
        results: Dict with model, masks, stats, and optionally perplexity
    """
    skip_patterns = skip_patterns or ['embed', 'lm_head', 'head', 'norm', 'layernorm', 'ln_']
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*70)
    print("SMART BINARIZATION PIPELINE")
    print("="*70)
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset}")
    print(f"Salient fraction: {p_global*100:.1f}%")
    print(f"Calibration samples: {nsamples}")
    print("="*70)
    
    # Step 1: Load model
    start_time = time.time()
    model = get_model(model_name)
    model.to(device)
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.1f}s")
    
    # Step 2: Load calibration data
    trainloader, testenc, tokenizer = get_calibration_data(dataset, model_name, nsamples, seed)
    
    # Step 3: Collect activation energies
    print("\n" + "-"*70)
    print("Step 1: Collecting activation energies...")
    print("-"*70)
    
    collector = ActivationCollector(model, skip_patterns=skip_patterns)
    activation_energies = collector.collect(trainloader, device=device)
    
    print(f"Collected energies for {len(activation_energies)} layers")
    
    # Step 4: Compute saliency
    print("\n" + "-"*70)
    print("Step 2: Computing saliency scores...")
    print("-"*70)
    
    S_dict, L_dict, alpha_dict = compute_all_saliency(model, activation_energies, skip_patterns)
    
    print(f"Computed saliency for {len(S_dict)} layers")
    
    # Analyze layer needs
    total_L = sum(L_dict.values())
    print("\nLayer need distribution (top 10):")
    sorted_layers = sorted(L_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    for name, L in sorted_layers:
        print(f"  {name}: {L:.2e} ({L/total_L*100:.1f}%)")
    
    # Step 5: Allocate budgets
    print("\n" + "-"*70)
    print("Step 3: Allocating budgets...")
    print("-"*70)
    
    sizes_dict = {}
    for name, module in model.named_modules():
        if name in S_dict:
            sizes_dict[name] = module.weight.numel()
    
    K_dict, K_total = allocate_budgets(L_dict, sizes_dict, p_global, min_frac, max_frac)
    N_total = sum(sizes_dict.values())
    
    print(f"Total weights: {N_total:,}")
    print(f"Salient weights: {K_total:,} ({K_total/N_total*100:.2f}%)")
    print(f"Binarized weights: {N_total - K_total:,} ({(N_total-K_total)/N_total*100:.2f}%)")
    
    # Step 6: Generate masks
    print("\n" + "-"*70)
    print("Step 4: Generating masks...")
    print("-"*70)
    
    mask_dict = generate_masks(S_dict, K_dict)
    
    # Verify masks
    for name, mask in mask_dict.items():
        expected = K_dict[name]
        actual = mask.sum().item()
        if expected != actual:
            print(f"Warning: Mask mismatch for {name}: expected {expected}, got {actual}")
    
    print(f"Generated masks for {len(mask_dict)} layers")
    
    # Save masks
    if save_masks:
        mask_path = os.path.join(output_dir, f"masks_{model_name.split('/')[-1]}_p{p_global}.pt")
        torch.save({
            'mask_dict': {k: v.cpu() for k, v in mask_dict.items()},
            'alpha_dict': {k: v.cpu() for k, v in alpha_dict.items()},
            'K_dict': K_dict,
            'L_dict': L_dict,
            'config': {
                'model': model_name,
                'dataset': dataset,
                'p_global': p_global,
                'nsamples': nsamples,
            }
        }, mask_path)
        print(f"Saved masks to {mask_path}")
    
    # Step 7: Apply binarization
    print("\n" + "-"*70)
    print("Step 5: Applying smart binarization...")
    print("-"*70)
    
    model = apply_smart_binarization(model, mask_dict, alpha_dict, use_ste=False, inplace=True)
    
    stats = get_binarization_stats(model)
    print(f"Global salient ratio: {stats['global_salient_ratio']*100:.2f}%")
    
    # Step 8: Evaluate
    results = {
        'model_name': model_name,
        'dataset': dataset,
        'p_global': p_global,
        'nsamples': nsamples,
        'K_total': K_total,
        'N_total': N_total,
        'actual_salient_ratio': K_total / N_total,
        'num_layers': len(mask_dict),
        'stats': stats,
    }
    
    if eval_ppl:
        print("\n" + "-"*70)
        print("Step 6: Evaluating perplexity...")
        print("-"*70)
        
        ppl = evaluate_perplexity(model, testenc, device, model.seqlen)
        results['perplexity'] = ppl
        
        print(f"\nPerplexity: {ppl:.2f}")
    
    # Save results
    results_path = os.path.join(output_dir, f"results_{model_name.split('/')[-1]}_p{p_global}.json")
    with open(results_path, 'w') as f:
        # Convert non-serializable values
        serializable = {k: v for k, v in results.items() if k != 'stats'}
        serializable['stats'] = {
            'global_salient_ratio': stats['global_salient_ratio'],
            'total_salient': stats['total_salient'],
            'total_weights': stats['total_weights'],
        }
        json.dump(serializable, f, indent=2)
    print(f"Saved results to {results_path}")
    
    print("\n" + "="*70)
    print("SMART BINARIZATION COMPLETE")
    print("="*70)
    
    return model, results


def main():
    parser = argparse.ArgumentParser(description="Smart Binarization with Activation-Aware Saliency")
    
    # Required arguments
    parser.add_argument("model", type=str, help="Model name (e.g., facebook/opt-125m)")
    parser.add_argument("dataset", type=str, choices=['wikitext2', 'c4'], 
                       help="Calibration dataset")
    
    # Binarization parameters
    parser.add_argument("--p_global", type=float, default=0.1,
                       help="Fraction of weights to keep salient (default: 0.1 = 10%%)")
    parser.add_argument("--nsamples", type=int, default=128,
                       help="Number of calibration samples (default: 128)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Constraints
    parser.add_argument("--min_frac", type=float, default=0.0001,
                       help="Minimum salient fraction per layer")
    parser.add_argument("--max_frac", type=float, default=0.9,
                       help="Maximum salient fraction per layer")
    
    # Evaluation
    parser.add_argument("--eval", action="store_true", help="Evaluate perplexity after binarization")
    parser.add_argument("--no_save", action="store_true", help="Don't save masks")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="./outputs/smart",
                       help="Output directory")
    
    args = parser.parse_args()
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    model, results = run_smart_binarization(
        model_name=args.model,
        dataset=args.dataset,
        p_global=args.p_global,
        nsamples=args.nsamples,
        seed=args.seed,
        device=device,
        min_frac=args.min_frac,
        max_frac=args.max_frac,
        save_masks=not args.no_save,
        eval_ppl=args.eval,
        output_dir=args.output_dir,
    )
    
    return results


if __name__ == "__main__":
    main()
