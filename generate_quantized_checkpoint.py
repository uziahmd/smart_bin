#!/usr/bin/env python
"""
Generate and save Smart Binarization quantized model checkpoint for comparison.
Uses GPTQ-based algorithm with standard parameters.
"""

import os
import sys
import json
import torch
import argparse
import time
from datetime import datetime

sys.path.insert(0, '/home/uzair/code/smart binarization')

def generate_pb_llm_quantized_checkpoint(
    model_name="facebook/opt-125m",
    dataset="wikitext2",
    quant_method="xnor",
    low_frac=0.8,
    high_bit=8,
    salient_metric="magnitude",
    nsamples=128,
    output_dir="./quantized_checkpoints"
):
    """
    Generate a Smart Binarization quantized checkpoint.
    
    Args:
        model_name: HF model ID
        dataset: Calibration dataset
        quant_method: Quantization method (xnor, sign, etc.)
        low_frac: Fraction of weights to binarize
        high_bit: Bits for salient weights
        salient_metric: How to detect salient weights
        nsamples: Number of calibration samples
        output_dir: Where to save checkpoint
    
    Returns:
        Path to saved checkpoint
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Run GPTQ-PB quantization
    os.chdir('/home/uzair/code/smart binarization/gptq_pb')
    os.makedirs('outputs/mask', exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Generating Smart Binarization Quantized Checkpoint")
    print(f"{'='*70}")
    print(f"Model: {model_name}")
    print(f"Calibration: {dataset} ({nsamples} samples)")
    print(f"Method: {quant_method}, low_frac={low_frac}, high_bit={high_bit}")
    print(f"Metric: {salient_metric}")
    
    cmd = [
        "python", "run.py",
        model_name,
        dataset,
        quant_method,
        f"--nsamples={nsamples}",
        f"--low_frac={low_frac}",
        f"--high_bit={high_bit}",
        f"--salient_metric={salient_metric}",
    ]
    
    print(f"Running: {' '.join(cmd)}\n")
    
    import subprocess
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print(result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr)
        print(f"✗ Quantization failed")
        return None
    
    # Extract perplexity from output
    ppl = None
    for line in result.stdout.split('\n'):
        if "Perplexity:" in line:
            try:
                ppl = float(line.split("Perplexity:")[1].strip())
            except:
                pass
    
    # Create checkpoint metadata
    checkpoint_name = f"{model_name.split('/')[-1]}_pb_llm_{quant_method}_frac{low_frac}_{salient_metric}"
    checkpoint_path = os.path.join(output_dir, checkpoint_name)
    os.makedirs(checkpoint_path, exist_ok=True)
    
    # Save metadata
    metadata = {
        "model_name": model_name,
        "algorithm": "pb_llm_gptq",
        "quantization_params": {
            "method": quant_method,
            "low_frac": low_frac,
            "high_bit": high_bit,
            "salient_metric": salient_metric,
        },
        "calibration": {
            "dataset": dataset,
            "nsamples": nsamples,
        },
        "results": {
            "perplexity": ppl,
        },
        "timestamp": datetime.now().isoformat(),
        "masks_path": os.path.abspath("outputs/mask"),
    }
    
    with open(os.path.join(checkpoint_path, "config.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✓ Checkpoint saved: {checkpoint_path}")
    print(f"✓ Perplexity (wikitext2): {ppl:.2f}")
    print(f"✓ Masks location: {metadata['masks_path']}")
    print(f"{'='*70}\n")
    
    return checkpoint_path, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Smart Binarization quantized checkpoint")
    parser.add_argument("--model", default="facebook/opt-125m", help="Model to quantize")
    parser.add_argument("--dataset", default="wikitext2", help="Calibration dataset")
    parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration samples")
    parser.add_argument("--low-frac", type=float, default=0.8, help="Fraction to binarize")
    parser.add_argument("--high-bit", type=int, default=8, help="Bits for salient weights")
    parser.add_argument("--salient-metric", default="magnitude", help="Saliency metric")
    parser.add_argument("--method", default="xnor", help="Quantization method")
    args = parser.parse_args()
    
    checkpoint_path, metadata = generate_pb_llm_quantized_checkpoint(
        model_name=args.model,
        dataset=args.dataset,
        quant_method=args.method,
        low_frac=args.low_frac,
        high_bit=args.high_bit,
        salient_metric=args.salient_metric,
        nsamples=args.nsamples,
    )
    
    if checkpoint_path:
        print(f"Checkpoint ready for evaluation at: {checkpoint_path}")
