#!/usr/bin/env python
"""
Comprehensive comparison: Vanilla vs PB-LLM Quantized Models

Generates a detailed comparison report with metrics:
- Perplexity (various datasets)
- Compression ratio
- Inference speed
- Memory usage
"""

import os
import json
import torch
import argparse
from datetime import datetime
from pathlib import Path

from evaluate_models import ModelEvaluator, run_comparison


def create_comparison_report(results, output_file=None):
    """
    Create a detailed comparison report from evaluation results.
    
    Args:
        results: List of result dicts from run_comparison()
        output_file: Optional file to save report to
    
    Returns:
        Formatted report string
    """
    
    report = []
    report.append("\n" + "="*80)
    report.append("MODEL COMPARISON REPORT")
    report.append("="*80)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Extract metrics
    models = {}
    for result in results:
        name = result['model_name']
        mtype = result['model_type']
        key = f"{name}_{mtype}"
        models[key] = result['results']
    
    # Comparison table
    report.append("PERPLEXITY (Lower is Better)")
    report.append("-" * 80)
    report.append(f"{'Model':<30} {'Type':<25} {'PPL':<15}")
    report.append("-" * 80)
    
    ppl_vanilla = None
    ppl_quantized = None
    
    for result in results:
        name = result['model_name'].split('/')[-1]
        mtype = result['model_type']
        ppl = result['results'].get('wikitext2', {}).get('perplexity')
        
        if ppl:
            if mtype == 'vanilla':
                ppl_vanilla = ppl
            elif mtype == 'pb_llm_quantized':
                ppl_quantized = ppl
            
            report.append(f"{name:<30} {mtype:<25} {ppl:<15.2f}")
    
    if ppl_vanilla and ppl_quantized:
        ppl_degradation = ((ppl_quantized - ppl_vanilla) / ppl_vanilla) * 100
        report.append("-" * 80)
        report.append(f"\nPPL Degradation: {ppl_degradation:+.2f}% (quantized vs vanilla)")
    
    # Memory comparison
    report.append("\n\nMEMORY USAGE")
    report.append("-" * 80)
    report.append(f"{'Model':<30} {'Type':<25} {'Memory (GB)':<15}")
    report.append("-" * 80)
    
    mem_vanilla = None
    mem_quantized = None
    
    for result in results:
        name = result['model_name'].split('/')[-1]
        mtype = result['model_type']
        mem = result['results'].get('memory', {}).get('estimated_memory_gb')
        
        if mem:
            if mtype == 'vanilla':
                mem_vanilla = mem
            elif mtype == 'pb_llm_quantized':
                mem_quantized = mem
            
            report.append(f"{name:<30} {mtype:<25} {mem:<15.3f}")
    
    if mem_vanilla and mem_quantized:
        compression_ratio = mem_vanilla / mem_quantized if mem_quantized > 0 else float('inf')
        compression_pct = ((mem_vanilla - mem_quantized) / mem_vanilla) * 100
        report.append("-" * 80)
        report.append(f"\nCompression: {compression_pct:.1f}% reduction ({compression_ratio:.2f}x smaller)")
    
    # Inference speed
    report.append("\n\nINFERENCE SPEED (Tokens per second)")
    report.append("-" * 80)
    report.append(f"{'Model':<30} {'Type':<25} {'Tokens/Sec':<15}")
    report.append("-" * 80)
    
    speed_vanilla = None
    speed_quantized = None
    
    for result in results:
        name = result['model_name'].split('/')[-1]
        mtype = result['model_type']
        speed = result['results'].get('inference', {}).get('tokens_per_sec')
        
        if speed:
            if mtype == 'vanilla':
                speed_vanilla = speed
            elif mtype == 'pb_llm_quantized':
                speed_quantized = speed
            
            report.append(f"{name:<30} {mtype:<25} {speed:<15.2f}")
    
    if speed_vanilla and speed_quantized:
        speedup = (speed_quantized - speed_vanilla) / speed_vanilla * 100
        report.append("-" * 80)
        report.append(f"\nSpeedup: {speedup:+.2f}% (quantized vs vanilla)")
    
    # Summary
    report.append("\n" + "="*80)
    report.append("SUMMARY")
    report.append("="*80)
    
    if ppl_vanilla and ppl_quantized:
        report.append(f"Vanilla PPL:     {ppl_vanilla:.2f}")
        report.append(f"Quantized PPL:   {ppl_quantized:.2f}")
        report.append(f"Degradation:     {ppl_degradation:+.2f}%")
    
    if mem_vanilla and mem_quantized:
        report.append(f"\nVanilla Memory:  {mem_vanilla:.3f} GB")
        report.append(f"Quantized Memory: {mem_quantized:.3f} GB")
        report.append(f"Compression:     {compression_pct:.1f}%")
    
    report.append("\n" + "="*80 + "\n")
    
    report_str = "\n".join(report)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_str)
        print(f"Report saved to: {output_file}")
    
    return report_str


def main():
    parser = argparse.ArgumentParser(description="Compare vanilla vs Smart Binarization quantized models")
    parser.add_argument("--model", default="facebook/opt-125m", help="Model to evaluate")
    parser.add_argument("--gen-quantized", action="store_true", help="Generate quantized checkpoint first")
    parser.add_argument("--nsamples", type=int, default=128, help="Calibration samples for quantization")
    args = parser.parse_args()
    
    # Generate quantized checkpoint if needed
    checkpoint_path = None
    if args.gen_quantized:
        print("Generating Smart Binarization quantized checkpoint...")
        import subprocess
        result = subprocess.run([
            "python", "generate_quantized_checkpoint.py",
            f"--model={args.model}",
            f"--nsamples={args.nsamples}",
        ], cwd="/home/uzair/code/smart binarization")
        
        checkpoint_path = f"./quantized_checkpoints/{args.model.split('/')[-1]}_pb_llm_xnor_frac0.8_magnitude"
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint not found at {checkpoint_path}")
            checkpoint_path = None
    else:
        # Try to find existing checkpoint
        checkpoint_dir = "quantized_checkpoints"
        if os.path.exists(checkpoint_dir):
            for item in os.listdir(checkpoint_dir):
                if "pb_llm" in item:
                    checkpoint_path = os.path.join(checkpoint_dir, item)
                    break
    
    # Prepare model configurations
    models_config = [
        {
            "name": f"{args.model.split('/')[-1]} (Vanilla)",
            "model_name": args.model,
            "type": "vanilla",
        },
        {
            "name": f"{args.model.split('/')[-1]} (Smart Binarization)",
            "model_name": args.model,
            "type": "pb_llm_quantized",
            "checkpoint": checkpoint_path,
        }
    ]
    
    # Run evaluation
    print(f"\nEvaluating models...")
    results = run_comparison(models_config)
    
    # Generate and print report
    report = create_comparison_report(
        results,
        output_file=f"./eval_results/comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    print(report)


if __name__ == "__main__":
    main()
