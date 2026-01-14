#!/usr/bin/env python
"""
Comprehensive test and benchmark script for PB-LLM quantization pipeline.
Tests multiple models, datasets, and configurations for comparisons.
"""

import os
import json
import time
import argparse
import subprocess
import sys
from datetime import datetime

# Test configurations
MODELS = [
    "facebook/opt-125m",
    # "facebook/opt-1.3b",  # Uncomment for larger tests
    # "huggyllama/llama-7b",  # Requires HF auth
]

DATASETS = [
    "wikitext2",
    # "ptb",  # Has dataset API issues
    # "c4",  # Very large
]

QUANT_METHODS = [
    "xnor",
    "sign",
    "no",
]

SALIENT_METRICS = [
    "magnitude",
    # "hessian",  # More expensive
]

LOW_FRACS = [0.5, 0.8, 0.9]

NSAMPLES = 16  # Small for quick tests; increase to 128 for thorough evaluation


def run_test(model, dataset, quant_method, low_frac, salient_metric, nsamples):
    """Run a single GPTQ-PB quantization test."""
    test_id = f"{model.split('/')[-1]}_{dataset}_{quant_method}_frac{low_frac}_{salient_metric}"
    print(f"\n{'='*70}")
    print(f"Test: {test_id}")
    print(f"Model: {model}")
    print(f"Dataset: {dataset}, Samples: {nsamples}")
    print(f"Method: {quant_method}, Low Frac: {low_frac}, Metric: {salient_metric}")
    print(f"{'='*70}")
    
    # Ensure output directories exist
    os.makedirs("/home/uzair/code/PB-LLM/gptq_pb/outputs/mask", exist_ok=True)
    
    cmd = [
        "python",
        "gptq_pb/run.py",
        model,
        dataset,
        quant_method,
        f"--nsamples={nsamples}",
        f"--low_frac={low_frac}",
        "--high_bit=8",
        f"--salient_metric={salient_metric}",
    ]
    
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd="/home/uzair/code/PB-LLM",
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        elapsed = time.time() - start_time
        
        # Extract perplexity from output
        ppl = None
        for line in result.stdout.split('\n'):
            if "Perplexity:" in line:
                try:
                    ppl = float(line.split("Perplexity:")[1].strip())
                except:
                    pass
        
        if result.returncode == 0:
            print(f"✓ PASSED (Time: {elapsed:.2f}s, PPL: {ppl})")
            return {
                "status": "pass",
                "time": elapsed,
                "perplexity": ppl,
                "test_id": test_id,
            }
        else:
            print(f"✗ FAILED")
            print(f"Error output:\n{result.stderr[-500:]}")  # Last 500 chars
            return {
                "status": "fail",
                "time": elapsed,
                "test_id": test_id,
                "error": result.stderr[-200:],
            }
    except subprocess.TimeoutExpired:
        print(f"✗ TIMEOUT (>600s)")
        return {
            "status": "timeout",
            "test_id": test_id,
        }
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return {
            "status": "error",
            "test_id": test_id,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="PB-LLM Comprehensive Test Suite")
    parser.add_argument("--quick", action="store_true", help="Run quick test (1 config)")
    parser.add_argument("--thorough", action="store_true", help="Run thorough test (all configs)")
    parser.add_argument("--nsamples", type=int, default=NSAMPLES, help="Number of calibration samples")
    args = parser.parse_args()
    
    # Determine test scope
    if args.quick:
        models = MODELS[:1]
        datasets = DATASETS[:1]
        quant_methods = ["xnor"]
        salient_metrics = ["magnitude"]
        low_fracs = [0.8]
    elif args.thorough:
        models = MODELS
        datasets = DATASETS
        quant_methods = QUANT_METHODS
        salient_metrics = SALIENT_METRICS
        low_fracs = LOW_FRACS
    else:
        models = MODELS[:1]
        datasets = DATASETS[:1]
        quant_methods = ["xnor"]
        salient_metrics = ["magnitude"]
        low_fracs = [0.8]
    
    results = []
    total_tests = len(models) * len(datasets) * len(quant_methods) * len(salient_metrics) * len(low_fracs)
    test_num = 0
    
    print(f"\n{'='*70}")
    print(f"PB-LLM Quantization Test Suite")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total tests to run: {total_tests}")
    print(f"{'='*70}\n")
    
    for model in models:
        for dataset in datasets:
            for quant_method in quant_methods:
                for salient_metric in salient_metrics:
                    for low_frac in low_fracs:
                        test_num += 1
                        print(f"[{test_num}/{total_tests}] ", end="")
                        result = run_test(
                            model, dataset, quant_method, low_frac,
                            salient_metric, args.nsamples
                        )
                        results.append(result)
    
    # Summary
    print(f"\n\n{'='*70}")
    print(f"Test Summary")
    print(f"{'='*70}")
    
    passed = sum(1 for r in results if r["status"] == "pass")
    failed = sum(1 for r in results if r["status"] == "fail")
    timeout = sum(1 for r in results if r["status"] == "timeout")
    error = sum(1 for r in results if r["status"] == "error")
    
    print(f"Passed:  {passed}/{total_tests}")
    print(f"Failed:  {failed}/{total_tests}")
    print(f"Timeout: {timeout}/{total_tests}")
    print(f"Error:   {error}/{total_tests}")
    
    # Save results
    results_file = f"/home/uzair/code/PB-LLM/test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    # Print perplexity comparison
    print(f"\n{'='*70}")
    print(f"Perplexity Results (lower is better)")
    print(f"{'='*70}")
    for r in results:
        if r["status"] == "pass" and r.get("perplexity"):
            print(f"{r['test_id']}: PPL = {r['perplexity']:.2f}")
    
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
