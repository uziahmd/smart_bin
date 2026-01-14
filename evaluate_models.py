#!/usr/bin/env python
"""
Unified evaluation framework for comparing:
1. Vanilla (unquantized) models
2. Smart Binarization quantized models
3. Future custom algorithms

Metrics: Perplexity on calibration datasets, downstream task accuracy
"""

import os
import json
import time
import argparse
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm


class ModelEvaluator:
    """Unified evaluator for any model type."""
    
    def __init__(self, model_name, model_type="vanilla", device="cuda:0"):
        """
        Args:
            model_name: HF model ID or path
            model_type: 'vanilla', 'pb_llm_quantized', or custom
            device: torch device
        """
        self.model_name = model_name
        self.model_type = model_type
        self.device = device
        self.model = None
        self.tokenizer = None
        self.results = defaultdict(dict)
        
    def load_vanilla_model(self):
        """Load unquantized model from Hugging Face."""
        print(f"Loading vanilla model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map=self.device,
        )
        self.model.eval()
        print(f"✓ Loaded: {self.model_name}")
        return self.model
    
    def load_pb_llm_quantized(self, quantized_checkpoint_path):
        """Load Smart Binarization quantized model from checkpoint."""
        print(f"Loading Smart Binarization quantized model from: {quantized_checkpoint_path}")
        # Load base model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map=self.device,
        )
        
        # Load quantized layers
        try:
            from utils import load_bnn
            self.model = load_bnn(self.model, quantized_checkpoint_path)
            print(f"✓ Loaded quantized model from: {quantized_checkpoint_path}")
        except Exception as e:
            print(f"⚠ Could not load quantized checkpoint: {e}")
            print("  Using vanilla model for comparison baseline")
        
        self.model.eval()
        return self.model
    
    @torch.no_grad()
    def evaluate_perplexity(self, dataset_name="wikitext2", seqlen=2048, num_samples=None):
        """
        Evaluate perplexity on a dataset.
        
        Args:
            dataset_name: 'wikitext2', 'wikitext-103-v1', or 'c4'
            seqlen: Sequence length for evaluation
            num_samples: Number of samples to evaluate (None = all)
        
        Returns:
            PPL: Perplexity value
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        print(f"\nEvaluating perplexity on {dataset_name} (seqlen={seqlen})...")
        
        # Load dataset
        if dataset_name == "wikitext2":
            testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            testenc = self.tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
        elif dataset_name == "wikitext-103-v1":
            testdata = load_dataset("wikitext", "wikitext-103-v1", split="test")
            testenc = self.tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        testenc = testenc.input_ids
        nsamples = testenc.numel() // seqlen
        if num_samples and num_samples < nsamples:
            nsamples = num_samples
        
        use_cache = self.model.config.use_cache
        self.model.config.use_cache = False
        
        nlls = []
        
        for i in tqdm(range(nsamples), desc=f"PPL {dataset_name}"):
            batch = testenc[:, (i * seqlen) : ((i + 1) * seqlen)].to(self.device)
            
            if batch.shape[1] < seqlen:
                continue
            
            outputs = self.model(batch)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:]
            
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            neg_log_likelihood = loss.float() * seqlen
            nlls.append(neg_log_likelihood)
        
        self.model.config.use_cache = use_cache
        
        ppl = torch.exp(torch.stack(nlls).sum() / (len(nlls) * seqlen)).item()
        self.results[dataset_name]["perplexity"] = ppl
        print(f"✓ {dataset_name} PPL: {ppl:.2f}")
        return ppl
    
    @torch.no_grad()
    def evaluate_memory_usage(self):
        """Get model memory usage."""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Estimate memory
        param_mem = total_params * 2 / (1024**3)  # float16 = 2 bytes
        
        self.results["memory"] = {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "estimated_memory_gb": param_mem,
        }
        
        print(f"Memory: {param_mem:.2f}GB ({total_params/1e6:.1f}M params)")
        return self.results["memory"]
    
    @torch.no_grad()
    def evaluate_inference_speed(self, prompt="Hello, my name is", num_tokens=50):
        """Evaluate inference speed."""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        print(f"Evaluating inference speed ({num_tokens} tokens)...")
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Warmup
        with torch.no_grad():
            _ = self.model.generate(inputs.input_ids, max_new_tokens=10)
        torch.cuda.synchronize()
        
        # Timed run
        start = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=num_tokens,
                do_sample=False,
            )
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        tokens_per_sec = num_tokens / elapsed
        self.results["inference"] = {
            "time_sec": elapsed,
            "tokens_per_sec": tokens_per_sec,
            "num_tokens": num_tokens,
        }
        
        print(f"✓ Inference: {tokens_per_sec:.2f} tokens/sec ({elapsed:.2f}s for {num_tokens} tokens)")
        return self.results["inference"]
    
    def get_summary(self):
        """Return results dictionary."""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "timestamp": datetime.now().isoformat(),
            "results": dict(self.results),
        }


def run_comparison(models_config, output_dir="./eval_results"):
    """
    Run comprehensive comparison of multiple models/algorithms.
    
    Args:
        models_config: List of dicts with keys:
            - name: model identifier
            - model_name: HF model ID
            - type: 'vanilla', 'pb_llm_quantized', etc.
            - checkpoint: optional checkpoint path
        output_dir: where to save results
    """
    
    os.makedirs(output_dir, exist_ok=True)
    all_results = []
    
    for config in models_config:
        print(f"\n{'='*70}")
        print(f"Evaluating: {config['name']} ({config['type']})")
        print(f"{'='*70}")
        
        evaluator = ModelEvaluator(
            model_name=config['model_name'],
            model_type=config['type'],
            device="cuda:0"
        )
        
        # Load model
        if config['type'] == 'vanilla':
            evaluator.load_vanilla_model()
        elif config['type'] == 'pb_llm_quantized':
            checkpoint = config.get('checkpoint')
            if checkpoint:
                evaluator.load_pb_llm_quantized(checkpoint)
            else:
                evaluator.load_vanilla_model()
        else:
            raise ValueError(f"Unknown model type: {config['type']}")
        
        # Run evaluations
        try:
            evaluator.evaluate_memory_usage()
            evaluator.evaluate_perplexity("wikitext2", num_samples=50)
            evaluator.evaluate_inference_speed()
        except Exception as e:
            print(f"✗ Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
        
        all_results.append(evaluator.get_summary())
        
        # Free memory
        del evaluator
        torch.cuda.empty_cache()
    
    # Save results
    results_file = os.path.join(
        output_dir,
        f"comparison_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {results_file}")
    print(f"{'='*70}")
    
    # Print summary table
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Model':<25} {'Type':<20} {'PPL (wikitext2)':<15} {'Memory (GB)':<15}")
    print("-"*70)
    
    for result in all_results:
        name = result['model_name'][:25]
        mtype = result['model_type']
        ppl = result['results'].get('wikitext2', {}).get('perplexity', 'N/A')
        mem = result['results'].get('memory', {}).get('estimated_memory_gb', 'N/A')
        
        if isinstance(ppl, float):
            ppl_str = f"{ppl:.2f}"
        else:
            ppl_str = str(ppl)
        
        if isinstance(mem, float):
            mem_str = f"{mem:.2f}"
        else:
            mem_str = str(mem)
        
        print(f"{name:<25} {mtype:<20} {ppl_str:<15} {mem_str:<15}")
    
    print("="*70)
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate and compare models")
    parser.add_argument("--vanilla", action="store_true", help="Test vanilla model only")
    parser.add_argument("--smart-bin", action="store_true", help="Test Smart Binarization quantized model only")
    parser.add_argument("--compare", action="store_true", help="Compare vanilla vs Smart Binarization")
    parser.add_argument("--model", type=str, default="facebook/opt-125m", help="Model to evaluate")
    args = parser.parse_args()
    
    if not (args.vanilla or args.smart_bin or args.compare):
        args.compare = True  # Default to comparison
    
    models_to_eval = []
    
    if args.vanilla or args.compare:
        models_to_eval.append({
            "name": f"{args.model.split('/')[-1]} (Vanilla)",
            "model_name": args.model,
            "type": "vanilla",
        })
    
    if args.smart_bin or args.compare:
        models_to_eval.append({
            "name": f"{args.model.split('/')[-1]} (Smart Binarization)",
            "model_name": args.model,
            "type": "smart_binarization_quantized",
            "checkpoint": None,  # Will implement checkpoint loading later
        })
    
    print(f"Models to evaluate: {len(models_to_eval)}")
    for m in models_to_eval:
        print(f"  - {m['name']} ({m['type']})")
    
    all_results = run_comparison(models_to_eval)
