#!/usr/bin/env python3
"""Evaluate QAT-trained models on perplexity."""

import sys
sys.path.append("..")
sys.path.append(".")

import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datautils import get_loaders
from tqdm import tqdm
import math


def evaluate_perplexity(model, testenc, device, seqlen=2048):
    """Evaluate perplexity on test data."""
    model.eval()
    nsamples = testenc.numel() // seqlen
    
    nlls = []
    with torch.no_grad():
        for i in tqdm(range(nsamples), desc="Evaluating"):
            batch = testenc[:, (i * seqlen):((i + 1) * seqlen)].to(device)
            
            with torch.cuda.amp.autocast():
                outputs = model(batch)
                lm_logits = outputs.logits
            
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:].contiguous()
            
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            nlls.append(loss.float().item())
    
    ppl = math.exp(sum(nlls) / len(nlls))
    return ppl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str, help="Path to QAT model")
    parser.add_argument("--dataset", type=str, default="wikitext2")
    parser.add_argument("--seqlen", type=int, default=2048)
    args = parser.parse_args()
    
    print(f"Loading model from: {args.model_path}")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    device = next(model.parameters()).device
    model.seqlen = args.seqlen
    
    # Get test data
    cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
    _, testenc = get_loaders(
        args.dataset,
        nsamples=128,
        seed=42,
        seqlen=args.seqlen,
        model=args.model_path,
        cache_dir=cache_dir
    )
    
    # Evaluate
    ppl = evaluate_perplexity(model, testenc.input_ids, device, args.seqlen)
    print(f"\n{'='*50}")
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset}")
    print(f"Perplexity: {ppl:.2f}")
    print(f"{'='*50}")
    
    return ppl


if __name__ == "__main__":
    main()
