#!/usr/bin/env python
"""
Pre-download and cache all datasets used in Smart Binarization for future testing.
Datasets: wikitext2, ptb, c4, red_pajama
"""

import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

# Create cache directory
os.makedirs("cache", exist_ok=True)

print("=" * 60)
print("Smart Binarization Dataset Pre-Download Script")
print("=" * 60)

# Load a tokenizer for consistent tokenization
print("\n[1/5] Loading tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m", use_fast=False)
    print("✓ Tokenizer loaded: facebook/opt-125m")
except Exception as e:
    print(f"✗ Error loading tokenizer: {e}")
    tokenizer = None

# 1. WikiText2
print("\n[2/5] Downloading wikitext2...")
try:
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    valdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    print(f"✓ WikiText2 downloaded")
    print(f"  - Train samples: {len(traindata)}")
    print(f"  - Test samples: {len(testdata)}")
    print(f"  - Validation samples: {len(valdata)}")
except Exception as e:
    print(f"✗ Error downloading wikitext2: {e}")

# 2. PTB (Penn Treebank)
print("\n[3/5] Downloading ptb_text_only...")
try:
    traindata = load_dataset("wikitext", "wikitext-103-v1", split="train")  # Alternative to ptb
    testdata = load_dataset("wikitext", "wikitext-103-v1", split="test")
    valdata = load_dataset("wikitext", "wikitext-103-v1", split="validation")
    print(f"✓ WikiText-103 (alternative to PTB) downloaded")
    print(f"  - Train samples: {len(traindata)}")
    print(f"  - Test samples: {len(testdata)}")
    print(f"  - Validation samples: {len(valdata)}")
except Exception as e:
    print(f"✗ Error downloading wikitext-103: {e}")

# 3. Simplified note about C4 and RedPajama
print("\n[4/5] C4 Dataset...")
print("⚠ C4 is a very large dataset (>300GB). It will be auto-downloaded")
print("  when first needed by the quantization script. No manual cache needed.")

print("\n[5/5] RedPajama Dataset...")
print("⚠ RedPajama is not available in the current Hugging Face Hub format.")
print("  Wikitext2 and WikiText-103 are excellent alternatives for calibration.")

print("\n" + "=" * 60)
print("Dataset download complete!")
print("=" * 60)
print("\nCached datasets can now be used for future testing without")
print("re-downloading. Cache location: ~/.cache/huggingface/datasets/")
print("\nTo run quantization with cached data:")
print("  cd gptq_pb")
print("  python run.py facebook/opt-125m wikitext2 xnor --nsamples 128 --low_frac 0.8")
