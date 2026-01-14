# Smart Binarization: Implementation Details

## Algorithm

### Activation-Aware Saliency Detection

The smart binarization method identifies which weights are most critical to preserve at full precision by considering both the binarization error and how much each weight is actually used during inference.

**Core Formula:**
$$S_{ij} = (\Delta W_{ij})^2 \times a_j$$

Where:
- $S_{ij}$ = saliency score for weight at position (i, j)
- $\Delta W_{ij} = W_{ij} - \alpha_j \times \text{sign}(W_{ij})$ = binarization error
- $\alpha_j = \text{mean}(|W_{:,j}|)$ = column-wise scaling factor
- $a_j = \mathbb{E}[x_j^2]$ = activation energy (collected during calibration)

### Pipeline Steps

1. **Calibration Pass**: Run model on calibration data, collect $a_j = \mathbb{E}[x_j^2]$ per layer
2. **Compute Saliency**: For each layer, compute $S_{ij}$ matrix and layer need $L = \sum S$
3. **Budget Allocation**: Distribute global budget $K$ across layers proportionally to need $L$
4. **Mask Generation**: Select top-$K_l$ weights per layer as salient
5. **Apply Binarization**: Keep salient weights at full precision, binarize rest as $\alpha_j \times \text{sign}(W_{ij})$

---

## Results (OPT-125M on WikiText-2)

| Method | Salient % | Perplexity | vs Vanilla |
|--------|-----------|------------|------------|
| Vanilla | 100% | 27.65 | — |
| **Smart** | **50%** | **34.09** | **+23.3%** |
| Smart | 20% | 507.87 | +1736.6% |
| PB-LLM (magnitude) | 50% | 616.08 | +2127.9% |
| PB-LLM (magnitude) | 20% | 1048.05 | +3690.0% |
| Smart | 10% | 1387.90 | +4919.0% |
| PB-LLM (magnitude) | 10% | 2287.56 | +8172.3% |

### Smart vs PB-LLM Comparison

| Salient % | Smart PPL | PB-LLM PPL | Improvement |
|-----------|-----------|------------|-------------|
| 10% | 1387.90 | 2287.56 | **1.6x better** |
| 20% | 507.87 | 1048.05 | **2.1x better** |
| 50% | 34.09 | 616.08 | **18x better** |

---

## How to Run

### Smart Binarization (Single Run)

```bash
python run_smart_binarization.py <model> <dataset> [options]

# Example: 10% salient weights with evaluation
python run_smart_binarization.py facebook/opt-125m wikitext2 --p_global 0.1 --eval

# Example: 50% salient weights, 256 calibration samples
python run_smart_binarization.py facebook/opt-125m wikitext2 --p_global 0.5 --nsamples 256 --eval
```

**Arguments:**
- `model`: HuggingFace model name (e.g., `facebook/opt-125m`, `meta-llama/Llama-2-7b-hf`)
- `dataset`: Calibration dataset (`wikitext2` or `c4`)
- `--p_global`: Fraction of weights to keep salient (default: 0.1)
- `--nsamples`: Number of calibration samples (default: 128)
- `--eval`: Evaluate perplexity after binarization
- `--output_dir`: Output directory (default: `./outputs/smart`)

### Compare Methods (Vanilla vs PB-LLM vs Smart)

```bash
python compare_methods.py <model> <dataset> [options]

# Example: Compare all methods at 10%, 20%, 50% salient
python compare_methods.py facebook/opt-125m wikitext2 \
    --methods vanilla pbllm smart \
    --p_global 0.1 0.2 0.5

# Example: Compare only smart vs pbllm at 20%
python compare_methods.py facebook/opt-125m wikitext2 \
    --methods pbllm smart \
    --p_global 0.2
```

**Arguments:**
- `--methods`: Methods to compare (`vanilla`, `pbllm`, `smart`)
- `--p_global`: List of salient fractions to test
- `--nsamples`: Calibration samples (default: 128)
- `--output_dir`: Output directory (default: `./comparison_results`)

---

## File Structure

```
smart_saliency/
├── __init__.py
├── activation_collector.py   # Collect a_j during calibration
├── saliency_scorer.py        # Compute S_ij and layer need L
├── budget_allocator.py       # Allocate K_l per layer
└── mask_generator.py         # Generate top-K masks

quant/
└── smart_quantizer.py        # SmartBinaryLinear layer

run_smart_binarization.py     # Main entry point
compare_methods.py            # Comparison framework
```
