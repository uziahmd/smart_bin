# Smart Binarization: LLM Quantization Framework

A comprehensive framework for developing and evaluating smart binarization strategies for large language models. This project combines GPTQ-based quantization with quantization-aware training (QAT) to achieve extreme low-bit quantization while maintaining model performance.

## 🎯 Project Overview

**Smart Binarization** implements partially-binarized LLM quantization, where a small ratio of salient weights are preserved in higher precision while the majority are binarized to extreme compression. The framework includes:

- ✅ **Vanilla baseline evaluation** (unquantized reference)
- ✅ **Smart Binarization quantized models** (current reference implementation)
- ✅ **Extensible framework** for custom quantization algorithms
- ✅ **Comprehensive evaluation tools** (perplexity, memory, speed)
- ✅ **Automated testing and reporting**

## 📊 Baseline Results (OPT-125M)

| Configuration | Perplexity | Memory | Speed | Status |
|---|---|---|---|---|
| **Vanilla** (baseline) | **28.62** | 250 MB | 396 tok/s | ✓ Reference |
| **Smart Binarization** (80%) | 858.57 | 125 MB | ~400 tok/s | ✓ Working |

**Note:** High PPL at 80% binarization is expected. Performance recovery through lower `low_frac`, hessian saliency, or QAT.

## 🚀 Quick Start

### Prerequisites
```bash
conda activate uzi  # Python 3.12 with all dependencies
```

### Evaluate Models
```bash
cd "/home/uzair/code/smart binarization"

# Vanilla model only
python evaluate_models.py --vanilla

# Generate quantized checkpoint and evaluate
python generate_quantized_checkpoint.py --nsamples 128
python evaluate_models.py --smart-bin

# Compare both models
python evaluate_models.py --compare

# Generate comprehensive comparison report
python final_comparison_report.py
```

### Run Full Quantization Pipeline
```bash
cd gptq_pb

# Standard parameters (80% binary, magnitude saliency)
python run.py facebook/opt-125m wikitext2 xnor \
  --nsamples 128 --low_frac 0.8 --high_bit 8 --salient_metric magnitude

# Better quality (50% binary, hessian saliency)
python run.py facebook/opt-125m wikitext2 xnor \
  --nsamples 128 --low_frac 0.5 --high_bit 8 --salient_metric hessian

# Try 2-bit quantization
python run.py facebook/opt-125m wikitext2 2bit \
  --nsamples 128 --low_frac 0.8 --high_bit 8 --salient_metric magnitude
```

## 📁 Project Structure

```
smart binarization/
├── evaluate_models.py              Main evaluation framework
├── compare_models.py               Model comparison runner
├── final_comparison_report.py       Report generator
├── generate_quantized_checkpoint.py Checkpoint generator
├── download_datasets.py            Dataset pre-cacher
├── test_suite.py                   Automated test suite
├── requirements.txt                Python dependencies
│
├── gptq_pb/                        Smart Binarization implementation
│   ├── run.py                      Main quantization script
│   ├── gptq.py                     LowHighGPT algorithm
│   ├── high_quant.py               High-bit quantizer
│   ├── low_quant.py                Low-bit quantizer
│   └── outputs/mask/               Generated masks
│
├── qat/                            Quantization-aware training
│   ├── run_qat.py
│   └── eval_after_qat.py
│
├── quant/                          Quantizer implementations
│   ├── quantizer.py                Binary/STE layers
│   └── outlier_quantizer.py        Outlier-aware quantizers
│
├── eval_results/                   Evaluation outputs
│   ├── comparison_results_*.json
│   ├── comparison_report_*.txt
│   └── final_comparison_*.txt
│
└── cache/                          Local dataset cache
```

## 🔧 Framework Architecture

### Evaluation Pipeline
```
ModelEvaluator class:
  ├── load_vanilla_model()           Load unquantized model
  ├── load_smart_binarization()      Load quantized checkpoint
  ├── evaluate_perplexity()          Measure on wikitext2/wikitext-103/c4
  ├── evaluate_memory_usage()        GPU/CPU memory profiling
  └── evaluate_inference_speed()     Tokens/second measurement
```

### Supported Models
- ✓ facebook/opt-125m (tested, fast)
- ✓ facebook/opt-1.3b (ready)
- ✓ facebook/opt-6.7b (ready)
- ✓ huggyllama/llama-7b (ready)

### Quantization Methods
- ✓ `xnor` - Binary quantization (XNOR operation)
- ✓ `sign` - Sign-based binary
- ✓ `2bit` - 2-bit quantization
- ✓ `4bit` - 4-bit quantization
- ✓ `no` - No quantization (baseline)

### Saliency Metrics
- ✓ `magnitude` - Weight magnitude ranking (fast)
- ✓ `hessian` - Hessian-based saliency (better quality, slower)

### Datasets
- ✓ **wikitext2** (cached) - 36K train, primary calibration
- ✓ **wikitext-103-v1** (cached) - 1.8M train, thorough evaluation
- ✓ **c4** (auto-download) - Large-scale pretraining corpus

## 💡 How to Add Custom Algorithms

### 1. Create Quantizer
```python
# quant/my_algorithm.py
from torch import nn
from quant.quantizer import BinaryInterface

class MyQuantizer(nn.Module, BinaryInterface):
    def __init__(self, weight, bias):
        super().__init__()
        self.weight = nn.Parameter(weight.data)
        self.bias = nn.Parameter(bias.data) if bias else None
    
    def forward(self, x):
        w = self.quantize_weights()  # Your quantization logic
        return F.linear(x, w, self.bias)
    
    def get_save_weight_dict(self):
        return {"weight": self.weight.data.half().cpu(), "bias": self.bias}
```

### 2. Update Evaluator
```python
# evaluate_models.py - add to load_vanilla_model()
elif config['type'] == 'my_algorithm':
    self.model = load_my_algorithm(config['checkpoint'])
```

### 3. Run Evaluation
```bash
python evaluate_models.py --compare
python final_comparison_report.py
```

Results automatically save to `eval_results/` with JSON and formatted text reports.

## 📈 Performance Optimization Guide

### Improve Quality from 858 PPL

1. **Lower binarization fraction**
   ```bash
   python run.py ... --low_frac 0.5  # Try 50% instead of 80%
   ```

2. **Better saliency detection**
   ```bash
   python run.py ... --salient_metric hessian  # vs magnitude
   ```

3. **Quantization-aware training**
   ```bash
   cd qat
   python run_qat.py facebook/opt-125m wikitext2 xnor
   ```

4. **Higher precision for salient weights**
   ```bash
   python run.py ... --high_bit 16  # vs 8-bit
   ```

5. **Multi-method comparison**
   - Test sign, 2bit, 4bit quantization
   - Compare results side-by-side

## 🧪 Testing

### Quick Test (1 minute)
```bash
python test_suite.py --quick
```

### Standard Test (5 minutes)
```bash
python test_suite.py
```

### Thorough Test (30 minutes)
```bash
python test_suite.py --thorough
```

### Manual Verification
```bash
python evaluate_models.py --vanilla   # Should get PPL ≈ 28.62
python evaluate_models.py --compare   # Compare both
python final_comparison_report.py     # Comprehensive summary
```

## 📚 Key Scripts Reference

### evaluate_models.py
```bash
# Evaluate vanilla model
python evaluate_models.py --vanilla

# Evaluate smart binarization
python evaluate_models.py --smart-bin

# Compare both (default)
python evaluate_models.py --compare

# Options
--model TEXT              Model ID (default: facebook/opt-125m)
--dataset TEXT            Dataset (default: wikitext2)
```

### generate_quantized_checkpoint.py
```bash
python generate_quantized_checkpoint.py \
  --model facebook/opt-125m \
  --dataset wikitext2 \
  --nsamples 128 \
  --low-frac 0.8 \
  --high-bit 8 \
  --salient-metric magnitude
```

### compare_models.py
```bash
python compare_models.py \
  --model facebook/opt-125m \
  --dataset wikitext2
```

### final_comparison_report.py
```bash
python final_comparison_report.py
# Generates comprehensive comparison with insights and next steps
```

## 📋 Development Roadmap

### Phase 1: Framework ✅
- [x] Vanilla baseline established (PPL: 28.62)
- [x] Smart Binarization working (PPL: 858.57 at 80%)
- [x] Evaluation infrastructure complete
- [x] Comparison framework ready

### Phase 2: Custom Algorithm (Next)
- [ ] Design algorithmic improvements
- [ ] Implement quantizer
- [ ] Test and compare vs baselines
- [ ] Optimize hyperparameters

### Phase 3: Scaling & Validation
- [ ] Test on larger models (opt-1.3b, opt-6.7b)
- [ ] Multi-method comparison (sign, 2bit, 4bit vs xnor)
- [ ] Performance benchmarking
- [ ] Paper preparation

## 🔍 Results & Artifacts

### Output Locations
- **Evaluation results**: `eval_results/`
  - `comparison_results_*.json` - Structured metrics (JSON)
  - `comparison_report_*.txt` - Formatted comparison
  - `final_comparison_*.txt` - Comprehensive summary with insights

- **Quantization outputs**: `gptq_pb/outputs/mask/`
  - Saved quantization masks for each layer

- **Cached datasets**: `~/.cache/huggingface/datasets/`
  - Preprocessed for fast loading

### Parse Results
```python
import json
with open('eval_results/comparison_results_*.json') as f:
    results = json.load(f)
    print(results['vanilla']['perplexity'])
    print(results['smart_binarization']['memory'])
```

## 💾 Environment & Dependencies

### Python Environment
```bash
conda activate uzi          # Python 3.12.12
pip install -r requirements.txt
```

### Key Dependencies
- **torch** >= 2.0.0
- **transformers** >= 4.30.0
- **datasets** >= 2.10.0
- **lm-eval** >= 0.4.0
- **bitsandbytes** >= 0.41.0

## 🤝 Contributing

To implement a new quantization algorithm:

1. Create quantizer class in `quant/`
2. Update `evaluate_models.py` with model type
3. Run evaluations with framework
4. Compare against baselines
5. Document methodology and results

## 📖 References

- **Original PB-LLM Paper**: [arxiv.org/abs/2310.00034](https://arxiv.org/abs/2310.00034)
  - Authors: Yuzhang Shang, Zhihang Yuan, Qiang Wu, Zhen Dong
  - Partially-Binarized LLMs via post-training quantization (GPTQ) and QAT

- **GPTQ Method**: General-Purpose Quantization for Large Language Models
- **Quantization-aware Training (QAT)**: For performance recovery

## ✨ Status

**Framework is complete and ready for algorithm development!**

- All baseline metrics locked
- Evaluation infrastructure validated
- Datasets cached for reproducibility
- Ready to implement and test custom algorithms

---

**Last Updated:** January 14, 2026  
**Project Status:** 🟢 Ready for development
