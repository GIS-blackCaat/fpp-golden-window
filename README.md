# FPP — Fieldcell Processing Protocol

**Loss Is Not Enough.** FPP is a 4-dimensional diagnostic framework that measures what loss cannot see: structural order, information capacity, layer differentiation, and metric honesty.

## Quick Start

```bash
pip install -r requirements.txt
python fpp_health.py --model Qwen/Qwen2.5-0.5B-Instruct
```

## ⏱️ 5-Minute Diagnostic Guide

New to FPP? Start here: **[5MIN_GUIDE.md](5MIN_GUIDE.md)** — load model → run FPP → read 4 indicators → decide what to do next. With decision tree and β safety quick-reference table.

## What You Get

- Architecture fingerprint (activation, attention, layers)
- FPP baseline (GS, MI, Phase, DC)
- Health status vs family norms
- Safe β range for momentum injection
- Optimization recommendations

## Examples

```bash
# Programmatic usage — call FPP from your own code
python example.py --model Qwen/Qwen2.5-0.5B-Instruct

# Fine-tuning monitor — Phase-guided checkpoint selection demo
python demo_finetune_monitor.py
```

## Paper

"Loss Is Not Enough: The Golden Window in Neural Network Training"
13 models, 5 architecture families, 7.5M to 14B parameters.

## Data

10 pre-computed health reports + 7B validation data + calibration database.

## Language

[中文版 (Chinese)](README_CN.md)

## License

MIT
