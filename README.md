# FPP — Fieldcell Processing Protocol

**Loss Is Not Enough.** FPP is a 4-dimensional diagnostic framework that measures what loss cannot see: structural order, information capacity, layer differentiation, and metric honesty.

> 13 models, 5 architecture families, 7.5M to 14B parameters. All experiments on a single GTX 1650 Ti (4GB VRAM).

---

## Quick Start

```bash
git clone https://github.com/GIS-blackCaat/fpp-golden-window.git
cd fpp-golden-window
pip install -r requirements.txt
python fpp_health.py --model Qwen/Qwen2.5-0.5B-Instruct
```

Works with any HuggingFace Transformers model (local path or model name). Diagnosis completes in ~5 seconds.

---

## ⏱️ 5-Minute Diagnostic Guide

New to FPP? Start here: **[5MIN_GUIDE.md](5MIN_GUIDE.md)** — load → run FPP → read 4 indicators → decide what to do next. Includes decision tree and β safety quick-reference table.

---

## What You Get

- **Architecture fingerprint** (activation, attention, layers)
- **FPP 4D baseline** (GS, MI, Phase, DC)
- **Health assessment** (vs family-specific norms)
- **Safe β range** (momentum injection safety)
- **Optimization recommendations** (whether/how to fine-tune, Phase-guided checkpointing)

## Examples

```bash
# Programmatic usage — call FPP from your own Python code
python example.py --model Qwen/Qwen2.5-0.5B-Instruct

# Fine-tuning monitor — Phase-guided checkpoint selection (with real EXP-23 data)
python demo_finetune_monitor.py
```

---

## The Four Metrics

| Metric | Name | What It Measures | TL;DR |
|:---|:---|:---|:---|
| **GS** | Green Symmetry | Time-reversal structural order (forward vs reverse pass) | Higher → healthier structure |
| **MI** | Mutual Information Retention | How much information survives from input to output | Architecture-determined; fine-tuning can't raise the ceiling |
| **Phase** | Phase Structure | Whether layers do different things (layer differentiation) | Peak = optimal checkpoint; dropping = stop training |
| **DC** | Deception Index | Whether Pearson correlation is lying about structural health | >0.3 → trust MI, not GS |

---

## β Safety Quick Reference

Before adding momentum to your optimizer, check this table. Same β=0.2: Qwen gains +19% GS, Gemma collapses.

| Your Model Family | Safe β Range | Expected Effect |
|:---|:---|:---|
| Qwen (SwiGLU+MHA) | [0.05, 0.20] | GS +19% |
| OPT (ReLU+MHA) | [0.01, 0.20] | GS +32% |
| TinyLlama / SmolLM2-360M (SwiGLU+GQA) | [0.01, 0.02] | Micro β only |
| Gemma (GeGLU) | [0.001, 0.02] | Ultra-micro; β>0.02 kills |
| SmolLM2-1.7B (SwiGLU+GQA) | ☠️ **NONE** | Any β causes collapse |
| Pythia (GELU+MHA) | [0.01, 0.02] | Conservative |

---

## Repo Contents

```
fpp_health.py              ← One-click health check (CLI)
fpp_metrics.py             ← Core GS/MI/Phase/DC implementation
fpp_health_dashboard.py    ← Health dashboard visualization
fpp_advantage_viz.py       ← FPP advantage comparison plots
example.py                 ← Programmatic usage demo
demo_finetune_monitor.py   ← Phase-guided checkpoint selection demo
data/                      ← 14 health reports + calibration DB + trajectories
figures/                   ← 4 paper figures (PDF vector)
tables/                    ← 2 LaTeX tables (10-model, method comparison)
5MIN_GUIDE.md              ← 5-minute diagnostic walkthrough
paper.pdf                  ← Full paper
```

---

## Paper

**"Loss Is Not Enough: The Golden Window in Neural Network Training"**

- 📄 **PDF**: [HuggingFace](https://huggingface.co/youyouYUE/golden-window)
- 🔬 **Key finding**: Training is not monotonic — neural networks go through Build → Collapse → Rebuild. Loss is blind to the collapse.
- 🪟 **Golden Window**: Structure peaks at ~step 1,000, then is sacrificed for marginal loss gains.
- 💻 **Hardware**: All experiments on consumer hardware. Large-model research isn't exclusive to large companies.

---

## Hardware Requirements

| Model Size | Minimum |
|:---|:---|
| ≤ 2B params | 4GB VRAM GPU or 8GB RAM CPU |
| 7B params | 16GB RAM (CPU inference) |
| 14B params | 32GB RAM (CPU inference) |

---

## Language

[中文版 (Chinese)](README_CN.md)

## License

MIT — do whatever you want, commercial-friendly.

## Citation

```bibtex
@article{yue2026goldenwindow,
  title={Loss Is Not Enough: The Golden Window in Neural Network Training},
  author={Yue, Song},
  journal={arXiv preprint},
  year={2026}
}
```
