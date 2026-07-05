# FPP — Fieldcell Processing Protocol

**Loss Is Not Enough.** FPP is a 4-dimensional diagnostic framework that measures what loss cannot see: structural order, information capacity, layer differentiation, and metric honesty.

> 13 models, 5 architecture families, 7.5M to 14B parameters. All experiments on a single GTX 1650 Ti (4GB VRAM).

---

## Quick Start

### Docker (one command, zero setup)

```bash
docker run -it ghcr.io/gis-blackcaat/fpp-golden-window:latest demo
```

Runs the full paper demo: health check → model comparison → Phase-guided checkpoint monitor → baseline table.

### Or: pip install locally

```bash
git clone https://github.com/GIS-blackCaat/fpp-golden-window.git
cd fpp-golden-window
pip install -r requirements.txt
python tools/fpp_health.py --model Qwen/Qwen2.5-0.5B-Instruct
```

Works with any HuggingFace Transformers model. Diagnosis completes in ~5 seconds.

---

## ⏱️ 5-Minute Diagnostic Guide

New to FPP? → **[docs/5MIN_GUIDE.md](docs/5MIN_GUIDE.md)** — load → run → read 4 indicators → decide what to do next. Decision tree + β safety table.

## 🤖 Claude Code Skill

This repo includes a Claude Code skill (`.claude/skills/fpp-expert.md`). After cloning, Claude Code auto-discovers it — just ask:

> "Is this model healthy?" / "When should I stop fine-tuning?" / "Can I add momentum?"

The skill provides instant access to: metric interpretation, family healthy ranges, β safety table, decision tree, and all 4 workflows.

---

## Repo Structure

```
tools/                     ← Main tools: health check, metrics, visualization
examples/                  ← Runnable example scripts
data/                      ← Health reports + experiments + database
docker/                    ← Dockerfile + compose + entrypoint
docs/                      ← Guide, install, paper
figures/                   ← 4 paper figures (PDF)
tables/                    ← 2 LaTeX tables
```

---

## The Four Metrics

| Metric | Name | What It Measures | TL;DR |
|:---|:---|:---|:---|
| **GS** | Green Symmetry | Time-reversal structural order | Higher → healthier structure |
| **MI** | Mutual Information Retention | Information survival from input to output | Architecture-determined ceiling |
| **Phase** | Phase Structure | Whether layers do different things | Peak = optimal checkpoint |
| **DC** | Deception Index | Whether Pearson is lying | >0.3 → trust MI, not GS |

---

## β Safety Quick Reference

Before adding momentum to your optimizer:

| Your Model Family | Safe β Range | Expected Effect |
|:---|:---|:---|
| Qwen (SwiGLU+MHA) | [0.05, 0.20] | GS +19% |
| OPT (ReLU+MHA) | [0.01, 0.20] | GS +32% |
| TinyLlama / SmolLM2-360M (SwiGLU+GQA) | [0.01, 0.02] | Micro β only |
| Gemma (GeGLU) | [0.001, 0.02] | Ultra-micro; β>0.02 kills |
| SmolLM2-1.7B (SwiGLU+GQA) | ☠️ **NONE** | Any β causes collapse |
| Pythia (GELU+MHA) | [0.01, 0.02] | Conservative |

---

## Examples

```bash
# Programmatic usage — call FPP from your own Python code
python examples/basic_usage.py --model Qwen/Qwen2.5-0.5B-Instruct

# Fine-tuning monitor — Phase-guided checkpoint selection (real EXP-23 data)
python examples/finetune_monitor.py
```

---

## Docker — Reproduce Paper Experiments

```bash
# Full demo (baseline + comparison + monitor + table)
docker run -it fpp-golden-window demo

# Check any HuggingFace model
docker run -it fpp-golden-window health Qwen/Qwen2.5-0.5B-Instruct

# Compare two models
docker run -it fpp-golden-window compare Qwen/Qwen2.5-0.5B TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Show 13-model baseline table
docker run -it fpp-golden-window table

# GPU acceleration
docker run -it --gpus all fpp-golden-window demo

# Interactive shell
docker run -it fpp-golden-window shell
```

Build from source:

```bash
docker build -t fpp-golden-window -f docker/Dockerfile .
# or: cd docker && docker compose up
```

---

## Paper

**"Loss Is Not Enough: The Golden Window in Neural Network Training"**

- 📄 **PDF**: [HuggingFace](https://huggingface.co/youyouYUE/golden-window)
- 🔬 Training is not monotonic — Build → Collapse → Rebuild. Loss is blind to the collapse.
- 🪟 Structure peaks at ~step 1,000 (golden window), then sacrificed for marginal loss gains.

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

MIT

## Citation

```bibtex
@article{yue2026goldenwindow,
  title={Loss Is Not Enough: The Golden Window in Neural Network Training},
  author={Yue, Song},
  journal={arXiv preprint},
  year={2026}
}
```
