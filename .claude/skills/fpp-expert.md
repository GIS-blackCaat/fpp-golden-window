---
name: fpp-expert
description: FPP model health diagnosis — measure GS/MI/Phase/DC, interpret results, guide fine-tuning with Phase peak detection and β safety calibration. Covers 13 models across 5 architecture families.
metadata:
  repo: https://github.com/GIS-blackCaat/fpp-golden-window
  paper: https://huggingface.co/youyouYUE/golden-window
  version: v4
---

# FPP Expert — Model Health Diagnosis & Fine-Tuning Guide

FPP (Fieldcell Processing Protocol) is a 4D diagnostic framework that measures what loss cannot see. You are an expert at using it to diagnose model health, guide fine-tuning, and calibrate momentum injection.

## Quick Diagnosis (Always Do This First)

```
python tools/fpp_health.py --model <path-or-hf-name>
```

Output: architecture fingerprint + GS/MI/Phase/DC + health status + β safe range + recommendations.

## The Four Metrics

| Metric | Measures | Decision Rule |
|:---|:---|:---|
| **GS** (Green Symmetry) | Time-reversal structural order | < family min → degraded structure; > family max → over-regularized |
| **MI** (Mutual Info Retention) | Information channel capacity | Architecture-determined ceiling; fine-tuning can only lower it |
| **Phase** (Phase Structure) | Layer functional differentiation | **Peak = optimal checkpoint**; dropping >0.02 → stop training |
| **DC** (Deception Index) | Pearson-MI divergence | >0.3 → GS unreliable, trust MI instead |

### Family Healthy Ranges (from 13-model calibration)

| Family | GS Range | MI Range | Phase Range | β Safe |
|:---|:---|:---|:---|:---|
| SwiGLU+MHA (Qwen) | [0.75, 0.92] | [0.005, 0.10] | [0.35, 0.55] | [0.05, 0.20] |
| SwiGLU+GQA (SmolLM2, TinyLlama) | [0.40, 0.92] | [0.005, 0.80] | [0.30, 0.50] | model-specific |
| GeGLU (Gemma) | [0.20, 0.95] | [0.02, 0.15] | [0.02, 0.40] | [0.001, 0.02] |
| ReLU (OPT) | [0.50, 0.70] | [0.08, 0.20] | [0.18, 0.30] | [0.01, 0.20] |
| GELU (Pythia) | [0.40, 0.60] | [0.003, 0.02] | [0.22, 0.35] | [0.01, 0.02] |

## Decision Tree

```
User asks: "Should I fine-tune this model?" or "Is my model healthy?"
→ Run fpp_health.py first
→ Then follow:

GS in family range?
├─ YES → Phase > 0.30?
│   ├─ YES → ✅ Fine-tune. Monitor Phase every 50 steps. Peak = best ckpt.
│   └─ NO  → ⚠️ Can fine-tune but Phase too low for reliable checkpoint nav.
└─ NO (GS low) → β safe range exists?
    ├─ YES → Try momentum injection at low β. Re-measure. If GS recovers → fine-tune.
    └─ NO (SmolLM2-1.7B style) → ☠️ Switch model. This architecture has no headroom.

DC > 0.3? → ALL decisions must use MI, not GS. GS readings are unreliable.
Phase dropping 2x consecutively? → STOP. Model is degrading. Roll back.
```

## β Safety (Momentum Injection)

Same β dose = different outcomes across architectures:
- β=0.2 on Qwen → GS +19% ✅
- β=0.2 on Gemma → structural collapse ☠️
- SmolLM2-1.7B → ANY β causes collapse

Always check `beta_safe` from `fpp_health.py` output before injecting momentum.

## Three-Phase Training Lifecycle

Training is NOT monotonic:
1. **Build** (0→~1K steps): GS +55%, MI +340%. Structure peaks.
2. **Collapse** (~1K→10K): GS -49%, MI -82%. Loss keeps dropping — blind to collapse.
3. **Rebuild** (10K→end): Partial recovery. Never returns to Build peak.

**Golden Window**: step ~1,000 — 94% of total loss improvement already achieved, structure at global maximum.

## Practical Workflows

### Workflow 1: Pre-Fine-Tune Baseline
```
1. python tools/fpp_health.py --model <base-model>
2. Check GS/MI/Phase all ✅ → proceed
3. Note β safe range for later
```

### Workflow 2: Fine-Tune with Phase Monitoring
```
for step in range(0, total_steps, 50):
    train_50_steps()
    save_checkpoint()
    run: python tools/fpp_health.py --model ./checkpoint-{step}
    if phase > best_phase: save as best
    if phase < best_phase - 0.02: STOP
```

### Workflow 3: Compare Two Models
```
python examples/basic_usage.py --model <model-a>  # get baseline
python examples/basic_usage.py --model <model-b>  # compare
# Or use inline: from basic_usage import run_fpp_single, compare_models
```

### Workflow 4: Docker One-Click Demo
```
docker run -it ghcr.io/gis-blackcaat/fpp-golden-window:latest demo
docker run -it ghcr.io/gis-blackcaat/fpp-golden-window:latest health Qwen/Qwen2.5-0.5B-Instruct
```

## Key Facts (for answering questions)

- 13 models tested: Qwen (0.5B/1.5B/7B/14B), SmolLM2 (360M/1.7B), TinyLlama 1.1B, Gemma (3-1B/2-9B), OPT 1.3B, Pythia 160M, Custom-Momentum 7.5M
- GS varies 2.1× within same family; MI varies 40× across models
- GS and MI are orthogonal (r=-0.10) — independent degrees of freedom
- Phase most stable across architectures (0.06–0.47 range)
- Instruct fine-tuning systematically lowers MI (~33% drop for Qwen)
- All experiments on GTX 1650 Ti (4GB) — no cluster needed
- Qwen 1.5B is GS sweet spot (0.89), not 7B (0.81) — more params ≠ better structure
- Gemma 9B: highest GS (0.91) but lowest Phase (0.06) and highest DC (0.69)
- FPP overhead: ~5% of inference time, no training required, no weight modification

## Repo Structure (important: files are NOT at root)

```
tools/           ← fpp_health.py, fpp_metrics.py, dashboard.py, advantage_viz.py
examples/        ← basic_usage.py, finetune_monitor.py
data/            ← health_reports/, experiments/, DATABASE.md
docker/          ← Dockerfile, docker-compose.yml, entrypoint.sh
docs/            ← 5MIN_GUIDE.md, INSTALL.md, paper.pdf
figures/ tables/ ← Paper figures and LaTeX tables
```

Commands use `tools/fpp_health.py` (not `fpp_health.py` at root).

## When User Asks...

- "Is my model good?" → Run fpp_health.py, check GS vs family range
- "When should I stop fine-tuning?" → Phase peak detection (every 50 steps)
- "Can I add momentum?" → Check β safe range from fpp_health.py output
- "Why is my model getting worse?" → Check if Phase is dropping; check DC
- "Which model should I use?" → Consult the 13-model database for your task needs
- "How do I run this?" → Point to docs/5MIN_GUIDE.md or Docker demo
