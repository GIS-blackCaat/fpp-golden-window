# FPP — Fieldcell Processing Protocol

**Loss Is Not Enough.** FPP is a 4-dimensional diagnostic framework that measures what loss cannot see: structural order, information capacity, layer differentiation, and metric honesty.

## Quick Start

```bash
pip install -r requirements.txt
python fpp_health.py --model Qwen/Qwen2.5-0.5B-Instruct
```

## What You Get

- Architecture fingerprint (activation, attention, layers)
- FPP baseline (GS, MI, Phase, DC)
- Health status vs family norms
- Safe β range for momentum injection
- Optimization recommendations

## Paper

"Loss Is Not Enough: The Golden Window in Neural Network Training"
13 models, 5 architecture families, 7.5M to 14B parameters.

## Data

10 pre-computed health reports + 7B validation data + calibration database.

## License

MIT
