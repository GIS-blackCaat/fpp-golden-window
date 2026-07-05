# FPP Model Health Check — Quick Start

## Install
pip install -r requirements.txt

## Run
python fpp_health.py --model /path/to/your/model

## Example
python fpp_health.py --model Qwen/Qwen2.5-0.5B-Instruct
python fpp_health.py --model ~/models/my-finetuned-model

## Output
- Architecture fingerprint (activation, attention, layer count)
- FPP 4D baseline (GS, MI, Phase, DC)
- Health status vs family norms
- Safe beta range for momentum injection
- Recommendations

## Requirements
- Python 3.10+
- 4GB+ RAM for <2B models
- 16GB+ RAM for 7B models (CPU inference)
