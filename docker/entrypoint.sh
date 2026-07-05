#!/bin/bash
# ─── FPP Golden Window — Docker Entrypoint ───
set -e

banner() {
    echo "══════════════════════════════════════════════════════════════"
    echo "  FPP Golden Window — Paper Experiment Reproduction"
    echo "  'Loss Is Not Enough' — Song Yue, 2026"
    echo "══════════════════════════════════════════════════════════════"
    echo ""
}

usage() {
    banner
    echo "Usage: docker run -it fpp-golden-window [COMMAND] [ARGS]"
    echo ""
    echo "Commands:"
    echo "  demo             Run full demo (default): baseline + compare + monitor"
    echo "  health MODEL     Run FPP health check on a HuggingFace model"
    echo "                   e.g.: health Qwen/Qwen2.5-0.5B-Instruct"
    echo "  compare A B      Compare two models side by side"
    echo "                   e.g.: compare Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-1.5B"
    echo "  scan MODEL       Show β safety range for a model"
    echo "  monitor          Demo Phase-guided fine-tuning checkpoint selection"
    echo "  table            Print the 13-model baseline table from the paper"
    echo "  shell            Drop into bash shell"
    echo "  help             Show this message"
    echo ""
    echo "GPU users: add '--gpus all' to docker run for CUDA acceleration."
}

case "${1:-demo}" in
    demo)
        banner
        echo ">>> Step 1/4: FPP Health Check (Qwen 0.5B Instruct)"
        echo "    (first run will download ~1GB model — cached thereafter)"
        echo ""
        python tools/fpp_health.py --model Qwen/Qwen2.5-0.5B-Instruct --n-runs 5
        echo ""
        echo ">>> Step 2/4: Baseline Comparison (TinyLlama vs Qwen 0.5B)"
        echo ""
        python -c "
import sys; sys.path.insert(0, '/app/examples')
from basic_usage import run_fpp_single, compare_models
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float16 if device == 'cuda' else torch.float32

def get_layers(model):
    for path in ['model.layers','model.decoder.layers','transformer.h']:
        obj = model
        try:
            for p in path.split('.'): obj = getattr(obj, p)
            return obj
        except: pass
    return None

text = 'The transformer architecture has become the foundation of modern AI systems.'

print('Loading Qwen 0.5B...')
m1 = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct', torch_dtype=dtype, trust_remote_code=True).to(device)
t1 = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct', trust_remote_code=True)
l1 = get_layers(m1)
r1 = run_fpp_single(m1, t1, l1, text, n=5)
del m1; torch.cuda.empty_cache() if device == 'cuda' else None

print('Loading TinyLlama...')
m2 = AutoModelForCausalLM.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0', torch_dtype=dtype, trust_remote_code=True).to(device)
t2 = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0', trust_remote_code=True)
l2 = get_layers(m2)
r2 = run_fpp_single(m2, t2, l2, text, n=5)
del m2

compare_models(r1, r2, 'Qwen 0.5B', 'TinyLlama 1.1B')
"
        echo ""
        echo ">>> Step 3/4: Phase-Guided Fine-Tuning Monitor"
        echo ""
        python examples/finetune_monitor.py
        echo ""
        echo ">>> Step 4/4: 13-Model Baseline Table"
        echo ""
        cat data/DATABASE.md
        echo ""
        banner
        echo "✅ Full demo complete!"
        echo ""
        echo "Next steps:"
        echo "  docker run -it fpp-golden-window health <YOUR_MODEL>"
        echo "  docker run -it fpp-golden-window compare <MODEL_A> <MODEL_B>"
        echo "  docker run -it fpp-golden-window shell"
        ;;

    health)
        shift
        MODEL="${1:-Qwen/Qwen2.5-0.5B-Instruct}"
        banner
        echo "Running FPP Health Check on: $MODEL"
        python tools/fpp_health.py --model "$MODEL"
        ;;

    compare)
        shift
        MODEL_A="${1}"
        MODEL_B="${2}"
        if [ -z "$MODEL_A" ] || [ -z "$MODEL_B" ]; then
            echo "Usage: compare <MODEL_A> <MODEL_B>"
            echo "  e.g.: compare Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-1.5B"
            exit 1
        fi
        banner
        echo "Comparing: $MODEL_A  vs  $MODEL_B"
        echo ""
        python -c "
import sys; sys.path.insert(0, '/app/examples')
from basic_usage import run_fpp_single, compare_models
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float16 if device == 'cuda' else torch.float32
text = 'The transformer architecture has become the foundation of modern AI systems.'

def get_layers(model):
    for path in ['model.layers','model.decoder.layers','transformer.h']:
        obj = model
        try:
            for p in path.split('.'): obj = getattr(obj, p)
            return obj
        except: pass
    return None

for label, name in [('Model A', '$MODEL_A'), ('Model B', '$MODEL_B')]:
    print(f'Loading {label}: {name}...')
    m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, trust_remote_code=True).to(device)
    t = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    l = get_layers(m)
    r = run_fpp_single(m, t, l, text, n=5)
    if label == 'Model A': ra = r
    else: rb = r
    del m; torch.cuda.empty_cache() if device == 'cuda' else None

compare_models(ra, rb, '$MODEL_A', '$MODEL_B')
"
        ;;

    scan)
        shift
        MODEL="${1:-Qwen/Qwen2.5-0.5B-Instruct}"
        banner
        echo "β Safety Scan on: $MODEL"
        echo ""
        echo "Note: β scanning requires actual fine-tuning — this shows the"
        echo "paper's pre-computed safe ranges for your model's architecture family."
        echo ""
        python tools/fpp_health.py --model "$MODEL" --verbose
        ;;

    monitor)
        banner
        python examples/finetune_monitor.py
        ;;

    table)
        banner
        echo "13-Model Baseline Table (from paper Table 2)"
        echo "============================================="
        echo ""
        cat data/DATABASE.md
        ;;

    shell)
        banner
        echo "Tools:  tools/fpp_health.py  tools/fpp_metrics.py"
        echo "Demos:  examples/basic_usage.py  examples/finetune_monitor.py"
        echo "Docs:   docs/5MIN_GUIDE.md  docs/paper.pdf"
        echo "Data:   data/health_reports/  data/experiments/"
        echo ""
        exec bash
        ;;

    help|--help|-h)
        usage
        ;;

    *)
        echo "Unknown command: $1"
        usage
        exit 1
        ;;
esac
