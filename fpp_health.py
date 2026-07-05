#!/usr/bin/env python3
"""
FPP V4 Health Check — 任何开源/自有模型一键体检

用法:
    python fpp_health.py --model /path/to/model
    python fpp_health.py --model Qwen/Qwen2.5-0.5B-Instruct
    python fpp_health.py --model ./my-finetuned-model

输出: 架构指纹 + FPP 4维基线 + 健康评估 + 安全优化建议
"""
import os, sys, time, json, argparse
import numpy as np
import torch

# Use local fpp_metrics.py if available
try:
    from fpp_metrics import pearson_r, mutual_info, deception_index
except ImportError:
    import os as _os
    _fpp_dir = _os.path.dirname(_os.path.abspath(__file__))
    if _os.path.exists(_os.path.join(_fpp_dir, 'fpp_metrics.py')):
        _os.sys.path.insert(0, _fpp_dir)
        from fpp_metrics import pearson_r, mutual_info, deception_index
    else:
        raise ImportError(
            "fpp_metrics.py not found. Please download it from:\n"
            "  https://github.com/GIS-blackCaat/fpp-golden-window\n"
            "and place it in the same directory as fpp_health.py."
        )

# ═══════════════════════════════════════════════════════════════
# Architecture Family Database (from 12-architecture study)
# ═══════════════════════════════════════════════════════════════
FAMILY_DB = {
    ('SwiGLU', 'MHA'): {
        'gs_healthy':   (0.75, 0.92),
        'mi_healthy':   (0.005, 0.10),
        'phase_healthy':(0.35, 0.55),
        'beta_safe':    (0.05, 0.20),   # Qwen-tested: safe across full range
        'known_models': ['Qwen2.5-0.5B', 'Qwen2.5-1.5B', 'Qwen2.5-7B', 'Qwen2.5-14B'],
    },
    ('SwiGLU', 'GQA'): {
        'gs_healthy':   (0.40, 0.92),
        'mi_healthy':   (0.005, 0.80),
        'phase_healthy':(0.30, 0.50),
        'beta_safe':    None,            # MODEL-SPECIFIC: must scan individually
        'known_models': ['SmolLM2-360M', 'SmolLM2-1.7B', 'TinyLlama-1.1B'],
        'warning':      'GQA models have highly variable β safety. SmolLM2-1.7B: NONE. Others: [0.01,0.02]. Individual scan REQUIRED.',
    },
    'GeGLU': {
        'gs_healthy':   (0.20, 0.95),
        'mi_healthy':   (0.02, 0.15),
        'phase_healthy':(0.02, 0.40),
        'beta_safe':    (0.001, 0.02),  # ultra-micro only
        'known_models': ['Gemma-3-1B', 'Gemma-2-9B'],
        'note':         'GeGLU was previously misclassified as GELU. Fixed in FPP V4.',
    },
    'GELU': {
        'gs_healthy':   (0.40, 0.60),
        'mi_healthy':   (0.003, 0.02),
        'phase_healthy':(0.22, 0.35),
        'beta_safe':    (0.01, 0.02),   # conservative, from Pythia-160M sweep
        'known_models': ['Pythia-160M', 'GPT-NeoX'],
    },
    'ReLU': {
        'gs_healthy':   (0.50, 0.70),
        'mi_healthy':   (0.08, 0.20),
        'phase_healthy':(0.18, 0.30),
        'beta_safe':    (0.01, 0.20),
        'known_models': ['OPT-1.3B'],
    },
}

def detect_activation_fn(model):
    """Detect activation function family, handling gated variants."""
    cfg = model.config
    model_type = getattr(cfg, 'model_type', '').lower()

    # 1. Check model type first (most reliable)
    if model_type in ('gemma', 'gemma2', 'gemma3', 'gemma3_text'):
        return 'GeGLU', 'geglu'

    # 2. Check config attributes
    act = None
    for attr in ['hidden_act', 'hidden_activation', 'activation_function']:
        if hasattr(cfg, attr):
            act = getattr(cfg, attr)
            if act is not None:
                break

    # 3. Determine if gated (geglu/glu variants) or standard
    if act:
        act_lower = act.lower()
        # Gated variants
        if 'geglu' in act_lower or 'gated' in act_lower:
            return 'GeGLU', act
        if 'swiglu' in act_lower:
            return 'SwiGLU', act

        # Standard activations
        if act_lower in ('silu', 'swish'):
            return 'SwiGLU', act
        if act_lower in ('gelu', 'gelu_new', 'gelu_pytorch_tanh', 'gelu_fast'):
            return 'GELU', act
        if act_lower in ('relu',):
            return 'ReLU', act

    # 4. Probe actual modules as fallback
    for module in model.modules():
        if isinstance(module, torch.nn.GELU):
            act = 'gelu'; break
        elif isinstance(module, torch.nn.ReLU):
            act = 'relu'; break
        elif isinstance(module, torch.nn.SiLU):
            act = 'silu'; break

    if act:
        return detect_activation_fn.__wrapped__(model) if hasattr(detect_activation_fn, '__wrapped__') else ('SwiGLU', act)

    return 'SwiGLU', 'unknown'  # default

def count_layers(model):
    """Count transformer layers regardless of model architecture."""
    for attr in ['model.layers', 'model.decoder.layers', 'transformer.h',
                 'transformer.layers', 'encoder.layers']:
        parts = attr.split('.')
        obj = model
        try:
            for p in parts: obj = getattr(obj, p)
            return len(obj), attr
        except: pass
    return 0, 'unknown'

def get_layer_list(model):
    """Return list of transformer layers."""
    for path in ['model.layers', 'model.decoder.layers', 'transformer.h',
                  'transformer.layers', 'encoder.layers']:
        parts = path.split('.')
        obj = model
        try:
            for p in parts: obj = getattr(obj, p)
            return obj
        except: pass
    return None

def run_fpp(model, tokenizer, layers, text, n=10):
    """Run n FPP measurements, return mean ± std for all 6 dimensions."""
    gs_v, mi_v, pv_v, dc_v, ph_v, ipr_v = [], [], [], [], [], []

    for _ in range(n):
        ids = tokenizer(text, return_tensors='pt', truncation=True,
                        max_length=64)['input_ids'].to(device)
        outs = {}
        hooks = []
        def hk(i):
            def h(_, __, o):
                outs[i] = o[0].detach() if isinstance(o, tuple) else o.detach()
                return None
            return h
        for i, _ in enumerate(layers):
            hooks.append(layers[i].register_forward_hook(hk(i)))

        with torch.no_grad():
            _ = model(ids); fwd = dict(outs)
            _ = model(torch.flip(ids, [1])); rev = dict(outs)
        for h in hooks: h.remove()

        fv = [fwd[i].float() for i in sorted(fwd.keys())]
        rv = [rev[i].float() for i in sorted(rev.keys())]

        gs = np.mean([pearson_r(fv[i].flatten().cpu().numpy(),
                                 rv[len(rv)-1-i].flatten().cpu().numpy())
                       for i in range(len(fv))])
        mi_vals = [mutual_info(fv[i].flatten().cpu().numpy()[:1000],
                               fv[-1].flatten().cpu().numpy()[:1000])
                    for i in range(len(fv))]
        mi = np.mean(mi_vals)
        pv = pearson_r(fv[0].flatten().cpu().numpy(),
                       fv[-1].flatten().cpu().numpy())
        dc = deception_index(pv, mi)

        nl = len(fv)
        corr = np.zeros((nl, nl))
        for i in range(nl):
            for j in range(nl):
                corr[i][j] = pearson_r(fv[i].flatten().cpu().numpy(),
                                        fv[j].flatten().cpu().numpy())
        ph = np.std(corr)
        ipr = 1.0 / max(np.sum(np.mean(np.abs(corr), axis=0) ** 2), 1e-8)

        gs_v.append(gs); mi_v.append(mi); pv_v.append(pv)
        dc_v.append(dc); ph_v.append(ph); ipr_v.append(ipr)

    return {
        'gs': float(np.mean(gs_v)), 'gs_std': float(np.std(gs_v)),
        'mi': float(np.mean(mi_v)), 'mi_std': float(np.std(mi_v)),
        'pv': float(np.mean(pv_v)), 'dc': float(np.mean(dc_v)),
        'ph': float(np.mean(ph_v)), 'ph_std': float(np.std(ph_v)),
        'ipr': float(np.mean(ipr_v)),
    }

def assess_health(fpp, family, fam_name):
    """Check FPP metrics against family-specific healthy ranges."""
    checks = {}
    gs_min, gs_max = family['gs_healthy']
    mi_min, mi_max = family['mi_healthy']
    ph_min, ph_max = family['phase_healthy']

    checks['gs'] = '✅' if gs_min <= fpp['gs'] <= gs_max else \
                   ('⚠️ LOW' if fpp['gs'] < gs_min else '⚠️ HIGH')
    checks['mi'] = '✅' if mi_min <= fpp['mi'] <= mi_max else \
                   ('⚠️ LOW' if fpp['mi'] < mi_min else '⚠️ HIGH')
    checks['phase'] = '✅' if ph_min <= fpp['ph'] <= ph_max else \
                      ('⚠️ LOW' if fpp['ph'] < ph_min else '⚠️ HIGH')

    # DC-based downgrade: high deception makes all Pearson-based metrics less reliable
    if fpp.get('dc', 0) > 0.5:
        checks['dc'] = '⚠️ VERY HIGH (>0.5) — Pearson may be unreliable'
    elif fpp.get('dc', 0) > 0.3:
        checks['dc'] = '⚠️ HIGH (>0.3)'

    all_ok = all(v == '✅' for v in [checks.get(k,'✅') for k in ['gs','mi','phase']])
    status = '🟢 HEALTHY' if all_ok else '🟡 NEEDS ATTENTION'
    if any('LOW' in v for v in checks.values()):
        status = '🔴 AT RISK'
    # DC severity: if DC>0.5, downgrade one more level
    if fpp.get('dc', 0) > 0.5 and status == '🟡 NEEDS ATTENTION':
        status = '🔴 AT RISK (metric reliability degraded)'

    return status, checks

def main():
    parser = argparse.ArgumentParser(description='FPP V4 Health Check')
    parser.add_argument('--model', required=True, help='Model path or HF name')
    parser.add_argument('--n-runs', type=int, default=10, help='FPP measurement repeats')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show per-input details')
    args = parser.parse_args()

    print("=" * 65)
    print("  FPP V4  HEALTH CHECK")
    print("=" * 65)
    print(f"  Model: {args.model}")
    print(f"  Time:  {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # ─── Load model ───
    global device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n[1/4] Loading model on {device}...")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token

    dtype = torch.float16 if device == 'cuda' else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype,
                                                  trust_remote_code=True).to(device)
    model.eval()

    # ─── Architecture fingerprint ───
    n_layers, layer_path = count_layers(model)
    layers = get_layer_list(model)
    if layers is None:
        print("  ❌ Cannot find transformer layers in this model")
        return

    fam_name, act_raw = detect_activation_fn(model)
    # Use composite key for SwiGLU family: distinguish MHA vs GQA
    try:
        first_attn = layers[0].self_attn if hasattr(layers[0], 'self_attn') else layers[0].attention
        has_gqa = hasattr(first_attn, 'num_key_value_heads')
        attn_type = f"GQA({first_attn.num_heads}/{first_attn.num_key_value_heads})" if has_gqa else "MHA"
    except:
        attn_type = "unknown"

    attn_simple = "GQA" if "GQA" in attn_type else "MHA"
    fam_key = (fam_name, attn_simple) if fam_name == 'SwiGLU' and attn_simple == 'GQA' else (fam_name, attn_simple)
    # For non-SwiGLU families, use the family name directly
    family = FAMILY_DB.get(fam_key, FAMILY_DB.get(fam_name, FAMILY_DB.get(('SwiGLU','MHA'))))

    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3 if device == 'cuda' else 0

    print(f"""
  ┌─ Architecture Fingerprint ─────────────────────────────┐
  │ Layers:      {n_layers} ({layer_path})
  │ Activation:  {act_raw} → Family: {fam_name}
  │ Attention:   {attn_type}
  │ Precision:   {str(dtype)}
  │ VRAM:        {vram:.1f} GB
  │ Known family members: {', '.join(family['known_models'][:3])}
  └────────────────────────────────────────────────────────┘""")

    # ─── FPP baseline ───
    print(f"\n[2/4] Running FPP baseline ({args.n_runs} measurements)...")

    normal_en = "The transformer architecture has become the foundation of modern AI systems."
    code = "def quick_sort(arr): return arr if len(arr) <= 1 else quick_sort([x for x in arr[1:] if x <= arr[0]])"
    noise = "zxcvbnm asdfghjkl !@#$%^&*() qwertyuiop poiuy trewq"

    results = {}
    for name, text in [('normal_en', normal_en), ('code', code), ('noise', noise)]:
        t0 = time.time()
        fpp = run_fpp(model, tokenizer, layers, text, n=args.n_runs)
        elapsed = time.time() - t0
        results[name] = fpp
        if args.verbose:
            print(f"  {name:12s} ({elapsed:.1f}s): GS={fpp['gs']:.4f} MI={fpp['mi']:.4f} "
                  f"Phase={fpp['ph']:.4f} DC={fpp['dc']:.4f} IPR={fpp['ipr']:.4f}")

    fpp = results['normal_en']

    # ─── Health assessment ───
    print(f"\n[3/4] Health Assessment (vs {fam_name} family standards)")
    status, checks = assess_health(fpp, family, fam_name)

    print(f"""
  ┌─ FPP Vital Signs ─────────────────────────────────────┐
  │ GS (结构有序度):     {fpp['gs']:.4f} ± {fpp['gs_std']:.4f}   {checks['gs']}
  │   family healthy:    [{family['gs_healthy'][0]:.2f}, {family['gs_healthy'][1]:.2f}]
  │ MI (信息通道):       {fpp['mi']:.4f} ± {fpp['mi_std']:.4f}   {checks['mi']}
  │   family healthy:    [{family['mi_healthy'][0]:.3f}, {family['mi_healthy'][1]:.2f}]
  │ Phase (层功能分化):  {fpp['ph']:.4f} ± {fpp['ph_std']:.4f}   {checks['phase']}
  │   family healthy:    [{family['phase_healthy'][0]:.2f}, {family['phase_healthy'][1]:.2f}]
  │ Deception (欺骗度):  {fpp['dc']:.4f}
  │ IPR (信息定位):      {fpp['ipr']:.4f}
  ├────────────────────────────────────────────────────────┤
  │ STATUS: {status}
  └────────────────────────────────────────────────────────┘""")

    # ─── Input sensitivity ───
    if args.verbose:
        print(f"\n  Input Sensitivity:")
        gs_base = results['normal_en']['gs']
        print(f"  {'Input':<12} {'GS':<10} {'Δ vs normal':<12} {'MI':<10}")
        for name in ['code', 'noise']:
            f = results[name]
            delta = f['gs'] - gs_base
            print(f"  {name:<12} {f['gs']:<10.4f} {delta:+.4f}         {f['mi']:<10.4f}")

    # ─── Recommendations ───
    print(f"\n[4/4] Recommendations")
    print("  " + "-" * 58)

    recs = []
    beta_safe = family.get('beta_safe')
    warning = family.get('warning', '')
    if beta_safe:
        recs.append(f"✅ Momentum injection SAFE: β ∈ [{beta_safe[0]:.2f}, {beta_safe[1]:.2f}]")
        if warning:
            recs.append(f"   ⚠️  {warning}")
    else:
        recs.append(f"❌ β safety is MODEL-SPECIFIC for this architecture. Run individual β sweep before injection.")
        if warning:
            recs.append(f"   ⚠️  {warning}")

    if fpp['ph'] > 0.35:
        recs.append("✅ FPP-guided fine-tuning available (Phase peak detection)")
    else:
        recs.append("⚠️  Phase too low for reliable checkpoint navigation")

    if fpp['dc'] > 0.3:
        recs.append("⚠️  High Deception Index — Pearson may be unreliable. Trust MI instead.")

    if fpp['gs'] < family['gs_healthy'][0]:
        recs.append(f"⚠️  GS below family normal — model may have degraded structure")
    elif fpp['gs'] > family['gs_healthy'][1]:
        recs.append(f"⚠️  GS above family normal — model may be over-regularized/rigid")

    for r in recs:
        print(f"  {r}")

    # ─── Output ───
    result = {
        'model': args.model,
        'architecture': {'layers': n_layers, 'activation': act_raw, 'family': fam_name,
                         'attention': attn_type},
        'fpp_baseline': fpp,
        'input_sensitivity': {k: {'gs': v['gs'], 'mi': v['mi'], 'phase': v['ph']}
                              for k, v in results.items()},
        'health': {'status': status, 'checks': checks},
        'recommendations': recs,
    }

    # Save to model directory or current dir
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            f"fpp_health_{os.path.basename(args.model.rstrip('/'))}.json")
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n  Report saved to: {out_path}")
    print("=" * 65)


if __name__ == '__main__':
    main()
