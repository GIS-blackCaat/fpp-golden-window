#!/usr/bin/env python3
"""
FPP Programmatic Usage Example
===============================
演示如何在自己的代码中调用 FPP，而非通过命令行。

场景:
  1. 加载任意 HuggingFace 模型
  2. 程序化运行 FPP 四维测量
  3. 获取健康评估和建议
  4. 用 FPP 对比两个模型 / checkpoint

要求: pip install -r requirements.txt
"""

import torch
import numpy as np
import os, sys
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─── 导入 FPP 核心函数 ───
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))
from fpp_metrics import pearson_r, mutual_info, deception_index

# ============================================================
# 第一部分：基础用法 — 测一个模型
# ============================================================

def run_fpp_single(model, tokenizer, layers, text, n=10):
    """
    对单个输入文本跑 n 次 FPP 测量。
    返回四维指标均值±标准差。
    """
    gs_vals, mi_vals, dc_vals, ph_vals = [], [], [], []

    for _ in range(n):
        # tokenize
        ids = tokenizer(text, return_tensors='pt', truncation=True,
                        max_length=64)['input_ids']

        # 注册 forward hooks 收集隐藏态
        outs = {}
        hooks = []
        def make_hook(idx):
            def hook_fn(_, __, output):
                outs[idx] = output[0].detach() if isinstance(output, tuple) else output.detach()
            return hook_fn
        for i, _ in enumerate(layers):
            hooks.append(layers[i].register_forward_hook(make_hook(i)))

        with torch.no_grad():
            # 正向 pass
            model(ids)
            fwd = dict(outs)

            # 反向 pass（倒序输入）
            model(torch.flip(ids, [1]))
            rev = dict(outs)

        for h in hooks:
            h.remove()

        # 按层索引排序
        fv = [fwd[i].float() for i in sorted(fwd.keys())]
        rv = [rev[i].float() for i in sorted(rev.keys())]
        L = len(fv)

        # ─── 计算 GS：正向第 i 层 vs 反向第 L-i 层的 Pearson ───
        gs = np.mean([
            pearson_r(fv[i].flatten().cpu().numpy(),
                      rv[L-1-i].flatten().cpu().numpy())
            for i in range(L)
        ])

        # ─── 计算 MI：每层输入-输出互信息均值 ───
        mi_vals_per_layer = [
            mutual_info(fv[i].flatten().cpu().numpy()[:1000],
                        fv[-1].flatten().cpu().numpy()[:1000])
            for i in range(L)
        ]
        mi = np.mean(mi_vals_per_layer)

        # ─── 计算 Phase：层间相关矩阵标准差 ───
        corr = np.zeros((L, L))
        for i in range(L):
            for j in range(L):
                corr[i, j] = pearson_r(fv[i].flatten().cpu().numpy(),
                                       fv[j].flatten().cpu().numpy())
        ph = np.std(corr)

        # ─── 计算 DC：Pearson vs MI 的差值 ───
        pv = pearson_r(fv[0].flatten().cpu().numpy(),
                       fv[-1].flatten().cpu().numpy())
        dc = deception_index(pv, mi)

        gs_vals.append(gs)
        mi_vals.append(mi)
        dc_vals.append(dc)
        ph_vals.append(ph)

    return {
        'gs':   float(np.mean(gs_vals)), 'gs_std':  float(np.std(gs_vals)),
        'mi':   float(np.mean(mi_vals)), 'mi_std':  float(np.std(mi_vals)),
        'phase':float(np.mean(ph_vals)), 'ph_std':  float(np.std(ph_vals)),
        'dc':   float(np.mean(dc_vals)),
    }


# ============================================================
# 第二部分：对比两个模型 / checkpoint
# ============================================================

def compare_models(result_a, result_b, name_a='Model A', name_b='Model B'):
    """对比两个 FPP 结果，输出差异分析。"""
    print(f"\n{'='*60}")
    print(f"  FPP Comparison: {name_a} vs {name_b}")
    print(f"{'='*60}")
    print(f"  {'Metric':<10} {name_a:<15} {name_b:<15} {'Δ':<10}")
    print(f"  {'-'*50}")

    for key, label in [('gs', 'GS'), ('mi', 'MI'), ('phase', 'Phase'), ('dc', 'DC')]:
        va = result_a[key]
        vb = result_b[key]
        delta = vb - va
        direction = '↑' if delta > 0 else '↓' if delta < 0 else '→'
        print(f"  {label:<10} {va:<15.4f} {vb:<15.4f} {delta:+.4f} {direction}")

    # 结构变化解读
    print(f"\n  Interpretation:")
    if result_b['gs'] > result_a['gs']:
        print(f"  ✅ GS improved: {name_b} has better structural order")
    elif result_b['gs'] < result_a['gs']:
        print(f"  ⚠️  GS degraded: {name_b} lost structural order")

    if result_b['phase'] < result_a['phase'] and result_a['phase'] > 0.30:
        print(f"  ⚠️  Phase dropped: {name_b} may have passed its optimal checkpoint")

    if result_b['dc'] > 0.3:
        print(f"  ⚠️  High DC in {name_b}: Pearson-based metrics may be unreliable")

    print(f"{'='*60}\n")


# ============================================================
# 第三部分：快速健康判断
# ============================================================

# 家族健康范围（来自 13 模型校准数据库）
FAMILY_HEALTHY = {
    ('SwiGLU', 'MHA'):  {'gs': (0.75, 0.92), 'mi': (0.005, 0.10), 'phase': (0.35, 0.55)},
    ('SwiGLU', 'GQA'):  {'gs': (0.40, 0.92), 'mi': (0.005, 0.80), 'phase': (0.30, 0.50)},
    'GeGLU':            {'gs': (0.20, 0.95), 'mi': (0.02, 0.15),  'phase': (0.02, 0.40)},
    'ReLU':             {'gs': (0.50, 0.70), 'mi': (0.08, 0.20),  'phase': (0.18, 0.30)},
    'GELU':             {'gs': (0.40, 0.60), 'mi': (0.003, 0.02), 'phase': (0.22, 0.35)},
}

def quick_health_check(fpp_result, family_key=('SwiGLU', 'MHA')):
    """用家族正常范围快速判断模型健康状态。"""
    ref = FAMILY_HEALTHY.get(family_key, FAMILY_HEALTHY[('SwiGLU', 'MHA')])

    def check(val, lo, hi):
        if lo <= val <= hi: return '✅'
        return '⚠️ LOW' if val < lo else '⚠️ HIGH'

    gs_ok = check(fpp_result['gs'], *ref['gs'])
    mi_ok = check(fpp_result['mi'], *ref['mi'])
    ph_ok = check(fpp_result['phase'], *ref['phase'])

    all_ok = all(v == '✅' for v in [gs_ok, mi_ok, ph_ok])
    status = '🟢 HEALTHY' if all_ok else '🟡 NEEDS ATTENTION'

    print(f"  GS:    {fpp_result['gs']:.4f}  {gs_ok}  (family: [{ref['gs'][0]:.2f}, {ref['gs'][1]:.2f}])")
    print(f"  MI:    {fpp_result['mi']:.4f}  {mi_ok}  (family: [{ref['mi'][0]:.3f}, {ref['mi'][1]:.2f}])")
    print(f"  Phase: {fpp_result['phase']:.4f}  {ph_ok}  (family: [{ref['phase'][0]:.2f}, {ref['phase'][1]:.2f}])")
    print(f"  DC:    {fpp_result['dc']:.4f}  {'⚠️ HIGH' if fpp_result['dc'] > 0.3 else '✅'}")
    print(f"  → STATUS: {status}")
    return status


# ============================================================
# 主程序：完整演示流程
# ============================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='FPP Programmatic Example')
    parser.add_argument('--model', default='Qwen/Qwen2.5-0.5B-Instruct',
                        help='Model to demo (HuggingFace name or local path)')
    parser.add_argument('--n-runs', type=int, default=5,
                        help='Number of FPP measurements')
    args = parser.parse_args()

    print("=" * 60)
    print("  FPP Programmatic Usage — Example")
    print("=" * 60)
    print(f"  Model: {args.model}")

    # ─── 加载模型 ───
    print("\n[1] Loading model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16 if device == 'cuda' else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True
    ).to(device)
    model.eval()

    # 获取层列表
    layers = None
    for path in ['model.layers', 'model.decoder.layers', 'transformer.h']:
        obj = model
        try:
            for p in path.split('.'): obj = getattr(obj, p)
            layers = obj; break
        except: pass
    print(f"  Device: {device}, Layers: {len(layers)}")

    # ─── 运行 FPP ───
    print(f"\n[2] Running FPP ({args.n_runs} measurements on 3 input types)...")

    test_inputs = {
        'natural':   "The transformer architecture has become the foundation of modern AI.",
        'code':      "def quick_sort(arr): return arr if len(arr) <= 1 else quick_sort([x for x in arr[1:] if x <= arr[0]])",
        'random':    "zxcvbnm asdfghjkl qwertyuiop poiuy trewq mnbvcxz",
    }

    results = {}
    for name, text in test_inputs.items():
        fpp = run_fpp_single(model, tokenizer, layers, text, n=args.n_runs)
        results[name] = fpp
        print(f"  {name:10s} → GS={fpp['gs']:.4f}, MI={fpp['mi']:.4f}, "
              f"Phase={fpp['phase']:.4f}, DC={fpp['dc']:.4f}")

    # ─── 健康检查 ───
    print(f"\n[3] Quick Health Check (vs SwiGLU+MHA family):")
    quick_health_check(results['natural'])

    # ─── 输入敏感度 ───
    print(f"\n[4] Input Sensitivity:")
    base_gs = results['natural']['gs']
    for name in ['code', 'random']:
        delta = results[name]['gs'] - base_gs
        print(f"  {name:10s} GS={results[name]['gs']:.4f} (Δ={delta:+.4f})")

    print(f"\n{'='*60}")
    print("  Done. See fpp_metrics.py for the underlying GS/MI/Phase/DC math.")
    print(f"{'='*60}")
