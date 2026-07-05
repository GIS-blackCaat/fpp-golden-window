#!/usr/bin/env python3
"""
Demo: FPP-Guided Fine-Tuning Monitor
=====================================
模拟微调过程中每 50 步用 FPP Phase 导航最优 checkpoint 选择。

真实场景:
  - 你在 LoRA 微调一个模型
  - 每 50 步保存一个 checkpoint
  - 对每个 checkpoint 跑 FPP，追踪 Phase
  - Phase 峰值 → 最优 checkpoint（比 Loss 最优点多保 30-40% 推理能力）

本 demo 使用 EXP-23 的真实实验数据演示这个流程。
数据来源: data/baseline_comparison.json (Qwen 0.5B LoRA, WikiText-2, 500 steps)

要求: pip install -r requirements.txt
"""

import json
import os
import sys
import numpy as np

# ─── 加载真实实验数据 ───
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
data_path = os.path.join(DATA_DIR, 'baseline_comparison.json')

if not os.path.exists(data_path):
    print("Data file not found. Run from repo root:")
    print("  python demo_finetune_monitor.py")
    sys.exit(1)

with open(data_path) as f:
    exp_data = json.load(f)

training_log = exp_data['training_log']  # 每 100 步的完整记录


# ============================================================
# 1. 回放训练轨迹
# ============================================================

print("=" * 65)
print("  FPP-Guided Fine-Tuning Monitor — Demo")
print("  Data: EXP-23, Qwen 0.5B LoRA, WikiText-2, 500 steps")
print("=" * 65)

print(f"\n{'Step':<8} {'Train Loss':<12} {'GS':<8} {'Phase':<8} {'MI':<8} {'DC':<8} {'Causal':<8}")
print("-" * 65)

for entry in training_log:
    step = entry['iter']
    loss = entry['train_loss']
    gs = entry['gs']
    phase = entry['phase']
    mi = entry['mi']
    dc = entry.get('deception', 0)
    causal = entry['causal']

    # 标记关键点
    flag = ''
    if step == 200:
        flag = ' ← Phase PEAK'
    elif step == 400:
        flag = ' ← Loss MIN'
    elif step == 500:
        flag = ' ← Val PPL MIN'

    print(f"  {step:<8} {loss:<12.4f} {gs:<8.4f} {phase:<8.4f} {mi:<8.4f} {dc:<8.4f} {causal:<8.0%}{flag}")


# ============================================================
# 2. 对比三种选点策略
# ============================================================

print(f"\n{'='*65}")
print("  三种 Checkpoint 选择策略对比")
print(f"{'='*65}")

strategies = {
    'A. Train Loss 最低 (iter 400)': {
        'causal': 0.067, 'loss': 2.967, 'desc': '你通常会选的'
    },
    'B. Val PPL 最低 (iter 500)': {
        'causal': 0.0, 'loss': 3.020, 'desc': '标准早停策略'
    },
    'C. FPP Phase 峰值 (iter 200)': {
        'causal': 0.267, 'loss': 3.109, 'desc': 'FPP 导航策略'
    },
}

print(f"\n  {'策略':<30} {'因果推理':<12} {'Train Loss':<12} {'备注':<20}")
print(f"  {'-'*65}")

best_causal = 0
best_strategy = ''
for name, info in strategies.items():
    print(f"  {name:<30} {info['causal']:<12.0%} {info['loss']:<12.4f} {info['desc']:<20}")
    if info['causal'] > best_causal:
        best_causal = info['causal']
        best_strategy = name


# ============================================================
# 3. 模拟 Phase 导航逻辑
# ============================================================

print(f"\n{'='*65}")
print("  Phase 导航规则（可直接嵌入你的训练循环）")
print(f"{'='*65}")

print("""
  # 伪代码 — 嵌入到你的微调脚本中:
  #
  # best_phase = 0
  # best_checkpoint = None
  #
  # for step in range(0, total_steps, 50):
  #     # ... 训练 50 步 ...
  #     save_checkpoint(step)
  #     fpp = run_fpp(model)
  #     phase = fpp['phase']
  #
  #     if phase > best_phase:
  #         best_phase = phase
  #         best_checkpoint = step
  #         print(f"🟢 New Phase peak at step {step}: {phase:.4f}")
  #     elif phase < best_phase - 0.02:
  #         print(f"🔴 Phase dropping — structure degrading. Stop?")
  #         # 可选: 直接停止训练
  #         break
  #
  # print(f"Best checkpoint: step {best_checkpoint}")
""")

# 在真实数据上验证这条规则
print("  用真实数据验证（Phase 峰值 = iter 200, Causal = 26.7%):")
phases = [e['phase'] for e in training_log]
peak_idx = np.argmax(phases)
peak_step = training_log[peak_idx]['iter']
peak_causal = training_log[peak_idx]['causal']

loss_min_idx = np.argmin([e['train_loss'] for e in training_log])
loss_min_step = training_log[loss_min_idx]['iter']
loss_min_causal = training_log[loss_min_idx]['causal']

print(f"  ✅ Phase 峰值: step {peak_step}, causal={peak_causal:.0%}")
print(f"  ❌ Loss 最低:  step {loss_min_step}, causal={loss_min_causal:.0%}")
print(f"  📈 FPP 优势:   +{peak_causal-loss_min_causal:.0%} 绝对百分点")
print(f"  📈 相对提升:   {(peak_causal/loss_min_causal - 1)*100:.0f}% (因果推理能力)" if loss_min_causal > 0 else "  📈 Loss 最低点推理能力为 0，Phase 峰值仍有 26.7%")


# ============================================================
# 4. 实战建议
# ============================================================

print(f"""
{'='*65}
  实战 Checklist
{'='*65}

  □ 微调前: 跑 python fpp_health.py --model <你的基座模型>
    确认 GS/MI/Phase 在家族正常范围

  □ 每 50 步: 保存 checkpoint + 跑 FPP
    python fpp_health.py --model ./checkpoint-{step}

  □ Phase 到达峰值: 评估下游任务 → 大概率就是最优模型

  □ Phase 连续 2 次下降 >0.02: 停止训练
    继续训练只会让结构退化

  ⚠ Loss 最低 ≠ 模型最好。Phase 峰值才是。
  ⚠ 想加动量注入? 先查 β 安全表 (见 5MIN_GUIDE.md)

{'='*65}
""")
