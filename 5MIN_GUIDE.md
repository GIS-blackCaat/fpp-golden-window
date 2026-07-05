# ⏱️ FPP 五分钟诊断指南

> 加载模型 → 跑 FPP → 看四个指标 → 决定下一步怎么调

---

## 第一步：安装 & 加载（30 秒）

```bash
pip install torch transformers numpy scipy scikit-learn
python fpp_health.py --model /path/to/your/model
```

支持本地路径和 HuggingFace 模型名。模型加载耗时取决于模型大小，FPP 测量本身只需 ~5 秒。

---

## 第二步：看懂输出（2 分钟）

跑完之后你会看到这个：

```
═══════════════════════════════════════════════════════════════════
  FPP V4  HEALTH CHECK
═══════════════════════════════════════════════════════════════════
  Model: Qwen/Qwen2.5-0.5B-Instruct

  ┌─ Architecture Fingerprint ─────────────────────────────────┐
  │ Layers:      24
  │ Activation:  silu → Family: SwiGLU
  │ Attention:   MHA
  └────────────────────────────────────────────────────────────┘

  ┌─ FPP Vital Signs ─────────────────────────────────────────┐
  │ GS (结构有序度):     0.81 ± 0.02   ✅
  │   family healthy:    [0.75, 0.92]
  │ MI (信息通道):       0.09 ± 0.01   ✅
  │   family healthy:    [0.005, 0.10]
  │ Phase (层功能分化):  0.42 ± 0.03   ✅
  │   family healthy:    [0.35, 0.55]
  │ Deception (欺骗度):  0.15
  ├────────────────────────────────────────────────────────────┤
  │ STATUS: 🟢 HEALTHY
  └────────────────────────────────────────────────────────────┘

  Recommendations:
  ✅ Momentum injection SAFE: β ∈ [0.05, 0.20]
  ✅ FPP-guided fine-tuning available (Phase peak detection)
```

四个指标的直觉理解：

| 指标 | 它告诉你什么 | 一句话 |
|:---|:---|:---|
| **GS** | 模型内部计算是否"对称"、有序 | 越高 → 结构越健康 |
| **MI** | 信息从输入到输出保留了多少 | 越高 → 信息通道越宽 |
| **Phase** | 24 层是否在做不同的事 | 越高 → 层分工越明确 |
| **DC** | Pearson 相关性是否在撒谎 | >0.3 → 不要轻信 GS |

---

## 第三步：对号入座（2 分钟）

### 你的 STATUS 是什么？

#### 🟢 HEALTHY（四个指标全绿）

**结论：模型结构健康。可以直接微调。**

下一步：
- 微调中每 50 步跑一次 FPP，盯着 **Phase**
- Phase 到达局部峰值时 → 评估下游任务 → 大概率是最优 checkpoint
- 比 Loss 最低点早 100-200 步，保住 30-40% 推理能力

#### 🟡 NEEDS ATTENTION（1-2 个指标黄了）

按症状查：

| 症状 | 可能原因 | 怎么办 |
|:---|:---|:---|
| **GS 偏低** | 结构有序度不足，可能是预训练不充分或架构问题 | ① 检查是否选了 GS 偏低的模型家族（如 Gemma 1B, SmolLM2-1.7B）；② 考虑换同家族 GS 更高的模型；③ 尝试轻量动量注入（先从 β=0.01 试） |
| **MI 偏低** | 信息通道窄，模型可能过度压缩 | ① MI 是架构决定的——微调改不了上限；② 如果下游任务需要长程信息保留（因果推理、长文本），换 MI 更高的模型；③ 短文本分类/补全对 MI 不敏感，可以继续用 |
| **Phase 偏低** | 层之间在干同一件事，分工退化 | ① 如果微调中 Phase 持续下降 → 立即停止，模型在退化；② 回退到 Phase 峰值 checkpoint；③ 降低学习率重新微调 |
| **DC 偏高 (>0.3)** | Pearson 在撒谎，GS 的实际可靠性下降 | ① **所有决策优先参考 MI 而不是 GS**；② 如果 MI 正常，模型仍可用，但不要基于 GS 做优化决策；③ 典型高 DC 模型：TinyLlama (DC=0.66) |

#### 🔴 AT RISK（多个指标红了）

**立即行动：**

1. **GS 和 MI 双低** → 模型结构严重受损。检查是否：加载了损坏的 checkpoint、量化精度过低、模型文件不完整
2. **Phase 极低 (<0.10)** → 层几乎完全同质化。这个模型的微调空间极其有限。**换模型。**
3. **DC > 0.5** → 度量系统不可信。不仅 GS，Phase 也可能受影响。需要 n≥20 的重复测量来确认任何结论。

---

## 第四步：速查表 — 你的模型该怎么调

### β（动量注入）安全范围

在微调优化器中加入动量项之前，先查这张表。同一个 β=0.2，Qwen +19%，Gemma 直接崩。

| 你的模型属于 | 安全 β | 预期 GS 提升 | 注意事项 |
|:---|:---|:---|:---|
| Qwen (SwiGLU+MHA) | [0.05, 0.20] | +19% | 7B 全范围安全 [0.01, 0.50] |
| OPT (ReLU+MHA) | [0.01, 0.20] | +32% | 老旧架构但动量响应最好 |
| TinyLlama / SmolLM2-360M (SwiGLU+GQA) | [0.01, 0.02] | 微量 | GQA 天生窄窗，别贪 |
| Gemma (GeGLU) | [0.001, 0.02] | 微量 | 超过 0.02 必崩 |
| SmolLM2-1.7B (SwiGLU+GQA) | ☠️ **禁止** | N/A | 任何 β 都会崩塌 |
| Pythia (GELU+MHA) | [0.01, 0.02] | 保守 | 小模型，保守注入 |

### 微调中 Phase 监控策略

```
每 50 步跑: python fpp_health.py --model ./checkpoint-XXX

Phase ↗ 上升  → ✅ 继续，结构在变好
Phase → 平稳  → ⚠️ 接近峰值，准备评估下游任务
Phase ↘ 下降  → 🔴 停止！回退到上一个 Phase 峰值 checkpoint
```

**实战案例**：Qwen 0.5B LoRA 微调 500 步。Phase 在第 250 步达峰 → 因果推理 40%。第 400 步 Loss 最低 → 因果推理 7%。第 500 步 Val PPL 最低 → 因果推理 0%。**Phase 峰值比 Loss 最优点多保住 33 个百分点的推理能力。**

### 不知道该不该调的决策树

```
你的模型 GS 在家族正常范围内？
  ├─ 是 → 直接微调。每 50 步看 Phase。Phase 峰值 = 最优。
  └─ 否 → GS 低于家族下限？
           ├─ 是 → 先扫 β。能不能用动量注入提升 GS？
           │        ├─ β 安全 → 注入后重新测 FPP，GS 恢复后再微调
           │        └─ β 不安全 → 换模型。这个架构没有优化空间。
           └─ GS 高于家族上限 → 模型可能过正则化/僵化。
                                降低学习率微调，密切关注 Phase 是否松动。
```

---

## 附录：命令行速查

```bash
# 基础诊断
python fpp_health.py --model ~/models/my-model

# 高精度模式（20 次重复测量，DC>0.3 时推荐）
python fpp_health.py --model ~/models/my-model --n-runs 20

# 详细输出（看每个输入类型的 GS/MI/Phase）
python fpp_health.py --model ~/models/my-model --verbose

# 微调中监控 checkpoint
python fpp_health.py --model ./checkpoint-250
```

---

## 一句话总结

**微调前跑一次 FPP → 知道基座模型有没有暗病。微调中每 50 步跑一次 → Phase 峰值就是最优 checkpoint。想加动量 → 先查 β 安全表。这 5 分钟省下的是几百步白训和部署后的事故。**

---

> 📄 论文：https://huggingface.co/youyouYUE/golden-window
> 💻 工具：https://github.com/GIS-blackCaat/fpp-golden-window
