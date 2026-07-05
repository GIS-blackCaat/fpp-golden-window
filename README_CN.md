# FPP — Fieldcell Processing Protocol（现场胞处理协议）

**Loss Is Not Enough（Loss 不够）。** FPP 是一个四维诊断框架，测量 Loss 看不到的东西：结构有序度、信息通道容量、层功能分化度、度量诚实度。

> 13 个模型，5 个架构家族，7.5M 到 14B 参数。全部实验在一台 GTX 1650 Ti（4GB 显存）上完成。

---

## 快速开始

```bash
git clone https://github.com/GIS-blackCaat/fpp-golden-window.git
cd fpp-golden-window
pip install -r requirements.txt
python fpp_health.py --model Qwen/Qwen2.5-0.5B-Instruct
```

支持任何 HuggingFace Transformers 模型（本地路径或模型名）。5 秒完成体检。

---

## ⏱️ 五分钟诊断指南

第一次用？从这里开始：**[5MIN_GUIDE.md](5MIN_GUIDE.md)** —— 加载模型 → 跑 FPP → 看四个指标 → 决定下一步怎么调。含决策树和 β 安全速查表。

---

## 你会得到什么

- **架构指纹**（激活函数、注意力机制、层数）
- **FPP 四维基线**（GS 结构有序度、MI 信息通道、Phase 层分化度、DC 欺骗指数）
- **健康评估**（vs 同家族正常范围）
- **β 安全范围**（动量注入的安全参数区间）
- **优化建议**（能不能微调、能不能加动量、Phase 峰值导航）

## 实例脚本

```bash
# 程序化调用 — 在自己的代码中使用 FPP
python example.py --model Qwen/Qwen2.5-0.5B-Instruct

# 微调监控演示 — Phase 导航最优 checkpoint 选择
python demo_finetune_monitor.py
```

---

## 四维指标

| 指标 | 中文名 | 测什么 | 一句话 |
|:---|:---|:---|:---|
| **GS** | 结构有序度 | 正向/反向计算的时间反演对称性 | 越高 → 结构越健康 |
| **MI** | 信息通道容量 | 信息从输入到输出保留了多少 | 越高 → 信息通道越宽，但上限由架构决定 |
| **Phase** | 层功能分化度 | 各层是否在做不同的事 | 越高 → 层分工越明确；峰值 = 最优 checkpoint |
| **DC** | 欺骗指数 | Pearson 相关性是否在撒谎 | >0.3 → 不要轻信 GS，优先参考 MI |

---

## β 安全速查表

在微调优化器中加入动量之前，先查这张表。同一个 β=0.2，Qwen +19%，Gemma 直接崩。

| 你的模型属于 | 安全 β 范围 | 预期效果 |
|:---|:---|:---|
| Qwen 系列 (SwiGLU+MHA) | [0.05, 0.20] | GS +19% |
| OPT (ReLU+MHA) | [0.01, 0.20] | GS +32% |
| TinyLlama / SmolLM2-360M (SwiGLU+GQA) | [0.01, 0.02] | 仅小 β 可用 |
| Gemma (GeGLU) | [0.001, 0.02] | 超微量，β>0.02 会死 |
| SmolLM2-1.7B (SwiGLU+GQA) | ☠️ **禁止** | 任何 β 都会崩塌 |
| Pythia (GELU+MHA) | [0.01, 0.02] | 保守注入 |

---

## 论文

**"Loss Is Not Enough: The Golden Window in Neural Network Training"**

- 📄 **论文 PDF**：[HuggingFace](https://huggingface.co/youyouYUE/golden-window)
- 🔬 **核心发现**：训练不是单调变好的——神经网络经历 Build → Collapse → Rebuild 三个阶段，Loss 看不到中间的结构崩塌
- 🪟 **黄金窗口**：step ~1,000 时结构质量达到峰值，之后被牺牲以换取边际 Loss 收益
- 💻 **硬件**：全部实验在消费级硬件上完成，大模型研究不属于大公司

---

## 数据

本仓库包含 10 个预计算健康报告 + 7B 验证数据 + 完整校准数据库。所有数据开放，无商业限制。

---

## 中文资源

完整的中文文档和工具包请见：[模型健康学](https://github.com/GIS-blackCaat/fpp-golden-window)（同仓库）

- 公众号系列文章（7 篇，面向工程师/研究者/开源团队）
- 知乎深度分析（工程师向，含完整数据表）

---

## 硬件要求

| 模型规模 | 最低配置 |
|:---|:---|
| ≤ 2B 参数 | 4GB 显存 GPU 或 8GB RAM CPU |
| 7B 参数 | 16GB RAM（CPU 推理） |
| 14B 参数 | 32GB RAM（CPU 推理） |

---

## 许可

MIT — 随便用，随便改，商业友好。

---

## 引用

```bibtex
@article{yue2026goldenwindow,
  title={Loss Is Not Enough: The Golden Window in Neural Network Training},
  author={Yue, Song},
  journal={arXiv preprint},
  year={2026}
}
```
