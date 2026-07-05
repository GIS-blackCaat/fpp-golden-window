# FPP — Fieldcell Processing Protocol（现场胞处理协议）

**Loss Is Not Enough（Loss 不够）。** FPP 是一个四维诊断框架，测量 Loss 看不到的东西：结构有序度、信息通道容量、层功能分化度、度量诚实度。

> 13 个模型，5 个架构家族，7.5M 到 14B 参数。全部实验在一台 GTX 1650 Ti（4GB 显存）上完成。

---

## 快速开始

### Docker（一行命令，零配置）

```bash
docker run -it ghcr.io/gis-blackcaat/fpp-golden-window:latest demo
```

跑完整演示流程：健康检查 → 模型对比 → Phase 导航监控 → 基线表。

### 或者：本地 pip 安装

```bash
git clone https://github.com/GIS-blackCaat/fpp-golden-window.git
cd fpp-golden-window
pip install -r requirements.txt
python tools/fpp_health.py --model Qwen/Qwen2.5-0.5B-Instruct
```

支持任何 HuggingFace Transformers 模型。5 秒完成体检。

---

## ⏱️ 五分钟诊断指南

第一次用？→ **[docs/5MIN_GUIDE.md](docs/5MIN_GUIDE.md)** —— 加载模型 → 跑 FPP → 看四个指标 → 决定下一步怎么调。含决策树和 β 安全速查表。

## 🤖 Claude Code Skill

仓库自带 Claude Code skill（`.claude/skills/fpp-expert.md`）。clone 后 Claude Code 自动发现，直接问：

> "这个模型健康吗？" / "微调什么时候该停？" / "能加动量吗？"

Skill 即时提供：指标解读、家族健康范围、β 安全表、决策树、4 个常用 workflow。

---

## 仓库结构

```
tools/                     ← 主工具：健康检查、核心指标、可视化
examples/                  ← 可运行的示例脚本
data/                      ← 健康报告 + 实验数据 + 评价数据库
docker/                    ← Dockerfile + compose + entrypoint
docs/                      ← 五分钟指南 + 安装说明 + 论文 PDF
figures/                   ← 4 张论文主图（PDF）
tables/                    ← 2 个 LaTeX 表格
```

---

## 四维指标

| 指标 | 中文名 | 测什么 | 一句话 |
|:---|:---|:---|:---|
| **GS** | 结构有序度 | 正向/反向计算的时间反演对称性 | 越高 → 结构越健康 |
| **MI** | 信息通道容量 | 信息从输入到输出保留了多少 | 架构决定上限，微调改不了 |
| **Phase** | 层功能分化度 | 各层是否在做不同的事 | 峰值 = 最优 checkpoint |
| **DC** | 欺骗指数 | Pearson 相关性是否在撒谎 | >0.3 → 优先参考 MI |

---

## β 安全速查表

在微调优化器中加入动量之前，先查这张表：

| 你的模型 | 安全 β 范围 | 预期效果 |
|:---|:---|:---|
| Qwen (SwiGLU+MHA) | [0.05, 0.20] | GS +19% |
| OPT (ReLU+MHA) | [0.01, 0.20] | GS +32% |
| TinyLlama / SmolLM2-360M (SwiGLU+GQA) | [0.01, 0.02] | 仅小 β 可用 |
| Gemma (GeGLU) | [0.001, 0.02] | 超微量，β>0.02 必崩 |
| SmolLM2-1.7B (SwiGLU+GQA) | ☠️ **禁止** | 任何 β 崩塌 |
| Pythia (GELU+MHA) | [0.01, 0.02] | 保守注入 |

---

## 实例脚本

```bash
# 程序化调用 — 在自己的 Python 代码中使用 FPP
python examples/basic_usage.py --model Qwen/Qwen2.5-0.5B-Instruct

# 微调监控演示 — Phase 导航最优 checkpoint 选择
python examples/finetune_monitor.py
```

---

## Docker — 一键复现论文实验

```bash
# 完整演示
docker run -it fpp-golden-window demo

# 测任意 HuggingFace 模型
docker run -it fpp-golden-window health Qwen/Qwen2.5-0.5B-Instruct

# 对比两个模型
docker run -it fpp-golden-window compare Qwen/Qwen2.5-0.5B TinyLlama/TinyLlama-1.1B-Chat-v1.0

# 打印 13 模型基线表
docker run -it fpp-golden-window table

# GPU 加速
docker run -it --gpus all fpp-golden-window demo

# 交互式 shell
docker run -it fpp-golden-window shell
```

从源码构建：

```bash
docker build -t fpp-golden-window -f docker/Dockerfile .
# 或者: cd docker && docker compose up
```

---

## 论文

**"Loss Is Not Enough: The Golden Window in Neural Network Training"**

- 📄 **PDF**：[HuggingFace](https://huggingface.co/youyouYUE/golden-window)
- 🔬 训练不是单调变好的——Build → Collapse → Rebuild 三个阶段，Loss 看不到崩塌
- 🪟 step ~1,000 时结构质量达到峰值（黄金窗口）

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

## 引用

```bibtex
@article{yue2026goldenwindow,
  title={Loss Is Not Enough: The Golden Window in Neural Network Training},
  author={Yue, Song},
  journal={arXiv preprint},
  year={2026}
}
```
