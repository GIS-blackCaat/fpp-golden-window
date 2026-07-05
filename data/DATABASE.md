# FPP V4 评价数据库

## 模型覆盖

**9 模型 × 4 架构家族 × 2 注意力类型 × 7.5M-1.7B 规模**

| # | 模型 | 家族 | 注意力 | 规模 | 训练 | GS | MI | Phase | DC |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| 1 | Qwen2.5-0.5B-Base | SwiGLU | MHA | 0.5B | 预训练 | 0.813 | 0.129 | 0.416 | 0.047 |
| 2 | Qwen2.5-0.5B-Inst | SwiGLU | MHA | 0.5B | +Instruct | 0.812 | 0.087 | 0.410 | 0.091 |
| 3 | Qwen2.5-1.5B-Inst | SwiGLU | MHA | 1.5B | +Instruct | 0.891 | 0.103 | 0.321 | 0.181 |
| 4 | SmolLM2-360M-Inst | SwiGLU | GQA | 360M | +Instruct | 0.890 | 0.069 | 0.311 | 0.396 |
| 5 | SmolLM2-1.7B-Inst | SwiGLU | GQA | 1.7B | +Instruct | 0.429 | 0.216 | 0.469 | 0.180 |
| 6 | TinyLlama-1.1B-Chat | SwiGLU | GQA | 1.1B | +Chat | 0.798 | 0.771 | 0.389 | 0.661 |
| 7 | Gemma-3-1B | GeGLU | MHA | 1.0B | 预训练 | 0.282 | 0.096 | 0.311 | 0.087 |
| 8 | OPT-1.3B | ReLU | MHA | 1.3B | 预训练 | 0.599 | 0.138 | 0.213 | 0.018 |
| 9 | YouYou-GPT2 | SwiGLU | MHA | 7.5M | 动量训入 | 0.236 | 0.301 | 0.181 | 0.173 |
| — | Pythia 160M | GELU | MHA | 160M | 预训练 | ~0.48 | ~0.01 | ~0.29 | — |
| — | FieldCell-d4096 | Laplacian | 无 | — | 多模态 | 0.48→0.01 | 0.04 | — | — |

## 安全 β 校准表

| 家族 | 安全 β 范围 | 代表模型 |
|:---|:---|:---|
| SwiGLU+MHA | [0.05, 0.20] | Qwen 两代 |
| SwiGLU+GQA | 需逐个扫描 | TinyLlama [0.01,0.02], SmolLM2 待测 |
| GeGLU | [0.001, 0.02] | Gemma，仅超微β安全 |
| ReLU | [0.01, 0.20] | OPT，动量响应最强(+32%) |
| GELU | 待测 | Pythia (tokenizer缺) |

## 核心发现

### 1. GS 由激活函数主导，但同族内可差2倍
- SwiGLU: 0.43-0.89 (2.1×)
- GeGLU: 0.28
- ReLU: 0.60
- 规模不是主因：360M的GS(0.89) > 1.7B的GS(0.43)

### 2. MI 跨模型差40倍，训练配方是主因
- TinyLlama MI=0.77 是 Qwen的40倍
- SmolLM2-360M MI=0.07 是最低的
- Instruct微调系统性降低MI (Qwen: 0.13→0.09, -33%)
- Chat微调可能推高MI

### 3. Phase 唯一跨架构稳定 (0.18-0.47)
- 所有模型Phase都在此范围
- 不受激活函数、规模、注意力类型显著影响

### 4. GS和MI正交 (r≈-0.10)
- 两个独立自由度，证实了V3的发现

### 5. 动量注入三模式
- Qwen型: β=0.2 → GS+19%，温和增强
- OPT型: β=0.1 → GS+32%，强烈正向
- TinyLlama型: 高MI模型，β需缩到0.01才安全
- Gemma型: 低GS模型，β需缩到0.001(千分之一)
- 训入权重(YouYou): 对GS无提升，但Phase+10%, DC-16%

### 6. 三阶段生命周期跨训练范式成立
- 从零训练(Pythia): Build→Collapse→Rebuild
- 微调(Qwen): Collapse→Rebuild (无Build)
- FieldCell多模态: Build→Collapse→死锁 (无Rebuild)

### 7. FieldCell vs Transformer 结构差异
- FieldCell: 时间对称初始值高(GS=0.67)，但SGD摧毁后无法恢复
- Transformer: 时间对称初始值低(GS=0.0)，但有三阶段自我修复
- FieldCell优势: 非Gram矩阵、O(d²)缩放、边→频率映射
- FieldCell劣势: 无Rebuild机制、Phase很低、未经大规模验证

## 评价体系能力

| 能回答的问题 | 答案来源 |
|:---|:---|
| 这个模型结构健康吗？ | GS vs 同家族基线 |
| 信息通道够不够？ | MI 绝对值 + 家族范围 |
| 能用动量注入吗？ | β安全校准表 |
| 微调什么时候该停？ | Phase峰检测 |
| 两个模型有多大差异？ | 5维雷达图对比 |
| 微调改变了什么？ | Before/After FPP |
| 这个架构和Transformer有什么不同？ | FieldCell vs 9模型对比 |
| Loss在骗我吗？ | DC>0.3 → Pearson不可靠 |

## 待补缺口

| 缺口 | 优先级 |
|:---|:---|
| GELU β扫描 (Pythia tokenizer) | P0 |
| SmolLM2 β扫描 (两个规模) | P0 |
| v3 FieldCell文本模型训练+FPP | P1 |
| 下游benchmark验证(MMLU/GSM8K) | P1 |
| 7B+ 规模验证 | P2 |
