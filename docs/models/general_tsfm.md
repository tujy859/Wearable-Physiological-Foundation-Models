# 通用时间序列基础模型 (General TSFM) 选型与跨界对比

> 近年来，基于自监督预训练的通用时间序列基础模型（General Time-Series Foundation Models, TSFM）在预测、插补与异常检测等通用时序任务上展现出强大的零样本（Zero-Shot）迁移能力。本专题系统性解构学术界与工业界代表性通用 TSFM 的底层架构与训练机制，并深入探讨将其跨界应用于**可穿戴生理信号**时的核心优势与“域鸿沟”（Domain Gap）。

---

## 📑 目录

- [1. 通用时序模型发展概况与技术路线分野](#1-通用时序模型发展概况与技术路线分野)
- [2. Google TimesFM —— 基于分块解码器的自回归预测基座](#2-google-timesfm--基于分块解码器的自回归预测基座)
- [3. Amazon Chronos & Chronos-Bolt —— 离散词元分箱与语言模型迁移](#3-amazon-chronos--chronos-bolt--离散词元分箱与语言模型迁移)
- [4. CMU Auton Lab MOMENT —— 多变量掩码自编码 (Patch-MAE) 全能底座](#4-cmu-auton-lab-moment--多变量掩码自编码-patch-mae-全能底座)
- [5. Huawei Noah's Ark Lab Mantis —— 面向时序分类的轻量判别基座](#5-huawei-noahs-ark-lab-mantis--面向时序分类的轻量判别基座)
- [6. Salesforce MOIRAI —— 任意变量多任务掩码预测基座](#6-salesforce-moirai--任意变量多任务掩码预测基座)
- [7. 通用时序基础模型横向对比矩阵](#7-通用时序基础模型横向对比矩阵)
- [8. 通用 TSFM 应用于穿戴生理信号的“域鸿沟”与适配建议](#8-通用-tsfm-应用于穿戴生理信号的域鸿沟与适配建议)

---

## 1. 通用时序模型发展概况与技术路线分野

时间序列不同于自然语言（离散且具严格语义句法）和图像（空间局部相关性）。通用时序基础模型主要演化出三大设计路线：

```text
通用时序基础模型 (General TSFM)
│
├── 连续值分块自回归预测 (Continuous Patch Autoregression): Google TimesFM
│   └── 优势: 保持原始数值连续性，无分箱精度损失，长程推断速度快。
│
├── 离散分箱与语言模型迁移 (Quantized Tokenization via LLM): Amazon Chronos, Chronos-Bolt
│   └── 优势: 借力成熟的 NLP Transformer (T5/GPT) 架构与交叉熵优化，分布拟合能力极强。
│
├── 掩码时序自编码与多任务表征 (Masked Autoencoding / MAE): CMU MOMENT, Salesforce MOIRAI
│   └── 优势: 原生支持任意位置缺失插补 (Imputation)、异常检测与下游客制化微调。
│
└── 几何自适应分块与时序判别学习 (Adaptive Patching & Classification): Huawei Mantis
    └── 优势: 专攻时序分类、动作识别与事件检测，小参数量支持端侧实时运行。
```

---

## 2. Google TimesFM —— 基于分块解码器的自回归预测基座

- **论文**: *A Decoder-Only Foundation Model for Time-series Forecasting* (ICML 2024 / arXiv:2310.10688)
- **研发团队**: Google Research
- **开源资源**: [GitHub: google-research/timesfm](https://github.com/google-research/timesfm) | Hugging Face 权重公开 (`google/timesfm-1.0-200m`, `timesfm-2.0-500m`)

### 核心架构与设计机制
1. **分块输入与残差 MLP (Patch-Level Tokenization)**:
   - 将 1D 连续时序切分为长度为 $P$（如 32）的非重叠分块（Patches）。
   - 通过输入残差 MLP 映射为隐空间 Token 嵌入，大幅降低了 Transformer 序列长度，使自注意力机制可处理长达数万步的历史上下文。
2. **纯解码器因果自回归 (Decoder-Only Causality)**:
   - 采用掩码多头自注意力（Causal Mask），只能基于过去的 Patch 预测未来的 Patch。
   - 输出头不仅支持单步/多步点预测，还支持分位数输出（Quantile Output），原生提供不确定性区间估计。
3. **百亿级跨领域预训练**:
   - 在涵盖 Google Trends、Wikipedia 流量、气象、电力、交通等超过 **1000 亿个时间点** 的真实与合成时序上训练。

---

## 3. Amazon Chronos & Chronos-Bolt —— 离散词元分箱与语言模型迁移

- **论文**: *Chronos: Learning the Language of Time Series* (ICML 2024) / *Chronos-Bolt* (2024-2025)
- **研发团队**: Amazon Web Services (AWS) AI Labs
- **开源资源**: [GitHub: amazon-science/chronos-forecasting](https://github.com/amazon-science/chronos-forecasting) | Hugging Face: `amazon/chronos-t5-tiny/small/base/large`, `amazon/chronos-bolt-small/base`

### 核心设计思想：时序即语言 (Time Series as Language)
1. **缩放与离散量化 (Scaling & Quantization)**:
   - 对输入序列进行局部均值方差缩放，随后通过均匀量化划分为固定数量的分箱（如 4096 个 Bins）。
   - 每个实数值被映射为一个唯一的词表 Token ID，彻底将时间序列数值回归转化为分类跨熵问题。
2. **经典 T5 编码器-解码器骨干**:
   - 直接复用预训练语言模型（T5-Tiny 20M 到 T5-Large 710M）的架构超参。
   - 目标函数为纯粹的 Next-Token 交叉熵损失，使模型对尖峰、突变等复杂非高斯分布具有优越的鲁棒性。
3. **Chronos-Bolt 极速推理演进**:
   - 原版自回归按点自回归推理较慢。Chronos-Bolt 引入 Patch 化双向编码器结合直接多步预测头，单次前向输出完整预测视界，推理速度提升超过 **250 倍**，显存开销骤降。

---

## 4. CMU Auton Lab MOMENT —— 多变量掩码自编码 (Patch-MAE) 全能底座

- **论文**: *MOMENT: A Family of Open Time-series Foundation Models* (ICML 2024)
- **研发团队**: Carnegie Mellon University (Auton Lab)
- **开源资源**: [GitHub: moment-timeseries-foundation-model/moment](https://github.com/moment-timeseries-foundation-model/moment) | Hugging Face: `AutonLab/MOMENT-1-large` (385M), `MOMENT-1-base`, `MOMENT-1-small`

### 核心架构与多任务全能性
1. **Patch-Based Masked Autoencoder (Patch-MAE)**:
   - 将时序划分为大小为 8 或 16 的 Patch。在预训练阶段对序列随机掩蔽高达 50%~70% 的 Patch。
   - 编码器采用标准双向 Transformer（12~24 层），解码器采用极轻量级线性投影头。
2. **多变量通道独立与解耦 (Channel-Independence)**:
   - 针对多维时序采用通道独立策略（Channel-Independent），不同传感器通道共享同一套网络权重，规避了不同通道维数与排列顺序导致的维度灾难。
3. **Time-series Pile 语料**:
   - 汇总了全球 13 个领域的公开时序，尤为可贵的是**大量吸纳了心电 (ECG)、脉搏 (PPG)、脑电 (EEG) 与人体活动 (HAR)** 生理医疗数据。
   - 原生支持四大任务基准：**长程预测 (Forecasting)、缺失补全 (Imputation)、异常检测 (Anomaly Detection) 及表征分类 (Classification)**。

---

## 5. Huawei Noah's Ark Lab Mantis —— 面向时序分类的轻量判别基座

- **论文**: *Mantis: Lightweight Foundation Model for Time Series Classification* (ICML 2026 / arXiv:2502.15637)
- **研发团队**: 华为诺亚方舟实验室 (Huawei Noah's Ark Lab), Inria 等
- **开源资源**: [GitHub: vfeofanov/mantis](https://github.com/vfeofanov/mantis) | PyPI: `pip install mantis-tsfm` | Hugging Face: `paris-noah/Mantis-8M`, `MantisPlus`

### 核心突破：填补时序大模型的“分类盲区”
- 以往时序基础模型（TimesFM, Chronos）多定位于**时序预测（Forecasting）**，而现实穿戴设备与物联网中最具商业价值的高频场景大量集中在**时序分类与判别**（如房颤心律失常筛查、人体姿态活动 HAR 识别、跌倒预警）。
- **Token Generator Unit (TGU)**:
  - 摒弃了固定长度切块，采用自适应多尺度 1D 卷积与差分分块，最大程度保留波形的局部极值与陡峭过渡特征。
- **8M 超轻量 ViT-1D 骨干与对比学习**:
  - 参数量仅 800 万，在 CPU 上推理仅需数十毫秒。在大量多样化时序上通过对比学习构建高判别度特征空间。
  - 冻结编码器提取特征后训练简单逻辑回归（Linear Probing），在 UCR/UEA 跨领域基准上击败众多大模型。

---

## 6. Salesforce MOIRAI —— 任意变量多任务掩码预测基座

- **论文**: *Unified Training of Universal Time Series Forecasting Transformers* (ICML 2024 / arXiv:2402.02592)
- **研发团队**: Salesforce AI Research
- **开源资源**: [GitHub: SalesForce/uni2ts](https://github.com/SalesforceAIResearch/uni2ts) | Hugging Face: `Salesforce/moirai-1.0-R-base/large`

### 核心架构：Any-Variate Transformer
1. **多尺寸分块投射 (Multi-Patch Size Projection)**:
   - 支持不同频率的时序动态选择 Patch 尺寸（8, 16, 32, 64 等），自适应高频震荡与低频平缓序列。
2. **任意变量掩码注意力 (Any-Variate Attention)**:
   - 将多变量时序在时间和变量两个维度展开为一维 Token 序列，构建跨时间与跨通道的混合全注意力网络，无需预先固定输入通道数。
3. **LOTSA 跨领域数据集**:
   - 包含 270 亿个观测值，覆盖金融、交通、能源与部分传感器日志。

---

## 7. 通用时序基础模型横向对比矩阵

| 模型名称 | 机构 | 核心范式 | 词元化机制 | 参数规模 | 优势任务 | 在穿戴/生理领域的适配性 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TimesFM** | Google | 自回归生成 (Decoder) | 连续值 Patch (32点) | 200M ~ 500M | 点预测、分位数预测 | 适中（适合宏观步数、昼夜作息趋势预测） |
| **Chronos** | Amazon | 自回归离散生成 (T5) | 4096 Bins 离散分箱 | 20M ~ 710M | 复杂非线性分布预测 | 较差（高频波形量化分箱易丢失收缩微形态） |
| **Chronos-Bolt** | Amazon | 单次前向多步生成 | Patch 离散化 | 40M ~ 200M | 极速高并发时序预测 | 适中（推断速度快，可用于小时级心率预警） |
| **MOMENT** | CMU | 掩码重建 (Patch-MAE) | 连续值 Patch (8/16点) | 385M (Large) | **信号补全、去噪、异常检测** | **极佳（Pile 含生理数据，适合信号插补）** |
| **Mantis** | 华为诺亚 | 几何对比学习 (ViT-1D) | 自适应卷积分块 (TGU) | **~8M (端侧友好)** | **事件分类、活动识别、异常筛查**| **极佳（适合手环端侧 HAR 与心律失常）** |
| **MOIRAI** | Salesforce | 任意变量掩码注意力 | 多尺度灵活 Patch | 14M ~ 311M | 跨传感器灵活预测 | 适中（通道注意力对强耦合生理流开销大） |

---

## 8. 通用 TSFM 应用于穿戴生理信号的“域鸿沟”与适配建议

尽管通用时序基础模型在宏观统计数据上表现亮眼，但将其**零样本直接迁移至智能手表或 CGM 传感器**时，存在显著的“域鸿沟”（Domain Gap）：

1. **频域伪影碰撞的不可知性**:
   - 通用模型（如 TimesFM）缺乏对人体骨骼运动动力学的先验。当手腕 PPG 中混杂着高达数十倍能量的跑步步频谐波时，通用自回归模型会把步频谐波视为主导周期进行外推，导致预测心率严重偏离真实生理节律。
2. **微形态切迹与动力学丢失**:
   - PPG 的重搏切迹（Dicrotic notch）、ECG 的 ST 段微小抬高仅占几个采样点。通用模型的固定粗粒度 Patch（如 32 或 64 点）或离散分箱（Binning）会直接抹平这些微弱但致命的病理形态。
3. **异构采样率与生理因果耦合**:
   - 穿戴场景中 PPG（100Hz）、ACC（50Hz）与温度（0.1Hz）多速率并存。通用模型大多采用通道独立（Channel-Independent）或等长重采样，无法利用 ACC 作为“运动伪影参考流”来主动消除 PPG 中的噪声。

### 落地适配工程建议
- **信号补全与去噪场景**: 优先选用 **MOMENT**，利用其 Patch-MAE 强大的局部隐空间插补能力；
- **端侧计步、动作识别与心律失常筛查**: 强烈推荐选用 **Huawei Mantis**，结合其轻量级 8M 体积与 TGU 局部几何感知能力，作为冻结特征提取器服务端侧分类；
- **连续生命体征高精度追踪与长程预测**: 建议以专用穿戴模型（如 **PaPaGei、Pulse-PPG、Watch-LSM**）为主干，通用模型作为高阶宏观趋势的辅助先验。
