# 通用时间序列基础模型 (General TSFM) 选型与跨界对比

基于自监督预训练的通用时间序列基础模型（General Time-Series Foundation Models, TSFM）在预测、插补与异常检测等通用时序任务上展现出零样本（Zero-Shot）迁移能力。本专题系统梳理学术界与工业界代表性通用 TSFM 的底层架构与训练机制，并探讨将其跨界应用于可穿戴生理信号分析时的适配优势与领域差异（Domain Gap）。

## 目录

- [1. 通用时序模型发展概况与技术路线分野](#1-通用时序模型发展概况与技术路线分野)
- [2. 经典模型剖析](#2-经典模型剖析)
  - [TimesFM](#timesfm)
  - [Chronos](#chronos)
  - [MOMENT](#moment)
  - [Mantis](#mantis)
  - [MOIRAI](#moirai)
  - [Timestral](#timestral)
  - [Kumo-Forecast](#kumo-forecast)
  - [TimeBraid](#timebraid)
  - [Time-LLM](#time-llm)
- [3. 通用时序基础模型横向对比矩阵](#3-通用时序基础模型横向对比矩阵)
- [4. 通用 TSFM 应用于穿戴生理信号的领域差异与适配建议](#4-通用-tsfm-应用于穿戴生理信号的领域差异与适配建议)
- [5. 核心文献与资源汇总](#5-核心文献与资源汇总)

---

## 1. 通用时序模型发展概况与技术路线分野

时间序列不同于自然语言（离散且具严格语义句法）和图像（二维空间局部相关性）。通用时序基础模型主要演化出以下设计路线：

1. **连续值分块与因果解码预测**（Continuous Patch & Causal Decoder）：如 Google TimesFM、Mistral Timestral。保留原始数值连续性，无分箱精度损失；Timestral 引入解耦预测跨度（DHF）与协变量支持；
2. **离散分箱与语言模型迁移**（Quantized Tokenization via LLM）：如 Amazon Chronos、Chronos-Bolt。借力成熟的 NLP Transformer 架构与交叉熵优化；
3. **掩码时序自编码与多任务表征**（Masked Autoencoding / MAE）：如 CMU MOMENT、Salesforce MOIRAI。支持任意位置缺失插补（Imputation）、异常检测与下游微调；
4. **几何自适应分块与时序判别学习**（Adaptive Patching & Classification）：如 Huawei Mantis。侧重时序分类、动作识别与事件检测，参数量紧凑适合端侧运行；
5. **检索增强与时序可解释预测**（Retrieval-Augmented TSFM）：如 NVIDIA Kumo-Forecast。结合神经网络预测与历史模式 $k$-NN 检索，内置特征与滞后归因层；
6. **跨模态时序重编程**（Cross-Modality Reprogramming via Text Prototypes）：如 Time-LLM。冻结大语言模型主干，通过词表原型软匹配进行特征对齐；
7. **多模态时序-语言深度交错基座**（Multimodal Interleaved TSFM）：如 UCSD / USC TimeBraid。构建连续数值与语言 Token 的双向跨注意力编织，支持假设性情景推演（What-If）。

---

## 2. 经典模型剖析

### TimesFM

Google Research · ICML 2024 · [论文](https://arxiv.org/abs/2310.10688) · [代码](https://github.com/google-research/timesfm)

![TimesFM 架构图](../figures/timesfm/overview.png)

图源：TimesFM 论文 Fig. 1（ICML 2024）

TimesFM 是 Google 开发的纯解码器因果时序预测基础模型家族（提供 200M 与 500M 规格）。

#### 核心设计机制

1. **分块输入与残差 MLP（Patch-Level Tokenization）**：
   - 将一维连续时序切分为长度为 $P$（如 32）的非重叠分块（Patches）；
   - 通过输入残差 MLP 映射为隐空间 Token 嵌入，降低 Transformer 序列长度，支持长程上下文输入；
2. **纯解码器因果自回归（Decoder-Only Causality）**：
   - 采用因果注意力掩码（Causal Mask），基于过去的 Patch 预测未来的 Patch；
   - 输出头支持单步与多步点预测，以及分位数输出（Quantile Output）以提供不确定性区间估计；
3. **多领域预训练语料**：
   - 在涵盖搜索趋势、维基百科流量、气象、电力、交通等超千亿时间点的真实与合成时序数据上进行训练。

---

### Chronos

Amazon Web Services · ICML 2024 · [论文](https://arxiv.org/abs/2403.07815) · [代码](https://github.com/amazon-science/chronos-forecasting)

![Chronos 流程架构图](../figures/chronos/overview.png)

图源：Chronos 论文 Fig. 1（ICML 2024）

Chronos 探索了将语言模型架构直接迁移至时间序列预测的可行性（提供 T5-Tiny 20M 至 T5-Large 710M 规格，以及加速版本 Chronos-Bolt）。

#### 核心设计思想

1. **缩放与离散量化（Scaling & Quantization）**：
   - 对输入序列进行局部均值方差缩放，随后通过均匀量化划分为固定数量的分箱（如 4,096 个 Bins）；
   - 将每个实数值映射为一个离散词元 ID，把时序回归问题转化为自回归分类交叉熵问题；
2. **复用 T5 编码器-解码器**：
   - 沿用标准预训练语言模型的架构超参，利用 Next-Token 预测损失优化分布拟合；
3. **Chronos-Bolt 加速演进**：
   - 后续演进版本 Chronos-Bolt 引入 Patch 化双向编码器配合直接多步预测头，单次前向输出完整预测视界，大幅提升了推理吞吐。

---

### MOMENT

Carnegie Mellon University (Auton Lab) · ICML 2024 · [论文](https://arxiv.org/abs/2402.04385) · [代码](https://github.com/moment-timeseries-foundation-model/moment)

![MOMENT 架构图](../figures/moment/overview.png)

图源：MOMENT 论文 Fig. 3（ICML 2024）

MOMENT 是基于 Patch-MAE 构建的多任务开源时间序列基础模型（提供 Small、Base 及 385M Large 规格）。

#### 核心设计机制

1. **Patch-Based Masked Autoencoder（Patch-MAE）**：
   - 将时序划分为大小为 8 或 16 的 Patch，预训练阶段对序列随机掩码 50%～70% 的 Patch；
   - 编码器采用标准双向 Transformer，解码器采用轻量级线性投影头重建被掩码部分；
2. **通道独立策略（Channel-Independence）**：
   - 多变量时序采用通道独立处理，不同传感器通道共享同一套网络参数，避免不同通道维数与排列顺序带来的维度适配问题；
3. **Time-series Pile 语料与多任务支持**：
   - 训练集整合了 13 个领域的公开时序，涵盖部分心电（ECG）、脉搏波（PPG）、脑电（EEG）与人体活动（HAR）数据；
   - 原生支持四项下游基准任务：长程预测（Forecasting）、缺失插补（Imputation）、异常检测（Anomaly Detection）与表征分类（Classification）。

---

### Mantis

华为诺亚方舟实验室 / Inria · ICML 2026 · [论文](https://arxiv.org/abs/2502.15637) · [代码](https://github.com/vfeofanov/mantis)

![Mantis 架构图](../figures/mantis/overview.png)

图源：Mantis 论文 Fig. 2（ICML 2026）

Mantis 针对多数时序基础模型聚焦于预测（Forecasting）而分类（Classification）支持不足的问题，提出了面向判别式任务的轻量时序基座（参数量约 8M）。

#### 核心设计机制

1. **Token Generator Unit (TGU)**：
   - 摒弃固定长度切块，采用多尺度 1D 卷积与差分分块提取局部极值与形态过渡特征；
2. **8M 轻量 ViT-1D 骨干与对比学习**：
   - 基于轻量一维视觉 Transformer 架构，通过自监督对比学习构建高判别度特征空间；
   - 在 CPU 上具备低延迟推理特性，适合端侧人体活动识别（HAR）与事件筛查。

---

### MOIRAI

Salesforce AI Research · ICML 2024 · [论文](https://arxiv.org/abs/2402.02592) · [代码](https://github.com/SalesforceAIResearch/uni2ts)

![MOIRAI 统一多变量架构图](../figures/moirai/overview.png)

图源：MOIRAI 论文 Fig. 2（ICML 2024）

MOIRAI 提出了面向任意变量、多任务的通用时序预测 Transformer 架构（提供 Small 14M、Base 91M 与 Large 311M 规格）。

#### 核心设计机制

1. **多尺寸分块投射（Multi-Patch Size Projection）**：
   - 支持不同采样频率的时序动态选择 Patch 尺寸（8, 16, 32, 64 等），适应不同变化速率的序列；
2. **任意变量掩码注意力（Any-Variate Attention）**：
   - 将多变量时序在时间和变量两个维度展开为一维 Token 序列，构建跨时间与跨通道的混合全注意力网络；
3. **LOTSA 数据集**：
   - 在包含 270 亿个观测值的 LOTSA 数据集上进行预训练，涵盖能源、交通与传感器记录。

---

### Timestral

Mistral AI · 2026 · [代码](https://github.com/mistralai/timestral) · [模型](https://huggingface.co/mistralai/Timestral-209M)

Timestral 是 Mistral AI 推出的原生零样本概率时序预测模型家族，提供 5M、28M、123M 与 209M 四个规格，支持纯 CPU 环境下的低延迟推理。

#### 核心设计机制

1. **解耦预测跨度机制（Decoupled Horizon Forecasting, DHF）**：
   - 传统因果解码器采用连续块掩码遮盖未来窗口，减少了历史上下文的利用率；
   - DHF 引入独立的可学习预测词元（Learned Forecast Tokens）读取完整历史序列，使历史观测值兼具完整上下文与监督训练目标；
2. **超长上下文与时间-变量双重注意力**：
   - 采用 32 时间步的 Patch 切分，支持最长 32,768 步历史上下文，单次前向可直接预测 1,024 步；
   - 融合时间注意力与跨变量注意力，支持单变量与多变量联合建模；
3. **全协变量支持与缺失值指示器**：
   - 支持历史协变量与已知未来协变量（如作息、用药/进食计划），并通过因果掩码阻断信息泄露；内置缺失值指示器，可直接处理存在间断的传感器序列；
4. **21 分位数概率输出**：
   - 基于 Pinball Loss 训练，每个预测时间步输出 21 个分位数（0.01 到 0.99），刻画预测的不确定性区间。

---

### Kumo-Forecast

NVIDIA · 2026 · [代码](https://github.com/NVIDIA/Kumo-TS) · [模型](https://huggingface.co/nvidia/Kumo-Forecast)

Kumo-Forecast 是 NVIDIA Kumo 企业级结构化数据基础模型家族中的多变量预测套件（基于 MOMENT-1-large 骨干，约 340M 参数）。

#### 核心设计机制

1. **冻结预训练编码器 + 可训练预测头**：
   - 预训练 Transformer Encoder 权重保持冻结，仅微调轻量级任务特定预测头，解耦通用表征提取与特定预测任务；
2. **领域感知表征与检索增强（DARR: Domain-Aware Representation and Retrieval）**：
   - 融合两条预测路径：直接模型预测（利用 Transformer 提取的局部表征）与历史模式检索（通过 $k$-NN 在历史数据库中动态检索相似的演化轨迹）；
3. **内置模型无关可解释性层（Built-in Interpretability Layer）**：
   - 原生支持特征归因、滞后步归因（Lag Attributions）与轨迹稳定性诊断，为预测结果提供依据支持；
4. **异常检测协同**：
   - 配合家族中的基于扩散模型的 `Kumo-Anomaly` 模块，支持多变量时序异常检测。

---

### TimeBraid

UC San Diego / USC · arXiv 2026 · [论文](https://arxiv.org/abs/2609.29792) · [代码](https://github.com/Aether-AI/TimeBraid)

![TimeBraid 架构图](../figures/timebraid/overview.png)

图源：TimeBraid 论文 Fig. 3（arXiv:2609.29792）

TimeBraid 提出了预训练时序基础模型与预训练大语言模型原生深度交错的跨模态架构（提供 1.2B、2.5B、6.7B 规格），旨在打破时序与自然语言之间的模态壁垒。

#### 核心设计机制

1. **双塔交错残差注意力（Interleaved Global Residual Attention）**：
   - 语言分支采用开源大语言模型（如 Qwen），时序分支采用连续值时序基础模型（如 TimesFM）；
   - 在网络的深度维度（如 28 层语言网络中交织 20 个时序融合块），文本 Token 与连续时序 Patch 进行双向跨注意力交互；
2. **统一提示词范式**：
   - 引入专用标记符 `<ts>` 与 `</ts>`，输入历史时序附带统计先验描述，支持时序问答与纯数值预测任务的统一表达；
3. **多任务自回归损失与受控反事实推演（Ctrl-F）**：
   - 语言输出使用交叉熵损失，时序输出采用多 Patch 预测回归头；
   - 验证了模型根据自然语言假设情景（如生活方式干预）推演未来对应时序趋势的能力。

---

### Time-LLM

Monash University / 蚂蚁集团 / HKUST · ICLR 2024 · [论文](https://arxiv.org/abs/2310.01728) · [代码](https://github.com/KimMeen/Time-LLM)

![Time-LLM 跨模态重编程架构图](../figures/time-llm/overview.png)

图源：Time-LLM 论文 Fig. 2（ICLR 2024）

Time-LLM 提出了跨模态时序重编程范式，旨在冻结大语言模型主干的前提下将连续时序映射至语言语义流形。

#### 核心设计机制

1. **输入切片与可逆标准化（Patching & RevIN）**：
   - 采用 RevIN 处理时序分布偏移，将连续序列划分为局部 Patch；
2. **文本原型重编程（Text Prototypes via Cross-Attention）**：
   - 从 LLM 词表中选取文本原型子集，通过多头跨注意力将时序 Patch 表示为词原型的软加权组合，保持与 LLM 输入嵌入空间的对齐；
3. **提示词前缀增强（Prompt-as-Prefix）**：
   - 拼接包含领域背景、任务设定与统计量（趋势、极值、周期）的结构化文本提示作为前缀；
4. **冻结骨干与轻量投影**：
   - 保持语言模型（LLaMA / GPT-2）主干参数冻结，仅更新重编程层与线性输出头（训练参数通常低于总参数量 1%）。

---

## 3. 通用时序基础模型横向对比矩阵

| 模型名称 | 主导机构 | 核心范式 | 词元化机制 | 参数规模 | 优势任务 | 在穿戴/生理领域的适配性 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TimesFM** | Google | 自回归因果生成 (Decoder) | 连续值 Patch (32 点) | 200M～500M | 点预测、分位数预测 | 适中（适合步数、作息等宏观趋势分析） |
| **Chronos** | Amazon | 离散分箱语言模型 (T5) | 4,096 Bins 离散量化 | 20M～710M | 复杂非线性分布预测 | 较弱（高频波形量化分箱易丢失收缩微形态） |
| **Chronos-Bolt** | Amazon | 非自回归多步预测 | Patch 离散化 | 40M～200M | 高吞吐时序预测 | 适中（适合小时级心率趋势预警） |
| **MOMENT** | CMU | 掩码重建 (Patch-MAE) | 连续值 Patch (8/16 点) | 385M (Large) | 信号插补、去噪、异常检测 | 良好（预训练包含生理数据，适合波形插补） |
| **Mantis** | 华为诺亚 | 几何对比学习 (ViT-1D) | 自适应卷积分块 (TGU) | 约 8M | 事件分类、活动识别、异常筛查 | 良好（轻量化适合端侧 HAR 与节律筛查） |
| **MOIRAI** | Salesforce | 任意变量掩码注意力 | 多尺度动态 Patch | 14M～311M | 跨传感器灵活预测 | 适中（跨变量注意力在密集通道下计算开销大） |
| **Timestral** | Mistral AI | 解耦预测跨度 (DHF) + 概率预测 | 连续 Patch + 预测词元 | 5M～209M | 概率预测、未来协变量注入 | 良好（适合 CGM 饮食/用药协变量血糖预测） |
| **Kumo-Forecast** | NVIDIA | 冻结 MOMENT + DARR 检索增强 | 连续 Patch + $k$-NN 检索 | 约 340M | 检索增强预测、可解释归因 | 良好（适合结合个体历史生理基线进行归因） |
| **Time-LLM** | Monash / 蚂蚁 | 跨模态时序重编程 | Patch + 文本原型匹配 | 冻结 7B (微调 <1%) | 少样本预测、长程外推 | 适中（适合低频宏观生理指标少样本冷启动） |
| **TimeBraid** | UCSD / USC | 多模态深度交错 (MoT) | 连续 Patch + 文本 Token | 1.2B～6.7B | 时序问答、反事实推演 | 良好（可作为时序与语言交织推理的底座方案） |

---

## 4. 通用 TSFM 应用于穿戴生理信号的领域差异与适配建议

### 1. 核心领域差异（Domain Gap）

将通用时序基础模型直接应用于智能手表（PPG / ACC）或连续血糖仪（CGM）时，通常存在以下领域差异：

1. **频域伪影与人体运动动力学**：
   - 通用模型缺少人体骨骼运动动力学先验。当手腕 PPG 中混杂高能量的跑步步频谐波时，通用自回归模型容易将运动步频视为主导周期进行外推，导致预测心率偏离真实生理节律；
2. **微形态切迹与病理特征敏感性**：
   - 脉搏波的重搏切迹（Dicrotic notch）、心电图的 ST 段微小变化仅占少数采样点。通用模型的粗粒度 Patch（如 32 或 64 点）或离散分箱（Binning）容易平滑掉这些形态特征；
3. **异构多采样率与传感器互补**：
   - 穿戴场景中 PPG（25～100 Hz）、ACC（25～50 Hz）与皮温（0.1 Hz）多速率并存。通用模型大多采用通道独立或重采样至统一频率，难以利用 ACC 作为运动伪影参考通道进行自适应降噪；
4. **非规则佩戴与长程物理断流**：
   - 通用模型多假设序列具备连续规则的时间网格。真实穿戴场景中受充电、脱表等影响存在频繁断流，需要连续物理时间位置编码进行补偿。

### 2. 落地适配工程建议

- **信号补全与去噪场景**：可选用 **MOMENT**，利用其 Patch-MAE 的隐空间插补能力；
- **端侧计步、动作识别与事件筛查**：可选用 **Huawei Mantis**（8M 参数），结合其 TGU 局部几何感知能力作为特征提取器用于端侧分类；
- **CGM 连续血糖概率预测与干预输入**：可选用 **Mistral Timestral**（5M / 28M），利用其对未来协变量（饮食碳水、胰岛素注射、运动计划）的原生支持与 21 分位数概率输出，推演未来血糖区间；
- **个体历史档案锚定与临床可解释归因**：可参考 **NVIDIA Kumo-Forecast** 的 DARR（时序检索增强）思路，利用 $k$-NN 从用户过往数周/数月的历史生理数据库中检索相似状态切面辅助预测，并借助滞后归因层提供解释依据；
- **健康问答与假设性推演（What-If）**：可借鉴 **TimeBraid** 的交错残差注意力与统一 Prompt 范式，探索“生理体征输入 + 生活干预文本 → 恢复轨迹预测与健康建议生成”的跨模态协同；
- **高频生理波形连续追踪**：仍建议以专有生理基础模型（如 **PaPaGei、Pulse-PPG、SleepFM**）为主干，通用 TSFM 可作为宏观趋势辅助参考。

---

## 5. 核心文献与资源汇总

| 项目 / 论文名称 | 主要机构 | 发表情况 | 核心定位与资源 |
| :--- | :--- | :--- | :--- |
| **TimesFM** | Google Research | ICML 2024 | 纯解码器因果时序预测基座 · [论文](https://arxiv.org/abs/2310.10688) · [代码](https://github.com/google-research/timesfm) |
| **Chronos** | Amazon AWS | ICML 2024 | 离散分箱与语言模型迁移预测基座 · [论文](https://arxiv.org/abs/2403.07815) · [代码](https://github.com/amazon-science/chronos-forecasting) |
| **MOMENT** | CMU Auton Lab | ICML 2024 | Patch-MAE 多任务通用时序基座 · [论文](https://arxiv.org/abs/2402.04385) · [代码](https://github.com/moment-timeseries-foundation-model/moment) |
| **Mantis** | 华为诺亚方舟实验室 | ICML 2026 | 8M 轻量级时序判别与分类基座 · [论文](https://arxiv.org/abs/2502.15637) · [代码](https://github.com/vfeofanov/mantis) |
| **MOIRAI** | Salesforce AI Research | ICML 2024 | 任意变量多任务掩码预测基座 · [论文](https://arxiv.org/abs/2402.02592) · [代码](https://github.com/SalesforceAIResearch/uni2ts) |
| **Timestral** | Mistral AI | 2026 | 解耦预测跨度 (DHF) 概率预测基座 · [代码](https://github.com/mistralai/timestral) · [模型](https://huggingface.co/mistralai/Timestral-209M) |
| **Kumo-Forecast** | NVIDIA | 2026 | DARR 检索增强与原生可解释时序预测套件 · [代码](https://github.com/NVIDIA/Kumo-TS) · [模型](https://huggingface.co/nvidia/Kumo-Forecast) |
| **TimeBraid** | UCSD / USC | arXiv 2026 | 时序与语言原生深度交错全能基座 · [论文](https://arxiv.org/abs/2609.29792) · [代码](https://github.com/Aether-AI/TimeBraid) |
| **Time-LLM** | Monash / 蚂蚁集团 | ICLR 2024 | 基于文本原型的跨模态时序重编程基座 · [论文](https://arxiv.org/abs/2310.01728) · [代码](https://github.com/KimMeen/Time-LLM) |
