# 连续血糖监测（CGM）基础模型 (Continuous Glucose Monitoring Foundation Models)

连续血糖监测（Continuous Glucose Monitoring, CGM）通过皮下微创电化学传感器每 5～15 分钟连续测定组织间液葡萄糖浓度，是糖尿病及代谢综合征数字化管理的核心监测手段。本专题系统拆解以血糖时序为核心表征对象的代表性基础模型与标准化评测基准，并结合多目标、多架构全因子受控实验观察，总结 CGM 表征学习的设计空间与范式演进。

---

## 目录

- [1. CGM 信号生理特性与时序表征痛点](#1-cgm-信号生理特性与时序表征痛点)
- [2. 代表性连续血糖基础模型与评测基准](#2-代表性连续血糖基础模型与评测基准)
  - [GlucoFM](#glucofm)
  - [CGM-JEPA](#cgm-jepa)
  - [GluFormer](#gluformer)
  - [CGMformer](#cgmformer)
  - [CGM-LSM](#cgm-lsm)
  - [GlucoBench](#glucobench)
- [3. 核心 CGM 基础模型与基准横向对比](#3-核心-cgm-基础模型与基准横向对比)
- [4. CGM 基础模型设计空间实证洞察](#4-cgm-基础模型设计空间实证洞察)

---

## 1. CGM 信号生理特性与时序表征痛点

相比高频心血管波形信号（如 25～100 Hz PPG 或 250～500 Hz ECG），CGM 时序呈现出低采样率、强昼夜节律及独特的皮下生化动力学特性：

1. **昼夜节律与 288 点固定时间网格**：
   - 主流商用 CGM（如 Dexcom G6/G7、Medtronic Guardian）采样间隔为 5 分钟（部分型号如早期 Abbott FreeStyle Libre 为 15 分钟）。
   - 单日 24 小时完整时程恰好包含 $24 \times 60 / 5 = 288$ 个采样点。血糖波动受下丘脑-垂体-肾上腺轴皮质醇分泌（黎明现象）、胰岛素敏感性昼夜周期及睡眠节律调控，具有显著的 24 小时周期性。
2. **皮下组织间液生化延迟与双尺度动力学叠加**：
   - 传感器测定的是皮下组织间液葡萄糖，相比静脉或毛细血管全血存在 5～15 分钟的生理扩散滞后。
   - 信号在物理上由两个时间尺度叠加而成：**慢速生理基线漂移**（空腹血糖稳态、基础胰岛素释放与胰岛素抵抗底色）与**高频瞬态事件冲击**（进食碳水化合物引发的餐后血糖尖峰、外源短效胰岛素推注、剧烈运动消耗，以及夜间睡眠受压导致的传感器局部缺血假性骤降 Compression Artifact）。
3. **非随机观测缺失（Observational Missingness）**：
   - 传感器探针脱落、蓝牙传输断连与更换传感器时的初始化校准期会导致时序碎片化。在 CGM 建模中，传统的稠密线性插值或样条填补会向缺失区段注入虚假的平滑先验，破坏真实的餐后突变陡峭度与夜间稳态方差特征。

---

## 2. 代表性连续血糖基础模型与评测基准

### GlucoFM

> **发表信息**：arXiv 2026（2026 年 5 月） · [论文](https://arxiv.org/abs/2605.30865) · [配套基准 GlucoFM-Bench](https://arxiv.org/abs/2606.06881)  
> **机构**：Google Research / 新南威尔士大学 (UNSW)  
> **预训练数据**：477 名受试者（累计约 10.9 万小时连续血糖记录）  
> **开源状态**：论文公开（官方代码在论文中承诺开源）

![GlucoFM 架构与预训练流程](../figures/glucofm/overview.png)

图源：GlucoFM 论文 Fig. 1（arXiv:2605.30865）

GlucoFM（*GlucoFM: A Dual-Stream Foundation Model for Continuous Glucose Monitoring*）针对血糖时序中“慢稳态基线”与“快瞬态冲击”混杂的痛点，提出了轻量级双流解耦联合嵌入预测架构（Dual-Stream JEPA）。

#### 2.1 双流生理分解 (Dual-Stream Decomposition)

GlucoFM 在输入层将原始血糖序列解耦为两个并行的物理分支：
1. **状态流（State Stream，慢生理稳态）**：
   - 引入**可学习因果高斯滤波器（Learnable Causal Gaussian Filter）** 提取低频生理基线；
   - 滤波核带宽约束在 $\sigma \in [2, 12]$ 个网格步（对应物理时间约 10～60 分钟），通过 Sigmoid 重参数化与端到端反向传播自适应学习最优平滑尺度。
2. **事件流（Event Stream，瞬态偏移）**：
   - 残差项定义为 $\text{Event}(t) = \text{Signal}(t) - \text{State}(t)$；
   - 专门捕捉餐后血糖急剧拉升、运动后快速下降以及夜间局部受压伪影等高频突变成分。

#### 2.2 24h 时间网格对齐与观测掩码显式保留

- 将不规则采样时间戳对齐至统一的 288 点单日昼夜网格，**严格分离时间网格对齐与数值插值**；
- 显式保留二值观测掩码 $M \in \{0, 1\}$：对于 $\le 1\text{ h}$ 的短时缺失，在网格内标注 $M=0$ 并将掩码一路传递至注意力层与损失函数；对于 $> 1\text{ h}$ 的长时间断连，直接执行物理切段以防跨段污染。

#### 2.3 双预训练目标 (JEPA 潜空间预测 + 时序动力学残差)

GlucoFM 摒弃了在原始数值空间逐点重建的 MAE 范式，转向隐空间非生成式预测：
1. **掩码上下文潜空间预测（MCR, Masked Contextual Latent Prediction）**：
   - 随机掩蔽 50%～60% 的时序分块（Patch），由双向编码器根据可见分块预测被掩蔽分块在指数移动平均（EMA，动量系数 $0.997$）目标编码器中的潜表征向量；
   - 采用 Smooth L1 损失，并按分块内部真实有效观测点的比例对损失进行加权。
2. **时序动力学残差转移预测（TD, Temporal Dynamics Residual Prediction）**：
   - 引入隐空间状态递推头：$S_{\tau+1} = S_\tau + g(S_\tau, E_\tau, \tau)$；
   - 迫使模型显式建模当前慢状态 $S_\tau$ 在快事件 $E_\tau$ 与时间相位 $\tau$ 驱动下向下一时刻状态 $S_{\tau+1}$ 演变的物理过程。

#### 2.4 模型规模与下游临床评测

- **轻量级参数设计**：可训练参数量仅 **0.72M**（3 层 Transformer、4 注意力头、特征维度 $D=128$、前馈维度 $\text{FFN}=256$），单张 H100 GPU 即可完成全量预训练。
- **下游表征评估**：在配套的 **GlucoFM-Bench**（涵盖 CGMacros、Hall、Stanford、Shanghai 4 个独立队列）7 项临床代谢下游任务（糖尿病诊断分级、胰岛素抵抗 HOMA-IR、胰岛 $\beta$ 细胞功能障碍、Glucotype 血糖变异分型、高脂血症、低血糖风险、肥胖）中，冻结表征线性探针的平均 PR-AUC 相比对比基线提升 **+4.1 个百分点**，参数量约为百兆级对比模型的 $1/200$。

---

### CGM-JEPA

> **发表信息**：CRUISE Research Group 2025 · [权重与数据集](https://huggingface.co/CRUISEResearchGroup/CGM-JEPA)  
> **机构**：新南威尔士大学 (UNSW CRUISE Research Group)  
> **预训练数据**：228 名受试者（公开多队列标准化重组语料）  
> **开源状态**：开源预训练权重与配套数据集（Hugging Face）

![CGM-JEPA 架构](../figures/cgm-jepa/overview.png)

图源：CGM-JEPA 架构示意图

CGM-JEPA 是 GlucoFM 的单流联合嵌入预测前作原型与开源对照基线，为社区提供了可直接调用的预训练权重与标准化数据集切分。

- **架构与配置**：采用 24 Patch × 12 步（每块对应 1 小时）的一维卷积分块、EMA 动量目标编码器（动量 $0.997$）与单层轻量 Predictor 骨架，参数量约 **0.5M**。
- **开源生态**：在 Hugging Face 完整开放了预训练模型权重、预训练语料库 `CGM-JEPA-Pretraining` 以及下游评测集 `CGM-JEPA-Downstream`，是复现潜空间预测与双流滤波消融实验的重要开源基座。

---

### GluFormer

> **发表信息**：Nature 2026（预印本 arXiv 2024） · [论文](https://www.nature.com/articles/s41586-025-08639-6) · [代码](https://github.com/Guylu/GluFormer)  
> **机构**：Pheno.AI / 魏茨曼科学研究所 (Weizmann Institute) / NVIDIA  
> **预训练数据**：10,812 名非糖尿病及前驱糖尿病受试者（超 1,000 万条连续血糖读数，人类表型组计划 HPP）  
> **开源状态**：开源评估与特征提取代码（预训练权重受 HPP 队列隐私协议限制需申请）

![GluFormer 架构与训练流程](../figures/gluformer/overview.png)

图源：GluFormer 论文 Fig. 1（Nature 2026 / arXiv:2408.11876）

GluFormer（*A Foundation Model for Continuous Glucose Monitoring and Metabolic Health*）是基于万人级前瞻性纵向队列训练的自回归生成式血糖基础模型。

#### 核心架构与自回归建模

- **分箱离散化词元（Quantization Tokenization）**：将连续血糖测量区间均匀量化映射为 460 个离散词元（Bins），从而将连续生理时序建模转化为标准的因果语言模型下一词元预测任务（Next-Token Prediction）。
- **超长上下文窗口**：支持长达 **1,200 个 Token（对应约 12.5 天连续监测）** 的上下文窗口，采用 16 层 GPT 式单向因果 Transformer（特征维度 1024，参数量约 **135M**）。
- **临床长程预后与多模态扩展**：
  - **跨年远期结局预测**：在长达 2～12 年的前瞻性随访中，仅凭入组基线期的 14 天 CGM 表征，即可显著超越传统临床指标（HbA1c、空腹血糖）预测受试者未来发生 2 型糖尿病、心血管疾病及视网膜病变的风险；
  - **个性化饮食响应生成**：通过交叉注意力融合餐食宏量营养素记录，可自回归生成受试者进食特定食物后的餐后血糖波动轨迹。

---

### CGMformer

> **发表信息**：National Science Review (NSR) 2025 · [论文](https://doi.org/10.1093/nsr/nwaf082) · [代码](https://github.com/YurunLu/CGMformer) · [权重](https://github.com/YurunLu/CGMformer)  
> **机构**：中国科学院微系统与信息技术研究所 / 上海交通大学医学院附属第六人民医院  
> **预训练数据**：58,847 名中国受试者（累计 131 万天真实世界临床 CGM 记录）  
> **开源状态**：开源代码与官方预训练权重

CGMformer（*CGMformer: A Foundation Model for Continuous Glucose Monitoring in Real-World Clinical Practice*）是面向临床真实世界诊疗闭环的大规模掩码自编码血糖基座。

- **掩码语言建模（BERT-style MLM）**：采用双向 Transformer 编码器结构（提供 0.85M 至 10M 等多种参数规格），将单日 288 点血糖值量化为 260 个离散 Token 并引入 `PAD` 与 `MASK` 特殊标记，通过上下文双向注意力还原被随机掩蔽的血糖读数。
- **大规模临床应用验证**：依托上海六院等中心近 5.9 万人的真实世界监测队列预训练，重点验证了院内血糖变异性分型、糖尿病视网膜病变（DR）筛查以及个性化降糖药物与胰岛素方案辅助决策能力。

---

### CGM-LSM

> **发表信息**：arXiv 2024 · [论文](https://arxiv.org/abs/2412.09727) · [代码](https://github.com/JHU-CDHAI/cgmlsm)  
> **机构**：约翰霍普金斯大学数字健康与人工智能中心 (JHU CDHAI)  
> **预训练数据**：592 名受试者（约 1,600 万条连续血糖读数）  
> **开源状态**：开源训练与推理代码

![CGM-LSM 数据流程与架构](../figures/cgm-lsm/overview.png)

图源：CGM-LSM 论文 Fig. 1（arXiv:2412.09727）

CGM-LSM（*CGM-LSM: A Language Model-Based Foundation Model for Continuous Glucose Monitoring*）探索将经典语言模型骨架直接迁移至短程血糖轨迹预测任务。

- **设计机制**：基于 GPT-2 Small 架构（约 **124M** 参数），将连续血糖值划分为 400 个离散分箱词元；输入序列设定为 26 小时（24 小时历史上下文 + 未来 2 小时自回归预测窗口）。
- **短程外推表现**：在 OhioT1DM 等公开基准上，针对未来 30 分钟、60 分钟与 120 分钟的短程连续血糖预测，均方根误差（RMSE）相比传统 ARIMA 与循环神经网络（RNN）基线降低达 48.51%。

---

### GlucoBench

> **发表信息**：ICLR 2024 · [论文](https://openreview.net/forum?id=u4UaZq1Goi) · [代码](https://github.com/IrinaStatsLab/GlucoBench)  
> **机构**：德州农工大学 (Texas A&M University, Irina Gaynanova Lab)  
> **数据规模**：整合 5 个公开 CGM 队列（覆盖健康、前驱糖尿病、T1D 与 T2D 超 500 名受试者）  
> **开源状态**：开源完整评测管线与基线库

![GlucoBench 任务流程](../figures/glucobench/overview.png)

图源：GlucoBench 论文 Fig. 2（ICLR 2024）

GlucoBench 针对早期血糖预测研究数据集切分不一、评价指标单薄的问题，建立了首个跨队列标准化评测基准：
- **统一数据协议**：将 Hall、Colas、OhioT1DM、Dubosson、Weinstock 等涵盖不同人群与采样设备的公开队列统一对齐至 5 分钟时间网格，严格区分受试者内（In-distribution）与跨受试者零样本（Out-of-distribution）测试集。
- **临床安全评价体系**：除了常规数值误差（RMSE、MAE、MARD），强制引入**克拉克误差网格（Clarke Error Grid, Zone A+B 临床安全区占比）**，防止模型在低血糖或高血糖极值区发生具有临床风险的预测偏差。

---

## 3. 核心 CGM 基础模型与基准横向对比

| 对比维度 | Google GlucoFM | CRUISE CGM-JEPA | Pheno.AI GluFormer | 中科院 CGMformer | JHU CGM-LSM |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **发表来源** | arXiv 2026 | Hugging Face 2025 | Nature 2026 | NSR 2025 | arXiv 2024 |
| **学习范式** | 双流 JEPA 潜空间预测 | 单流 JEPA 潜空间预测 | GPT 式因果自回归生成 | BERT 式 MLM 掩码重构 | GPT-2 式因果自回归 |
| **词元化方式** | 连续 Conv1d + 双流滤波 | 连续 Conv1d 分块 | 460 Bins 离散化 | 260 Bins 离散化 | 400 Bins 离散化 |
| **参数量** | **0.72M** | **约 0.5M** | 约 135M | 0.85M～10M | 约 124M |
| **预训练规模** | 477 人（10.9 万小时） | 228 人（公开重组语料） | 10,812 人（>1,000 万读数） | 58,847 人（131 万天） | 592 人（1,600 万读数） |
| **时间上下文** | 单日 24 小时（288 点） | 单日 24 小时（288 点） | **1,200 点（约 12.5 天）** | 单日 24 小时（288 点） | 26 小时（24h 历史 + 2h 预测） |
| **缺失值处理** | 二值观测掩码显式保留 | 观测掩码保留 | 线性插值 | `PAD` 特殊标记填充 | 连续完整性切片过滤 |
| **核心优势任务** | 临床代谢表型线性探针判别 | 开源重训与迁移评测基线 | 2～12 年远期结局与饮食模拟 | 院内分型、并发症筛查与用药 | 30～120 分钟短程血糖外推 |
| **开源状态** | 论文公开（代码承诺开源） | 代码、数据与权重完全开源 | 代码开源（权重受限） | 代码与官方权重完全开源 | 代码开源 |

---

## 4. CGM 基础模型设计空间实证洞察

在基于 3 类自监督目标（JEPA 潜空间预测 vs. 原始值掩码重建 Recon vs. 因果自回归 Causal）× 3 类骨干架构（标准 Transformer vs. 双流 GlucoFM vs. 1D 卷积网络）的全因子矩阵（30 组受控预训练模型、三轨综合评测）实证研究中，CGM 表征学习展现出以下三条核心规律：

### 洞察 1：预训练目标函数决定下游表征上限

- **连续值掩码重建（MSE Recon）在高阶临床判别任务中表现最弱**：单纯迫使解码器在数值空间逐点拟合血糖读数，会使模型过度关注局部平滑插值先验，反而钝化了反映胰岛 $\beta$ 细胞功能衰退的宏观动态特征。
- **JEPA 潜空间预测与因果预测更具判别力**：JEPA（MCR + TD）迫使编码器在高维隐空间捕捉生理状态转移趋势，在 Hall Glucotype 血糖重度变异分型任务中，线性探针 AUROC 达到 **0.984**（相比随机初始化基线 0.841 提升显著）。

### 洞察 2：双流分解的物理归纳偏置增益

- 无论在 Transformer 还是 1D-CNN 骨干中，引入因果高斯滤波将慢速生理状态流（$\sigma \in [2, 12]$）与快速事件流显式剥离，均能为时序插补与趋势外推提供有效的物理归纳偏置。
- 在未观测区段时序插补探针评测中，双流分解架构将平均绝对误差（MAE）从单流基线的 23.8 mg/dL 降低至 **16.6 mg/dL（误差相对下降 30.3%）**。

### 洞察 3：显式观测掩码优于稠密线性插值

- 真实世界 CGM 记录的稀疏缺失反映了传感器脱落、受压与低功耗采样特性。将二值观测掩码 $M \in \{0, 1\}$ 显式传入注意力机制与损失函数，能让模型聚焦于高置信度真实观测区间建立动力学表征，相比预先进行线性插值的方案具有更强的跨设备泛化鲁棒性。
