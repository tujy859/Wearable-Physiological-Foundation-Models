# 智能手表与腕戴生理基础模型深度解析 (Smartwatch & Wrist Physiological Foundation Models)

> 腕戴设备（智能手表、智能手环）已成为普及度最高的消费级生理监测终端。本专题深入解构针对光电容积脉搏波 (PPG)、三轴加速度计 (ACC/IMU) 及微弱体表心电 (ECG) 研发的专用时序基础模型，梳理其对抗运动伪影、多速率对齐及端侧轻量化部署的底层技术演进。

---

## 📑 目录

- [1. 腕戴生理时序的独特性与核心技术挑战](#1-腕戴生理时序的独特性与核心技术挑战)
- [2. Google SensorFM (2026) —— 万亿分钟级全模态基座](#2-google-sensorfm-2026--万亿分钟级全模态基座)
- [3. Nokia Bell Labs PaPaGei (ICLR 2025) —— 开源脉搏形态基座](#3-nokia-bell-labs-papagei-iclr-2025--开源脉搏形态基座)
- [4. UIUC Pulse-PPG (UbiComp 2025) —— 野外非受限抗伪影基座](#4-uiuc-pulse-ppg-ubicomp-2025--野外非受限抗伪影基座)
- [5. Samsung Research xMAE & HiMAE (ICML 2024 / ICLR 2025) —— 端侧跨模态与分层架构](#5-samsung-research-xmae--himae-icml-2024--iclr-2025--端侧跨模态与分层架构)
- [6. Stanford Medicine SleepFM (Nature Medicine) —— 多器官耦合睡眠基座](#6-stanford-medicine-sleepfm-nature-medicine--多器官耦合睡眠基座)
- [7. 空间与运动判别基座：LIMU-BERT 与 Huawei Mantis](#7-空间与运动判别基座limu-bert-与-huawei-mantis)
- [8. 核心基础模型横向对比矩阵](#8-核心基础模型横向对比矩阵)
- [9. 对自建手表大模型 (Watch-LSM) 的工程架构启示](#9-对自建手表大模型-watch-lsm-的工程架构启示)

---

## 1. 腕戴生理时序的独特性与核心技术挑战

与抽象的通用单变量时序（如金融气象）不同，智能手表采集的生理流具备强物理与生物先验：

1. **异构采样率与多模态物理对齐 (Multi-Rate Heterogeneity)**:
   - **PPG (光电脉搏)**: 常见采样率 25Hz、50Hz、64Hz 或 100Hz，波形由微血管收缩期充血与舒张期弹性回缩构成，包含收缩峰（Systolic Peak）、重搏切迹（Dicrotic Notch）及舒张峰（Diastolic Peak）。
   - **ACC (三轴加速度计)**: 常见采样率 25Hz 或 50Hz，记录人体在空间三维的正交加速度矢量 $(a_x, a_y, a_z)$。
   - **ECG (偶发心电)**: 250Hz~1000Hz，用户主动按压表冠/电极时触发。
   - **低频慢速指标**: 皮肤温度 (Skin Temp, 0.1Hz)、皮电反应 (EDA, 4Hz)、气压高度计 (1Hz)。
2. **严重的运动伪影与频域碰撞 (Spectral Collision)**:
   - 手臂摆动、跑步落足引起的微小形变与静脉血液惯性流动，使光学传感器接收到的漫反射光被剧烈调制。
   - **致命碰撞**: 日常跑步步频谐波（1.0~3.5 Hz，即 60~210 SPM）与人体运动心率主频（1.2~3.2 Hz，即 70~190 BPM）完全重叠。传统经验带通滤波或峰值拾取（Peak-Picking）极易把步频误当心率，导致误差超过 50 BPM。
3. **极度受限的端侧计算资源**:
   - 智能手表电池通常仅 200~500 mAh，运行在 ARM Cortex-M 或低功耗 Cortex-A 核心上，要求模型参数小（<5M）、推理延迟低（<5ms）且满足整型定点量化。

---

## 2. Google SensorFM (2026) —— 万亿分钟级全模态基座

- **论文**: *Towards a General Intelligence and Interface for Wearable Health Data* (arXiv:2605.22759)
- **研发团队**: Google Research & Fitbit Health AI
- **发表年份**: 2026 年
- **开源状态**: 论文已发布；模型权重与代码闭源（商业生态核心资产）

### 核心规模与数据构成
- **数据总量**: 超过 **1 万亿分钟（>1,000,000,000,000 分钟）** 的真实可穿戴多模态时序。
- **用户基数**: 超过 **500 万** 名经过知情同意的 Fitbit 与 Google Pixel Watch 全球真实用户。
- **传感器模态**: 绿光/红外 PPG、三轴 ACC、EDA、皮肤温度、气压高度计。

### 核心架构与自监督算法
- **统一时间分块 (Multi-Rate Patch Tokenization)**:
  - 摒弃了简单粗暴的插值重采样，采用固定物理时间窗口（Physical Time Slice）作为统一的 Patch 跨度。
  - 为不同速率的传感器配备独立的线性投影嵌入层（Sensor-Specific Embedding），附加时间位置编码后输入统一的双向 Transformer 编码器。
- **复合预训练目标**:
  - **Masked Multimodal Modeling**: 在时域和模态域进行随机多维掩码，训练 Transformer 联合重建被遮蔽的生理片断。
  - **Cross-Modal Infilling**: 模拟硬件传感器掉线或低功耗节能策略（例如夜间关闭高耗电 PPG 仅开启低功耗 ACC），由存活模态在隐空间推断缺失模态表征。

### 下游任务与 Scaling 验证
- 在 **35 个下游健康与行为预测基准** 上进行了迁移评估，横跨四大领域：
  1. *心血管健康*: 静息心率、运动恢复心率、HRV、血压趋势、最大摄氧量 (VO2 Max)。
  2. *代谢与活动*: 步频能量消耗、高强度运动类型识别、代谢综合征风险评估。
  3. *睡眠与神经*: 睡眠深浅分期、REM 快速眼动期、阻塞性睡眠呼吸暂停。
  4. *情绪与压力*: EDA 突变检测与全天候生理压力指数。
- **关键结论**: 在 35 项任务中的 **34 项** 击败了此前针对单一任务精调的专用模型，在生理信号领域首次完整验证了 Scaling Law 的泛化飞轮。

---

## 3. Nokia Bell Labs PaPaGei (ICLR 2025) —— 开源脉搏形态基座

- **论文**: *PaPaGei: Open Foundation Models for Optical Physiological Signals* (ICLR 2025 / arXiv:2410.20542)
- **团队**: Nokia Bell Labs, University of Cambridge, Dartmouth College
- **开源资源**: [GitHub](https://github.com/Nokia-Bell-Labs/papagei-foundation-model) | [Zenodo 权重 (Record 13983110)](https://zenodo.org/records/13983110)

### 核心架构: ResNet1D-MoE
- **输入设定**: 10 秒长度的单通道 PPG 信号，统一重采样至 125 Hz（单样本 1250 点），并经过自适应 Z-score 标准化。
- **网络结构**:
  - 18 个 1D 残差卷积块（ResNet Blocks），基础特征通道 32。
  - 引入 3 个 Mixture of Experts (MoE) 专家分支，基于瞬态脉搏形态动态路由特征。
  - 全局平均池化输出 **512 维** 潜空间表征。参数量仅约 **1.5M**。

### 形态感知对比学习 (Morphology-Aware Pretraining)
- **破除经典对比学习在生理信号上的缺陷**:
  - 传统 SimCLR/BYOL 若将“同一用户不同时刻”视作正样本，会强制拉近该用户的平静心率波形与剧烈运动波形，从而抹杀血管阻抗、瞬时搏动的微形态特征。
- **形态引导配对采样 (Morphology-Guided Pair Sampling)**:
  - 提取波形收缩期上升斜率、舒张期凹槽深浅等生理先验，构建连续动态对比正负样本对，迫使模型保留心血管动力学特异性。

### 下游评测
- 整合 10 个公开数据集（VitalDB, MIMIC, CAPNO, WESAD, PPG-DaLiA 等），覆盖 20 项下游任务：
  - 心率回归 (HR Regression, MAE ~ 1.2-3.5 BPM)
  - 血压估计 (收缩压/舒张压 MAE ~ 5-8 mmHg)
  - 血管弹性生理年龄 (Vascular Age)
  - 睡眠呼吸暂停检测 (Sleep Apnea, F1 > 0.82)

---

## 4. UIUC Pulse-PPG (UbiComp 2025) —— 野外非受限抗伪影基座

- **论文**: *Pulse-PPG: An Open-Source Field-Trained PPG Foundation Model for Wearable Applications Across Lab and Field Settings*
- **团队**: University of Memphis & UIUC (Saha, Xu, Rehg, Kumar 等)
- **开源资源**: [GitHub](https://github.com/maxxu05/pulseppg) | Zenodo 权重开源

### 解决临床数据到手表的域漂移 (Clinical-to-Field Domain Shift)
- **临床数据的虚假干净**: MIMIC、VitalDB 中患者多处于麻醉或静卧状态，探头紧贴指尖，波形几乎无运动伪影。而在真实手腕上，受试者跑步、洗手、打字时信号信噪比急剧恶化。
- **大规模真实野外语料**: 采集 120 名受试者连续 100 天自由生活（Free-living）的真实腕戴数据，有效野外 PPG 数据量达 **2 亿秒（约 5.5 万小时）**。

### 相对对比学习 (Relative Contrastive Learning, RelCon)
- 放弃离散的 Positive/Negative 二元分类损失。
- 引入连续相似度衰减机制：将**时序物理间隔**与**三轴加速度计测算的受试者瞬时运动量**联合编码为连续权重，构建具备自适应温度调节的连续对比损失：
  $$\mathcal{L}_{\text{RelCon}} = - \sum_{i} \log \frac{\sum_{j} w(i, j) \cdot \exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k} \exp(\text{sim}(z_i, z_k)/\tau)}$$
- **架构**: 12 层 1D ResNet，采用大卷积核（Kernel Size = 11）以充分覆盖跨心动周期的长程时间依赖。在野外跨域生物年龄与高噪心率追踪中显著胜过仅在临床数据上训练的对比基线。

---

## 5. Samsung Research xMAE & HiMAE (ICML 2024 / ICLR 2025) —— 端侧跨模态与分层架构

- **团队**: Samsung Research (Health AI Lab)
- **论文**: xMAE (ICML 2024), HiMAE (ICLR 2025)
- **产品导向**: Galaxy Watch 等消费级智能穿戴设备的端侧低功耗实时推断。

### 5.1 xMAE: 跨模态虚拟传感 (PPG $\to$ 虚拟 ECG)
- **临床矛盾**: 心电图 (ECG) 具备极高的心脏电生理诊断价值，但智能手表要求用户另一手长按表圈电极形成回路，无法做到全天候被动连续监测；而背部光学 PPG 可 24 小时被动采集，但信息维度较低、噪声剧烈。
- **训练方案**:
  - 基于 9,400 小时配对同步采集的 ECG-PPG 数据集。
  - 将高精度 ECG 信号大比例掩码（如 75%），以配对的腕部 PPG 为上下文条件，训练 Cross-Modal Transformer 重建被掩码的 ECG 潜表征。
- **落地效果**: 推理期无需用户触碰表圈，仅凭日常 PPG 即可连续提取富含心脏电生理（如 QRS 复合波间期、心肌缺血特征）的高阶潜表征。

### 5.2 HiMAE: 分层多尺度端侧轻量化 (ICLR 2025)
- **生理时间尺度的分层建模**:
  - **尺度 1（微观，0.5 ~ 2 秒）**: 逐拍心跳微波形特征，服务于瞬时心率、心率变异性 (HRV)、脉搏波传导时间 (PWTT)。
  - **尺度 2（介观，5 ~ 30 秒）**: 肢体步态运动节奏与呼吸包络调制，服务于计步、能量消耗与呼吸率。
  - **尺度 3（宏观，5 ~ 30 分钟）**: 生理状态稳态演变与昼夜节律，服务于睡眠分期、全天压力评估。
- **端侧超低功耗优化**: 针对手表 ARM Cortex-M / Cortex-A 芯片进行定点整型量化与算子融合，单次前向推理延迟 **小于 1 毫秒**。

---

## 6. Stanford Medicine SleepFM (Nature Medicine) —— 多器官耦合睡眠基座

- **论文**: *SleepFM: A Multi-modal Foundation Model for Sleep Analysis*
- **发表期刊**: Nature Medicine (2024-2026)
- **团队**: Stanford Medicine (Emmanuel Mignot, James Zou 教授团队)
- **开源状态**: 论文公开，代码开源，数据集需通过协议申请。

### 数据规模与留一对比学习 (Leave-One-Out Contrastive Learning, LOO)
- **数据语料**: 来自 65,000 名受试者的 60 万小时夜间多导睡眠监测 (PSG) 数据。
- **多器官模态**: 包含脑神经（EEG）、心血管（ECG/PPG）、呼吸力学（胸腹呼吸带/气流）与肌肉活动（EMG）。
- **留一对比学习范式**:
  - 在每个 30 秒分析窗口中，随机掩码掉一个生理模态（如遮蔽脉搏或遮蔽呼吸）。
  - 通过对比损失拉近剩余多模态的联合表征与被掩码模态单独编码出的表征。
  - 迫使模型在隐空间显式建模大脑-心血管-呼吸三大器官在睡眠阶段的深层生物耦合动力学。

### 突破性远期疾病风险预测
SleepFM 不仅实现了高精度自动化睡眠分期（Wake, N1, N2, N3, REM），更令人瞩目的是仅凭单夜多模态生理表征，即可对未来数年甚至数十年的重大慢性疾病进行生存风险评估：
- **痴呆症 (Dementia)**: C-index 0.85
- **全因死亡率 (All-cause Mortality)**: C-index 0.84
- **心肌梗死 (Myocardial Infarction)**: C-index 0.81
- **心房颤动 (Atrial Fibrillation)**: C-index 0.78

---

## 7. 空间与运动判别基座：LIMU-BERT 与 Huawei Mantis

除了以心血管 PPG 为核心的模型外，针对手腕运动加速度 (ACC) 与时序判别分类的模型同样不可或缺：

### 7.1 LIMU-BERT (UbiComp 2021)
- **核心定位**: 针对三轴加速度计与陀螺仪的空间 IMU 表征模型。
- **原理**: 借鉴 BERT 的 Masked Sensor Modeling (MSM)，将三轴运动矢量分片掩码后重构，有效剥离受试者个体身份生物特征（跨用户泛化），提炼纯净的高阶步态动力学表征。在走路、跑步、骑行的人体活动识别 (HAR) 线性探测中达到 92.1% 的准确率。

### 7.2 Huawei Noah's Ark Lab: Mantis (ICML 2026)
- **核心定位**: 时序领域首个专注于**分类与判别性表征**的轻量级基础模型（参数量约 8M）。
- **核心组件**:
  - **Token Generator Unit (TGU)**: 自适应多尺度卷积与差分分块，解决固定 Patch 丢失局部波峰波谷的问题。
  - **多变量通道适配器 (Multichannel Projectors)**: 原生支持手环三轴 ACC 或多导联生理信号。
- **手环端侧实测优势**: 在 CPU 上单次前向仅需 **19.8ms**，在日常运动状态识别中线性探测准确率高达 **99.7%**，是手环端侧防误计步与心律失常快速判别的利器。

---

## 8. 核心基础模型横向对比矩阵

| 维度 / 特征 | **SensorFM** | **PaPaGei** | **Pulse-PPG** | **Samsung xMAE/HiMAE** | **SleepFM** | **Huawei Mantis** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **机构团队** | Google Research | Nokia Bell Labs | UIUC / Memphis | Samsung Research | Stanford Medicine | 华为诺亚方舟实验室 |
| **核心模态** | PPG + ACC + EDA + Temp | PPG (单通道) | 腕部高噪 PPG | PPG $\to$ 虚拟 ECG | EEG + ECG + PPG + Resp | 多变量 (ACC / 生理) |
| **预训练规模** | 1万亿分钟 (500万人) | 5.7万小时 (10个公开集)| 5.5万小时 (野外100天)| 9,400小时同步数据 | 60万小时 (6.5万人) | 多领域通用分类语料 |
| **网络骨干** | Patch Transformer | ResNet1D-MoE | 12层 1D-ResNet | 分层 Hierarchical Trans | 多模态对比 Transformer | 8M ViT-1D + TGU |
| **自监督目标** | 掩码建模 + 跨模态填补 | 形态引导对比学习 | 连续相对对比 (RelCon) | 跨模态掩码重构 | 留一对比学习 (LOO) | 几何自监督对比学习 |
| **抗伪影机制** | 万亿级数据 Scaling | 形态过滤配对 | 结合运动强度的软对比 | 跨模态先验约束 | 多器官互信息补偿 | 多尺度差分局部卷积 |
| **部署延迟** | 云端基座 | 端侧轻量 (~1.5M) | 端侧中等 (~2M) | **极低 (<1ms)** | 云端专业分析 | 端侧极低 (~19.8ms) |
| **开源状态** | 🔴 仅论文 | 🟢 代码+Zenodo权重 | 🟢 代码+Zenodo权重 | 🔴 工业闭源 | 🟢 代码开源 | 🟢 Hugging Face 开源 |

---

## 9. 对自建手表大模型 (Watch-LSM) 的工程架构启示

基于上述顶会与工业界成果，构建生产级智能手表生理基础模型应坚决遵循以下四条准则：

1. **物理时序等长分块 (Physical-Time Patch Tokenization)**:
   - 绝不使用任意重采样抹平采样率。定义固定物理时长（例如 $\Delta t = 0.25\text{s}$），PPG 采 16 点、ACC 采 8 点，分别通过模态专用 Conv1d 投影为等长的 Patch Token 序列。
2. **双流交叉消除运动伪影 (Dual-Stream Cross-Attention)**:
   - 将 ACC 流作为强抗噪的“运动参考基准”，通过交叉注意力（Cross-Attention）自适应指引 PPG 流衰减由手臂甩动引起的同频步频谐波。
3. **复合损失约束**:
   - 单纯依赖 MAE 容易重建出平滑模糊波形，必须结合频域保真损失（$\mathcal{L}_{\text{Spectral}}$）以保全脉搏重搏切迹与心跳间期微形态。
4. **判别与生成解耦**:
   - 端侧日常活动分类与心律失常筛查优先采用 Mantis 式判别骨干；而高精度连续生命体征（实时心率、血压波形）追踪则采用双流生成式重构骨干。
