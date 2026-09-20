# 智能手表与腕戴生理基础模型深度解析 (Smartwatch & Wrist Physiological Foundation Models)

> 腕戴设备（智能手表、智能手环）已成为普及度最高的消费级生理监测终端。本专题深入解构针对光电容积脉搏波 (PPG)、三轴加速度计 (ACC/IMU) 及微弱体表心电 (ECG) 研发的专用时序基础模型，梳理其对抗运动伪影、多速率对齐及端侧轻量化部署的底层技术演进。

---

## 📑 目录

- [1. 腕戴生理时序的独特性与核心技术挑战](#1-腕戴生理时序的独特性与核心技术挑战)
- [2. Google SensorFM (2026) —— 万亿分钟级全模态基座](#2-google-sensorfm-2026--万亿分钟级全模态基座)
- [3. Nokia Bell Labs PaPaGei (ICLR 2025) —— 开源脉搏形态基座](#3-nokia-bell-labs-papagei-iclr-2025--开源脉搏形态基座)
- [4. UIUC Pulse-PPG (UbiComp 2025) —— 野外非受限抗伪影基座](#4-uiuc-pulse-ppg-ubicomp-2025--野外非受限抗伪影基座)
- [5. Samsung Research xMAE & HiMAE (ICML 2024 / ICLR 2025) —— 端侧跨模态与分层架构](#5-samsung-research-xmae--himae-icml-2024--iclr-2025--端侧跨模态与分层架构)
- [6. Stanford Medicine SleepFM & SleepFM-2 (Nature Medicine / arXiv 2026) —— 多器官耦合与穿戴迁移基座](#6-stanford-medicine-sleepfm--sleepfm-2-nature-medicine--arxiv-2026--多器官耦合与穿戴迁移基座)
  - [6.1 SleepFM 1.0：留一对比学习与单夜预测 130+ 疾病](#61-sleepfm-10留一对比学习与单夜预测-130-疾病)
  - [6.2 SleepFM-2：200 万小时数据、手腕 PPG 迁移与“物理生理代理通道桥”](#62-sleepfm-2200-万小时数据手腕-ppg-迁移与物理生理代理通道桥)
  - [6.3 关键技术辩析：手腕 ACC 基础模型能否直接识别“微觉醒 (Cortical Arousals)”？](#63-关键技术辩析手腕-acc-基础模型能否直接识别微觉醒-cortical-arousals)
- [7. 空间与运动判别基座：LIMU-BERT 与 Huawei Mantis](#7-空间与运动判别基座limu-bert-与-huawei-mantis)
- [8. 清华大学 UniCardio (Nature MI) —— 统一心血管多模态扩散基础模型](#8-清华大学-unicardio-nature-mi--统一心血管多模态扩散基础模型)
- [9. 北京大学 PPGFlowECG (2025/2026) —— 潜空间整流流 PPG 转 ECG 跨模态生成框架](#9-北京大学-ppgflowecg-20252026--潜空间整流流-ppg-转-ecg-跨模态生成框架)
- [10. 北京大学 AnyPPG (KDD 2026) —— 心电引导预训练的光电多器官基座大模型](#10-北京大学-anyppg-kdd-2026--心电引导预训练的光电多器官基座大模型)
- [11. 端侧轻量化与生物年龄前沿：PPG-Distill 与 Apple PpgAge](#11-端侧轻量化与生物年龄前沿ppg-distill-与-apple-ppgage)
- [12. Stanford OpenMHC (2026) —— 超大规模开源穿戴基础模型生态基座](#12-stanford-openmhc-2026--超大规模开源穿戴基础模型生态基座)
- [13. 核心基础模型横向对比矩阵](#13-核心基础模型横向对比矩阵)
- [14. 对自建手表大模型 (Watch-LSM) 的工程架构启示](#14-对自建手表大模型-watch-lsm-的工程架构启示)


---

## 1. 腕戴生理时序的独特性与核心技术挑战

与抽象的通用单变量时序（如金融气象）不同，智能手表采集的生理流具备强物理与生物先验：

1. **异构采样率与多模态物理对齐 (Multi-Rate Heterogeneity)**:
   - **PPG (光电脉搏)**: 常见采样率 25Hz、50Hz、64Hz 或 100Hz，波形由微血管收缩期充血与舒张期弹性回缩构成，包含收缩峰（Systolic Peak）、重搏切迹（Dicrotic Notch）及舒张峰（Diastolic Peak）。
   - **ACC (三轴加速度计)**: 常见采样率 25Hz 或 50Hz，记录人体在空间三维的正交加速度矢量 $(a_x, a_y, a_z)$。
   - **ECG (偶发心电)**: 250Hz～1000Hz，用户主动按压表冠/电极时触发。
   - **低频慢速指标**: 皮肤温度 (Skin Temp, 0.1Hz)、皮电反应 (EDA, 4Hz)、气压高度计 (1Hz)。
2. **严重的运动伪影与频域碰撞 (Spectral Collision)**:
   - 手臂摆动、跑步落足引起的微小形变与静脉血液惯性流动，使光学传感器接收到的漫反射光被剧烈调制。
   - **致命碰撞**: 日常跑步步频谐波（1.0～3.5 Hz，即 60～210 SPM）与人体运动心率主频（1.2～3.2 Hz，即 70～190 BPM）完全重叠。传统经验带通滤波或峰值拾取（Peak-Picking）极易把步频误当心率，导致误差超过 50 BPM。
3. **极度受限的端侧计算资源**:
   - 智能手表电池通常仅 200～500 mAh，运行在 ARM Cortex-M 或低功耗 Cortex-A 核心上，要求模型参数小（<5M）、推理延迟低（<5ms）且满足整型定点量化。

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
  - 心率回归 (HR Regression, MAE \~ 1.2-3.5 BPM)
  - 血压估计 (收缩压/舒张压 MAE \~ 5-8 mmHg)
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
  - **尺度 1（微观，0.5～2 秒）**: 逐拍心跳微波形特征，服务于瞬时心率、心率变异性 (HRV)、脉搏波传导时间 (PWTT)。
  - **尺度 2（介观，5～30 秒）**: 肢体步态运动节奏与呼吸包络调制，服务于计步、能量消耗与呼吸率。
  - **尺度 3（宏观，5～30 分钟）**: 生理状态稳态演变与昼夜节律，服务于睡眠分期、全天压力评估。
- **端侧超低功耗优化**: 针对手表 ARM Cortex-M / Cortex-A 芯片进行定点整型量化与算子融合，单次前向推理延迟 **小于 1 毫秒**。

---

## 6. Stanford Medicine SleepFM & SleepFM-2 (Nature Medicine / arXiv 2026) —— 多器官耦合与穿戴迁移基座

> 💡 完整睡眠与心肺多模态基础模型专题（含 SleepFM 1.0 与 SleepFM-2 详细数学原理、双流 LOO+MAE 架构、BioSerenity 队列及临床 NSRR 数据集）请查阅：📖 [docs/models/sleep_models.md](sleep_models.md)

- **论文**: 
  - SleepFM 1.0: *SleepFM: A Multi-modal Foundation Model for Sleep Analysis* (Nature Medicine, 2024)
  - SleepFM-2: *Learning transferable human physiology from two million hours of sleep with SleepFM-2* (arXiv:2609.06849, 2026.09)
- **研发团队**: Stanford Medicine (Emmanuel Mignot, James Zou 教授团队) 联合 DTU、BrainCapture 等
- **开源状态**: 论文公开，代码开源 ([github.com/zou-group/sleepfm-v2-public](https://github.com/zou-group/sleepfm-v2-public))。

### 6.1 SleepFM 1.0：留一对比学习与单夜预测 130+ 疾病
- **数据语料**: 来自 65,000 名受试者的近 60 万小时夜间多导睡眠监测 (PSG) 数据；
- **留一对比学习范式 (LOO-CL)**:
  - 针对脑神经（EEG）、心血管（ECG/PPG）、呼吸力学（胸腹呼吸带/气流）与肌肉活动（EMG）多分支，在 30 秒窗口随机掩码某一器官模态，拉近剩余模态联合嵌入与目标模态的距离；
  - 首次在隐空间显式建模大脑-心血管-呼吸三大系统的生理因果耦合；
- **长期疾病风险预测**:
  - 痴呆症 (Dementia): C-index 0.85
  - 全因死亡率 (All-cause Mortality): C-index 0.84
  - 心肌梗死 (Myocardial Infarction): C-index 0.81
  - 心房颤动 (Atrial Fibrillation): C-index 0.78

### 6.2 SleepFM-2：200 万小时数据、手腕 PPG 迁移与“物理生理代理通道桥”
2026 年 9 月发布的 SleepFM-2 是生理基座领域的颠覆性飞跃，其成果对**消费级智能手表/手环算法研发**具有直接的指导意义：

1. **架构极速瘦身与 1 秒超细粒度 Tokenization**:
   - 采用现代 LLaMA 骨干（RMSNorm、SwiGLU、RoPE、无偏置线性层），编码器参数量从 4.83M 锐减至 **2.57M**（减重 47%）；
   - Token 步长从 5 秒精细化至 1 秒，并融合 LOO-CL 对比与 MAE 局部重构，支持微觉醒、肢体抽搐及呼吸暂停微事件高保真打分；
   - 提取出解释 58% 方差的 **PC1 通用生理风险轴**，并成功预测 215 种长期新发慢病；
2. **手腕光电容积脉搏波 (Wrist PPG) 零样本直接微调**:
   - 针对消费级手表采集的单通道手腕红外 PPG，SleepFM-2 冻结编码器零样本微调的 5 分类睡眠分期 Macro-F1 达到 **0.531**（对比端到端从头训练的 0.175，净增 3 倍以上），证明临床 PSG 学习到的心血管脉搏表征可以直接泛化至手腕传感器；
3. **颠覆性创新：手腕三轴加速度计 (ACC) 的“物理生理代理通道桥” (Surrogate Channels Bridge)**:
   - **核心困局**：智能手表的日常体动加速度计（ACC）与医院 PSG **完全没有重叠物理通道**，传统穿戴研究认为必须收集数万小时手表专用加速度数据重新训练模型；
   - **SleepFM-2 物理桥接算法**：利用经典生理物理学滤波从手腕微动中解耦内源性生理节律：
     - **呼吸代理通道 (Surrogate Respiration)**：将三轴加速度在 **0.1～0.6 Hz** 带通滤波捕获胸廓呼吸运动诱发的手腕微动，在 60 秒内通过相关性融合为主通道，直接接入 PSG 胸腹呼吸输入端；
     - **心动冲击代理通道 (Surrogate SCG)**：将三轴在 **3.5～14 Hz** 带通滤波捕获心脏每搏射血引起的微弱体动（心冲击信号 Seismocardiogram），计算三轴 L2 范数后在 **0.5～3.5 Hz** 带通滤波提取心跳包络，直接接入 PSG 心电/PPG 输入端；
   - **实证突破**：
     - 在 6 个真实穿戴加速度队列上，5 分类睡眠分期 AUROC 达到 **0.708～0.852**（超越从头训练的 0.590～0.690）；
     - 在 **UK Biobank 10 万人**为期 7 天的腕戴加速度计长期随访中，针对 390 种疾病发病预测，SleepFM-2 平均 C-index 达到 **0.687**，**直接打平了专门在 UK Biobank 10 万人加速度计上自监督预训练的专用基础模型（0.688）**！
   - **对 Watch-LSM 的工程启示**：这一机制证明了消费级设备无需耗费巨资重复采集百万小时专有数据，利用物理滤波代理桥即可无缝“借用”临床超大规模严肃医学多导模型的深厚生理表征，实现降维打击。

### 6.3 关键技术辩析：手腕 ACC 基础模型能否直接识别“微觉醒 (Cortical Arousals)”？

在将生理基础模型落地到智能手表/手环端时，一个关键工程困惑是：**既然手腕 ACC 能通过物理滤波代理桥实现宏观 5 分类睡眠分期，它是否能进一步直接用来识别临床微觉醒？**

#### 1. 临床判读金标准冲突（微觉醒是“脑电”事件而非“体动”事件）
- **AASM 医学金标准**：皮质微觉醒（Cortical Arousals）被严格定义为**脑电（EEG）发生 $\ge 3$ 秒的高频突变**（$\alpha / \theta$ 或 $>16\text{ Hz}$ 快波）；
- **物理失明与高漏报（False Negative）**：临床中存在极其普遍的**“无动微觉醒（Sub-movement Arousals）”**——中枢皮层已经唤醒、交感神经剧烈兴奋，但受试者四肢肌肉依然保持松弛，手腕**完全静止**。纯 ACC 对此类事件在物理上是彻底失明的；
- **误报与高虚警（False Positive）**：无害的生理性翻身、深睡浅睡交替期肢体小抽搐会产生巨大的加速度尖峰，极易被纯 ACC 算法误判为微觉醒；
- **SleepFM-2 的实证界限**：论文中微觉醒识别达到 F1 0.601（战胜人类资深技师）是基于**包含 EEG/EMG 的临床多导 PSG** 评估的；而在手腕 ACC 上仅测试了宏观 30 秒睡眠分期与慢病风险预测，官方从未宣称纯 ACC 能独立识别微觉醒。

#### 2. Watch-LSM 的工业级破局路线：PPG + ACC 联合检测【自主神经微觉醒 (Autonomic Arousals)】
智能手表端工业级落地的真正正确解法，是利用手腕 **PPG 光电容积脉搏波** 捕获微觉醒激发的交感神经外周级联反应，并用 **ACC 过滤伪影**：
1. **PPG 捕捉自主神经唤醒特征**：
   - **外周血管急剧收缩 (PWA Drop)**：微觉醒引发交感风暴，手腕末梢微循环骤缩，PPG 脉搏波交流幅度在数秒内跌落 $>30\%$；
   - **心率突增 (HR Surge)**：瞬时心率在 5～10 秒内跳高 10～20 bpm；
2. **ACC 充当“伪影过滤器与分型器”**：
   - **ACC 剧烈运动 + PPG 突变** $\to$ 伴随体动的强觉醒（Movement-associated Arousal / Awakening）；
   - **ACC 几乎静止 + PPG 强脉搏波幅衰减与心率跳升** $\to$ **高置信度判定为无体动型自主神经微觉醒**；
   - **ACC 剧烈震荡但无交感脉搏变化** $\to$ 被动位移或体动伪影，予以剔除。

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

## 8. 清华大学 UniCardio (Nature MI) —— 统一心血管多模态扩散基础模型

- **论文**: *Versatile cardiovascular signal generation with a unified diffusion transformer*
- **发表期刊**: **Nature Machine Intelligence** (2025/2026, [DOI: 10.1038/s42256-025-01147-y](https://doi.org/10.1038/s42256-025-01147-y) / [arXiv: 2505.22306](https://arxiv.org/abs/2505.22306))
- **研发团队**: 清华大学计算机系 / 心理与认知科学系 / 人工智能研究院 (Zehua Chen, Yuyang Miao, **王立元 (Liyuan Wang)**, 朱军教授等) 联合北京安贞医院、英国帝国理工学院 (Danilo P. Mandic 教授)
- **开源资源**: 论文公开，官方代码与权重持续释出中

### 8.1 核心突破：统一生成范式 (Many-to-Any Generative Modeling)
传统穿戴模型将信号处理任务机械割裂：去噪（Denoising）、插补（Imputation）与模态翻译（Translation，如 PPG $\to$ ECG）往往需要针对性训练完全独立的特定模型。
**UniCardio 提出了首个多模态心血管扩散 Transformer (Diffusion Transformer, DiT) 统一生成基座**：
- 联合建模光电脉搏 (PPG)、心电 (ECG) 与连续动脉血压波 (BP) 的多模态联合条件分布；
- **全任务一网打尽**: 在统一的无条件前向加噪与条件反向去噪框架下，同时实现低信噪比滤波去噪、间歇性丢点填补、以及任意“条件模态组 $\to$ 目标模态组”的连续信号生成（例如腕部 PPG 虚拟生成诊断级 ECG，或 PPG+ECG 无创推断连续血压波）。

### 8.2 核心架构设计与轻量化增量
1. **多尺度 1D 卷积分块编码器**:
   - 针对不同生理模态配置专用 1D-CNN 编码器，内置多组不同尺度的卷积核，自适应提取逐拍微形态与长程心血管节律。
2. **任务特异性注意力掩码 (Task-Specific Attention Masks)**:
   - 在统一 DiT 主干中，引入任务导向的注意力掩码，精确约束从“条件 Token”到“目标 Token”的信息流向，防止跨任务跨模态的信息穿透与负迁移。
3. **极低端侧增量开销 (仅 0.3M / 模态)**:
   - 核心 Transformer 骨干完全共享；当穿戴设备新增一种物理传感器（例如从 PPG 拓展到 ECG 或袖带压）时，仅需接入对应的一对轻量 Encoder 和 MLP Decoder（单模态增量参数仅约 **0.3M**），极大降低了智能手表的端侧存储与部署压力。

### 8.3 持续学习机制 (Continual Learning) 破解组合爆炸
- **穿戴动态组合困境**: 现实中传感器频繁断连与切换（例如白天单 PPG、运动时加 ACC、睡眠时偶发 ECG），可能存在成倍的条件组合。直接联合训练会导致训练样本失衡与灾难性遗忘。
- **阶段式持续学习范式**:
  - 由王立元教授等引入持续学习策略，按“单条件 $\to$ 双条件 $\to$ 多条件”阶段式递进学习；
  - 配合学习率退火（LR Scheduling）、回放缓冲区（Replay Composition）与任务注意力隔离，使模型能够终身动态接入新传感器，不仅无遗忘，而且先修模态能显著正向迁移促进复杂多模态任务的表现（两条件/三条件任务 RMSE 分别降低 31.7% 与 36.0%）。

### 8.4 临床病理保真度与实证表现
- **安贞医院临床专家验证**: 由腕部 PPG 虚拟翻译出的 ECG 波形，精准再现了**心肌缺血 ST 段位移、双相 P 波、T 波倒置、房性期前收缩 (APC) 及心房颤动 (AF)** 等危急病理微形态。
- **下游跨域零样本迁移**:
  - 在 PTB-XL 数据集上实现高保真度心电去噪与异常分类；
  - 在 MIMIC PERform AF 队列上实现基于腕部 PPG 的高灵敏度房颤筛查；
  - 在 WESAD 数据集上，将手环脉搏心率（HR）估计误差大幅降低；
  - 在 MIMIC 队列上，利用 PPG/ECG 无创推断连续收缩压 (SBP) 与舒张压 (DBP) 表现极其稳健。
- **超快推断延迟**: 通过高效一阶 ODE 采样器加速，4 秒时序切片的推断延迟在端侧仅需 **< 0.4 秒**，完全满足日常实时生命体征监测需求。

---

## 9. 北京大学 PPGFlowECG (2025/2026) —— 潜空间整流流 PPG 转 ECG 跨模态生成框架

- **论文**: *PPGFlowECG: Latent Rectified Flow with Cross-Modal Encoding for PPG-Guided ECG Generation and Cardiovascular Disease Detection*
- **发表状态**: arXiv:2502.14856 / OpenReview (2025/2026)
- **研发团队**: 北京大学健康医疗大数据国家研究院 / 人工智能研究院数字健康实验室（PKUDigitalHealth, 洪申达教授团队）
- **开源资源**: [GitHub: PKUDigitalHealth/PPGFlowECG](https://github.com/PKUDigitalHealth/PPGFlowECG)

### 9.1 临床背景与核心痛点
- **可穿戴健康监测的“模态鸿沟”**：智能手表等腕戴设备普及了光电容积脉搏波（PPG）的日常全天候采集，但临床心血管疾病（CVD）的确诊金标准依然是体表心电图（ECG）。
- **PPG $\to$ ECG 跨模态翻译的两大核心壁垒**：
  1. **跨模态物理与语义未对齐**：PPG 测量末梢血管容积波动阻抗，ECG 记录心肌除极与复极的电生理向量传导，二者物理起源异质，传统时序转换模型易产生严重的语义错位与相位伪影；
  2. **高维时序细粒度保真度难以兼顾**：传统 GAN 模型极易发生模式崩溃（Mode Collapse），而常规像素/信号级扩散模型（DDPM）反向去噪采样步数过多（数十至上百步），无法满足端侧与近线实时生成需求。

### 9.2 两阶段生成架构设计 (CardioAlign + Latent Rectified Flow)

```text
[Stage 1: 跨模态对齐]
PPG Waveform ──► [ CardioAlign Encoder ] ──► 变分参数 (μ_p, σ_p) ──► 潜向量 z_ppg
ECG Waveform ──► [ CardioAlign Encoder ] ──► 变分参数 (μ_e, σ_e) ──► 潜向量 z_ecg
                        ▲                           │
                        └────── 潜空间分布对齐 ──────┘ (L2 + KL 对齐损失)

[Stage 2: 潜空间整流流生成]
标准高斯噪声 X_0 ~ N(0, I) ────────────────────────────────────────┐
                                                                   ▼
条件引导 z_ppg ──────────────────────► [ Latent Rectified Flow (ODE 直线传输) ]
                                                                   │
                                                                   ▼
                                                          生成心电潜向量 z_ecg*
                                                                   │
                                                                   ▼
                                                          [ 1D-CNN ECG 解码器 ]
                                                                   │
                                                                   ▼
                                                          高保真诊断级 ECG 波形
```

1. **Stage 1: CardioAlign 编码器与潜空间生理对齐**:
   - 采用参数绑定（Parameter-Tied）的双分支 Encoder-Decoder 结构，将 PPG 与 ECG 统一映射至低维紧凑的生理潜空间 $\mathcal{Z}$ 中；
   - 编码器输出变分高斯参数 $\mu$ 与 $\sigma$，通过重参数化采样潜向量 $z = \mu + \sigma \odot \epsilon$；
   - **潜空间分布对齐损失 ($\mathcal{L}_{\text{align}}$)**：同时联合最小化均值欧氏距离 $||\mu_{\text{PPG}} - \mu_{\text{ECG}}||_2^2$ 与潜分布对称 KL 散度 $\mathcal{D}_{\text{KL}}$，促使潜空间过滤掉外周光电伪影，仅保留纯净的心肌电动力学生理不变量。
2. **Stage 2: Latent Rectified Flow（潜空间整流流/流匹配）**:
   - 摒弃了传统扩散模型曲折反向 SDE 路径，Rectified Flow 学习沿**直线轨迹（Straight-line Trajectory）**将标准高斯噪声分布 $X_0 \sim \mathcal{N}(0, I)$ 传输至目标 ECG 潜分布 $X_1 \sim q(z_{\text{ECG}})$：
     $$\frac{\mathrm{d}X_t}{\mathrm{d}t} = v_\theta(X_t, t, z_{\text{PPG}})$$
   - 以 PPG 潜向量 $z_{\text{PPG}}$ 作为强条件引导速度场网络；
   - **常微分方程（ODE）极速采样**：得益于直线矢量场，模型仅需 **1～4 步**数值积分（Euler 或 RK45 求解器）即可完成确定性潜表征生成，推理速度相比多步扩散模型提升一个数量级以上。
3. **高保真 ECG 波形还原**:
   - 生成的潜向量 $z_{\text{ECG}}^*$ 输入专用 1D 卷积心电解码器，还原出连续的诊断级体表心电波形。

### 9.3 评测体系与临床 CVD 筛查实证
- **超大规模急诊级真实评测集**:
  - **MC-MED 数据集**：洪申达团队首次在该任务中引入涵盖 **11.8 万名急诊患者、超 1000 万对真实 PPG-ECG 配对样本** 的临床急诊库，具有明确的专家 CVD 确诊标签；
  - **VitalDB / MIMIC-AFib / BIDMC**：跨中心、多硬件域的泛化性能全面领跑。
- **基于心电基座模型 ECGFounder 的高阶特征评估**:
  - 引入心电基座模型 **ECGFounder** 作为特征提取器，计算生成波形的 Fréchet Inception Distance (FID)，在特征保真度与形态结构指标（Pearson 峰值相关系数 $r$、SNR、RMSE）上显著超越现有基线。
- **临床医生双盲图灵测试与疾病辅助筛查**:
  - 联合心内科临床医生进行波形盲测，生成的 ECG 在 P 波、QRS 波群宽度及 ST-T 改变上具备极高可读性；
  - 在房颤（AFib）、房性/室性期前收缩（APC/PVC）、室上性心动过速（SVT）等急性心律失常与心肌缺血检测中，下游分类 AUC 逼近临床真实 ECG 上限。

---

## 10. 北京大学 AnyPPG (KDD 2026) —— 心电引导预训练的光电多器官基座大模型

- **论文**: *AnyPPG: A PPG Foundation Model*
- **发表会议**: **ACM SIGKDD 2026** / arXiv:2511.01747
- **研发团队**: 北京大学健康医疗大数据国家研究院 / 计算机学院 / 人工智能研究院（Guangkun Nie, Xiaocheng Fang, 洪申达教授等）
- **开源资源**: [GitHub: Ngk03/AnyPPG](https://github.com/Ngk03/AnyPPG)

### 10.1 核心定位与范式跃升：从“心率脉搏”迈向“全身画像”
- 过去的穿戴脉搏基座模型（如 PaPaGei、Pulse-PPG）大多局限于单一心血管下游任务（如心率追踪、血压回归、血管弹性年龄）；
- **AnyPPG 首次系统性论证了：PPG 可以作为多器官整体健康画像（Holistic Multi-Organ Health Profiling）的通用数字生物标志物**。通过深度跨模态预训练，使单模态手腕 PPG 具备了对心血管系统以外的全身系统性慢性疾病进行早期无创筛查的能力。

### 10.2 心电引导的跨模态对比预训练 (ECG-Guided Pretraining)
- **数据资产规模**:
  - 聚合了超 **100,000 小时** 的高精度同步 PPG-ECG 配对时序数据，涵盖 MC-MED、MIMIC、VitalDB 等多中心临床与生理波形库。
- **为什么必须由 ECG 引导对比学习？**
  - 单模态自监督（如纯粹的 PPG 掩码重建 MAE 或数据增强对比学习）极易走入局部极小值——模型会倾向于拟合受试者的皮肤静态色泽、探头接触压力或手臂运动高频伪影；
  - 同步采集的 ECG 是反映心脏中枢电生理活动的绝对金标准，不存在外周光吸收伪影。AnyPPG 采用**深层对比学习（Contrastive Learning）**，强制拉近时间同步的 PPG 与 ECG 在高维潜表征空间中的距离，迫使 PPG 编码器学会过滤表面运动噪声，自适应提取与心室除极、血管顺应性及自主神经中枢调控相关的深层生理动力学本质。

### 10.3 核心网络架构
- **Net1D 1D-ResNet 双分支编码器**:
  - 为 PPG 与 ECG 模态分别配置专用的 1D-ResNet 骨干网络（单模态分支参数量约 **5.85M**）；
  - 包含 7 个卷积阶段（Convolutional Stages），层层抽取微血管收缩峰、重搏切迹及脉冲波传导时间（PWTT）多尺度特征；
  - 经过全局平均池化（Global Mean Pooling）后输出 **1024 维** 的高阶特征向量。
- **轻量投影头与对比表征**:
  - 采用双层 MLP 投影头：$\text{Linear}(1024 \to 512) \to \text{GELU} \to \text{Linear}(512 \to 256)$；
  - 输出 $\ell_2$ 归一化的 256 维潜表征用于跨模态 InfoNCE 对比损失计算。在端侧推理时仅需保留 PPG 单分支（\~5.85M），即插即用。

### 10.4 突破性多器官表型评测 (Beyond Cardiovascular)
AnyPPG 经冻结表征（Frozen Representation）与极简线性探针（Linear Probe）微调，展现出惊人的全身多器官疾病跨域泛化能力：
1. **心血管中枢核心任务**:
   - 心力衰竭（Heart Failure）、原发性高血压（Hypertension）、心律失常等分类指标全面刷新 SOTA。
2. **突破心血管界限的非心血管系统表型**:
   - **慢性肾脏病 (Chronic Kidney Disease, CKD)**: 敏锐捕捉到微血管重塑、硬化与水钠潴留对外周脉搏微形态的微弱时域调制；
- **帕金森病 (Parkinson's Disease)**: 识别由于中枢神经退行导致的自主神经功能紊乱（Autonomic Dysregulation）在末梢血管舒缩节律中的特异性反映；
   - 强力证明了智能手表光电脉搏波在日常慢病长程管理与神经退行性疾病早筛中的颠覆性科研与应用价值。

---

## 11. 端侧轻量化与生物年龄前沿：PPG-Distill 与 Apple PpgAge

### 11.1 Emory University PPG-Distill (2025) —— 脉搏大模型多层次端侧知识蒸馏
- **论文**: *PPG-DISTILL: Efficient Photoplethysmography Signals Analysis via Foundation Model Distillation*
- **研发团队**: Emory University (Juntong Ni, Wei Jin 教授团队) 联合医学院
- **开源代码**: [GitHub: LingFengGold/PPG-Distill](https://github.com/LingFengGold/PPG-Distill)
- **技术突破**:
  - 针对大型脉搏基础模型（如 PaPaGei、Pulse-PPG）难以常驻端侧嵌入式 MCU 的瓶颈，首创**三级多尺度知识蒸馏体系**：
    1. **预测级蒸馏 (Prediction-Level)**: 迁移下游多任务软标签分布；
    2. **特征级蒸馏 (Feature-Level)**: 在潜空间拉近紧凑学生网络与大模型教师网络的表示距离；
    3. **形态与节律蒸馏 (Morphology & Rhythm Distillation)**: 专门保留局部脉搏收缩峰微形态与长程心拍周期性结构。
  - **实测表现**: 在心率估计与心房颤动分类中，学生模型推理加速 **7 倍**，内存开销骤降 **19 倍**，性能较直接训练提升达 **21.8%**。

### 11.2 Apple PpgAge 与血管生物年龄 (Nature Medicine 2025/2026)
- **研发团队**: Apple Health 团队
- **数据规模**: 依托 **213,593 名 Apple Health Study** 真实世界佩戴者手腕 PPG。
- **临床核心发现**:
  - 训练深度模型从手表日常 PPG 预测受试者实际年龄（MAE = 2.43 年）；
  - **血管年龄差 (PPG Age Gap)**：若模型预测年龄显著高于受试者生理年龄（提示微血管弹性硬化与内皮衰退），其未来罹患重大心脑血管事件的风险显著激增（**风险比 Hazard Ratio = 1.46**）。
  - 确立了可穿戴光电脉搏大模型作为**全生命周期血管老化数字标志物**的黄金地位。

---

## 12. Stanford OpenMHC (2026) —— 超大规模开源穿戴基础模型生态基座

- **论文**: *OpenMHC: Accelerating the Science of Wearable Foundation Models* (arXiv:2607.16235)
- **研发团队**: 斯坦福大学医学院 (Euan Ashley 实验室) 联合帝国理工学院、柏林夏里特医学院
- **开源资源**: 💻 [GitHub: AshleyLab/OpenMHC](https://github.com/AshleyLab/OpenMHC) | 🤗 [Hugging Face: MyHeartCounts](https://huggingface.co/MyHeartCounts) | 📄 [arXiv:2607.16235](https://arxiv.org/abs/2607.16235)

### 12.1 破局意义：打破科技巨头对穿戴数据的垄断
在可穿戴健康 AI 领域，工业界科技巨头（如 Google SensorFM 的 1 万亿分钟、Apple WBM 的百万级 Apple Watch 数据）长期构筑了坚固的商业与隐私壁垒，数据与模型权重双闭源，导致学术界长期处于“只能看工业界刷榜、无法在海量真实生活数据上预训练”的困境。

**OpenMHC（Open MyHeartCounts）是迄今为止全球规模最大、向合格研究人员开放访问的真实世界可穿戴基础模型基准数据集与开源实现**：
- **数据体量**: 包含 **11,894 名** 经过知情同意的受试者，横跨美、英、香港等地区，记录了 **>6,700 万小时（282 万人·天）** 的分钟级连续传感器时序（日历时间跨越长达 13 年）；
- **全景传感器通道**: 覆盖 **19 个连续生理与运动通道**（步数、心率、心率变异性、睡眠分期、高强度锻炼能耗等），并与 **169 维深度变量**（自述健康问卷、生活方式、情绪心理及 HealthKit 长期档案）建立严密对应。

### 12.2 三大统一评测赛道 (OpenMHC Benchmark)
OpenMHC 规范化了可穿戴基础模型的三大核心评估协议：
1. **Track 1: 预测性任务 (Predictive Tasks)**：包含 32 项涵盖心血管健康、代谢风险与日常行为模式的下游判别任务；
2. **Track 2A: 生成式多变量插补 (Generative Imputation)**：模拟现实佩戴中因洗澡、充电、断连造成的复杂数据空洞，评估模型重建时序动力学的能力；
3. **Track 2B: 时序自回归未来预测 (Generative Forecasting)**：评估模型在零样本或少样本条件下推演未来心率与运动趋势的能力。

### 12.3 工业级基础模型开源复现与对齐
OpenMHC 团队不仅开源了数据基准，更在统一框架下完整复现并开源了前沿可穿戴基础模型：
- **开源版 Apple WBM (Wearable Base Model)** 与 **LSM-2**：让学术界首次能够在统一标准下对标工业级骨干；
- **跨领域时序大模型接入**: 集成了 **Chronos-2** 与 **Toto** 等前沿自回归时序基座；
- **经典强基线**: 配备了提取 495 个手工生理特征的 **XGBoost**、卷积时序分类器 **MultiRocket** 及非均匀缺失时序模型 **GRU-D**。

> [!NOTE]
> **历史先驱沿革：My Heart Counts (2015) 与 Apple Heart Study (2017) 的交相辉映**  
> - **My Heart Counts (MHC, 2015)**：苹果发布开源医学研究框架 **ResearchKit** 时的首批 5 款先驱 App 之一，由斯坦福医学院 Euan Ashley 教授主导，率先探索智能手机传感器（三轴加速度计/协处理器）在大规模人群中被动采集运动与心肺耐力（6MWT）。这项长达十余年的长程队列最终演进成了今天的 **OpenMHC**。
> - **Apple Heart Study (AHS, 2017)**：苹果官方资助、斯坦福医学院执行的 41.9 万人超大型心房颤动（AFib）临床试验，凭借严格的远程心电贴片（ePatch）核验闭环，直接促成 Apple Watch 房颤脉搏预警算法获得美国 FDA 医疗器械认证（NEJM 2019）。
> - 两者共同奠定了当今智能手表从“玩具计步器”迈向“严肃医学基础模型”的科学基石。

---

## 13. 核心基础模型横向对比矩阵

| 维度 / 特征 | **SensorFM** | **OpenMHC (WBM/LSM)** | **PaPaGei** | **Pulse-PPG** | **PPG-Distill** | **UniCardio** | **PPGFlowECG** | **AnyPPG** | **Samsung xMAE/HiMAE** | **SleepFM-1 / 2** | **Huawei Mantis** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **机构团队** | Google Research | **Stanford / Imperial** | Nokia Bell Labs | UIUC / Memphis | **Emory Univ** | **清华大学 / 安贞医院** | **北京大学 (PKU Health)** | **北京大学 (PKU Health)** | Samsung Research | Stanford Medicine | 华为诺亚方舟实验室 |
| **核心模态** | PPG+ACC+EDA+Temp | **19通道腕部+169维变量** | PPG (单通道) | 腕部高噪 PPG | **腕戴 PPG** | **PPG + ECG + BP** | **PPG $\to$ 诊断级 ECG** | **PPG (心电引导)** | PPG $\to$ 虚拟 ECG | PSG + 腕部 PPG/ACC 代理桥 | 多变量 (ACC / 生理) |
| **预训练规模** | 1万亿分钟 (500万人) | **>6700万小时 (1.2万人)** | 5.7万小时 (10个公开集)| 5.5万小时 (野外100天)| 多任务穿戴语料 | **339小时三模态全时程**| **千万级配对 (MC-MED 11.8万人)** | **>100,000 小时同步数据** | 9,400小时同步数据 | **200+万小时 (23.5万人)** | 多领域通用分类语料 |
| **网络骨干** | Patch Transformer | **WBM / LSM-2 / Chronos-2** | ResNet1D-MoE | 12层 1D-ResNet | **极紧凑 1D 学生网络**| **多模态扩散 DiT** | **CardioAlign + Rectified Flow** | **Net1D 1D-ResNet 双分支** | 分层 Hierarchical Trans | **2.57M LLaMA Transformer** | 8M ViT-1D + TGU |
| **自监督目标** | 掩码建模 + 跨模态填补 | **多任务表征+自回归预测** | 形态引导对比学习 | 连续相对对比 (RelCon) | **多尺度三级知识蒸馏**| **条件扩散去噪+持续学习**| **潜空间分布对齐+整流流生成** | **跨模态深层 InfoNCE 对比** | 跨模态掩码重构 | **LOO-CL 对比 + 局部 MAE** | 几何自监督对比学习 |
| **抗伪影机制** | 万亿级数据 Scaling | **真实野外全生命周期滤波** | 形态过滤配对 | 结合运动强度的软对比 | 结构化节律形态迁移 | **多模态互信息互补生成**| **参数绑定潜空间生理不变量** | **中枢心电无伪影电信号锚定** | 跨模态先验约束 | **多器官互信息 + 物理代理桥** | 多尺度差分局部卷积 |
| **部署延迟** | 云端基座 | **端云协同评测基准** | 端侧轻量 (\~1.5M) | 端侧中等 (\~2M) | **极低 (内存降19x)** | **极低增量 (\~0.3M/模态)**| **1～4步极速ODE (<0.1s)** | **端侧单分支 (\~5.85M)** | **极低 (<1ms)** | **冻结编码器 (轻量2.57M)** | 端侧极低 (\~19.8ms) |
| **开源状态** | 🔴 仅论文 | 🟢 **代码+权重+基准开源** | 🟢 代码+Zenodo权重 | 🟢 代码+Zenodo权重 | 🟢 代码开源 (GitHub) | 🟢 论文公开 / 代码开源中 | 🟢 代码开源 (GitHub) | 🟢 代码开源 (GitHub) | 🔴 工业闭源 | 🟢 **GitHub (zou-group)** | 🟢 Hugging Face 开源 |

---

## 14. 对自建手表大模型 (Watch-LSM) 的工程架构启示

> 💡 本文提炼的端侧生理表征架构与抗伪影工程实现已完整落地于开源实验子仓库：[**Watch_LSM**](https://github.com/tujy859/Watch_LSM)

基于上述顶会与工业界成果，构建生产级智能手表生理基础模型应坚决遵循以下五条准则：

1. **物理时序等长分块 (Physical-Time Patch Tokenization)**:
   - 绝不使用任意重采样抹平采样率。定义固定物理时长（例如 $\Delta t = 0.25\text{s}$），PPG 采 16 点、ACC 采 8 点，分别通过模态专用 Conv1d 投影为等长的 Patch Token 序列。
2. **双流交叉消除运动伪影 (Dual-Stream Cross-Attention)**:
   - 将 ACC 流作为强抗噪的“运动参考基准”，通过交叉注意力（Cross-Attention）自适应指引 PPG 流衰减由手臂甩动引起的同频步频谐波。
3. **复合损失约束**:
   - 单纯依赖 MAE 容易重建出平滑模糊波形，必须结合频域保真损失（$\mathcal{L}_{\text{Spectral}}$）以保全脉搏重搏切迹与心跳间期微形态。
4. **吸收 UniCardio 的动态模态接入与防遗忘思想**:
   - 采用轻量化模态独立编码器结合任务注意力掩码，让用户在增删传感器（如外接胸带、贴片或脱下表带）时免于全模型重训。
5. **判别与生成解耦**:
   - 端侧日常活动分类与心律失常筛查优先采用 Mantis 式判别骨干；而高精度连续生命体征（实时心率、血压波形、虚拟心电）追踪则采用双流生成式重构或扩散骨干。
