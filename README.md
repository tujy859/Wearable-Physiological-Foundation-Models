# Awesome Wearable & Physiological Foundation Models

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Tracking](https://img.shields.io/badge/Status-Actively%20Maintained-blue.svg)]()

> 精选追踪学术界与工业界关于 **智能手表、连续血糖监测（CGM）及可穿戴生理信号基础模型（Foundation Models）** 的前沿论文、开源代码、预训练权重、基准数据集与实战解析。

---

## 📑 目录导航

- [📌 为什么需要可穿戴生理基础模型？](#-为什么需要可穿戴生理基础模型)
- [🗺️ 技术分类全景图 (Taxonomy)](#️-技术分类全景图-taxonomy)
- [📊 核心基础模型全景横评表](#-核心基础模型全景横评表)
- [⌚ 1. 智能手表与腕戴生理模型 (Smartwatch & Wrist)](#-1-智能手表与腕戴生理模型-smartwatch--wrist)
- [🩸 2. 连续血糖监测基础模型 (Continuous Glucose Monitoring, CGM)](#-2-连续血糖监测基础模型-continuous-glucose-monitoring-cgm)
- [📈 3. 通用时间序列基础模型 (General Time-Series FM)](#-3-通用时间序列基础模型-general-time-series-fm)
- [💤 4. 睡眠与心肺多模态模型 (Sleep & Cardiopulmonary)](#-4-睡眠与心肺多模态模型-sleep--cardiopulmonary)
- [🧠 5. 可穿戴健康推理与系统级生理大模型 (Wearable Health Reasoning & Universal Health FMs)](#-5-可穿戴健康推理与系统级生理大模型-wearable-health-reasoning--universal-health-fms)
- [📦 6. 开源数据集与评测基准 (Datasets & Benchmarks)](#-6-开源数据集与评测基准-datasets--benchmarks)
- [🛠️ 7. 动手实战与子工程关联 (Hands-on Labs)](#️-7-动手实战与子工程关联-hands-on-labs)
- [🤝 参与贡献 (Contributing)](#-参与贡献-contributing)



---

## 📌 为什么需要可穿戴生理基础模型？

通用时序大模型（如 TimesFM、Chronos）多将时序视为抽象的数值序列。然而，智能手表、手环、CGM 传感器等设备产生的生理信号具有极强的物理学与生物医学特异性：

1. **异构采样率与多模态物理对齐**：原始 PPG（25～100Hz）、ACC（25～50Hz）、ECG（250～1000Hz）、CGM（5分钟/点）以及派生指标（心率 1Hz、睡眠 30s）。通用模型难以直接处理如此大跨度的物理时间窗口。
2. **严重的运动伪影与频域碰撞**：手腕日常挥动或剧烈跑步产生的步频谐波（1～3.5 Hz）与心率脉搏波频段高度重叠。
3. **生物非平稳性与昼夜节律**：血糖的餐后突变漂移与 24 小时昼夜波动具有明确的生理因果机制。

**可穿戴生理信号基础模型的核心目标**：通过自监督预训练（SSL），从海量无标注生理时序中学习泛化的生理表征，从而仅用轻量级线性探针（Linear Probe）或极少微调即可服务于心率追踪、心律失常筛查、糖尿病与代谢风险预测、睡眠分期等数十种下游健康任务。

---

## 🗺️ 技术分类全景图 (Taxonomy)

```text
Wearable Physiological Foundation Models
│
├── 按硬件设备与传感器模态
│   ├── 智能手表/手环 (Wrist): PPG + 3-Axis ACC/IMU + ECG + Skin Temp + EDA
│   ├── 微创/穿戴传感 (Subcutaneous): CGM (血糖 5min 网格) + 汗液生物标志物
│   └── 临床与家庭睡眠设备: 多导睡眠监测 (PSG: EEG/EOG/EMG/ECG/Resp)
│
├── 按预训练学习范式 (Pretraining Paradigms)
│   ├── 掩码时序重建 (Masked Autoencoding, MAE): MOMENT, LIMU-BERT, xMAE
│   ├── 相对对比学习 (Relative Contrastive Learning): PaPaGei, Pulse-PPG
│   ├── 潜空间联合预测 (Latent Prediction / JEPA): GlucoFM
│   └── 离散分箱自回归 (Next-Token Autoregression): Chronos, GluFormer, CGM-LSM
│
└── 按认知与计算层级 (Cognitive & Computational Hierarchy)
    ├── Layer 3 认知推理与智能体 (Reasoning & Agent Systems): WearableQA, HEARTS
    ├── Layer 2 系统级全景健康模型 (Universal Health World Models): RisQ
    ├── Layer 1 端侧物理波形表征 (Signal Representation & Edge FMs): SensorFM, UniCardio, PaPaGei
    └── Layer 0 开放数据与生态基底 (Open Data & Infrastructure): Stanford OpenMHC
```

---

## 📊 核心基础模型全景横评表

> 状态标记说明：🟢 官方公开开源 | 🟡 申请开放 / 部分代码 | 🔴 闭源未公开

| 模型名称 | 机构 / 团队 | 发表时间 / 会议期刊 | 主要模态 | 架构类型 | 参数量 | 预训练数据规模 | 代码状态 | 权重状态 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :---: | :---: |
| **SensorFM** | Google Research | arXiv 2026 | PPG+ACC+EDA+Temp | Patch-Transformer | 大规模 | 1万亿分钟 (>500万人) | 🔴 | 🔴 |
| **Apple WBM** | Apple Health | 2024-2026 | Watch PPG+ACC+ECG | Multi-task Trans | - | 百万级 Apple Watch 数据 | 🔴 | 🔴 |
| **PaPaGei** | Nokia Bell Labs | ICLR 2025 | PPG (光电脉搏) | ResNet1D-MoE | \~1.5M | 5.7万小时 (10个公开库) | 🟢 | 🟢 (Zenodo) |
| **Pulse-PPG** | UIUC / Memphis | UbiComp 2025 | 腕部野外高噪 PPG | 1D-ResNet (RelCon) | \~2M | 5.5万小时 (真实自由生活) | 🟢 | 🟢 (Zenodo) |
| **UniCardio** | 清华大学 / 安贞医院 | Nature MI 2025 | PPG+ECG+BP | 统一多模态扩散 DiT | 增量0.3M/模态| 339小时三模态全时程 | 🟢 | 🟢 (论文公开) |
| **PPGFlowECG** | 北京大学 (PKU Health) | arXiv 2025/2026 | PPG $\to$ 诊断级 ECG | 潜空间整流流 (Rectified Flow) | 紧凑级 | 千万级配对 (MC-MED 11.8万人) | 🟢 | 🟢 (GitHub) |
| **AnyPPG** | 北京大学 (PKU Health) | KDD 2026 | 腕戴 PPG (心电引导) | Net1D 双分支 ResNet | 5.85M/分支 | >10万小时同步数据 | 🟢 | 🟢 (GitHub) |
| **Samsung xMAE** | Samsung Research | ICML 2024 | PPG $\to$ 虚拟 ECG | 跨模态 MAE | 端侧优化 | 9,400 小时配对数据 | 🔴 | 🔴 |
| **Samsung HiMAE** | Samsung Research | ICLR 2025 | 腕部多尺度生理 | 分层多尺度编码器 | <1ms延迟 | 真实手表自由生活数据 | 🔴 | 🔴 |
| **GlucoFM** | Google / UNSW | arXiv 2026 | CGM (5min 血糖) | 双流 JEPA 预测 | **0.72M** | 10.9万小时 (477人) | 🟡 (承诺开源) | 🔴 |
| **GluFormer** | Pheno.AI / Weizmann | Nature 2026 | CGM (15min 血糖) | GPT 式自回归 | 135M | >1000万读数 (10,812人) | 🟢 | 🔴 (HPP受限) |
| **CGMformer** | 中科院 / 上海六院 | NSR 2025 | CGM (5min 血糖) | BERT 式 MLM | 0.85M～10M | 131万天 (5.9万人) | 🟢 | 🟢 (GitHub) |
| **CGM-LSM** | JHU CDHAI | arXiv 2024 | CGM 血糖 | GPT-2 自回归 | \~124M | 1600万读数 (592人) | 🟢 (无数据) | 🔴 |
| **CGM-JEPA** | CRUISE Lab | 2025 | CGM (5min 血糖) | 潜空间 JEPA 预测 | \~0.5M | 228人 (开源重训基座) | 🟢 | 🟢 (HF) |
| **PPG-Distill** | Emory University | 2025 | 腕戴 PPG (脉搏) | 跨尺度知识蒸馏 | 极轻量 (<0.5M) | 多中心穿戴基准 | 🟢 | 🟢 (GitHub) |
| **SleepFM-1 / 2** | Stanford Medicine | Nat Med 24 / arXiv 26 | EEG+ECG+PPG+Resp+ACC | LOO-CL + 局部 MAE | **2.57M** (LLaMA) | **200万小时 (23.5万人)** | 🟢 | 🟢 (GitHub) |
| **SleepMaMi** | 首尔大学 (SNU) | ICML 2026 | PSG (EEG+ECG+Resp) | 宏微观双编码器 (MAE+CL) | \~15M | 15.8万小时 (2万人) | 🟢 | 🟢 (GitHub) |
| **LIMU-BERT** | 厦门大学等 | UbiComp 2021 | 3轴 ACC + Gyro | Sensor-BERT | 轻量级 | 多源 IMU 无标注数据 | 🟢 | 🟢 |
| **TimesFM** | Google Research | ICML 2024 / v2.0 | 通用单变量时序 | 解码器自回归 | 200M | 1000亿点 | 🟢 | 🟢 (HF) |
| **Chronos** | Amazon Research | ICML 2024 / Bolt | 通用单变量时序 | 离散分箱 T5/Encoder | 20M～710M | 泛领域时序语料 | 🟢 | 🟢 (HF) |
| **MOMENT** | CMU Auton Lab | ICML 2024 | 通用多变量时序 | Patch-MAE | 385M | Time-series Pile (含生理) | 🟢 | 🟢 (HF) |
| **OpenMHC** | Stanford / Imperial | arXiv 2026 | 19维腕戴时序+169维变量 | 开源基础模型生态基座 | WBM / LSM-2 | >6,700万小时 (11,894人) | 🟢 | 🟢 (Dataverse/HF) |
| **RisQ** | TUM / Helmholtz | 2026 | 全基因组+生化+EHR+穿戴 | 全景健康世界模型 | 统一跨模态表征 | 74.5万人 (UKB+All of Us)| 🟢 (论文公开) | 🔴 (顶刊在审) |
| **WearableQA** | Meta AI | arXiv 2026.09 | 500天日常16维指标+血检 | 纵向健康推理基准 | 14大主流LLM | 200人真实长程 (4,084题) | 🟢 (论文公开) | 🔴 (基准数据) |
| **HEARTS** | Yang AI Lab | ICML 2026 | 20种生理模态(ECG/PPG/CGM等) | 四层认知推理金字塔 | 多架构+CodeAct | 16大开源库 (20,226样本) | 🟢 | 🟢 (HF/Web) |

---


## ⌚ 1. 智能手表与腕戴生理模型 (Smartwatch & Wrist)

详细分析与网络细节请查阅：📖 [docs/models/smartwatch_models.md](docs/models/smartwatch_models.md)

### 工业级与旗舰全能多模态巨座
- **Google SensorFM** (2026): 可穿戴健康领域的万亿分钟里程碑。基于动态重采样解决 PPG/ACC/EDA 采样率异构问题，在 35 个健康与行为基准中大幅超越专用模型。
- **清华大学 UniCardio** (Nature MI 2025/2026): 清华大学朱军教授、王立元教授团队联合北京安贞医院研发的**统一心血管多模态扩散基础模型**。首创统一扩散 Transformer (DiT) 框架融合去噪、插补与 PPG $\to$ 诊断级 ECG / 连续血压 BP 的跨模态生成，并引入持续学习范式以极低增量参数（0.3M/模态）支持端侧传感器的动态热插拔与终身防遗忘。
- **Apple WBM & PPG/ECG FMs**: 深入探索日常无感监测与心律失常预警，通过海量真实世界日常佩戴数据学习个体基线与行为动力学。
- **Samsung xMAE & HiMAE** (ICML 24 / ICLR 25): 专为智能手表端侧计算优化。xMAE 利用连续 PPG 虚拟重构偶发高精度 ECG；HiMAE 提出多尺度分层时间架构，延迟小于 1ms。

### 开源先锋代表
- **Stanford OpenMHC** (arXiv 2026): 斯坦福医学院 Euan Ashley 团队联合帝国理工发布的**首个超大规模开源可穿戴基础模型基座与生态**。依托 10 余年真实世界生活时序（>6700 万佩戴小时、11,894 名受试者、19 维连续传感器通道及 169 维 HealthKit 变量），不仅打破了 Google SensorFM 与 Apple WBM 的数据垄断，更在统一基准下开源复现了工业级 WBM、LSM-2 及 Chronos-2 等基础模型，规范化了下游判别、生成式插补与未来预测三大赛道。
  > [!NOTE]
  > **先驱临床沿革**：OpenMHC 的数据底座源自 2015 年苹果发布 ResearchKit 时的首批先驱应用 **My Heart Counts (MHC)**；而同期苹果官方开展的 41.9 万人 **Apple Heart Study (AHS, 2017)** 则通过严格远程心电贴片闭环促成 Apple Watch 房颤预警算法获得美国 FDA 认证。两项研究共同奠定了穿戴设备迈向严肃医学基础模型的基石。
- **北京大学 PPGFlowECG** (2025/2026): 北大洪申达团队研发的 PPG 转 ECG 跨模态生成框架。首创 CardioAlign 编码器与潜空间整流流（Latent Rectified Flow），仅需 1～4 步 ODE 直线传输即可从可穿戴 PPG 极速合成高保真诊断级心电波形，依托千万级急诊数据集 MC-MED 在房颤、心梗等疾病筛查与医生盲测中表现优异。
- **北京大学 AnyPPG** (KDD 2026): 北大洪申达团队推出的通用光电脉搏基座大模型。基于超 10 万小时同步脉搏-心电数据进行跨模态对比预训练，突破传统单一心血管任务，首次实现对慢性肾病（CKD）、帕金森病等全身多器官复杂表型的无创筛查。
- **PaPaGei** (ICLR 2025): Nokia Bell Labs 与剑桥联合发布，首个开源通用光电生理基础模型。采用 ResNet1D-MoE 架构，参数量仅 1.5M，在心率、血压、血管年龄等 20 个下游任务表现卓越。
- **Pulse-PPG** (UbiComp 2025): UIUC 主导，针对真实野外高噪手腕 PPG 提出相对对比学习（RelCon），有效克服真实生活中的剧烈运动伪影。
- **PPG-Distill** (Emory University, 2025): 针对穿戴设备端侧资源受限难题，首创面向光电脉搏大模型的三级知识蒸馏框架（预测/特征/波形形态蒸馏），在保持心率与房颤高精度判别的同时实现 7 倍推理加速与 19 倍内存节省。
- **Apple PpgAge & WBM** (Nature Medicine 2025/2026): 基于 21 万人 Apple Health Study 真实世界手腕脉搏波，验证了 PPG 潜表征独立于日历年龄评估血管老化程度（Vascular Age）与心血管发病风险（HR = 1.46）的临床有效性。
- **LIMU-BERT** (UbiComp): 针对 IMU/加速度计的传感器表征模型，实现与个体身份解耦的高阶步态和运动模式提取。

---

## 🩸 2. 连续血糖监测基础模型 (Continuous Glucose Monitoring, CGM)

详细分析与基准横评请查阅：📖 [docs/models/cgm_models.md](docs/models/cgm_models.md)

- **Google GlucoFM** (arXiv 2605.30865, 2026):
  - 针对 CGM 专有的 5 分钟离散网格与间质液延迟物理特性，首创双流联合预测架构（Dual-Stream JEPA），参数量仅 **0.72M**。
  - 巧妙融合全局自注意力（捕获全天昼夜节律动力学）与局部卷积（捕捉餐后血糖陡峭尖峰），在 22 个少样本/跨设备临床血糖任务中超越多变量模型。
- **GluFormer** (Nature 2026, Pheno.AI / Weizmann Institute):
  - 基于 GPT 式因果自回归架构（1.35 亿参数），在大规模真实世界人类表型计划 (HPP) 10,812 人连续 15 分钟 CGM 数据上完成预训练。
  - 不仅能实现极高精度的 24 小时血糖轨迹生成与低血糖预警，更能直接从血糖波动潜表征中零样本推断内脏脂肪、肝脂肪变性及未来糖尿病发展风险。
- **CGMformer** (中科院 / 上海六院, National Science Review 2025):
  - 依托中国人群超大规模动态血糖队列（131 万天监测记录、59,000+ 受试者），构建了覆盖正常糖耐量、糖尿病前期及 1/2 型糖尿病全谱系语料库。
  - 采用 BERT 式掩码语言模型（MLM）预训练，在多中心验证中显著提升了隐匿性糖尿病视网膜病变与微血管并发症的早筛灵敏度。
- **CGM-JEPA** (2025):
  - 首个完全基于开源血糖数据重训的非生成式 JEPA 基础模型，证明潜空间预测比像素级还原更能抵御传感器漂移噪声。

---

## 📈 3. 通用时序基础模型 (General Time-Series FMs)

详细分析与适用性评估请查阅：📖 [docs/models/general_tsfm.md](docs/models/general_tsfm.md)

- **TimesFM** (Google): 采用分块解码器架构，在巨量时序数据上预训练，提供强大的零样本点预测与概率区间预测。
- **Chronos & Chronos-Bolt** (Amazon): 将连续时间序列离散化为词元（Tokenization via Bins），利用语言模型（T5 骨干）进行自回归时间序列预测。
- **CMU MOMENT & MANTIS**: 基于 Patch-MAE 理念，覆盖多变量重建、异常检测与分类，预训练数据 Time-series Pile 包含大量生理信号。

---

## 💤 4. 睡眠与心肺多模态模型 (Sleep & Cardiopulmonary)

详细分析与网络细节请查阅：📖 [docs/models/sleep_models.md](docs/models/sleep_models.md)

- **Stanford Medicine SleepFM & SleepFM-2** (Nature Medicine 2024 / arXiv 2609.06849, 2026.09):
  - **从 60 万到 200 万小时多器官生理基座**：依托 26 个独立队列、282,511 份完整夜间 PSG 记录（预训练 23.5 万份），联合建模脑电（EEG）、眼电（EOG）、心电（ECG/PPG）、肌电（EMG）与呼吸气流/胸腹阻抗，在完全保留的 Harvard HSP 独立医疗中心盲测中验证了强大的真实泛化力。
  - **LLaMA 现代架构与双流联合自监督 (LOO-CL + MAE)**：骨干采用 RMSNorm、SwiGLU 与 RoPE，编码器参数量精简至 **2.57M**（减重 47%），Token 时间步长细化至 **1 秒**；对比分支对齐全局多器官互补性，掩码自编码分支重构局部微观波形形态。
  - **四大临床事件打分闭环**：在微觉醒（Arousals, F1 0.60）、周期性肢体运动（PLMS, F1 0.60）及呼吸事件（Apnea/Hypopnea）上打分达到甚至超越人类资深睡眠技师水平。
  - **PheWAS 215 种新发疾病预测与通用生理风险轴 (PC1)**：联合年龄/性别/BMI 预测 215 种长期疾病（C-index $\ge 0.75$），全面超越涵盖 480 维专业工程特征全家桶；提取出解释 58% 方差的通用生理风险轴，关联全脑 Sigma 波空间失谐与催眠密度熵增。
  - **消费级穿戴迁移神级突破**：手腕 PPG 零样本微调睡眠分期 Macro-F1 达 **0.531**；手腕加速度计首创“物理生理代理通道桥”（0.1～0.6 Hz 提取呼吸，3.5～14 Hz 提取心动冲击 SCG），在 **UK Biobank 10 万人 390 种疾病预测**中直接打平专有加速度计模型！
  - **开源代码**：[github.com/zou-group/sleepfm-v2-public](https://github.com/zou-group/sleepfm-v2-public)
- **SleepMaMi** (首尔大学, ICML 2026):
  - **宏观-微观层级双编码器架构**：针对整夜宏观睡眠时序与局部微观瞬变波形尺度断层的痛点，提出 Macro-Encoder（结合年龄/性别/BMI 人口统计学先验建模全夜周期节律）与 Micro-Encoder（局部 MAE 重构与多模态对比学习）。
  - **少样本多中心迁移**：在 20,000+ 份 PSG 记录（约 15.8 万小时）上预训练，仅需 1% 标注即可匹敌全监督模型。
- **U-Sleep & 开源基准网络** (Nature npj Digital Medicine):
  - 基于全卷积 1D U-Net 的端到端多通道睡眠分期架构，具备极强的跨传感器配置与动态导联适应能力。

---

## 🧠 5. 可穿戴健康推理与系统级生理大模型 (Wearable Health Reasoning & Universal Health FMs)

### 5.1 系统级多模态健康世界模型 (Universal Health & Cross-Disease Risk FMs)

详细技术深度解构与架构剖析请查阅：📖 [docs/models/universal_health_models.md](docs/models/universal_health_models.md)

长期以来，临床流行病学陷于“单一疾病孤岛（Disease Silos）”的局限（如 Framingham 仅算心梗、SCORE2 仅算中风、FINDRISC 仅算糖尿病）。2026 年，慕尼黑工业大学（TUM）、亥姆霍兹慕尼黑中心（Helmholtz Munich）、哈佛大学及斯坦福医学 AIMI 团队提出了革命性的**人体健康世界模型**范式：

- **TUM / Helmholtz RisQ** (medRxiv / Research Square 2026):
  - **75 万人双中心零样本验证**：依托英国生物样本库（UK Biobank 48.8 万人）与美国国立卫生研究院（All of Us 25.7 万人跨族裔多中心），首次证实了**“人类健康全景跨疾病共享潜在结构（Shared Structure of Human Health）”**的客观存在。
  - **自然语言提示词驱动 (Promptable Risk Query)**：打破固定输出分类头，支持临床医生以自然语言 Prompt 自由定义查询——在 1 年、5 年或 10 年等任意时间跨度下，对全生命周期上百种 ICD-10 疾病进行零样本发病风险推演。
  - **医学风险预测领域的缩放定律 (Scaling Law)**：首次在医学领域严格证明了类似大语言模型的 Scaling Law——预训练覆盖的疾病谱系越宽，模型对未知疾病的零样本表征迁移能力与判别精度越高。
  - **微观单基因功能缺失突变 (LoF) 的跨系统病理锚定**：即便在没有任何临床生化化验的极端场景下，仅凭 *LDLR*（家族性高胆固醇血症）或 *HBB*（地中海贫血）等单基因变异，模型即可在潜在空间自发推演并映射至心血管与血液系统的全谱系并发症。
  - **穿戴动态遥测的未来中枢**：系统阐明了静态基因组（先天底色）、临床血液生化（稳态快照）与智能手表/体动仪（动态遥测探针）的共生关系，将可穿戴设备正式升级为通用医学大模型的高频动态感知中枢。

---

### 5.2 认知推理与智能体评测基准 (Longitudinal Health Reasoning Benchmarks)

详细技术深度解构与架构剖析请查阅：📖 [docs/models/health_reasoning_llms.md](docs/models/health_reasoning_llms.md)

随着穿戴设备从“偶发读数”走向“全生命周期纵向监测”，学术界与工业界（Meta、MIT、Yang AI Lab 等）于 2025–2026 年迎来了**从“信号级单点预测与表征提取”向“认知级多模态健康因果推理”**的重大范式跃迁：

- **Meta WearableQA** (arXiv:2609.05405, 2026):
  - **首个真实世界长程健康推理基准**：基于 200 名真实用户最长 500 天的连续穿戴记录（16 维每日心肺、运动、睡眠指标）、17 项临床化验血液面板及大样本群体分位数先验，构建 4,084 道 10 选 1 深度推理选择题。
  - **$2 \times 2$ 解耦评测与双重锚定**：严密分离纯数据计算推理（Data Reasoning）与临床病理机制推理（Health Reasoning）；采用“顶级医学指南文献”与“超大规模人群统计显著（$p<0.001$）”双重锚定真实标签。
  - **实证揭示**：顶级大模型（Gemini 3.1 Pro、GPT-5）准确率在 60%～72% 之间，中小开源模型甚至不足 40%（随机基线 10%），揭示了 LLM 在长程连续数字计算与微弱生理异常捕捉上的严重短板。
- **Yang AI Lab HEARTS** (ICML 2026 / arXiv:2603.06638):
  - **首个覆盖全谱系健康时序的大模型评测体系**：涵盖 16 个开源数据集、12 个健康领域、20 种生理模态（ECG、EEG、PPG、EMG、CGM、呼吸音等），采样率跨越每日聚合到 48 kHz 高频，包含 110 项任务与 20,226 个评测样本。
  - **四层认知阶梯 (Hierarchical Spectrum)**：首次将健康时序评测系统化抽象为 $\text{Perception (底层感知)} \to \text{Inference (状态推断)} \to \text{Generation (波形生成)} \to \text{Deduction (纵向因果演化)}$。
  - **关键“解毒”洞察**：纯 LLM 在多数健康时序任务上明显落后于专用轻量时序基础模型（如 PaPaGei、MOMENT），且在高频长序列上倾向于采用“低复杂度启发式作弊（复制/简单插值）”。
  - **CodeAct 神经符号协作**：验证了 LLM 必须借助 CodeAct 范式，作为中枢调度 Python 科学计算库（SciPy、NeuroKit2、BioSPPy）执行精确数值运算，才能真正实现高可靠健康时序分析。

---

## 📦 6. 开源数据集与评测基准 (Datasets & Benchmarks)

详细数据集下载指引与预处理代码请查阅：📖 [docs/datasets/wearable_datasets.md](docs/datasets/wearable_datasets.md)

| 领域 / 模态 | 数据集名称 | 采集设备 / 团队 | 核心传感器与采样率 | 样本规模 | 适用任务与特色 | 获取方式 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **⌚ 手腕多模态** | **OpenMHC** | Stanford (Ashley组) / 帝国理工 | 19维腕戴连续时序 + 169维变量 | 11,894人 (>6700万h, 13年) | 首个超大规模开源穿戴基础模型生态基座 (插补/预测/评估) | 🟢 开放免费 (Dataverse/HF) |
| **⌚ 手腕多模态** | **PhysioNet Wrist** | PhysioNet | PPG 256Hz, ACC 256Hz, ECG | 8人 (走跑骑运动) | 运动伪影消除、心率连续回归金标准 | 🟢 开放直下 |
| **⌚ 手腕多模态** | **PPG-DaLiA** | Fraunhofer / UCI | 手腕 E4 (PPG 64Hz, ACC 32Hz, EDA) | 15人 (真实自由生活) | 日常非受限活动心率监测、动态去噪 | 🟢 开放直下 |
| **⌚ 手腕多模态** | **WESAD** | UCI / Bosch | 手腕 E4 + 胸戴 RespiBAN | 15人 (受控压力诱发) | 情绪识别、心理压力 (Stress) 状态分类 | 🟢 开放直下 |
| **⌚ 手腕多模态** | **TROIKA** | IEEE TBME 2015 | 双通道手腕 PPG + 3轴 ACC | 12人 (跑步机高动态) | 运动伪影频域碰撞与心率追踪经典开山集 | 🟢 开放直下 |
| **⌚ 手腕多模态** | **BIDMC PPG** | 哈佛医学院 BIDMC | 手指 PPG 125Hz, ECG, 阻抗呼吸 | 53人 (8分钟高精波形) | 脉搏微形态分析、呼吸率 (RR) 估计基准 | 🟢 开放直下 |
| **🩸 CGM 连续血糖**| **CGMacros** | PhysioNet (2024) | 双 CGM (5/15min) + 高清餐食照 | 45人 (带生化血检) | 饮食营养摄入响应、生化探针多任务 | 🟢 开放直下 |
| **🩸 CGM 连续血糖**| **ShanghaiT1/T2DM**| 上海六院 (包玉倩团队) | 雅培瞬感 Libre (15min 连续间质液) | 112人 (14天连续) | 缺失值插补、糖尿病分型与并发症探针 | 🟢 开放直下 |
| **🩸 CGM 连续血糖**| **Hall Glucotypes** | 斯坦福大学 (Snyder组)| Dexcom G4 (5min 密集网格) | 57人 (10.5万读数) | 血糖波动分型 (Glucotype)、无监督表征 | 🟢 开放直下 |
| **🩸 CGM 连续血糖**| **BIG IDEAs** | 杜克大学 (PhysioNet) | Dexcom G6 (5min) + 手环 E4 | 16人 (双设备佩戴) | 穿戴光电-皮下间质血糖跨模态关联 | 🟢 开放直下 |
| **🩸 CGM 连续血糖**| **Colas DFA** | PLOS ONE (2019) | 微创 CGM (5min 网格) | 208人 (>9,500小时) | 自由生活大规模预训练、长程稳定性 | 🟢 开放直下 |
| **🩸 CGM 连续血糖**| **OhioT1DM** | 俄亥俄大学 / BGLP | Dexcom 5min + 胰岛素/碳水记录 | 12人 (8周连续时程) | 30～120min 短程血糖自回归预测标准集 | 🟡 学术申请 |
| **🩸 CGM 连续血糖**| **Weinstock 2016** | T1D Exchange / JAEB | Dexcom G4 (5min) 长期监测 | 226人 (>1.2亿读数) | 老年高危人群夜间无症状低血糖筛查 | 🟡 学术申请 |
| **🩸 CGM 连续血糖**| **Glucose-ML** | Augmented Health Lab | 自动化集成 20+ 个公开数据集 | 4,300+人 (44.9M点) | 一键下载、单位自动对齐与标准化集合库 | 🟢 GitHub 开源 |
| **🏥 临床高精基准**| **MC-MED** | 北京大学 (洪申达团队) | 急诊监护 PPG 100Hz+, ECG, 呼吸 | 11.8万人 (>1000万对) | PPG $\to$ ECG 跨模态生成、急诊重症筛查 | 🟡 凭证申请 |
| **🏥 临床高精基准**| **VitalDB** | 首尔大学医院 | 500Hz 动脉血压波, PPG, ECG | >10,000 例手术患者 | 血管弹性、连续无创血压金标准映射 | 🟢 开放 API |
| **💤 睡眠多导 PSG**| **SHHS** | 美国 NIH / NHLBI | 全套临床 PSG (EEG/ECG/Resp/EMG) | 5,804人 (多年随访) | 睡眠呼吸暂停、心脑血管死亡长期队列 | 🟢 NSRR 申请 |
| **💤 睡眠多导 PSG**| **MESA** | 美国 NHLBI / 多中心 | 完整 PSG + 7天手腕体动仪 | 2,237人 (多族裔) | 多族裔睡眠结构、动脉粥样硬化结局 | 🟢 NSRR 申请 |
| **🌐 系统级全景队列**| **UK Biobank** | 英国 MRC / Wellcome | 手腕高频体动 100Hz + WGS + EHR | 10万人体动 / 50万人全景 | 系统级健康世界模型 (RisQ基石)、全生命周期疾病风险 | 🔴 严格申请 (AMS/RAP云端) |
| **🌐 系统级全景队列**| **All of Us** | 美国 NIH | 电子健康档案 + 多组学 + 穿戴体动 | 25万+人 (多族裔) | 多族裔系统健康模型零微调外推验证金标准 | 🟡 平台认证 (Workbench) |
| **🧠 健康推理基准**| **WearableQA** | Meta AI (2026) | 16维每日聚合指标 + 17项血检面板 | 200人 (500天轨迹, 4,084题) | 真实用户长程健康推理、因果归因与临床指南对齐 | 📄 [arXiv:2609.05405](https://arxiv.org/abs/2609.05405) |
| **🧠 健康推理基准**| **HEARTS** | Yang AI Lab (ICML 2026) | 20种生理模态(ECG/PPG/EEG/CGM等) | 16大开源库 (20,226样本) | 全谱系生理时序四层认知推理阶梯、CodeAct Agent 协同 | 🟢 [GitHub](https://github.com/yang-ai-lab/HEARTS) / [Web](https://yang-ai-lab.github.io/HEARTS/) |

---

## 🛠️ 7. 动手实战与子工程关联 (Hands-on Labs)

本项目与核心实验复现子工程联动，提供端到端真实世界实操代码：

- ⌚ **智能手表工程实践自建库**: [**Watch_LSM**](https://github.com/tujy859/Watch_LSM)
  - 包含真实智能手表运动伪影碰撞分析（跑步场景心率 MAE 从 51 BPM 降至 7.5 BPM）；
  - 包含生产级双流跨模态自监督训练框架 `watch_lsm`。
- 🩸 **CGM 基础模型横评与复现**: [**CGM_FM**](https://github.com/tujy859/CGM_FM)
  - 深入评测 GlucoFM、GluFormer 与 CGM-JEPA 的表征迁移能力。
- 💻 **极简 Demo 脚本**（开箱即用体验）：
  - 详见 `notebooks/`（使用已公开开源权重进行 10 行代码特征提取与预测）。

---

## 🤝 参与贡献 (Contributing)

欢迎提交 Issue 或 Pull Request 推荐最新的顶会论文或开源模型！具体规范请见 [CONTRIBUTING.md](CONTRIBUTING.md)。
