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
- [📦 5. 开源数据集与评测基准 (Datasets & Benchmarks)](#-5-开源数据集与评测基准-datasets--benchmarks)
- [🛠️ 6. 动手实战与子工程关联 (Hands-on Labs)](#️-6-动手实战与子工程关联-hands-on-labs)
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
└── 按模型定位与计算层级
    ├── 云端通用超大模型 (Cloud Scale): Google SensorFM (>1 Trillion min), SleepFM
    ├── 工业界专用生态模型 (Industry Ecosystem): Apple WBM, Samsung xMAE/HiMAE
    └── 端侧超轻量模型 (Edge / On-Device): PaPaGei, GlucoFM (0.72M)
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
| **CGM-JEPA** | CRUISE Lab | 2025 | CGM (5min 血糖) | 潜空间 JEPA 预测 | ~0.5M | 228人 (开源重训基座) | 🟢 | 🟢 (HF) |
| **PPG-Distill** | Emory University | 2025 | 腕戴 PPG (脉搏) | 跨尺度知识蒸馏 | 极轻量 (<0.5M) | 多中心穿戴基准 | 🟢 | 🟢 (GitHub) |
| **SleepFM** | Stanford Medicine | Nat Med 2024-2026 | EEG+ECG+PPG+Resp | 留一对比学习 (LOO) | 基础模型 | 60万小时 (6.5万人) | 🟢 | 🟡 (受限开放) |
| **SleepMaMi** | 首尔大学 (SNU) | ICML 2026 | PSG (EEG+ECG+Resp) | 宏微观双编码器 (MAE+CL) | ~15M | 15.8万小时 (2万人) | 🟢 | 🟢 (GitHub) |
| **LIMU-BERT** | 厦门大学等 | UbiComp 2021 | 3轴 ACC + Gyro | Sensor-BERT | 轻量级 | 多源 IMU 无标注数据 | 🟢 | 🟢 |
| **TimesFM** | Google Research | ICML 2024 / v2.0 | 通用单变量时序 | 解码器自回归 | 200M | 1000亿点 | 🟢 | 🟢 (HF) |
| **Chronos** | Amazon Research | ICML 2024 / Bolt | 通用单变量时序 | 离散分箱 T5/Encoder | 20M～710M | 泛领域时序语料 | 🟢 | 🟢 (HF) |
| **MOMENT** | CMU Auton Lab | ICML 2024 | 通用多变量时序 | Patch-MAE | 385M | Time-series Pile (含生理) | 🟢 | 🟢 (HF) |
| **MANTIS** | CMU | 2024-2025 | 通用时序多任务 | 统一潜表征 | 多规格 | 跨领域时序 | 🟢 | 🟢 (HF) |

---

## ⌚ 1. 智能手表与腕戴生理模型 (Smartwatch & Wrist)

详细分析与网络细节请查阅：📖 [docs/models/smartwatch_models.md](docs/models/smartwatch_models.md)

### 工业级与旗舰全能多模态巨座
- **Google SensorFM** (2026): 可穿戴健康领域的万亿分钟里程碑。基于动态重采样解决 PPG/ACC/EDA 采样率异构问题，在 35 个健康与行为基准中大幅超越专用模型。
- **清华大学 UniCardio** (Nature MI 2025/2026): 清华大学朱军教授、王立元教授团队联合北京安贞医院研发的**统一心血管多模态扩散基础模型**。首创统一扩散 Transformer (DiT) 框架融合去噪、插补与 PPG $\to$ 诊断级 ECG / 连续血压 BP 的跨模态生成，并引入持续学习范式以极低增量参数（0.3M/模态）支持端侧传感器的动态热插拔与终身防遗忘。
- **Apple WBM & PPG/ECG FMs**: 深入探索日常无感监测与心律失常预警，通过海量真实世界日常佩戴数据学习个体基线与行为动力学。
- **Samsung xMAE & HiMAE** (ICML 24 / ICLR 25): 专为智能手表端侧计算优化。xMAE 利用连续 PPG 虚拟重构偶发高精度 ECG；HiMAE 提出多尺度分层时间架构，延迟小于 1ms。

### 开源先锋代表
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
  - 架构创新：**双流动力学分解**（慢速生理状态流 + 快速事件突变流）。
  - 学习范式：非生成式 JEPA 潜空间预测，参数量仅 **0.72M**，以小博大超越百兆级模型。
- **GluFormer** (Pheno.AI / Weizmann / NVIDIA, **Nature 2026**):
  - 采用自回归 Next-token 预测（1200 token 上下文 ≈ 12.5 天），擅长长程代谢结局与心血管远期风险预测。
- **CGMformer** (中科院 / 上海六院, NSR 2025):
  - 基于 BERT 掩码重构架构，依托 5.9 万人真实世界数据，全面覆盖糖尿病筛查、分型及并发症管理。
- **CGM-LSM** (JHU CDHAI, 2024):
  - 聚焦短程血糖自回归预测（30min～2h），在 OhioT1DM 上大幅降低均方根误差。
- **CGM-JEPA** (CRUISE Research Group, 2025):
  - **GlucoFM 同门开源基准**：在 Hugging Face 完整开源模型权重与预训练数据集，验证了 JEPA 潜表征在 24h 血糖网格下的高迁移能力，是复现非生成式血糖基座的开源基石。
- **GlucoBench** (Texas A&M Irina Gaynanova Lab, **ICLR 2024**):
  - 首个系统的连续血糖预测基准套件与公开数据集聚合库，规范了多中心标准化评价协议。

---

## 📈 3. 通用时间序列基础模型 (General Time-Series FM)

详细分析请查阅：📖 [docs/models/general_tsfm.md](docs/models/general_tsfm.md)

- **TimesFM** (Google): 采用分块解码器架构，在巨量时序数据上预训练，提供强大的零样本点预测与概率区间预测。
- **Chronos & Chronos-Bolt** (Amazon): 将连续时间序列离散化为词元（Tokenization via Bins），利用语言模型（T5 骨干）进行自回归时间序列预测。
- **CMU MOMENT & MANTIS**: 基于 Patch-MAE 理念，覆盖多变量重建、异常检测与分类，预训练数据 Time-series Pile 包含大量生理信号。

---

## 💤 4. 睡眠与心肺多模态模型 (Sleep & Cardiopulmonary)

详细分析与网络细节请查阅：📖 [docs/models/sleep_models.md](docs/models/sleep_models.md)

- **Stanford Medicine SleepFM** (Nature Medicine 2024-2026):
  - **首个多器官耦合睡眠基座**：依托 6.5 万名受试者、近 60 万小时夜间多导睡眠监测 (PSG) 数据，联合建模脑神经（EEG）、心血管（ECG/PPG）、呼吸力学（腹胸呼吸带/气流）与肌电（EMG）。
  - **留一对比学习 (LOO-CL)**：在 30 秒窗口内随机遮蔽某一器官模态，拉近该模态表征与剩余多模态联合表征的距离，迫使网络捕获大脑-心脏-肺部之间的深层生物物理耦合与生理共振。
  - **单夜预测远期疾病**：不仅实现高精度的睡眠分期与呼吸暂停筛查，更凭单夜生理表征即可预测 130+ 种未来重大慢性疾病（阿尔茨海默/痴呆症 C-index 0.85、全因死亡率 0.84、心肌梗死 0.81、心房颤动 0.78）。
- **SleepMaMi** (首尔大学, ICML 2026):
  - **宏观-微观层级双编码器架构**：针对整夜宏观睡眠时序与局部微观瞬变波形尺度断层的痛点，提出 Macro-Encoder（结合年龄/性别/BMI 人口统计学先验建模全夜周期节律）与 Micro-Encoder（局部 MAE 重构与多模态对比学习）。
  - **少样本多中心迁移**：在 20,000+ 份 PSG 记录（约 15.8 万小时）上预训练，仅需 1% 标注即可匹敌全监督模型。
- **U-Sleep & 开源基准网络** (Nature npj Digital Medicine):
  - 基于全卷积 1D U-Net 的端到端多通道睡眠分期架构，具备极强的跨传感器配置与动态导联适应能力。

---

## 📦 5. 开源数据集与评测基准 (Datasets & Benchmarks)

详细数据集下载指引与预处理代码请查阅：📖 [docs/datasets/wearable_datasets.md](docs/datasets/wearable_datasets.md)

| 领域 / 模态 | 数据集名称 | 采集设备 / 团队 | 核心传感器与采样率 | 样本规模 | 适用任务与特色 | 获取方式 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
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

---

## 🛠️ 6. 动手实战与子工程关联 (Hands-on Labs)

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
