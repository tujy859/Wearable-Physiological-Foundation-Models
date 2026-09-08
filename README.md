# Awesome Wearable & Physiological Foundation Models

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Tracking](https://img.shields.io/badge/Status-Actively%20Maintained-blue.svg)]()

> 精选追踪学术界与工业界关于**智能手表、连续血糖监测（CGM）及可穿戴生理信号基础模型（Foundation Models）**的前沿论文、开源代码、预训练权重、基准数据集与实战解析。

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

1. **异构采样率与多模态物理对齐**：原始 PPG（25~100Hz）、ACC（25~50Hz）、ECG（250~1000Hz）、CGM（5分钟/点）以及派生指标（心率 1Hz、睡眠 30s）。通用模型难以直接处理如此大跨度的物理时间窗口。
2. **严重的运动伪影与频域碰撞**：手腕日常挥动或剧烈跑步产生的步频谐波（1~3.5 Hz）与心率脉搏波频段高度重叠。
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
| **PaPaGei** | Nokia Bell Labs | ICLR 2025 | PPG (光电脉搏) | ResNet1D-MoE | ~1.5M | 5.7万小时 (10个公开库) | 🟢 | 🟢 (Zenodo) |
| **Pulse-PPG** | UIUC / Memphis | UbiComp 2025 | 腕部野外高噪 PPG | 1D-ResNet (RelCon) | ~2M | 5.5万小时 (真实自由生活) | 🟢 | 🟢 (Zenodo) |
| **Samsung xMAE** | Samsung Research | ICML 2024 | PPG $\to$ 虚拟 ECG | 跨模态 MAE | 端侧优化 | 9,400 小时配对数据 | 🔴 | 🔴 |
| **Samsung HiMAE** | Samsung Research | ICLR 2025 | 腕部多尺度生理 | 分层多尺度编码器 | <1ms延迟 | 真实手表自由生活数据 | 🔴 | 🔴 |
| **GlucoFM** | Google / UNSW | arXiv 2026 | CGM (5min 血糖) | 双流 JEPA 预测 | **0.72M** | 10.9万小时 (477人) | 🟡 (承诺开源) | 🔴 |
| **GluFormer** | Pheno.AI / Weizmann | Nature 2026 | CGM (15min 血糖) | GPT 式自回归 | 135M | >1000万读数 (10,812人) | 🟢 | 🔴 (HPP受限) |
| **CGMformer** | 中科院 / 上海六院 | NSR 2025 | CGM (5min 血糖) | BERT 式 MLM | 0.85M~10M | 131万天 (5.9万人) | 🟢 | 🟢 (GitHub) |
| **CGM-LSM** | JHU CDHAI | arXiv 2024 | CGM 血糖 | GPT-2 自回归 | ~124M | 1600万读数 (592人) | 🟢 (无数据) | 🔴 |
| **SleepFM** | Stanford Medicine | Nat Med 2024-2026 | EEG+ECG+PPG+Resp | 留一对比学习 (LOO) | 基础模型 | 60万小时 (6.5万人) | 🟢 | 🟡 (受限开放) |
| **LIMU-BERT** | 厦门大学等 | UbiComp 2021 | 3轴 ACC + Gyro | Sensor-BERT | 轻量级 | 多源 IMU 无标注数据 | 🟢 | 🟢 |
| **TimesFM** | Google Research | ICML 2024 / v2.0 | 通用单变量时序 | 解码器自回归 | 200M | 1000亿点 | 🟢 | 🟢 (HF) |
| **Chronos** | Amazon Research | ICML 2024 / Bolt | 通用单变量时序 | 离散分箱 T5/Encoder | 20M~710M | 泛领域时序语料 | 🟢 | 🟢 (HF) |
| **MOMENT** | CMU Auton Lab | ICML 2024 | 通用多变量时序 | Patch-MAE | 385M | Time-series Pile (含生理) | 🟢 | 🟢 (HF) |
| **MANTIS** | CMU | 2024-2025 | 通用时序多任务 | 统一潜表征 | 多规格 | 跨领域时序 | 🟢 | 🟢 (HF) |

---

## ⌚ 1. 智能手表与腕戴生理模型 (Smartwatch & Wrist)

详细分析与网络细节请查阅：📖 [docs/models/smartwatch_models.md](docs/models/smartwatch_models.md)

### 工业级全模态巨座
- **Google SensorFM** (2026): 可穿戴健康领域的万亿分钟里程碑。基于动态重采样解决 PPG/ACC/EDA 采样率异构问题，在 35 个健康与行为基准中大幅超越专用模型。
- **Apple WBM & PPG/ECG FMs**: 深入探索日常无感监测与心律失常预警，通过海量真实世界日常佩戴数据学习个体基线与行为动力学。
- **Samsung xMAE & HiMAE** (ICML 24 / ICLR 25): 专为智能手表端侧计算优化。xMAE 利用连续 PPG 虚拟重构偶发高精度 ECG；HiMAE 提出多尺度分层时间架构，延迟小于 1ms。

### 开源先锋代表
- **PaPaGei** (ICLR 2025): Nokia Bell Labs 与剑桥联合发布，首个开源通用光电生理基础模型。采用 ResNet1D-MoE 架构，参数量仅 1.5M，在心率、血压、血管年龄等 20 个下游任务表现卓越。
- **Pulse-PPG** (UbiComp 2025): UIUC 主导，针对真实野外高噪手腕 PPG 提出相对对比学习（RelCon），有效克服真实生活中的剧烈运动伪影。
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
  - 聚焦短程血糖自回归预测（30min~2h），在 OhioT1DM 上大幅降低均方根误差。

---

## 📈 3. 通用时间序列基础模型 (General Time-Series FM)

详细分析请查阅：📖 [docs/models/general_tsfm.md](docs/models/general_tsfm.md)

- **TimesFM** (Google): 采用分块解码器架构，在巨量时序数据上预训练，提供强大的零样本点预测与概率区间预测。
- **Chronos & Chronos-Bolt** (Amazon): 将连续时间序列离散化为词元（Tokenization via Bins），利用语言模型（T5 骨干）进行自回归时间序列预测。
- **CMU MOMENT & MANTIS**: 基于 Patch-MAE 理念，覆盖多变量重建、异常检测与分类，预训练数据 Time-series Pile 包含大量生理信号。

---

## 📦 5. 开源数据集与评测基准 (Datasets & Benchmarks)

详细数据集下载指引与预处理代码请查阅：📖 [docs/datasets/wearable_datasets.md](docs/datasets/wearable_datasets.md)

| 数据集名称 | 采集设备 / 方式 | 核心模态 | 规模 / 受试者 | 适用任务 | 获取方式 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PhysioNet Wrist PPG** | 手腕光电 + 胸部心电 | 64Hz PPG + 32Hz ACC + ECG | 步行、跑步、骑行运动 | 运动心率探测、去噪 | PhysioNet 公开 |
| **PPG-DaLiA** | Empatica E4 手环 | PPG, ACC, EDA, Temp | 15名受试者自由生活 | 日常活动心率回归 | UCI 开放下载 |
| **WESAD** | 手腕 E4 + 胸戴 RespiBAN | PPG, EDA, EMG, Temp, ACC | 15名受试者压力诱发实验 | 情绪压力检测 | UCI 开放下载 |
| **Shanghai T2DM** | 雅培瞬感 CGM (15min) | 连续皮下间质血糖 | 110名患者 | CGM 重建、代谢表型 | 公开申请 |
| **OhioT1DM** | Medtronic CGM (5min) | CGM + 胰岛素剂量 + 碳水 | 12名 T1D 患者连续监测 | 实时血糖预测 | 申请许可 |

---

## 🛠️ 6. 动手实战与子工程关联 (Hands-on Labs)

本项目与核心实验复现子工程联动，提供端到端真实世界实操代码：

- ⌚ **智能手表工程实践自建库**: [`Watch_LSM`](../Watch_LSM)
  - 包含真实智能手表运动伪影碰撞分析（跑步场景心率 MAE 从 51 BPM 降至 7.5 BPM）；
  - 包含生产级双流跨模态自监督训练框架 `watch_lsm`。
- 🩸 **CGM 基础模型横评与复现**: [`cgm_fm`](../cgm_fm)
  - 深入评测 GlucoFM、GluFormer 与 CGM-JEPA 的表征迁移能力。
- 💻 **极简 Demo 脚本**（开箱即用体验）：
  - 详见 `notebooks/`（使用已公开开源权重进行 10 行代码特征提取与预测）。

---

## 🤝 参与贡献 (Contributing)

欢迎提交 Issue 或 Pull Request 推荐最新的顶会论文或开源模型！具体规范请见 [CONTRIBUTING.md](CONTRIBUTING.md)。
