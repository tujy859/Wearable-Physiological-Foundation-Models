# 可穿戴生理时序公开数据集与评测基准汇总 (Wearable Physiological Datasets & Benchmarks)

> 可穿戴基础模型的训练与评估高度依赖高质量、多模态、贴近真实生活（Free-living）的生理时序数据集。本专题系统梳理 **智能手表腕部多模态（PPG/ACC/ECG/EDA）** 与 **连续血糖监测（CGM）** 两大核心领域的经典公开数据集、获取途径、授权协议与数据预处理标准流水线。

---

## 📑 目录

- [1. 智能手表与腕部生理数据集 (PPG, ACC, EDA, ECG)](#1-智能手表与腕部生理数据集-ppg-acc-eda-ecg)
- [2. 连续血糖监测数据集 (Continuous Glucose Monitoring, CGM)](#2-连续血糖监测数据集-continuous-glucose-monitoring-cgm)
- [3. 常用临床多模态参考基准库 (Clinical High-Precision Waveforms)](#3-常用临床多模态参考基准库-clinical-high-precision-waveforms)
- [4. 睡眠与心肺多导生理监测数据集 (Polysomnography & Sleep Dynamics)](#4-睡眠与心肺多导生理监测数据集-polysomnography--sleep-dynamics)
- [5. 数据集全景汇总与获取指引一览表](#5-数据集全景汇总与获取指引一览表)
- [6. 数据清洗与标准化工具链 (Harmonization Tools)](#6-数据清洗与标准化工具链-harmonization-tools)

---

## 1. 智能手表与腕部生理数据集 (PPG, ACC, EDA, ECG)

腕戴数据集的核心挑战在于**真实运动伪影（Motion Artifacts, MA）** 与海量真实生活自由佩戴（Free-living）数据的缺乏。以下精选数据集兼顾了超大规模开源基准、真实非受限野外场景与受控实验：

### 1.1 Stanford OpenMHC (Open MyHeartCounts, 2026) —— 全球最大开源穿戴基础模型基准
- **提供团队**: 斯坦福大学医学院 (Euan Ashley 实验室) 联合帝国理工学院、柏林夏里特医学院
- **发表情况**: arXiv:2607.16235 (2026) | [GitHub: AshleyLab/OpenMHC](https://github.com/AshleyLab/OpenMHC)
- **数据规模**: **11,894 名**真实受试者，记录超过 **6,700 万佩戴小时（282 万人·天）**，时间跨度达 13 年。
- **传感器与通道 (19 维)**:
  - 智能手机与 Apple Watch 分钟级时序：步数、心率（HR）、心率变异性（HRV）、静息心率、睡眠分期（浅睡/深睡/REM）、运动能量消耗、环境光等；
  - 关联 **169 维临床与生活方式变量**（问卷调查、健康感知、焦虑抑郁量表、心血管慢病史、HealthKit 档案）。
- **实战获取方式 (极低门槛，完全免费)**:
  - **途径 A: `OpenMHC-XS` (极轻量子集，约 1.9 GB)**：
    无需审核，官方提供 Python API 本地一键下载解压：
    ```python
    import openmhc
    openmhc.download_dataset(version="xs", dest="~/.cache/openmhc/data-xs")
    ```
  - **途径 B: `OpenMHC-Full` (完整科研集，约 38 GB)**：
    托管于 **Harvard Dataverse (哈佛学术数据网盘)**。在 `dataverse.harvard.edu` 注册账号并在线签署标准数据使用协议（DUA），生成 API Token 后即可直接批量下载至本地个人电脑或服务器。
- **评测赛道**: 规范化了 Track 1（32项下游健康预测）、Track 2A（多变量生成式插补）、Track 2B（未来时序自回归预测）。

### 1.2 PhysioNet Wrist PPG Exercise Dataset
- **提供机构**: PhysioNet (CCALab)
- **获取方式**: [PhysioNet 开放获取](https://physionet.org/content/wrist/) (无需伦理申请)
- **受试者与运动协议**: 8 名健康受试者在跑步机或阻力自行车上进行走路、跑步（最高 15 km/h）、低/高阻力骑行。
- **传感器通道与采样率**:
  - 手腕 PPG (256 Hz)
  - 手腕三轴加速度计 ACC (256 Hz)
  - 手腕三轴陀螺仪 Gyro (256 Hz)
  - 胸部佩戴实验室金标准心电 ECG (256 Hz)
- **典型应用**: 算法对抗严重运动伪影的黄金评测基准；心率（HR）连续回归与去噪。

### 1.3 PPG-DaLiA (Daily Life Activities)
- **提供机构**: 德国弗劳恩霍夫应用信息技术研究所 (Fraunhofer FIT) / UCI 机器学习知识库
- **获取方式**: [UCI Machine Learning Repository 开放下载](https://archive.ics.uci.edu/dataset/495/ppg+dalia)
- **数据规模**: 15 名受试者，单人采集时长约 2.5 小时，总计时长超 36 小时。
- **场景设计**: 包含 8 种日常真实生活活动（在桌面办公、走楼梯、户外骑行、在咖啡馆交谈、吃午餐、开车等）。
- **硬件设备**:
  - 手腕端: Empatica E4 手环（PPG 64Hz, ACC 32Hz, EDA 4Hz, 皮肤温度 4Hz）。
  - 参考金标准: 胸带 RespiBAN（高精 ECG 700Hz，用于计算真实心率基准）。

### 1.4 WESAD (Wearable Stress and Affect Detection)
- **提供机构**: UCI 机器学习库 / 德国博世与乌尔姆大学
- **获取方式**: [UCI 开放下载](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection)
- **数据内容**: 15 名受试者在受控实验室压力诱发实验（特里尔社会压力测试 TSST、观看有趣/悲伤视频片段）中的多模态生理记录。
- **传感器**: 手腕 Empatica E4 + 胸戴 RespiBAN（涵盖 PPG, 3-轴 ACC, EDA, 肌电 EMG, 体温 Temp, 呼吸气流 Resp）。
- **典型应用**: 情绪状态识别、生理心理压力剧增（Stress Detection）分类基准。

### 1.5 TROIKA
- **核心定位**: 运动伪影消除与跑步心率跟踪的开山基准（IEEE TBME 2015）。
- **特点**: 包含 12 名受试者在跑步机从 1～2 km/h 加速到 12～15 km/h 的双通道手腕 PPG 与三轴 ACC 信号，配对胸部心电。

---

## 2. 连续血糖监测数据集 (Continuous Glucose Monitoring, CGM)

连续血糖数据具有 288 点昼夜网格周期性、餐后突发动态及传感器脱落等特性。

### 2.1 ShanghaiT1DM & ShanghaiT2DM (上海六院队列)
- **来源与引用**: 上海交通大学附属第六人民医院（包玉倩教授团队），发表于 *Scientific Data* (2023)
- **获取入口**: [Figshare Record 20444397](https://figshare.com/articles/dataset/ShanghaiT1DM_T2DM_data/20444397) (CC BY 4.0 开放协议，直接下载压缩包)
- **受试者规模**:
  - `ShanghaiT1DM`: 12 名 1 型糖尿病患者连续监测记录；
  - `ShanghaiT2DM`: 100 名 2 型糖尿病患者连续监测记录（单人持续 14 天）。
- **传感器规格**: 雅培瞬感 (Abbott FreeStyle Libre)，原生采样间隔为 15 分钟。
- **标签信息**: 随附极具价值的临床化验表单（`Summary.xlsx`），包含受试者的空腹血糖 (FPG)、糖化血红蛋白 (HbA1c)、空腹胰岛素、C 肽、甘油三酯、总胆固醇等全套血检生化指标。

### 2.2 Hall et al. Glucotypes (斯坦福健康队列)
- **来源与引用**: Hall et al., *Glucotypes reveal new patterns of glucose dysregulation*, **PLoS Biology** (2018)
- **获取入口**: [PLoS Biology 补充材料 S1 Data (TSV)](https://journals.plosopen.org/) (公开可直接抓取)
- **受试者规模**: 57 名健康人、前驱糖尿病及未确诊糖尿病受试者，共 105,426 行 5 分钟网格读数。
- **核心价值**: 提出了基于血糖动态变异度（MAGE、标准差）的 **Glucotype（低/中/重度波动表型）** 分型，是 CGM 基础模型评测无监督代谢分型的标准基准。

### 2.3 BIG IDEAs Glycemic Wearable Dataset
- **来源与引用**: 杜克大学 BIG IDEAs Lab, *PhysioNet* (Version 1.1.2)
- **获取入口**: [PhysioNet 开放项目](https://physionet.org/content/big-ideas-glycemic-wearable/1.1.2/)
- **受试者与模态**: 16 名受试者同时佩戴 Dexcom G6（5分钟血糖）与 Empatica E4 手环（PPG/ACC/EDA/Temp）。
- **核心价值**: 连接微创间质血糖与腕戴非侵入生理指标的罕见多模态配对数据。

### 2.4 CGMacros (双设备与饮食多模态队列)
- **来源与引用**: *PhysioNet* (Version 1.0.0)
- **获取入口**: [PhysioNet 开放项目](https://physionet.org/content/cgmacros/1.0.0/)
- **规模与特色**: 45 名非糖尿病健康受试者同时佩戴 Dexcom G6（5分钟网格）与 Abbott Libre Pro（15分钟网格），并记录了每餐的高清餐食照片、精确宏量营养素（碳水/蛋白质/脂肪克数）及入组生化血检（HbA1c, HOMA-IR, 血脂四项）。

### 2.5 Colas et al. (DFA 自由生活长时程队列)
- **来源与引用**: Colas et al., *PLOS ONE* (2019)
- **获取入口**: PLOS ONE 补充材料 `Colas_DFA_S1_Data.zip` (公开)
- **规模**: 208 名健康自由生活受试者，单人监测时长超 14 天，累计有效时长超 9,500 小时。

### 2.6 OhioT1DM (短程血糖预测基准)
- **来源**: 俄亥俄大学与 Blood Glucose Level Prediction Challenge (BGLP)
- **获取方式**: [OhioT1DM 官方申请页](http://smarthealth.cs.ohio.edu/OhioT1DM-dataset.html)（需签署学术研究数据使用协议 DUA，审核通常 1～2 个工作日通过）
- **规模与模态**: 12 名 1 型糖尿病患者为期 8 周的高密集数据（Dexcom 5分钟 CGM + 胰岛素基础率/大剂量泵注 + 碳水摄入估计 + 自报运动）。
- **地位**: 评估 30/60/120 分钟血糖实时自回归预测（Forecasting）的国际公认基准。

### 2.7 Weinstock et al. 2016 (T1D Exchange 老年长程连续监测队列)
- **来源与引用**: Weinstock et al., *The Journal of Clinical Endocrinology & Metabolism* (2016) / JAEB 临床研究中心
- **获取入口**: [T1D Exchange / JAEB Center 公开申请通道](https://public.jaeb.org/) (或经由 GlucoBench 预处理工具链获取，本地 CSV 约 124MB)
- **受试者规模**: 226 名高龄（$\ge 60$ 岁）1 型糖尿病患者，长达 26 周的日常佩戴记录，累计有效数据量超 **1.2 亿个采样点**。
- **核心价值**: 针对老年糖尿病患者最凶险的“无症状夜间严重低血糖（Nocturnal Hypoglycemia）”提供规模最大的真实世界连续基准，也是 GlucoBench 评测模型长期抗漂移能力的核心测试集。

### 2.8 GlucoBench (ICLR 2024 连续血糖基准套件与数据集合)
- **来源与代码**: Texas A&M 大学 Irina Gaynanova 教授团队 ([GitHub: IrinaStatsLab/GlucoBench](https://github.com/IrinaStatsLab/GlucoBench))
- **发表会议**: **ICLR 2024**
- **核心贡献**: 首次系统性清洗、对齐并汇总了国际上主要的公开 CGM 数据集，提出了标准化的时序外推（Forecasting）评测协议，消除了传统算法在不同私有子集上各自为战、无法公平对比的行业痛点。

### 2.9 受控大型研究队列（需正式伦理审查申请）
- **AI-READI**: NIH 旗舰项目，针对 2,280 名 2 型糖尿病、前驱糖尿病与健康人的长程 CGM + 视网膜影像 + 全基因组多模态队列（申请入口: `ai-readi.org`）。
- **T1DEXI & T1DEXIP**: JAEB 临床研究中心，针对 497 名患者在自由生活与运动条件下的 CGM 连续记录（申请入口: `jaeb.org`）。

---

## 3. 常用临床多模态参考基准库 (Clinical High-Precision Waveforms)

用于对穿戴模型进行预训练（如 PaPaGei 初始预训练）或提取高精度形态基准：

1. **VitalDB**:
   - 包含首尔大学医院 10,000 余例非心脏手术患者在手术室内的超高精度多通道监测数据（500Hz 动脉侵入血压波、指尖 PPG、高导联心电、脑电双频指数）。
   - 获取方式: [VitalDB 开放数据平台](https://vitaldb.net/)，支持 Python API `pip install vitaldb`。
2. **MIMIC-III / MIMIC-IV Waveform Database**:
   - 麻省理工学院 (MIT) 与贝斯以色列女伯爵医学中心 (BIDMC) 联合建立的重症监护多模态数据库，包含数万名 ICU 患者的高频脉搏与心电波形。
   - 获取方式: PhysioNet 凭证访问（需完成 CITI 临床研究伦理认证）。
3. **MC-MED (Multimodal Clinical Monitoring in the Emergency Department)**:
   - 由北京大学洪申达团队整理并发布在 PhysioNet，采集自大型学术医学中心急诊科（ED）2020～2022 年间 118,385 例急诊就诊记录；
   - 提取了超过 **1000 万对严格同步配对** 的高频 PPG 与 ECG 信号，且融合了人口统计学、病史用药、化验血检、影像诊断以及经医生标注的心血管疾病（CVD）明确标签；
   - 获取方式: [PhysioNet 凭证访问](https://physionet.org/content/mc-med/)（需完成 CITI 伦理认证并签署 DUA）。
4. **BIDMC PPG and Respiration Dataset**:
   - 源自哈佛医学院附属贝斯以色列女伯爵医学中心（BIDMC）重症监护数据库（从 MIMIC-II 抽取的生理基准子集）；
   - 包含 53 位重症患者、每段 8 分钟的高采样率（125Hz）严格同步的 PPG、ECG 及阻抗呼吸波形，配有人工呼吸周期精细标注；
   - 获取方式: [PhysioNet 开放直接下载](https://physionet.org/content/bidmc/1.0.0/)（完全公开，遵循 ODC-BY 协议）。
5. **UK Biobank (UKB) 穿戴体动与全景多组学队列**:
   - 包含 10 万名受试者连续 7 天佩戴手腕研究级高频三轴加速度计（Axivity AX3, 100Hz）的真实生活记录，且深度链接了 50 万人的全基因组测序（WGS）、30 余项临床血液生化指标以及全生命周期 NHS 电子病历；
   - **获取方式与准入红线 (严格受控)**:
     - 必须由大学/科研机构全职学者（PI）通过 [UKB Access Management System (AMS)](https://www.ukbiobank.ac.uk/enable-your-research/apply-for-access) 提交正式课题提案；
     - 需由所在大学科研院/法务部门正式签署 Material Transfer Agreement (MTA) 法律协议，并缴纳 £2,000～£9,000+ 英镑（折合数万至近十万元人民币）的数据准入管理费；
     - **强制云端封闭计算**: 自 2024 年起，超大规模时序与组学数据禁止下载至本地，强制在 **UKB Research Analysis Platform (RAP, 基于 AWS / DNAnexus)** 云端环境中租赁计算实例运行，严禁私自分发扩散。
6. **All of Us (美国国立卫生研究院多族裔精准医学队列)**:
   - 超过 25 万名受试者的跨族裔生物样本、长程电子健康档案与生活方式追踪，为系统级健康模型（如 RisQ）提供零微调外部验证金标准；
   - 获取方式: 通过 [All of Us Research Program Workbench](https://www.researchallofus.org/) 注册认证后在云端工作台访问。

---

## 4. 睡眠与心肺多导生理监测数据集 (Polysomnography & Sleep Dynamics)

睡眠时序是评测脑-心-肺跨器官耦合、自主神经调节及远期慢性疾病生存风险的核心金标准：

1. **SHHS (Sleep Heart Health Study)**:
   - **主导机构**: 美国国立卫生研究院 (NIH / NHLBI) 资助的多中心流行病学队列；
   - **样本与规模**: 5,804 名中老年受试者，包含长达数十年的心脑血管疾病发病随访；
   - **传感器模态**: 全套未压缩科研级夜间 PSG（脑电 C3/C4、眼电 EOG、下颌肌电 EMG、心电 ECG、胸腹电阻抗呼吸带、口鼻热敏气流及透射脉搏血氧）；
   - **获取方式**: [美国国家睡眠研究资源库 (NSRR, sleepdata.org)](https://sleepdata.org/datasets/shhs) 签署学术协议后免费开放直下。
2. **MESA (Multi-Ethnic Study of Atherosclerosis) Sleep**:
   - **主导机构**: 美国 NHLBI 多族裔动脉硬化长程队列；
   - **样本与规模**: 2,237 名来自白人、非裔、西语裔与亚裔的多族裔受试者；
   - **模态特色**: 完整夜间 PSG 同步配对为期 7 天的手腕体动记录（Actigraphy），具备连接临床 PSG 金标准与可穿戴日常活动的高价值桥梁；
   - **获取方式**: [NSRR 开放获取](https://sleepdata.org/datasets/mesa)。
3. **PhysioNet Sleep-EDF (Expanded)**:
   - **主导机构**: 荷兰神经研究所 / PhysioNet；
   - **规模**: 197 份整夜 PSG 记录（涵盖健康受试者与助眠药物使用者）；
   - **模态**: 双通道脑电（Fpz-Cz / Pz-Oz）、水平眼电、下颌肌电与口鼻呼吸；
   - **获取方式**: [PhysioNet 开放直下](https://physionet.org/content/sleep-edfx/1.0.0/)（完全公开，算法快速基准验证首选）。

---

## 5. 数据集全景汇总与获取指引一览表

| 领域 / 模态 | 数据集名称 | 机构 / 团队 | 核心传感器与采样率 | 样本规模 | 获取难度 | 适用研究任务 |
| :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| **手腕多模态** | **OpenMHC** | Stanford (Ashley组) | 19维连续通道+169维变量 | 11,894人 (>6700万h) | 🟢 开放免费 (Dataverse) | 基础模型预训练、多变量插补与预测 |
| **手腕多模态** | **PhysioNet Wrist** | PhysioNet | PPG 256Hz, ACC 256Hz, ECG | 8人 (走跑骑) | 🟢 开放直下 | 运动伪影消除、心率连续回归 |
| **手腕多模态** | **PPG-DaLiA** | Fraunhofer / UCI | PPG 64Hz, ACC 32Hz, EDA | 15人 (自由生活) | 🟢 开放直下 | 日常真实心率监测、去噪 |
| **手腕多模态** | **WESAD** | UCI / Bosch | PPG, ACC, EDA, EMG, Temp | 15人 (受控压力) | 🟢 开放直下 | 心理压力检测、情绪分类 |
| **手腕多模态** | **TROIKA** | IEEE TBME | 双通道 PPG, 3轴 ACC | 12人 (跑步机) | 🟢 开放直下 | 运动心率追踪算法经典基准 |
| **CGM 血糖** | **CGMacros** | PhysioNet (2024) | 双 CGM (5/15min) + 高清餐食照 | 45人 (带生化血检)| 🟢 开放直下 | 饮食营养摄入响应、生化指标探针 |
| **CGM 血糖** | **ShanghaiT1/T2DM**| 上海六院 (包玉倩组) | 瞬感 Libre (15min 连续间质液) | 112人 (14天) | 🟢 开放直下 | 缺失重构、生化表型探针 |
| **CGM 血糖** | **Hall Glucotypes** | Stanford (Snyder组) | Dexcom (5min 网格) | 57人 (10.5万点)| 🟢 开放直下 | 血糖波动分型、无监督聚类 |
| **CGM 血糖** | **BIG IDEAs** | Duke (Dunson组) | Dexcom 5min + 手环 E4 | 16人 (双设备) | 🟢 开放直下 | 穿戴光电-皮下间质跨模态 |
| **CGM 血糖** | **Colas DFA** | PLOS ONE | CGM 5min 自由生活 | 208人 (>9500h) | 🟢 开放直下 | 基础模型大规模自监督预训练 |
| **CGM 血糖** | **OhioT1DM** | 俄亥俄大学 | Dexcom 5min + 胰岛素/碳水 | 12人 (8周时程) | 🟡 学术申请 | 30～120min 血糖自回归预测 |
| **CGM 血糖** | **Weinstock 2016** | T1D Exchange / JAEB | Dexcom G4 (5min) 长期监测 | 226人 (>1.2亿读数) | 🟡 学术申请 | 老年高危人群夜间无症状低血糖筛查 |
| **CGM 血糖** | **Glucose-ML** | Augmented Health Lab | 自动化集成 20+ 个公开数据集 | 4,300+人 (44.9M点) | 🟢 GitHub 开源 | 20+ 数据集一键下载与标准化集合 |
| **临床生理基准**| **VitalDB** | 首尔大学医院 | 动脉波 500Hz, PPG, ECG | >10,000 例手术 | 🟢 开放 API | 血管弹性、连续血压金标准 |
| **临床生理基准**| **MC-MED** | 北京大学 (洪申达组) | PPG 100Hz+, ECG, 呼吸等 | 11.8万人 (>1000万对) | 🟡 凭证申请 | 跨模态生成、急诊心血管筛查 |
| **临床生理基准**| **BIDMC PPG** | 哈佛医学院 BIDMC | PPG 125Hz, ECG 125Hz, Resp | 53人 (8分钟高精记录) | 🟢 开放直下 | 呼吸率估计、脉搏波形态基准 |
| **睡眠多导 PSG**| **SHHS** | 美国 NIH / NHLBI | 完整 PSG (EEG/ECG/Resp/EMG) | 5,804人 (多年随访) | 🟢 NSRR 申请 | 跨器官耦合、慢病死亡长期随访 |
| **睡眠多导 PSG**| **MESA** | 美国 NHLBI / 多中心 | 完整 PSG + 7天手腕体动仪 | 2,237人 (多族裔) | 🟢 NSRR 申请 | 多族裔睡眠结构、动脉硬化风险 |
| **睡眠多导 PSG**| **Sleep-EDF** | 荷兰神经所 / PhysioNet | 双通道 EEG, EOG, EMG, 呼吸 | 197人整夜记录 | 🟢 开放直下 | 快速原型验证、跨域分期对比 |
| **系统级全景队列**| **UK Biobank** | 英国 MRC / Wellcome | 手腕高频体动 100Hz + WGS + EHR | 10万人体动/50万人全景 | 🔴 严格申请 (AMS/RAP) | 系统级健康世界模型、远期发病风险 |
| **系统级全景队列**| **All of Us** | 美国 NIH | 电子健康档案 + 多组学 + 穿戴 | 25万+人 (多族裔) | 🟡 平台认证 (Workbench)| 跨族裔通用健康模型外推基准 |

---

## 6. 数据清洗与标准化工具链 (Harmonization Tools)

### 5.1 连续血糖标准化自动化套件：Glucose-ML Project
- **项目仓库**: [GitHub: Augmented-Health-Lab/Glucose-ML-Project](https://github.com/Augmented-Health-Lab/Glucose-ML-Project) (Emory 大学 Prioleau 组，MIT 协议)
- **两大核心自动化脚本**:
  1. `auto-download-open-datasets.py`: 一键自动化爬取并解压 14 个常用开源 CGM 数据集；
  2. `auto-harmonize-CGML-datasets.py`: 自动识别并纠正多达 20 余个数据集中的异构列名（如时间戳字符串格式、单位换算 $\text{mg/dL} \leftrightarrow \text{mmol/L}$，换算系数 $1\text{ mmol/L} = 18.018\text{ mg/dL}$）。
- **统一化三列 CSV 规范**:
  ```csv
  subject_id,timestamp,glucose_value
  shanghai_01,2021-05-10 08:00:00,112.5
  shanghai_01,2021-05-10 08:05:00,115.0
  ```

### 5.2 腕部生理信号质量评估 (Signal Quality Index, SQI)
在将原始高频 PPG/ECG 输入大模型前，过滤信噪比极低或佩戴脱落片段至关重要：
1. **时域偏度与峭度 (Skewness & Kurtosis SQI)**:
   - 正常心脏脉搏充盈波形具有正偏斜特性（$S_{\text{sqi}} > 0$），若运动伪影导致波形平底或杂乱震荡，偏度将显著异化。
2. **零交叉与能量比 (Zero-Crossing & Perfusion Index)**:
   - 灌注指数 $\text{PI} = (AC / DC) \times 100\%$，若 $\text{PI} < 0.1\%$ 通常提示探头与皮肤发生脱开悬空。
3. **标准化 1D Patch 窗口对齐协议**:
   - 设定基础时间片 $\tau = 0.25\text{s}$：
     - PPG (64 Hz) $\to$ 每片 16 个点；
     - ACC (32 Hz) $\to$ 每片 8 个点；
   - 保证进入 Transformer 或 1D-ResNet 骨干时的多模态 Token 数量完全等长对齐。
