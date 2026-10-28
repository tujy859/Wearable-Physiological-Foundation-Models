# 可穿戴与生理时序公开数据集

可穿戴生理基础模型的预训练与下游评测依赖多模态、多场景的生理时序数据集。本专题梳理**智能手表与腕部多模态（PPG/ACC/EDA/ECG）**、**连续血糖监测（CGM）**、**临床监护与诊断级心电**以及**睡眠多导与大型人群队列**四类公开数据集，说明其传感器配置、样本规模、获取流程与标准化预处理方法。

## 目录

- [1. 腕部生理与日常活动数据集](#1-腕部生理与日常活动数据集)
- [2. 连续血糖监测数据集与基准](#2-连续血糖监测数据集与基准)
- [3. 临床监护与心电数据集](#3-临床监护与心电数据集)
- [4. 睡眠多导与大型人群队列](#4-睡眠多导与大型人群队列)
- [5. 数据集全景汇总表](#5-数据集全景汇总表)
- [6. 数据清洗与信号质量控制](#6-数据清洗与信号质量控制)

---

## 1. 腕部生理与日常活动数据集

腕戴生理信号分析的主要难点在于运动伪影（Motion Artifacts, MA）干扰以及自由生活（Free-living）长程标注数据的匮乏。以下数据集覆盖了大规模自由佩戴队列、日常活动心率监测与实验室压力诱发场景。

### OpenMHC

Stanford University / Imperial College London / Charité · 2026 · [主页/代码](https://github.com/AshleyLab/OpenMHC) · [论文](https://arxiv.org/abs/2607.16235)

OpenMHC（Open MyHeartCounts）源自斯坦福大学 MyHeartCounts 移动健康研究项目，面向可穿戴基础模型提供大规模真实世界多模态时序与标准化评测基准。

- **样本规模**：共 **11,894 名**受试者，累计超过 6,700 万佩戴小时（约 282 万人·天），时间跨度达 13 年。
- **传感器与变量**：包含智能手机与 Apple Watch 采集的 19 维分钟级时序通道（步数、心率、心率变异性 HRV、静息心率、浅睡/深睡/REM 睡眠分期、运动能量消耗、环境光等），并关联 169 维临床与生活方式变量（问卷调查、健康感知、焦虑抑郁量表、心血管病史及 HealthKit 记录）。
- **评测赛道**：划分为 Track 1（32 项下游健康与行为预测）、Track 2A（多变量缺失插补）与 Track 2B（未来时序预测）。
- **获取方式**：提供两级开放获取途径。轻量级子集 `OpenMHC-XS`（约 1.9 GB）无需审批，可通过官方 Python 工具包直接下载：

```python
import openmhc
openmhc.download_dataset(version="xs", dest="~/.cache/openmhc/data-xs")
```

完整数据集 `OpenMHC-Full`（约 38 GB）托管于 Harvard Dataverse，注册账号并在线签署标准数据使用协议（DUA）后可通过 API Token 批量下载。

### PhysioNet Wrist PPG

University of Manchester · 2017 · [主页/下载](https://physionet.org/content/wrist/1.0.0/) · [论文](https://doi.org/10.3390/data2010001)

PhysioNet Wrist PPG During Exercise 数据集由曼彻斯特大学 Jarchi 与 Casson 采集，用于评估运动状态下腕部光电脉搏波去噪与心率跟踪算法。

- **样本规模**：8 名健康受试者（3 男 5 女，年龄 22-32 岁），在跑步机与功率自行车上依次完成步行、慢跑/快跑、低阻力骑行与高阻力骑行 4 类运动，单次活动时长 4-10 min。
- **传感器与采样率**：左腕佩戴脉搏血氧与惯性传感装置，同步记录手腕 PPG、三轴低噪声加速度计、三轴宽量程加速度计与三轴陀螺仪信号，胸部通过 Actiwave 记录单导联参考心电（ECG）并提供人工复核的 R 波峰值标注；所有通道采样率均为 **256 Hz**。
- **获取方式**：遵循 ODC-BY 1.0 协议，在 PhysioNet 页面直接下载，无需注册或申请。

### PPG-DaLiA

University of Siegen / Bosch · 2019 · [主页/下载](https://archive.ics.uci.edu/dataset/495/ppg+dalia) · [论文](https://doi.org/10.3390/s19143079)

PPG-DaLiA（PPG Dataset for Motion Compensation and Heart Rate Estimation in Daily Life Activities）由锡根大学与博世研究院联合发布，聚焦贴近真实日常生活的腕部心率估计。

- **样本规模**：15 名健康受试者（8 女 7 男，年龄 21-55 岁），每人连续采集约 2.5 小时，累计总时长约 **36 小时**。实验协议涵盖 8 类日常活动：静坐、上下楼梯、桌式足球、户外骑行、驾驶汽车、午餐、步行及办公桌工作。
- **传感器与采样率**：手腕佩戴 Empatica E4 手环（PPG 64 Hz、三轴 ACC 32 Hz、皮肤电活动 EDA 4 Hz、皮肤温度 4 Hz）；胸部佩戴 RespiBAN（ECG 700 Hz、呼吸与三轴 ACC 700 Hz），以胸部 ECG 提取的心率作为参考真值。
- **获取方式**：托管于 UCI Machine Learning Repository，直接下载。

### WESAD

University of Siegen / Bosch · 2018 · [主页/下载](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection) · [论文](https://doi.org/10.1145/3242969.3242985)

WESAD（Wearable Stress and Affect Detection）由锡根大学与博世研究院在 ACM ICMI 2018 发布，是可穿戴情绪识别与心理压力检测的常用公开数据集。

- **样本规模**：**15 名**受试者在受控实验室环境下完成三态诱发实验，包括基线中性态（Baseline）、特里尔社会压力测试（TSST 公开演讲与心算，诱发 Stress）以及观看幽默视频片段（诱发 Amusement），中间穿插冥想恢复阶段并配对 PANAS、STAI、SAM 等心理问卷自评。
- **传感器与采样率**：手腕佩戴 Empatica E4（PPG 64 Hz、三轴 ACC 32 Hz、EDA 4 Hz、皮温 4 Hz）；胸部佩戴 RespiBAN（ECG、EDA、肌电 EMG、呼吸气流 Resp、体温 Temp 及三轴 ACC，采样率均为 700 Hz）。
- **获取方式**：托管于 UCI Machine Learning Repository，直接下载。

### TROIKA

Samsung Research America / 国防科技大学 · 2015 · [主页/下载](https://zenodo.org/records/3902660) · [论文](https://doi.org/10.1109/TBME.2014.2359372)

TROIKA 数据集随 IEEE TBME 2015 同名论文及 2015 IEEE Signal Processing Cup 竞赛发布，用于评估剧烈跑步运动下的腕部 PPG 心率追踪算法。

- **样本规模**：**12 名**健康男性受试者（年龄 18-35 岁），在跑步机上按“静息 30 s → 6-8 km/h 慢跑 1 min → 12-15 km/h 快跑 1 min → 6-8 km/h 慢跑 1 min → 12-15 km/h 快跑 1 min → 静息 30 s”的变速协议各采集 5 min 波形。
- **传感器与采样率**：手腕双通道绿光 PPG（波长 515 nm）、手腕三轴加速度计（ACC）以及胸部单导联湿电极参考 ECG，所有信号采样率均为 125 Hz，按 8 s 滑动窗口（步长 2 s）提供参考心率真值。
- **获取方式**：可通过 Zenodo 归档库直接下载。

### CAS-BP

中国科学院深圳先进技术研究院 / 华为 · 2023 · [主页/下载](https://github.com/zdzdliu/CAS-BP) · [论文](https://doi.org/10.1109/JBHI.2023.3287313)

CAS-BP 是面向智能手表无袖带血压（Cuffless Blood Pressure）连续回归与多日随访校准评估的大规模验证数据集。

- **样本规模**：公开申请子集（Dataset-1）包含 **1,272 名**受试者（平均年龄 50.16 岁，66.82% 女性，25.63% 高血压患者），完成跨度 4 周的 4 次随访测量（第 0、7、14、21 天），累计 12,624 条有效测量记录。
- **传感器与金标准**：采用智能手表实验设备同步采集手腕多波长 PPG 与单导联 ECG 波形（原始采样率 1000 Hz，单次测量约 2 min，官方同时提供提炼后的 31 维生理与脉搏波传导时间 PTT 特征），配对双人柯氏音水银血压计听诊收缩压（SBP）与舒张压（DBP）读数，遵循 IEEE 1708 与 ANSI/AAMI/ISO 81060-2 测量规范。
- **评测协议**：
  - **无校准评估**（Calibration-free）：跨受试者直接从生理特征或波形预测血压绝对值。受个体血管刚度与接触阻抗差异影响，主流模型在该设定下的收缩压 MAE 通常在 7.5-11.0 mmHg 之间。
  - **基线校准评估**（Calibrated）：以第 0 天袖带基准值 $\text{BaseSBP}$ 为个性化锚点，测试阶段结合无校准预测值进行后处理校准（如 $\text{SBP}_{\text{cal}} = 0.3 \times \text{SBP}_{\text{free}} + 0.7 \times \text{BaseSBP}$），可将静息队列收缩压与舒张压 MAE 降低至 3.0-4.5 mmHg。
- **获取方式**：GitHub 仓库直接提供示例数据、解析脚本与 31 维特征说明；完整原始波形（Dataset-1）需填写仓库中的数据访问申请表向作者团队免费申请。基于该数据集的血压大模型微调方法参见 [CBPM-LLaMA](../models/health_reasoning_llms.md#cbpm-llama)。

---

## 2. 连续血糖监测数据集与基准

连续血糖监测（CGM）数据通常按 5 min（每日 288 个点）或 15 min（每日 96 个点）网格采样，具有昼夜节律性、餐后快速爬升及传感器脱落缺失等特点。

<a id="shanghait1dm"></a>
<a id="shanghait2dm"></a>

### ShanghaiT1DM / ShanghaiT2DM

上海交通大学医学院附属第六人民医院 · 2023 · [主页/下载](https://figshare.com/articles/dataset/ShanghaiT1DM_T2DM_data/20444397) · [论文](https://doi.org/10.1038/s41597-023-01940-7)

ShanghaiT1DM 与 ShanghaiT2DM 由上海交通大学医学院附属第六人民医院包玉倩、赵蔚菁团队发布于 *Scientific Data*，是中国人群临床糖尿病患者连续血糖与生化检验配对数据集。

- **样本规模**：`ShanghaiT1DM` 包含 **12 名** 1 型糖尿病患者（共 16 段监测记录），`ShanghaiT2DM` 包含 **100 名** 2 型糖尿病患者（共 109 段监测记录），单次连续佩戴时长为 3-14 天。
- **传感器与临床标签**：采用雅培瞬感（Abbott FreeStyle Libre）采集皮下间质液葡萄糖，采样间隔为 15 min。数据集同步记录了指尖毛细血管血糖（CBG）、每日膳食与降糖用药/胰岛素注射记录，并在汇总表（`Summary.xlsx`）中提供空腹血糖（FPG）、糖化血红蛋白（HbA1c）、空腹胰岛素、C 肽、甘油三酯、总胆固醇等全套入院血检生化指标。
- **获取方式**：以 CC BY 4.0 协议托管于 Figshare，直接下载压缩包。

### Hall Glucotypes

Stanford University · 2018 · [主页/下载](https://doi.org/10.1371/journal.pbio.2005143.s010) · [论文](https://doi.org/10.1371/journal.pbio.2005143)

Hall 等人（Michael Snyder 实验室）于 2018 年在 *PLOS Biology* 发表的血糖波动表型（Glucotypes）研究队列，常用于评估 CGM 基础模型在非糖尿病人群中的代谢风险分层能力。

- **样本规模**：**57 名**无既往糖尿病确诊史的受试者（涵盖正常糖耐量、前驱糖尿病及筛查发现的 2 型糖尿病个体），佩戴约 2-4 周，共 105,426 行有效采样点。
- **传感器与任务**：采用 Dexcom G4 连续记录 5 min 间隔血糖，包含三种标准化早餐干预实验（玉米片加牛奶、花生酱面包、蛋白能量棒）。作者据此提出低、中、重度三种动态波动表型（Glucotype），是 CGM 无监督聚类与前驱糖尿病表征探针的常用基准。
- **获取方式**：作为论文补充材料（S1 Data TSV 文件）公开，直接下载。

### BIG IDEAs

Duke University · 2023 · [主页/下载](https://physionet.org/content/big-ideas-glycemic-wearable/1.1.2/) · [论文](https://doi.org/10.13026/aw6y-fc44)

BIG IDEAs Lab Glycemic Variability and Wearable Device Data 由杜克大学 Jessilyn Dunn 实验室发布，提供皮下间质液 CGM 与腕戴生理传感器的同步配对记录。

- **样本规模**：**16 名**受试者（正常血糖与前驱糖尿病个体，HbA1c 5.2%-6.4%），连续共同佩戴 8-10 天。
- **传感器与采样率**：同时佩戴 Dexcom G6 连续血糖仪（每 5 min 记录一次血糖）与 Empatica E4 手环（PPG 64 Hz、三轴 ACC 32 Hz、EDA 4 Hz、皮肤温度 4 Hz），并附带标准化早餐与日常饮食日志。
- **获取方式**：遵循 ODC-BY 1.0 协议托管于 PhysioNet，直接下载。

### CGMacros

Texas A&M University / Sansum Diabetes Research Institute · 2025 · [主页/下载](https://physionet.org/content/cgmacros/1.0.0/) · [论文](https://doi.org/10.1038/s41597-025-05851-7)

CGMacros 由德州农工大学 Gutierrez-Osuna 等人发布于 PhysioNet 与 *Scientific Data*，聚焦饮食宏量营养素摄入与餐后血糖及心率动态响应。

- **样本规模**：**45 名**受试者（15 名健康成人、16 名前驱糖尿病患者、14 名 2 型糖尿病患者），每人连续监测约 10 天。
- **传感器与多模态标注**：受试者同时佩戴两款 CGM 设备——Dexcom G6 Pro（5 min 间隔）与 Abbott FreeStyle Libre Pro（15 min 间隔），以及 Fitbit 智能手表（分钟级心率与步数）。数据集附带每餐高清餐食照片、精确宏量营养素克数（碳水化合物、蛋白质、脂肪、膳食纤维与总热量）、肠道菌群 16S rRNA 丰度及空腹血检生化指标（HbA1c、空腹胰岛素、HOMA-IR、血脂四项）。
- **获取方式**：以 CC BY 4.0 协议托管于 PhysioNet，直接下载。

### Colas DFA

Hospital Universitario de Móstoles / Universidad Rey Juan Carlos · 2019 · [主页/下载](https://doi.org/10.1371/journal.pone.0225817.s001) · [论文](https://doi.org/10.1371/journal.pone.0225817)

Colás 等人在 *PLOS ONE* 2019 发表的去趋势波动分析（DFA）研究队列，记录了代谢高危门诊人群在自由生活状态下的连续血糖轨迹。

- **样本规模**：**208 名**具有 2 型糖尿病高危因素的非糖尿病门诊受试者，单人连续监测 24-48 小时，累计有效时长约 9,500 小时，并包含中位 30 个月的 2 型糖尿病新发随访结局。
- **传感器与采样率**：采用 Medtronic iPro 连续血糖监测系统，采样间隔为 5 min。
- **获取方式**：作为 *PLOS ONE* 论文补充材料（S1 Data 压缩包）公开，直接下载。

### OhioT1DM

Ohio University · 2018 / 2020 · [主页/申请](http://smarthealth.cs.ohio.edu/OhioT1DM-dataset.html) · [论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC6299888/)

OhioT1DM 由俄亥俄大学 Marling 与 Bunescu 团队针对 Blood Glucose Level Prediction（BGLP）挑战赛构建，是评估 30-60 min 短程血糖自回归预测算法的国际公认基准。

- **样本规模**：共 **12 名**接受胰岛素泵治疗的 1 型糖尿病患者（2018 批次 6 人 + 2020 批次 6 人），每人连续采集 **8 周**自由生活数据，按受试者时间序列划分为训练集与测试集。
- **传感器与模态**：Medtronic Enlite CGM（5 min 血糖采样）、胰岛素泵基础率与大剂量（Bolus）输注记录、患者自报进餐时间与碳水化合物摄入克数估计，以及腕戴手环生理与体动信号（2018 批次为 Basis Peak 心率、皮温、EDA 与步数；2020 批次为 Empatica Embrace 三轴 ACC 与 EDA）。
- **获取方式**：通过俄亥俄大学官方页面在线填写并签署学术研究数据使用协议（DUA）后免费获取。

### Weinstock 2016

T1D Exchange / Jaeb Center for Health Research · 2016 · [主页/申请](https://public.jaeb.org/) · [论文](https://doi.org/10.2337/dc15-1426)

Weinstock 等人依托 T1D Exchange 临床网络在 *Diabetes Care*（2016）发表的老年 1 型糖尿病严重低血糖病例对照队列，是评估无症状夜间低血糖与长序列血糖预测的代表性临床数据集。

- **样本规模**：**201 名**高龄（年龄 ≥ 60 岁、病程 ≥ 20 年）1 型糖尿病患者（101 例近一年发生过严重低血糖的病例组与 100 例近三年无严重低血糖的对照组；GlucoBench 清洗后保留 200 人），每人佩戴盲态 CGM 约 **14 天**（人均约 3,000 个采样点，总计约 60 万个有效读数）。
- **传感器与临床标签**：采用盲态 Dexcom G4 Platinum 采集 5 min 间隔血糖，配对严重低血糖发作史、低血糖无感知评分（Clarke 问卷）、认知功能测试及 HbA1c 等临床指标。
- **获取方式**：托管于 Jaeb Center for Health Research 公共数据平台（可直接下载原始研究归档包，或通过 GlucoBench 预处理脚本自动拉取格式化 CSV）。

### GlucoBench

Texas A&M University · 2024 · [主页/代码](https://github.com/IrinaStatsLab/GlucoBench) · [论文](https://arxiv.org/abs/2410.05780)

GlucoBench 由德州农工大学 Irina Gaynanova 团队（Sergazinov 等人）发表于 ICLR 2024，提供面向连续血糖预测的标准化数据集清洗流水线与评测基准。

- **收录范围**：统一汇总并对齐了 **5 个**涵盖 1 型糖尿病、2 型糖尿病与非糖尿病人群的公开 CGM 数据集（Hall、Colás、Dubosson、Broll、Weinstock），清洗后包含超过 **500 名**受试者的 5 min 网格血糖序列。
- **评测设计**：定义了受试者内（In-domain）与跨受试者零样本外推（Out-of-domain）两类短程预测协议（30 min 与 60 min 预测跨度），统一报告 RMSE、MAPE 以及反映临床风险的 Clarke Error Grid 指标。
- **获取方式**：开源 GitHub 仓库提供自动化下载、插值对齐与基线训练代码，直接获取。

### Glucose-ML

Emory University / Dartmouth College (Augmented Health Lab) · 2025 · [主页/代码](https://github.com/Augmented-Health-Lab/Glucose-ML-Project) · [论文](https://arxiv.org/abs/2507.14077)

Glucose-ML 由 Temiloluwa Prioleau 教授领衔的 Augmented Health Lab 发布，是面向糖尿病人工智能研发的大规模纵向 CGM 数据集集合与自动化标准化工具库。

- **集合规模**：整合了 **20 余个**公开糖尿病与连续血糖监测研究队列（其中论文核心评测基准包含 10 个公开队列、2,500 余名受试者、超 3,800 万个血糖采样点；在线平台扩展集合累计覆盖 **4,300 余名**受试者、超 33 万监测人·天与约 **4,490 万个**血糖采样点），横跨 T1D、T2D、前驱糖尿病与健康对照人群。
- **获取方式**：代码与自动化拉取脚本在 GitHub 以 MIT 协议开源，支持一键下载其中开放许可的子集并转换为统一三列 CSV 格式（详见第 6.1 节）。

此外，针对需要更大规模多模态临床数据队列的研究，**AI-READI**（NIH Bridge2AI 项目，目标覆盖约 4,000 名 T2D、前驱糖尿病与健康受试者的 Dexcom G6 CGM、Garmin 手环、视网膜影像与全基因组数据，当前已发布超 1,000 人批次，申请入口：[aireadi.org](https://aireadi.org/)）与 **T1DEXI / T1DEXIP**（Jaeb Center 组织的 497 名成人及青少年 1 型糖尿病患者 4 周自由生活运动与 CGM 队列，申请入口：[public.jaeb.org](https://public.jaeb.org/)）均可通过在线签署数据使用协议申请获取。

---

## 3. 临床监护与心电数据集

临床重症监护室（ICU）、急诊科（ED）与手术室采集的多通道床旁波形具有高信噪比、严格跨通道硬件同步以及完整的电子健康档案（EHR）诊断标签，常作为可穿戴生理基础模型的预训练底座或跨模态对齐真值。

### MC-MED

Stanford University · 2025 · [主页/申请](https://physionet.org/content/mc-med/) · [论文](https://doi.org/10.1038/s41597-025-05419-5)

MC-MED（Multimodal Clinical Monitoring in the Emergency Department）由斯坦福大学医学院 David Kim 与 Pranav Rajpurkar 团队（Kansal 等人，*Scientific Data* 2025）构建并发布于 PhysioNet，是将急诊连续床旁生理波形与全量电子病历深度互联的多模态数据集。

- **样本规模**：采集自斯坦福医疗中心 2020-2022 年成人急诊科 **118,385 次就诊记录**（对应 **70,545 名**独立患者；其中包含连续生理波形的子集覆盖超过 5.3 万次就诊，可切分出超 1,000 万段同步波形对）。
- **传感器与采样率**：床旁监护仪连续采集 II 导联心电（ECG，**500 Hz**）、指尖光电脉搏波（PPG / Pleth，**125 Hz**）与阻抗呼吸波形（Resp，**125 Hz**），以及分钟级生命体征数值序列（心率、无创血压、血氧饱和度、体温）。
- **临床上下文标签**：同步关联患者人口学信息、急诊分诊危重度指数（ESI）、主诉文本、既往病史（ICD-10）、在院给药时间戳、实验室血检结果、放射影像报告及离院/住院转归。该数据集已被用作 PPG-LM、PPGFlowECG、AnyPPG 等多个脉搏与跨模态基础模型的预训练或评测数据源。
- **获取方式**：PhysioNet 凭证访问（Credentialed Access），需注册 PhysioNet 账号、完成 CITI 临床研究伦理培训认证并签署数据使用协议（DUA）。

### VitalDB

Seoul National University Hospital · 2022 · [主页/下载](https://vitaldb.net/) · [论文](https://doi.org/10.1038/s41597-022-01411-5)

VitalDB 由首尔大学医院麻醉与疼痛医学科 Lee 等人发布于 *Scientific Data*，是手术室围术期高保真多参数生命体征与波形数据库。

- **样本规模**：收录 **6,388 例**非心脏手术患者从入室到出室的全程监测记录，涵盖 196 项术中监护通道、73 项围术期临床特征与 34 项时序血检指标。
- **传感器与采样率**：包含有创桡动脉血压波形（ABP，**500 Hz**）、指尖光电脉搏波（PPG，**500 Hz**）、多导联心电（ECG，**500 Hz**）、二氧化碳波形（Capnography）、脑电双频指数（BIS）及麻醉呼吸机力学参数，是训练 PPG 到动脉血压波形重建与无袖带连续血压模型的常用高精度参考库。
- **获取方式**：遵循 CC BY-NC-SA 4.0 协议开放获取，可通过官方网站、PhysioNet 镜像或官方 Python 库（`pip install vitaldb`）直接按需拉取通道数据。

### BIDMC PPG

University of Oxford / BIDMC · 2016 · [主页/下载](https://physionet.org/content/bidmc/1.0.0/) · [论文](https://doi.org/10.1109/TBME.2016.2613124)

BIDMC PPG and Respiration Dataset 由牛津大学 Pimentel 等人从 MIMIC-II 重症监护数据库中抽取并精细标注，广泛用于脉搏波形态分析与呼吸率估计算法评测。

- **样本规模**：**53 名**成年重症监护住院患者，每人一段固定 **8 min** 的连续同步记录。
- **传感器与采样率**：包含指尖 PPG、II 导联 ECG 与胸腹阻抗呼吸波（Resp），采样率均为 **125 Hz**，同时提供 1 Hz 生命体征数值序列及两位专家独立标注的单次呼吸周期时间戳。
- **获取方式**：遵循 ODC-BY 1.0 协议托管于 PhysioNet，直接下载。

### MIMIC-IV-ECG

MIT / Beth Israel Deaconess Medical Center · 2023 · [主页/申请](https://physionet.org/content/mimic-iv-ecg/1.0/) · [论文](https://doi.org/10.13026/4nqg-sb35)

MIMIC-IV-ECG 由麻省理工学院计算生理学实验室（MIT LCP）与贝斯以色列女伯爵医学中心（BIDMC）联合发布，是当前规模最大的与住院电子病历关联的公开诊断级心电图库。

- **样本规模**：包含 **800,035 条**诊断级 12 导联心电记录，覆盖 **161,352 名**独立患者。
- **传感器与采样率**：标准 12 导联静息 ECG，采样率为 **500 Hz**，每条记录时长为 **10 s**。数据集提供心电图机自动测量指标（RR 间期、PR 间期、QRS 时限、QT/QTc 间期、电轴）与自由文本机器诊断报告，并可通过 `subject_id` 与 MIMIC-IV 临床数据库中的 ICD 诊断、用药、化验、超声心动图及院内/院外死亡结局直接关联（此外，同系列的 `MIMIC-III / MIMIC-IV Waveform Database` 还提供数万名 ICU 患者的多小时床旁连续 ECG 与 PPG 监护波形）。
- **获取方式**：PhysioNet 凭证访问（Credentialed Access），需完成 CITI 培训并签署 DUA（注：MIMIC-IV-ECG 本身自 2023 年底起已在 PhysioNet 开放波形直接下载，但关联 MIMIC-IV 临床诊断表仍需凭证权限）。

### PTB-XL

Physikalisch-Technische Bundesanstalt (PTB) / Fraunhofer HHI · 2020 · [主页/下载](https://physionet.org/content/ptb-xl/1.0.3/) · [论文](https://doi.org/10.1038/s41597-020-0495-6)

PTB-XL 由德国联邦物理技术研究院（PTB）与弗劳恩霍夫 HHI 研究所（Wagner 等人，*Scientific Data* 2020）发布，是心电分类与基础模型线性探针评测最常用的多标签基准库。

- **样本规模**：来自 **18,885 名**患者的 **21,837 条**临床 12 导联 ECG 记录（男女比例均衡，年龄覆盖 0-95 岁）。
- **传感器与标注体系**：每条记录时长 10 s，同时提供 **500 Hz** 与 **100 Hz** 两个下采样版本。标签由最多两位心脏病专家依据 SCP-ECG 标准标注，包含 **71 种**细粒度心电语句（44 种诊断类、19 种形态类、12 种节律类），并聚合为 5 大诊断超类：正常（NORM）、心肌梗死（MI）、ST/T 改变（STTC）、传导阻滞（CD）与心室肥厚（HYP）。官方提供按受试者严格解耦的 10 折划分（`strat_fold` 1-8 训练、9 验证、10 测试）。
- **获取方式**：遵循 CC BY 4.0 协议托管于 PhysioNet，直接下载。

### CODE-15%

Universidade Federal de Minas Gerais (TNMG) / Uppsala University · 2020 · [主页/下载](https://zenodo.org/records/4916206) · [论文](https://doi.org/10.1038/s41467-020-15432-4)

CODE-15% 是从巴西米纳斯吉拉斯州远程医疗网络（TNMG）逾 230 万份临床心电库（CODE 队列）中按患者分层抽样 15% 构成的公开大型心电数据集。

- **样本规模**：包含 **233,770 名**患者的 **345,779 条** 12 导联 ECG 记录。
- **传感器与临床标签**：采样率为 **400 Hz**，记录时长为 7-10 s（在 HDF5 文件中统一零填充至 4,096 个采样点）。数据集附带患者年龄、性别、6 类常见心律失常与传导异常标签（1 度房室传导阻滞、右束支传导阻滞、左束支传导阻滞、窦性心动过缓、房颤、窦性心动过速）以及长达数年的随访全因死亡率与生存时间标签，另配套独立的 827 例专家精标测试集（[Zenodo Record 3765717](https://zenodo.org/records/3765717)）。
- **获取方式**：遵循 CC BY 4.0 协议托管于 Zenodo，直接下载。

<a id="chapman-shaoxing"></a>

### Chapman-Shaoxing / Ningbo

Chapman University / 绍兴市人民医院 / 宁波市第一医院 · 2020 · [主页/下载](https://physionet.org/content/ecg-arrhythmia/1.0.0/) · [论文](https://doi.org/10.1038/s41597-020-0386-x)

Chapman-Shaoxing 与 Ningbo 数据集（PhysioNet 合并版本名为 *A large scale 12-lead electrocardiogram database for arrhythmia study*）由查普曼大学联合绍兴市人民医院、宁波市第一医院构建，提供了大规模东亚人群临床心律失常样本。

- **样本规模**：合并库共包含 **45,152 条**患者级 12 导联 ECG 记录（其中绍兴市人民医院子集 10,646 人，宁波市第一医院子集 34,905 人，去重合并后为 45,152 条）。
- **传感器与采样率**：采用 GE MUSE 心电系统采集，采样率 **500 Hz**，每条记录时长 **10 s**，由临床专家采用 SNOMED-CT 标准编码标注数十种心律失常与形态学异常，常用于检验心电基础模型的跨人种与跨中心外部泛化性能。
- **获取方式**：遵循 CC BY 4.0 协议托管于 PhysioNet，直接下载。

---

## 4. 睡眠多导与大型人群队列

整夜多导睡眠图（Polysomnography, PSG）与国家级前瞻性生物样本库记录了脑、心、肺跨器官耦合动态及数十年慢病结局，是构建睡眠基础模型（如 SleepFM）和系统级健康风险模型的底层数据支柱。

### SHHS

NIH / NHLBI (National Sleep Research Resource) · 1997 · [主页/申请](https://sleepdata.org/datasets/shhs) · [论文](https://doi.org/10.1093/sleep/20.12.1077)

SHHS（Sleep Heart Health Study）是由美国国立心肺血液研究所（NHLBI）资助的多中心前瞻性队列，旨在研究睡眠呼吸障碍与心血管疾病发病及死亡风险的关联。

- **样本规模**：基线队列（SHHS Visit 1，1995-1998 年）收录 **5,804 名** 40 岁及以上中老年受试者的家庭整夜 PSG 记录，其中 3,295 人在 2001-2003 年完成了第二次整夜随访（Visit 2），并附带长达 15 年以上的心血管事件与全因死亡随访结局。
- **传感器与采样率**：采用 Compumedics P-Series 采集未压缩整夜多导信号，包含双通道脑电（EEG C3/A2 与 C4/A1，125 Hz）、双侧眼电（EOG，50 Hz）、下颌肌电（EMG，125 Hz）、心电（ECG，125 Hz 或 250 Hz）、胸腹呼吸感应体积描记带（10 Hz）、口鼻热敏气流（10 Hz）、体位及指尖脉搏血氧饱和度（SpO2，1 Hz），配有按 R&K 标准人工逐 30 s 帧标注的睡眠分期与呼吸暂停/微觉醒事件。
- **获取方式**：托管于美国国家睡眠研究资源库（NSRR），注册账号并在线提交简短学术数据使用申请（DUA）后免费下载 EDF 原始波形。

### MESA

NIH / NHLBI (National Sleep Research Resource) · 2015 · [主页/申请](https://sleepdata.org/datasets/mesa) · [论文](https://doi.org/10.5665/sleep.4732)

MESA Sleep（Multi-Ethnic Study of Atherosclerosis Sleep Ancillary Study）是美国多族裔动脉粥样硬化队列的睡眠子研究，兼具临床整夜 PSG 与连续 7 天腕部体动加光照记录。

- **样本规模**：**2,237 名**中老年受试者（涵盖白人、非裔、西语裔与华裔四个族裔，年龄 54-93 岁），其中 2,056 人拥有有效整夜 PSG 记录，2,159 人拥有配对的 7 天腕部体动仪记录。
- **传感器与模态**：整夜家庭 PSG 包含三通道脑电（Fz-Cz、Cz-Oz、C4-M1，256 Hz）、双侧 EOG（256 Hz）、下颌 EMG（256 Hz）、单导联 ECG（256 Hz）、胸腹呼吸带、鼻压力气流与指尖脉搏波（PPG，256 Hz）；腕戴端采用 Philips Respironics Actiwatch Spectrum 连续 7 天记录 30 s 帧手腕活动计数（Actigraphy）与环境白光强度，是连接临床 PSG 与可穿戴日常睡眠/心率估计的桥梁队列。
- **获取方式**：托管于 NSRR（`sleepdata.org`），在线签署学术 DUA 后免费下载。

### Sleep-EDF

St. Antonius Hospital / Leiden University Medical Center · 2000 / 2013 · [主页/下载](https://physionet.org/content/sleep-edfx/1.0.0/) · [论文](https://doi.org/10.1109/10.867928)

Sleep-EDF Expanded 由荷兰 Kemp 等人发布于 PhysioNet，是睡眠分期算法快速原型验证与跨域泛化基准测试最常用的公开数据集。

- **样本规模**：包含 **197 份**整夜 PSG 记录（由 `Sleep Cassette` 子集 78 名健康受试者的 153 晚家庭记录与 `Sleep Telemetry` 子集 22 名轻度入睡困难受试者在替马西泮/安慰剂对照下的 44 晚医院记录组成）。
- **传感器与采样率**：包含双通道脑电（Fpz-Cz 与 Pz-Oz，**100 Hz**）、水平眼电（EOG，**100 Hz**）、颏下肌电（EMG）及口鼻呼吸与体温（1 Hz），配有按 R&K 标准人工标注的 30 s 睡眠分期（W、R、N1、N2、N3、N4）。
- **获取方式**：遵循 ODC-BY 1.0 协议托管于 PhysioNet，直接下载。

### UK Biobank

UK Biobank / University of Oxford · 2015 / 2017 · [主页/申请](https://www.ukbiobank.ac.uk/enable-your-research/apply-for-access) · [论文](https://doi.org/10.1371/journal.pone.0169649)

英国生物银行（UK Biobank, UKB）是全球规模最大的前瞻性人群多组学与数字表型生物样本库之一。

- **样本规模**：总队列涵盖约 **50 万名**招募时年龄在 40-69 岁的英国受试者；其中 **103,712 名**受试者在 2013-2015 年完成了为期 **7 天**的腕部高频加速度计连续佩戴，另有约 8.5 万名受试者采集了静息与运动心电图（ECG）。
- **传感器与关联模态**：腕部体动子队列采用 Axivity AX3 三轴加速度计（采样率 **100 Hz**，动态范围 ±8 g），记录无压缩原始重力加速度波形；同时全队列关联全基因组测序（WGS）、血浆蛋白质组学、代谢组学、心脏与脑部磁共振成像（MRI）、30 余项临床血液生化指标以及跨度逾 15 年的英国国家医疗服务体系（NHS）住院、急诊与死亡登记（ICD-10）。
- **获取方式**：采用严格的机构申请与云端封闭分析机制（Restricted Access）。需由高校或科研机构研究员通过 UKB Access Management System（AMS）提交研究提案，由所在单位法务签署材料转移协议（MTA）并缴纳数据访问管理费；自 2024 年起，原始大规模时序与组学数据须在 **UKB Research Analysis Platform**（RAP，基于 AWS / DNAnexus）云端工作台内分析，禁止下载至本地分发。

### All of Us

NIH All of Us Research Program · 2019 · [主页/申请](https://www.researchallofus.org/) · [论文](https://doi.org/10.1056/NEJMsr1809937)

All of Us 是美国国立卫生研究院（NIH）发起的国家级多族裔精准医学纵向队列，强调代表性不足人群（Underrepresented in Biomedical Research）的覆盖。

- **样本规模**：已招募超过 **80 万名**受试者（公开云端科研工作台 Curated Data Repository 已开放超 **40 万名**受试者的电子健康档案与超 24.5 万份全基因组测序，其中数万名受试者通过 Bring-Your-Own-Device 或项目配发计划贡献了长达数年的 **Fitbit** 可穿戴连续追踪数据）。
- **传感器与关联模态**：可穿戴子集包含 Fitbit 分钟级心率、步数、活动代谢当量（METs）与睡眠分期序列，深度关联 OMOP 标准化电子病历（诊断、处方、化验、手术操作）、体格检查、生物样本库与社会决定因素（SDOH）问卷。
- **获取方式**：受控云端工作台访问（Registered / Controlled Tier）。授权研究机构学者在完成伦理培训与身份认证后，须在官方 **All of Us Researcher Workbench**（基于 Google Cloud 的 Terra 平台）内通过 Jupyter / RStudio 访问与分析数据，不可导出个体级原始记录。

---

## 5. 数据集全景汇总表

| 领域 / 模态 | 数据集名称 | 机构 | 核心传感器与采样率 | 样本规模 | 获取方式 | 适用任务 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 腕部多模态 | [OpenMHC](#openmhc) | Stanford / Imperial / Charité | 19 维分钟级时序 + 169 维临床变量 | 11,894 人（>6,700 万小时） | 开放下载（XS 直接下载 / Full 注册） | 基础模型预训练、多变量插补与预测 |
| 腕部多模态 | [PhysioNet Wrist PPG](#physionet-wrist-ppg) | Univ. of Manchester | 腕部 PPG/ACC/Gyro 256 Hz + 胸 ECG 256 Hz | 8 人（走、跑、骑行） | 开放直接下载（PhysioNet） | 运动伪影消除、运动心率回归 |
| 腕部多模态 | [PPG-DaLiA](#ppg-dalia) | Univ. of Siegen / Bosch | 腕部 PPG 64 Hz、ACC 32 Hz、EDA 4 Hz + 胸 ECG 700 Hz | 15 人（每人约 2.5 小时，共 36 小时） | 开放直接下载（UCI） | 日常活动心率监测与去噪 |
| 腕部多模态 | [WESAD](#wesad) | Univ. of Siegen / Bosch | 腕部 E4（PPG/ACC/EDA/Temp）+ 胸 RespiBAN 700 Hz | 15 人（实验室压力与情绪诱发） | 开放直接下载（UCI） | 心理压力检测、三态情绪分类 |
| 腕部多模态 | [TROIKA](#troika) | Samsung Research / 国防科大 | 腕部双通道 PPG 125 Hz、ACC 125 Hz + 胸 ECG 125 Hz | 12 人（跑步机最高 15 km/h） | 开放直接下载（Zenodo） | 剧烈跑步心率跟踪基准 |
| 腕部多模态 | [CAS-BP](#cas-bp) | 中科院深圳先进院 / 华为 | 手表多波长 PPG 1000 Hz + ECG 1000 Hz + 听诊血压 | 1,272 人（4 周随访，12,624 次测量） | 示例开放 / 完整库填写申请（GitHub） | 无袖带血压回归、脉搏传导时间建模 |
| CGM 血糖 | [ShanghaiT1DM / T2DM](#shanghait1dm--shanghait2dm) | 上海六院 | Abbott Libre（15 min 间质液血糖）+ 血检生化 | 112 人（12 例 T1D + 100 例 T2D，3-14 天） | 开放直接下载（Figshare） | 血糖预测、缺失插补、生化指标探针 |
| CGM 血糖 | [Hall Glucotypes](#hall-glucotypes) | Stanford Univ. | Dexcom G4（5 min）+ 标准化早餐餐食记录 | 57 人（105,426 个采样点） | 开放直接下载（PLOS Biology 附件） | 血糖动态波动分型、前驱糖尿病聚类 |
| CGM 血糖 | [BIG IDEAs](#big-ideas) | Duke Univ. | Dexcom G6（5 min）+ Empatica E4 手环 | 16 人（连续配对 8-10 天） | 开放直接下载（PhysioNet） | 腕戴光电与皮下间质血糖跨模态建模 |
| CGM 血糖 | [CGMacros](#cgmacros) | Texas A&M / Sansum | 双 CGM（5/15 min）+ Fitbit + 餐食照片与宏量营养素 | 45 人（每人约 10 天，含血检与菌群） | 开放直接下载（PhysioNet） | 个性化营养餐后血糖响应建模 |
| CGM 血糖 | [Colas DFA](#colas-dfa) | Hosp. Univ. de Móstoles | Medtronic iPro（5 min） | 208 人（每人 24-48 小时，共约 9,500 小时） | 开放直接下载（PLOS ONE 附件） | 2 型糖尿病发病风险预测、CGM 预训练 |
| CGM 血糖 | [OhioT1DM](#ohiot1dm) | Ohio Univ. | Medtronic CGM（5 min）+ 胰岛素泵 + 碳水 + 手环 | 12 人（每人 8 周） | 签署学术 DUA 申请 | 30-60 min 短程血糖自回归预测基准 |
| CGM 血糖 | [Weinstock 2016](#weinstock-2016) | T1D Exchange / Jaeb Center | 盲态 Dexcom G4（5 min）+ 低血糖与认知评估 | 201 人（老年 T1D，每人约 14 天） | 开放下载（Jaeb / GlucoBench） | 老年高危人群严重低血糖风险分析 |
| CGM 基准 | [GlucoBench](#glucobench) | Texas A&M Univ. | 整合 5 个公开 CGM 队列（统一 5 min 网格） | >500 人 | 开放代码与脚本（GitHub） | 跨队列标准化血糖预测评测基准 |
| CGM 集合 | [Glucose-ML](#glucose-ml) | Emory / Dartmouth | 自动化整合 20 余个公开 CGM 数据集 | 4,300+ 人（约 4,490 万采样点） | 开放代码与脚本（GitHub） | 大规模 CGM 基础模型预训练与标准化 |
| 临床监护 | [MC-MED](#mc-med) | Stanford Univ. | 床旁 ECG 500 Hz、PPG 125 Hz、Resp 125 Hz + 急诊 EHR | 118,385 次急诊就诊（70,545 人） | PhysioNet 凭证访问（CITI + DUA） | 脉搏与心电基础模型预训练、急诊预测 |
| 临床监护 | [VitalDB](#vitaldb) | 首尔大学医院 | 有创动脉波 ABP 500 Hz、PPG 500 Hz、ECG 500 Hz | 6,388 例非心脏手术 | 开放下载 / Python API | PPG 转血压波形重建、围术期监护 |
| 临床监护 | [BIDMC PPG](#bidmc-ppg) | Univ. of Oxford / BIDMC | 指尖 PPG 125 Hz、ECG 125 Hz、阻抗呼吸波 125 Hz | 53 人（每人 8 min） | 开放直接下载（PhysioNet） | 呼吸率估计、脉搏波形态分析基准 |
| 临床心电 | [MIMIC-IV-ECG](#mimic-iv-ecg) | MIT / BIDMC | 12 导联诊断级 ECG 500 Hz（10 s）+ 住院 EHR | 800,035 条记录（161,352 人） | 波形开放下载 / 关联 EHR 需凭证 | 大规模心电自监督预训练与多病种筛查 |
| 临床心电 | [PTB-XL](#ptb-xl) | 德国 PTB / Fraunhofer HHI | 12 导联 ECG 500 Hz / 100 Hz（10 s，71 类 SCP 标签） | 21,837 条记录（18,885 人） | 开放直接下载（PhysioNet） | 多标签心电分类与基础模型微调基准 |
| 临床心电 | [CODE-15%](#code-15) | 巴西 TNMG / Uppsala Univ. | 12 导联 ECG 400 Hz（7-10 s）+ 死亡率随访 | 345,779 条记录（233,770 人） | 开放直接下载（Zenodo） | 心电预训练缩放定律与生存分析 |
| 临床心电 | [Chapman-Shaoxing / Ningbo](#chapman-shaoxing--ningbo) | Chapman Univ. / 绍兴与宁波医院 | 12 导联 ECG 500 Hz（10 s，SNOMED-CT 标签） | 45,152 条记录 | 开放直接下载（PhysioNet） | 东亚人群心律失常分类与跨中心验证 |
| 睡眠多导 | [SHHS](#shhs) | NIH / NHLBI（NSRR） | 整夜家庭 PSG（EEG/EOG/EMG/ECG 125-250 Hz、Resp、SpO2） | 5,804 人（含 >15 年心血管随访） | NSRR 注册签署 DUA | 睡眠分期、脑心肺耦合与慢病生存预测 |
| 睡眠多导 | [MESA](#mesa) | NIH / NHLBI（NSRR） | 整夜 PSG（256 Hz，含指尖 PPG）+ 7 天腕部体动仪 | 2,237 人（四族裔队列） | NSRR 注册签署 DUA | 临床 PSG 与腕戴睡眠/脉搏跨模态对齐 |
| 睡眠多导 | [Sleep-EDF](#sleep-edf) | St. Antonius / LUMC | 双通道 EEG 100 Hz、EOG 100 Hz、EMG、呼吸 | 197 份整夜 PSG 记录 | 开放直接下载（PhysioNet） | 睡眠分期算法快速验证基准 |
| 人群队列 | [UK Biobank](#uk-biobank) | UK Biobank / Univ. of Oxford | 腕部三轴 ACC 100 Hz（7 天）+ 全基因组 + MRI + EHR | 103,712 人体动（总队列 50 万人） | 机构课题申请 + RAP 云端封闭计算 | 运动生物标志物、全谱系疾病风险建模 |
| 人群队列 | [All of Us](#all-of-us) | 美国 NIH | Fitbit 多年连续序列 + 全基因组 + OMOP EHR | 40 万+ 人 EHR（数万人含 Fitbit） | 机构注册认证 + Workbench 云端计算 | 跨族裔精准医学与长程健康模型验证 |

---

## 6. 数据清洗与信号质量控制

### 6.1 连续血糖标准化流程

不同厂家的 CGM 设备在采样间隔（Dexcom 5 min vs. Abbott Libre 15 min）、血糖单位（$\text{mg/dL}$ vs. $\text{mmol/L}$）及导出表头命名上存在差异。在构建跨队列预训练语料时，通常采用 [Glucose-ML](#glucose-ml) 与 [GlucoBench](#glucobench) 的标准化流程：

1. **自动化拉取与列名对齐**：通过 `Glucose-ML-Project` 提供的 `auto-download-open-datasets.py` 与 `auto-harmonize-CGML-datasets.py` 脚本，将异构原始表统一重映射为标准三列结构（受试者 ID、ISO 时间戳、以 $\text{mg/dL}$ 为单位的血糖值，其中单位换算关系为 $1\text{ mmol/L} = 18.018\text{ mg/dL}$）：

```csv
subject_id,timestamp,glucose_value
shanghai_01,2021-05-10 08:00:00,112.5
shanghai_01,2021-05-10 08:05:00,115.0
```

2. **时间网格重采样与缺失处理**：将时间戳对齐至固定 5 min 网格；对短于 30-45 min 的短暂传感器丢包采用线性或样条插值，对超过阈值的长段脱落则切分为独立连续子序列，避免在自回归或掩码预训练中引入伪造趋势。

### 6.2 腕部生理信号质量评估与窗口对齐

将高频原始 PPG/ECG 波形送入编码器前，需过滤佩戴松动、传感器悬空与重度运动伪影片段，并统一多通道时间片（Patch）粒度：

1. **时域偏度与峭度指标**（Skewness & Kurtosis SQI）：正常动脉脉搏充盈波呈现收缩期快速上升、舒张期缓慢下降的正偏斜形态（偏度 $S_{\text{sqi}} > 0$）。当剧烈运动伪影导致波形削顶或随机震荡时，偏度与峭度会偏离正常生理范围，可据此在滑窗阶段剔除低质量片段。
2. **灌注指数与接触检测**（Perfusion Index, PI）：灌注指数定义为交流脉动分量与直流基线分量之比 $\text{PI} = (\text{AC} / \text{DC}) \times 100\%$。当 $\text{PI} < 0.1\%$ 时，通常表明光电探头与手腕皮肤脱离或局部外周灌注过低。
3. **多模态时间片对齐**：针对采样率不同的同步通道，按固定物理时间窗 $\tau$（例如 $\tau = 0.25\text{ s}$）切分 1D Patch——此时 64 Hz 的 PPG 每片包含 16 个采样点，32 Hz 的三轴 ACC 每片包含 8 个采样点，从而保证不同模态投影后的 Token 序列在时间轴上严格对齐。

---

> **欢迎交流与共建**  
> 本文由开源知识库项目 [Wearable-Physiological-Foundation-Models](https://github.com/tujy859/Wearable-Physiological-Foundation-Models) 整理策划。该项目持续追踪全球在智能手表、连续血糖监测（CGM）以及多模态生理健康基础模型领域的最新突破。如果您对可穿戴生理计算、时序健康基模与数字生物标志物有任何想法，欢迎在仓库提 Issue 交流！
