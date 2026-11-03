# 可穿戴与生理时序公开数据集

可穿戴生理基础模型的预训练与下游评测依赖多模态、多场景的生理时序数据集。本专题按信号采集形态与模态覆盖划分为五个部分：**腕部生理与日常活动数据集**、**连续血糖监测（CGM）数据集**、**临床连续 PPG 与多模态监护数据集**、**睡眠多导与大型人群队列**以及**心电信号数据集**，并在各条目及汇总表中标注 `含 PPG`、`PPG+ECG 配对` 与 `纯 ECG` 属性，便于区分支持连续背景采集的脉搏波数据与纯心电数据。

## 目录

- [1. 腕部生理与日常活动数据集](#1-腕部生理与日常活动数据集)
- [2. 连续血糖监测数据集与基准](#2-连续血糖监测数据集与基准)
- [3. 临床连续 PPG 与多模态监护数据集](#3-临床连续-ppg-与多模态监护数据集)
- [4. 睡眠多导与大型人群队列](#4-睡眠多导与大型人群队列)
- [5. 心电信号数据集](#5-心电信号数据集)
- [6. 数据集全景汇总表](#6-数据集全景汇总表)
- [7. 数据清洗与信号质量控制](#7-数据清洗与信号质量控制)

---

## 1. 腕部生理与日常活动数据集

腕戴生理信号分析的主要难点在于运动伪影（Motion Artifacts, MA）干扰以及自由生活（Free-living）长程标注数据的匮乏。以下数据集覆盖了大规模自由佩戴队列、日常活动心率监测与实验室压力诱发场景。

### OpenMHC

`含 PPG` · Stanford University / Imperial College London / Charité · 2026 · [主页/代码](https://github.com/AshleyLab/OpenMHC) · [论文](https://arxiv.org/abs/2607.16235)

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

`PPG+ECG 配对` · University of Manchester · 2017 · [主页/下载](https://physionet.org/content/wrist/1.0.0/) · [论文](https://doi.org/10.3390/data2010001)

PhysioNet Wrist PPG During Exercise 数据集由曼彻斯特大学 Jarchi 与 Casson 采集，用于评估运动状态下腕部光电脉搏波去噪与心率跟踪算法。

- **样本规模**：8 名健康受试者（3 男 5 女，年龄 22-32 岁），在跑步机与功率自行车上依次完成步行、慢跑/快跑、低阻力骑行与高阻力骑行 4 类运动，单次活动时长 4-10 min。
- **传感器与采样率**：左腕佩戴脉搏血氧与惯性传感装置，同步记录手腕 PPG、三轴低噪声加速度计、三轴宽量程加速度计与三轴陀螺仪信号，胸部通过 Actiwave 记录单导联参考心电（ECG）并提供人工复核的 R 波峰值标注；所有通道采样率均为 **256 Hz**。
- **获取方式**：遵循 ODC-BY 1.0 协议，在 PhysioNet 页面直接下载，无需注册或申请。

### PPG-DaLiA

`PPG+ECG 配对` · University of Siegen / Bosch · 2019 · [主页/下载](https://archive.ics.uci.edu/dataset/495/ppg+dalia) · [论文](https://doi.org/10.3390/s19143079)

PPG-DaLiA（PPG Dataset for Motion Compensation and Heart Rate Estimation in Daily Life Activities）由锡根大学与博世研究院联合发布，聚焦贴近真实日常生活的腕部心率估计。

- **样本规模**：15 名健康受试者（8 女 7 男，年龄 21-55 岁），每人连续采集约 2.5 小时，累计总时长约 **36 小时**。实验协议涵盖 8 类日常活动：静坐、上下楼梯、桌式足球、户外骑行、驾驶汽车、午餐、步行及办公桌工作。
- **传感器与采样率**：手腕佩戴 Empatica E4 手环（PPG 64 Hz、三轴 ACC 32 Hz、皮肤电活动 EDA 4 Hz、皮肤温度 4 Hz）；胸部佩戴 RespiBAN（ECG 700 Hz、呼吸与三轴 ACC 700 Hz），以胸部 ECG 提取的心率作为参考真值。
- **获取方式**：托管于 UCI Machine Learning Repository，直接下载。

### WESAD

`PPG+ECG 配对` · University of Siegen / Bosch · 2018 · [主页/下载](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection) · [论文](https://doi.org/10.1145/3242969.3242985)

WESAD（Wearable Stress and Affect Detection）由锡根大学与博世研究院在 ACM ICMI 2018 发布，是可穿戴情绪识别与心理压力检测的常用公开数据集。

- **样本规模**：**15 名**受试者在受控实验室环境下完成三态诱发实验，包括基线中性态（Baseline）、特里尔社会压力测试（TSST 公开演讲与心算，诱发 Stress）以及观看幽默视频片段（诱发 Amusement），中间穿插冥想恢复阶段并配对 PANAS、STAI、SAM 等心理问卷自评。
- **传感器与采样率**：手腕佩戴 Empatica E4（PPG 64 Hz、三轴 ACC 32 Hz、EDA 4 Hz、皮温 4 Hz）；胸部佩戴 RespiBAN（ECG、EDA、肌电 EMG、呼吸气流 Resp、体温 Temp 及三轴 ACC，采样率均为 700 Hz）。
- **获取方式**：托管于 UCI Machine Learning Repository，直接下载。

### TROIKA

`PPG+ECG 配对` · Samsung Research America / 国防科技大学 · 2015 · [主页/下载](https://zenodo.org/records/3902660) · [论文](https://doi.org/10.1109/TBME.2014.2359372)

TROIKA 数据集随 IEEE TBME 2015 同名论文及 2015 IEEE Signal Processing Cup 竞赛发布，用于评估剧烈跑步运动下的腕部 PPG 心率追踪算法。

- **样本规模**：**12 名**健康男性受试者（年龄 18-35 岁），在跑步机上按“静息 30 s → 6-8 km/h 慢跑 1 min → 12-15 km/h 快跑 1 min → 6-8 km/h 慢跑 1 min → 12-15 km/h 快跑 1 min → 静息 30 s”的变速协议各采集 5 min 波形。
- **传感器与采样率**：手腕双通道绿光 PPG（波长 515 nm）、手腕三轴加速度计（ACC）以及胸部单导联湿电极参考 ECG，所有信号采样率均为 125 Hz，按 8 s 滑动窗口（步长 2 s）提供参考心率真值。
- **获取方式**：可通过 Zenodo 归档库直接下载。

### CAS-BP

`PPG+ECG 配对` · 中国科学院深圳先进技术研究院 / 华为 · 2023 · [主页/下载](https://github.com/zdzdliu/CAS-BP) · [论文](https://doi.org/10.1109/JBHI.2023.3287313)

CAS-BP 是面向智能手表无袖带血压（Cuffless Blood Pressure）连续回归与多日随访校准评估的大规模验证数据集。

- **样本规模**：公开申请子集（Dataset-1）包含 **1,272 名**受试者（平均年龄 50.16 岁，66.82% 女性，25.63% 高血压患者），完成跨度 4 周的 4 次随访测量（第 0、7、14、21 天），累计 12,624 条有效测量记录。
- **传感器与金标准**：采用智能手表实验设备同步采集手腕多波长 PPG 与单导联 ECG 波形（原始采样率 1000 Hz，单次测量约 2 min，官方同时提供提炼后的 31 维生理与脉搏波传导时间 PTT 特征），配对双人柯氏音水银血压计听诊收缩压（SBP）与舒张压（DBP）读数，遵循 IEEE 1708 与 ANSI/AAMI/ISO 81060-2 测量规范。
- **评测协议**：
  - **无校准评估**（Calibration-free）：跨受试者直接从生理特征或波形预测血压绝对值。受个体血管刚度与接触阻抗差异影响，主流模型在该设定下的收缩压 MAE 通常在 7.5-11.0 mmHg 之间。
  - **基线校准评估**（Calibrated）：以第 0 天袖带基准值 $\text{BaseSBP}$ 为个性化锚点，测试阶段结合无校准预测值进行后处理校准（如 $\text{SBP}_{\text{cal}} = 0.3 \times \text{SBP}_{\text{free}} + 0.7 \times \text{BaseSBP}$），可将静息队列收缩压与舒张压 MAE 降低至 3.0-4.5 mmHg。
- **获取方式**：GitHub 仓库直接提供示例数据、解析脚本与 31 维特征说明；完整原始波形（Dataset-1）需填写仓库中的数据访问申请表向作者团队免费申请。基于该数据集的血压大模型微调方法参见 [CBPM-LLaMA](../models/health_reasoning_llms.md#cbpm-llama)。

### WildPPG

`PPG+ECG 配对` · ETH Zürich (SIPLab) · NeurIPS 2024 · [主页/代码](https://github.com/eth-siplab/WildPPG) · [论文](https://arxiv.org/abs/2405.08083)

WildPPG 由苏黎世联邦理工学院（ETH Zürich）SIPLab 团队发布于 NeurIPS 2024 Datasets and Benchmarks Track，专门针对非受控真实野外环境下的剧烈运动、高海拔与极端温差干扰构建多部位光电脉搏基准。

- **样本规模**：**16 名**受试者完成从苏黎世往返少女峰（Jungfraujoch，海拔最高 3,571 米）的全天户外行程，单人连续记录约 13.5 小时，累计总时长 **216 小时**，涵盖徒步登山、上下楼梯、步行、进餐及乘坐火车/缆车等多样化真实生活状态。
- **传感器与采样率**：受试者在**头部、胸骨、手腕与脚踝 4 个身体部位**同步佩戴定制穿戴节点，采集三波长 PPG（红光、绿光、红外，**128 Hz**）、三轴加速度计（ACC，**128 Hz**）、皮肤温度与气压海拔高度，并以胸部单导联心电（ECG，**128 Hz**）作为心率与心动周期参考真值。
- **获取方式**：以开放协议托管于 GitHub 与 Hugging Face（`eth-siplab/WildPPG`），直接下载。

### BUT-PPG

`PPG+ECG 配对` · Brno University of Technology · 2021 · [主页/下载](https://physionet.org/content/butppg/2.0.0/) · [论文](https://doi.org/10.3390/s21113832)

BUT-PPG（Brno University of Technology Smartphone PPG Database）由布尔诺理工大学 Nemcova 等人构建并发布于 PhysioNet，聚焦移动终端光电脉搏波的信号质量评估（SQI）与抗伪影心率估计。

- **样本规模**：2.0 版本包含 **48 名**健康受试者（年龄 19-76 岁）的 **3,888 段** 10 s 同步配对记录，采集条件覆盖静息及故意引入的走动、交谈、咳嗽、大笑、手指按压力度变化与环境光突变等伪影场景。
- **传感器与标注**：采用智能手机摄像头（Xiaomi Mi9 / Huawei P20 Pro，采样率 **30 Hz**）采集指尖或耳部 PPG 波形，胸部通过医疗级心电仪（Bittium Faros，**1000 Hz**）与三轴加速度计（**100 Hz**）同步采集参考信号。每段记录均附带多位专家共识标注的二元信号质量标签（Good / Poor）、人工核验的参考心率以及血压、血氧饱和度（SpO2）与血糖参考值。
- **获取方式**：遵循 ODC-BY 1.0 协议托管于 PhysioNet，直接下载。

### RealWorldPPG

`含 PPG` · Menoufia University · 2019 · [主页/下载](https://data.mendeley.com/datasets/yynb8t9x3d/3) · [论文](https://doi.org/10.17632/yynb8t9x3d.3)

RealWorldPPG（Real-World PPG Dataset）由埃及曼努菲亚大学 Siam 等人构建并发布于 Mendeley Data，是物联网便携传感条件下的短时指端光电脉搏波形库，被纳入 [PPG-Bench](../models/smartwatch_models.md#ppg-bench) 统一评测套件。

- **样本规模**：采集自 **35 名**受试者（年龄 10-75 岁，平均年龄 28.4 岁），每人包含 50-60 段独立记录，总计 **2,074 段** PPG 波形（官方标准划分为训练集 1,374 段、测试集 700 段）。
- **传感器与采样率**：采用便携式光电脉搏传感器以 **50 Hz** 采样率采集指端外周血容积脉搏波，单段记录时长为 **6 s**（300 个采样点），常用于评估轻量级脉搏特征提取、心率回归与生物身份识别（Biometric Identification）模型。
- **获取方式**：遵循 CC BY 4.0 协议托管于 Mendeley Data，直接下载。

### ECSMP

`PPG+ECG 配对` · 重庆邮电大学 / 西南大学 · 2021 · [主页/下载](https://data.mendeley.com/datasets/f9bf7b6264/2) · [论文](https://doi.org/10.3389/fnins.2021.751441)

ECSMP（A dataset on emotion, cognition, sleep, and multi-model physiological signals）由高竹等人发布于 *Frontiers in Neuroscience* 与 Mendeley Data，面向情绪、认知、睡眠与心理健康的跨模态生理分析。

- **样本规模**：**89 名**健康大学生受试者完成静息基线测试、6 类基本情绪视频片段诱发（中性、恐惧、悲伤、快乐、愤怒、厌恶）及认知测试，并配套抑郁自评量表（SDS）、匹兹堡睡眠质量指数（PSQI）、正负性情绪量表（PANAS）及森田神经质问卷标注。
- **传感器与采样率**：手腕佩戴 Empatica E4 手环（PPG **64 Hz**、三轴 ACC **32 Hz**、EDA **4 Hz**、皮温 **4 Hz**），同步记录前额脑电（EEG）与胸部单导联心电（ECG）。在 [PPG-Bench](../models/smartwatch_models.md#ppg-bench) 中常用于检验脉搏基础模型在情绪唤醒、心理压力及抑郁倾向筛查上的表征迁移能力。
- **获取方式**：遵循 CC BY 4.0 协议托管于 Mendeley Data，直接下载。

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

`含 PPG` · Duke University · 2023 · [主页/下载](https://physionet.org/content/big-ideas-glycemic-wearable/1.1.2/) · [论文](https://doi.org/10.13026/aw6y-fc44)

BIG IDEAs Lab Glycemic Variability and Wearable Device Data 由杜克大学 Jessilyn Dunn 实验室发布，提供皮下间质液 CGM 与腕戴生理传感器的同步配对记录。

- **样本规模**：**16 名**受试者（正常血糖与前驱糖尿病个体，HbA1c 5.2%-6.4%），连续共同佩戴 8-10 天。
- **传感器与采样率**：同时佩戴 Dexcom G6 连续血糖仪（每 5 min 记录一次血糖）与 Empatica E4 手环（PPG 64 Hz、三轴 ACC 32 Hz、EDA 4 Hz、皮肤温度 4 Hz），并附带标准化早餐与日常饮食日志。
- **获取方式**：遵循 ODC-BY 1.0 协议托管于 PhysioNet，直接下载。

### CGMacros

`含 PPG` · Texas A&M University / Sansum Diabetes Research Institute · 2025 · [主页/下载](https://physionet.org/content/cgmacros/1.0.0/) · [论文](https://doi.org/10.1038/s41597-025-05851-7)

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

`含 PPG` · Ohio University · 2018 / 2020 · [主页/申请](http://smarthealth.cs.ohio.edu/OhioT1DM-dataset.html) · [论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC6299888/)

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
- **获取方式**：代码与自动化拉取脚本在 GitHub 以 MIT 协议开源，支持一键下载其中开放许可的子集并转换为统一三列 CSV 格式（详见第 7.1 节）。

此外，针对需要更大规模多模态临床数据队列的研究，**AI-READI**（NIH Bridge2AI 项目，目标覆盖约 4,000 名 T2D、前驱糖尿病与健康受试者的 Dexcom G6 CGM、Garmin 手环、视网膜影像与全基因组数据，当前已发布超 1,000 人批次，申请入口：[aireadi.org](https://aireadi.org/)）与 **T1DEXI / T1DEXIP**（Jaeb Center 组织的 497 名成人及青少年 1 型糖尿病患者 4 周自由生活运动与 CGM 队列，申请入口：[public.jaeb.org](https://public.jaeb.org/)）均可通过在线签署数据使用协议申请获取。

---

## 3. 临床连续 PPG 与多模态监护数据集

重症监护室（ICU）、急诊科（ED）与手术室床旁监护仪采集的连续光电脉搏波（PPG）及同步参考心电信号（ECG）、有创动脉血压波（ABP）具有高信噪比、长时程连续采样以及完整的电子健康档案（EHR）标签，是训练脉搏波基础模型（如 PaPaGei、SigmaPPG、PPG-LM）以及 PPG-ECG 跨模态对齐模型（如 AnyPPG、PPGFlowECG）的主力数据底座。

### MC-MED

`PPG+ECG 配对` · Stanford University · 2025 · [主页/申请](https://physionet.org/content/mc-med/) · [论文](https://doi.org/10.1038/s41597-025-05419-5)

MC-MED（Multimodal Clinical Monitoring in the Emergency Department）由斯坦福大学医学院 David Kim 与 Pranav Rajpurkar 团队（Kansal 等人，*Scientific Data* 2025）构建并发布于 PhysioNet，是将急诊连续床旁生理波形与全量电子病历深度互联的多模态数据集。

- **样本规模**：采集自斯坦福医疗中心 2020-2022 年成人急诊科 **118,385 次就诊记录**（对应 **70,545 名**独立患者；其中包含连续生理波形的子集覆盖超过 5.3 万次就诊，可切分出超 1,000 万段同步波形对）。
- **传感器与采样率**：床旁监护仪连续采集 II 导联心电（ECG，**500 Hz**）、指尖光电脉搏波（PPG / Pleth，**125 Hz**）与阻抗呼吸波形（Resp，**125 Hz**），以及分钟级生命体征数值序列（心率、无创血压、血氧饱和度、体温）。
- **临床上下文标签**：同步关联患者人口学信息、急诊分诊危重度指数（ESI）、主诉文本、既往病史（ICD-10）、在院给药时间戳、实验室血检结果、放射影像报告及离院/住院转归。该数据集已被用作 PPG-LM、PPGFlowECG、AnyPPG 等多个脉搏与跨模态基础模型的预训练或评测数据源。
- **获取方式**：PhysioNet 凭证访问（Credentialed Access），需注册 PhysioNet 账号、完成 CITI 临床研究伦理培训认证并签署数据使用协议（DUA）。

### VitalDB

`PPG+ECG 配对` · Seoul National University Hospital · 2022 · [主页/下载](https://vitaldb.net/) · [论文](https://doi.org/10.1038/s41597-022-01411-5)

VitalDB 由首尔大学医院麻醉与疼痛医学科 Lee 等人发布于 *Scientific Data*，是手术室围术期高保真多参数生命体征与波形数据库。

- **样本规模**：收录 **6,388 例**非心脏手术患者从入室到出室的全程监测记录，涵盖 196 项术中监护通道、73 项围术期临床特征与 34 项时序血检指标。
- **传感器与采样率**：包含有创桡动脉血压波形（ABP，**500 Hz**）、指尖光电脉搏波（PPG，**500 Hz**）、多导联心电（ECG，**500 Hz**）、二氧化碳波形（Capnography）、脑电双频指数（BIS）及麻醉呼吸机力学参数，是训练 PPG 到动脉血压波形重建与无袖带连续血压模型的常用高精度参考库。
- **获取方式**：遵循 CC BY-NC-SA 4.0 协议开放获取，可通过官方网站、PhysioNet 镜像或官方 Python 库（`pip install vitaldb`）直接按需拉取通道数据。

### BIDMC PPG

`PPG+ECG 配对` · University of Oxford / BIDMC · 2016 · [主页/下载](https://physionet.org/content/bidmc/1.0.0/) · [论文](https://doi.org/10.1109/TBME.2016.2613124)

BIDMC PPG and Respiration Dataset 由牛津大学 Pimentel 等人从 MIMIC-II 重症监护数据库中抽取并精细标注，广泛用于脉搏波形态分析与呼吸率估计算法评测。

- **样本规模**：**53 名**成年重症监护住院患者，每人一段固定 **8 min** 的连续同步记录。
- **传感器与采样率**：包含指尖 PPG、II 导联 ECG 与胸腹阻抗呼吸波（Resp），采样率均为 **125 Hz**，同时提供 1 Hz 生命体征数值序列及两位专家独立标注的单次呼吸周期时间戳。
- **获取方式**：遵循 ODC-BY 1.0 协议托管于 PhysioNet，直接下载。

### PulseDB

`PPG+ECG 配对` · Texas A&M University (PulseLab) · 2023 · [主页/代码](https://github.com/pulselabteam/PulseDB) · [论文](https://doi.org/10.3389/fdgth.2022.1090854)

PulseDB 由德州农工大学 Wang 等人发表于 *Frontiers in Digital Health*，是从 MIMIC-III Waveform Database Matched Subset 与 VitalDB 两大重症/围术期库中经过严格质量控制与特征点对齐提炼而成的超大规模标准化无袖带血压基准库。

- **样本规模**：涵盖 **5,361 名**受试者的 **5,245,454 段**高质量 10 s 同步生理波形切片，并附带受试者年龄、性别、身高、体重与 BMI 等人口学信息（其 VitalDB 子集亦被 [PPG 生物年龄横评](../models/smartwatch_models.md#开源脉搏基础模型生物年龄横评) 用作血管生物年龄跨域评测基准）。
- **传感器与标注**：每段 10 s 切片均统一重采样至 **125 Hz**，包含同步配对的心电（ECG）、指尖光电脉搏波（PPG）与有创动脉血压波（ABP），并提供逐拍收缩压（SBP）、舒张压（DBP）、平均动脉压（MAP）真值以及标准化的有校准（Calibration-based）与无校准（Calibration-free）训练集/测试集划分脚本。
- **获取方式**：开源 GitHub 仓库提供全套构建脚本与预处理好的数据分片下载链接。

### PPG-BP

`含 PPG` · 桂林电子科技大学 / University of British Columbia · 2018 · [主页/下载](https://figshare.com/articles/dataset/PPG-BP_Database_zip/5459299) · [论文](https://doi.org/10.1038/sdata.2018.20)

PPG-BP Database 由梁永波、陈真诚与 Mohamed Elgendi 等人发布于 *Scientific Data*，是面向短时指端光电脉搏波无创血压估计与心血管慢病筛查的常用公开临床数据集。

- **样本规模**：采集自桂林市人民医院 **219 名**住院与门诊受试者（年龄 20-89 岁，平均 57 岁，涵盖正常血压、高血压前期、1/2 期高血压及糖尿病、脑梗死等慢病患者），每人连续采集 3 段高质量波形，共 **657 段** PPG 记录。
- **传感器与临床标签**：采用指夹式光电传感器（SMPLUS SEP9AF-2）以 **1000 Hz** 高采样率采集指尖 PPG 信号，单段时长 **2.1 s**（2,100 个采样点），同步配对欧姆龙上臂式电子血压计测量的收缩压（SBP）、舒张压（DBP）、心率以及住院临床疾病诊断标签。
- **获取方式**：遵循 CC BY 4.0 协议托管于 Figshare，直接下载。

### MIMIC-III Waveform

`PPG+ECG 配对` · MIT Laboratory for Computational Physiology / BIDMC · 2016 / 2020 · [主页/下载](https://physionet.org/content/mimic3wdb/1.0/) · [匹配子集](https://physionet.org/content/mimic3wdb-matched/1.0/) · [论文](https://doi.org/10.1038/sdata.2016.35)

MIMIC-III Waveform Database 由麻省理工学院计算生理学实验室与贝斯以色列女伯爵医学中心联合发布，是单导联心电与光电脉搏波基础模型（如 [PaPaGei](../models/smartwatch_models.md#papagei)、[SigmaPPG](../models/smartwatch_models.md#sigmappg)、[HeartGPT](../models/ecg_models.md#heartgpt)、[HeartBERT](../models/ecg_models.md#heartbert)）使用率最高的重症监护连续波形库。

- **样本规模**：全库收录约 **30,000 名**重症监护室（ICU）患者的 **67,830 条**多小时连续床旁监护记录；其中与 MIMIC-III 临床电子病历精确关联的匹配子集（Matched Subset）包含 **10,282 名**患者的 **22,317 条**波形记录。此外，后续扩充的 `MIMIC-IV Waveform Database` 进一步提供了无重叠新批次 ICU 监护序列。
- **传感器与采样率**：床旁监护仪以 **125 Hz** 同步采集至多 8 个通道的生理波形，核心包含多导联心电（主要为 II、V、AVR 导联）、指尖光电容积脉搏波（PPG / Pleth）、有创桡动脉血压波（ABP）与胸腹阻抗呼吸波（Resp），并配套 1 Hz 或每分钟生命体征趋势数值序列。
- **获取方式**：遵循 ODC-BY 1.0 协议托管于 PhysioNet，波形数据直接开放下载（若需关联 MIMIC-III 住院诊断与用药表则需 PhysioNet 凭证权限）。

### WAVES

`PPG+ECG 配对` · Stanford University (Lucile Packard Children's Hospital) · 2023 · [主页/申请](https://stanfordmedicine.box.com/v/waves-dataset) · [论文](https://doi.org/10.1038/s41597-023-02374-3)

WAVES（Waveform Dataset for Pediatric Critical Care）由斯坦福大学卢西尔·帕卡德儿童医院发布于 *Scientific Data*，是面向儿科住院与重症监护场景的大规模连续心电与脉搏波形数据库。

- **样本规模**：涵盖 **50,400 名**儿科住院患者（年龄从新生儿至 18 岁青少年）在长达 14 年间采集的约 **55 万条连续心电（ECG）记录**与 **51 万条光电脉搏波（PPG）记录**，单次记录平均连续时长约 **7 小时**。
- **传感器与采样率**：床旁监护仪连续采集多导联心电（ECG，**125 Hz**，部分达 250/500 Hz）、指尖/足端光电脉搏波（PPG，**125 Hz**）、呼吸波与有创血压波形，是研究儿童生长发育期心率/脉搏形态演变及评估成人基础模型跨年龄泛化的重要临床资源。
- **获取方式**：受限申请访问（Restricted Access），需向斯坦福医学院团队提交学术研究申请并签署数据使用协议（DUA）。

### MOVER

`PPG+ECG 配对` · University of California, Irvine Medical Center · 2023 · [主页/申请](https://mover.ics.uci.edu/) · [论文](https://doi.org/10.1093/jamiaopen/ooad084)

MOVER（Medical Informatics Operating Room Vitals and Events Repository）由加州大学欧文分校（UCI）医学中心发布，是与全套围术期电子健康档案关联的大型成人手术室生理波形库。

- **样本规模**：收录 **58,799 名**接受外科手术患者的 **83,468 台次手术**全程监护记录。
- **传感器与临床标签**：包含手术期间连续采集的心电波形（ECG，**180 Hz 或 300 Hz**）、指尖光电脉搏波（PPG，**100 Hz**）与有创动脉血压波（ABP），并深度关联术前合并症、术中麻醉给药剂量、输血补液事件、呼吸机参数及术后并发症/ICU 转归。
- **获取方式**：在官方网站在线签署学术研究数据使用协议（DUA）后免费下载。

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

`PPG+ECG 配对` · NIH / NHLBI (National Sleep Research Resource) · 2015 · [主页/申请](https://sleepdata.org/datasets/mesa) · [论文](https://doi.org/10.5665/sleep.4732)

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

### DREAMT

`PPG+ECG 配对` · Duke University (BIG IDEAs Lab) · 2024 · [主页/下载](https://physionet.org/content/dreamt/2.1.0/) · [论文](https://arxiv.org/abs/2405.13086)

DREAMT（Dataset for Real-time sleep stage EstimAtion using Multisensor wearable Technology）由杜克大学 Jessilyn Dunn 实验室（Wang 等人）发布于 PhysioNet，提供了睡眠障碍门诊患者整夜腕戴多传感器与临床金标准多导睡眠图（PSG）的严格同步记录。

- **样本规模**：采集自杜克大学医疗系统睡眠实验室的 **100 名**受试者（涵盖阻塞性睡眠呼吸暂停 OSA、打鼾、呼吸急促等真实睡眠障碍人群），记录整夜约 8 小时同步睡眠过程。
- **传感器与采样率**：受试者左手腕佩戴 Empatica E4 手环（绿光/红光 PPG **64 Hz**、三轴 ACC **32 Hz**、皮肤电活动 EDA **4 Hz**、皮肤温度 **4 Hz**），同时接受临床标准整夜 PSG 监测（EEG、EOG、颏下与腿部 EMG、ECG、口鼻气流、胸腹呼吸带及血氧 SpO2，提供 100 Hz 对齐版本与 64 Hz 穿戴切片版本）。每 30 s 睡眠帧均由专业睡眠技师按 AASM 标准标注睡眠分期（W、N1、N2、N3、R）及呼吸暂停低通气指数（AHI）、微觉醒事件与临床肥胖/慢病元数据，被 [HiMAE](../models/smartwatch_models.md#himae) 与 [PPG-Bench](../models/smartwatch_models.md#ppg-bench) 广泛用作腕部睡眠分期与呼吸暂停评测基准。
- **获取方式**：托管于 PhysioNet（注册账号并在线签署标准数据使用协议 DUA 后即可下载）。

### UK Biobank

UK Biobank / University of Oxford · 2015 / 2017 · [主页/申请](https://www.ukbiobank.ac.uk/enable-your-research/apply-for-access) · [论文](https://doi.org/10.1371/journal.pone.0169649)

英国生物银行（UK Biobank, UKB）是全球规模最大的前瞻性人群多组学与数字表型生物样本库之一。

- **样本规模**：总队列涵盖约 **50 万名**招募时年龄在 40-69 岁的英国受试者；其中 **103,712 名**受试者在 2013-2015 年完成了为期 **7 天**的腕部高频加速度计连续佩戴，另有约 8.5 万名受试者采集了静息与运动心电信号（ECG）。
- **传感器与关联模态**：腕部体动子队列采用 Axivity AX3 三轴加速度计（采样率 **100 Hz**，动态范围 ±8 g），记录无压缩原始重力加速度波形；同时全队列关联全基因组测序（WGS）、血浆蛋白质组学、代谢组学、心脏与脑部磁共振成像（MRI）、30 余项临床血液生化指标以及跨度逾 15 年的英国国家医疗服务体系（NHS）住院、急诊与死亡登记（ICD-10）。
- **获取方式**：采用严格的机构申请与云端封闭分析机制（Restricted Access）。需由高校或科研机构研究员通过 UKB Access Management System（AMS）提交研究提案，由所在单位法务签署材料转移协议（MTA）并缴纳数据访问管理费；自 2024 年起，原始大规模时序与组学数据须在 **UKB Research Analysis Platform**（RAP，基于 AWS / DNAnexus）云端工作台内分析，禁止下载至本地分发。

### All of Us

`含 PPG` · NIH All of Us Research Program · 2019 · [主页/申请](https://www.researchallofus.org/) · [论文](https://doi.org/10.1056/NEJMsr1809937)

All of Us 是美国国立卫生研究院（NIH）发起的国家级多族裔精准医学纵向队列，强调代表性不足人群（Underrepresented in Biomedical Research）的覆盖。

- **样本规模**：已招募超过 **80 万名**受试者（公开云端科研工作台 Curated Data Repository 已开放超 **40 万名**受试者的电子健康档案与超 24.5 万份全基因组测序，其中数万名受试者通过 Bring-Your-Own-Device 或项目配发计划贡献了长达数年的 **Fitbit** 可穿戴连续追踪数据）。
- **传感器与关联模态**：可穿戴子集包含 Fitbit 分钟级心率、步数、活动代谢当量（METs）与睡眠分期序列，深度关联 OMOP 标准化电子病历（诊断、处方、化验、手术操作）、体格检查、生物样本库与社会决定因素（SDOH）问卷。
- **获取方式**：受控云端工作台访问（Registered / Controlled Tier）。授权研究机构学者在完成伦理培训与身份认证后，须在官方 **All of Us Researcher Workbench**（基于 Google Cloud 的 Terra 平台）内通过 Jupyter / RStudio 访问与分析数据，不可导出个体级原始记录。

---

## 5. 心电信号数据集

以下数据集仅包含医院 10 秒 12 导联静息心电信号或单导联动态胸贴心电信号（不含 PPG 通道），主要用于训练心电基础模型、评测心律失常与结构性心脏病分类，或在跨模态训练中作为心电教师网络的预训练语料。

### MIMIC-IV-ECG

`纯 ECG` · MIT / Beth Israel Deaconess Medical Center · 2023 · [主页/申请](https://physionet.org/content/mimic-iv-ecg/1.0/) · [论文](https://doi.org/10.13026/4nqg-sb35)

MIMIC-IV-ECG 由麻省理工学院计算生理学实验室（MIT LCP）与贝斯以色列女伯爵医学中心（BIDMC）联合发布，是当前规模最大的与住院电子病历关联的公开诊断级心电信号库。

- **样本规模**：包含 **800,035 条**诊断级 12 导联心电记录，覆盖 **161,352 名**独立患者。
- **传感器与采样率**：标准 12 导联静息 ECG，采样率为 **500 Hz**，每条记录时长为 **10 s**。数据集提供心电图机自动测量指标（RR 间期、PR 间期、QRS 时限、QT/QTc 间期、电轴）与自由文本机器诊断报告，并可通过 `subject_id` 与 MIMIC-IV 临床数据库中的 ICD 诊断、用药、化验、超声心动图及院内/院外死亡结局直接关联（此外，同系列的 `MIMIC-III / MIMIC-IV Waveform Database` 还提供数万名 ICU 患者的多小时床旁连续 ECG 与 PPG 监护波形）。
- **获取方式**：PhysioNet 凭证访问（Credentialed Access），需完成 CITI 培训并签署 DUA（注：MIMIC-IV-ECG 本身自 2023 年底起已在 PhysioNet 开放波形直接下载，但关联 MIMIC-IV 临床诊断表仍需凭证权限）。

### PTB-XL

`纯 ECG` · Physikalisch-Technische Bundesanstalt (PTB) / Fraunhofer HHI · 2020 · [主页/下载](https://physionet.org/content/ptb-xl/1.0.3/) · [论文](https://doi.org/10.1038/s41597-020-0495-6)

PTB-XL 由德国联邦物理技术研究院（PTB）与弗劳恩霍夫 HHI 研究所（Wagner 等人，*Scientific Data* 2020）发布，是心电分类与基础模型线性探针评测最常用的多标签基准库。

- **样本规模**：来自 **18,885 名**患者的 **21,837 条**临床 12 导联 ECG 记录（男女比例均衡，年龄覆盖 0-95 岁）。
- **传感器与标注体系**：每条记录时长 10 s，同时提供 **500 Hz** 与 **100 Hz** 两个下采样版本。标签由最多两位心脏病专家依据 SCP-ECG 标准标注，包含 **71 种**细粒度心电语句（44 种诊断类、19 种形态类、12 种节律类），并聚合为 5 大诊断超类：正常（NORM）、心肌梗死（MI）、ST/T 改变（STTC）、传导阻滞（CD）与心室肥厚（HYP）。官方提供按受试者严格解耦的 10 折划分（`strat_fold` 1-8 训练、9 验证、10 测试）。
- **获取方式**：遵循 CC BY 4.0 协议托管于 PhysioNet，直接下载。

### CODE-15%

`纯 ECG` · Universidade Federal de Minas Gerais (TNMG) / Uppsala University · 2020 · [主页/下载](https://zenodo.org/records/4916206) · [论文](https://doi.org/10.1038/s41467-020-15432-4)

CODE-15% 是从巴西米纳斯吉拉斯州远程医疗网络（TNMG）逾 230 万份临床心电库（CODE 队列）中按患者分层抽样 15% 构成的公开大型心电数据集。

- **样本规模**：包含 **233,770 名**患者的 **345,779 条** 12 导联 ECG 记录。
- **传感器与临床标签**：采样率为 **400 Hz**，记录时长为 7-10 s（在 HDF5 文件中统一零填充至 4,096 个采样点）。数据集附带患者年龄、性别、6 类常见心律失常与传导异常标签（1 度房室传导阻滞、右束支传导阻滞、左束支传导阻滞、窦性心动过缓、房颤、窦性心动过速）以及长达数年的随访全因死亡率与生存时间标签，另配套独立的 827 例专家精标测试集（[Zenodo Record 3765717](https://zenodo.org/records/3765717)）。
- **获取方式**：遵循 CC BY 4.0 协议托管于 Zenodo，直接下载。

<a id="chapman-shaoxing"></a>

### Chapman-Shaoxing / Ningbo

`纯 ECG` · Chapman University / 绍兴市人民医院 / 宁波市第一医院 · 2020 · [主页/下载](https://physionet.org/content/ecg-arrhythmia/1.0.0/) · [论文](https://doi.org/10.1038/s41597-020-0386-x)

Chapman-Shaoxing 与 Ningbo 数据集（PhysioNet 合并版本名为 *A large scale 12-lead electrocardiogram database for arrhythmia study*）由查普曼大学联合绍兴市人民医院、宁波市第一医院构建，提供了大规模东亚人群临床心律失常样本。

- **样本规模**：合并库共包含 **45,152 条**患者级 12 导联 ECG 记录（其中绍兴市人民医院子集 10,646 人，宁波市第一医院子集 34,905 人，去重合并后为 45,152 条）。
- **传感器与采样率**：采用 GE MUSE 心电系统采集，采样率 **500 Hz**，每条记录时长 **10 s**，由临床专家采用 SNOMED-CT 标准编码标注数十种心律失常与形态学异常，常用于检验心电基础模型的跨人种与跨中心外部泛化性能。
- **获取方式**：遵循 CC BY 4.0 协议托管于 PhysioNet，直接下载。

### HEEDB

`纯 ECG` · Harvard Medical School (MGH / BWH) / Emory University · 2024 · [主页/申请](https://bdsp.io/content/heedb/) · [论文](https://doi.org/10.60508/rv6h-7d10)

HEEDB（Harvard-Emory ECG Database）由麻省总医院（MGH）、布莱根妇女医院（BWH）与埃默里大学联合构建并托管于 Brain Data Science Platform（BDSP），是目前全球向学术界开放申请的规模最大的千万级临床 12 导联心电数据库，直接支撑了 [ECGFounder](../models/ecg_models.md#ecgfounder) 与 [ECG-CPC](../models/ecg_models.md#ecg-cpc) 等千万级心电基础模型的预训练。

- **样本规模**：汇总了来自美国多中心医疗系统的 **11,739,868 条（约 1,170 万条）** 诊断级 12 导联心电记录，覆盖 **2,120,264 名（约 210 万名）** 独立患者，时间跨度超过 30 年。
- **传感器与标注体系**：标准 10 秒 12 导联静息 ECG，采样率为 **250 Hz 或 500 Hz**。数据集配套 GE Marquette 12SL 自动分析程序生成的结构化诊断陈述码、全套心电传导间期测量值（心率、PR、QRS、QT/QTc、P/QRS/T 电轴）以及关联的住院 ICD-9/ICD-10 诊断与死亡率随访信息。
- **获取方式**：凭证申请访问（Credentialed Access），研究人员可通过 BDSP 平台提交线上申请并签署数据使用协议后获取云端或下载访问权限。

### EchoNext

`纯 ECG` · Columbia University Irving Medical Center · 2025 · [主页/申请](https://physionet.org/content/echonext/1.0.0/) · [代码](https://github.com/PierreElias/EchoNext) · [论文](https://doi.org/10.13026/3ykd-bf14)

EchoNext 由哥伦比亚大学欧文医学中心 Pierre Elias 团队构建并发布于 PhysioNet，是专门面向由常规 12 导联心电信号筛查隐匿性结构性心脏病（Structural Heart Disease, SHD）的大规模心电-超声心动图配对基准库。

- **样本规模**：包含 **36,286 名**患者的 **100,000 条** 12 导联心电记录，每条心电信号均严格配对短期内完成的经胸超声心动图（TTE）定量结果。
- **传感器与金标准标签**：输入为 **250 Hz**、10 秒的标准 12 导联 ECG；标签端提供由心脏超声确诊的 **11 类结构性心脏病二元与定量指标**，包括左室射血分数降低（LVEF ≤ 40% / ≤ 50%）、右心室收缩功能障碍、中重度主动脉瓣狭窄（AS）、主动脉瓣反流（AR）、二尖瓣反流（MR）、三尖瓣反流（TR）、肺动脉高压、左心室室壁增厚（肥厚）及心包积液，在 [ECG-CPC](../models/ecg_models.md#ecg-cpc) 评测基准中被用作核心结构性心脏病迁移任务。
- **获取方式**：PhysioNet 凭证访问（Credentialed Access），完成 CITI 培训并签署 DUA 后下载。

### SPH

`纯 ECG` · 山东省立医院 · 2022 · [主页/下载](https://springernature.figshare.com/collections/A_large-scale_multi-label_12-lead_electrocardiogram_database_with_standardized_diagnostic_statements/5779802) · [论文](https://doi.org/10.1038/s41597-022-01403-5)

SPH（Shandong Provincial Hospital ECG Database）由山东省立医院刘澄玉、刘常春等人发布于 *Scientific Data*，是遵循国际标准化诊断语句体系的大规模中国成人多标签 12 导联心电信号库。

- **样本规模**：收录 2019-2020 年间采集自 **24,666 名**患者的 **25,770 条**临床 12 导联心电记录。
- **传感器与标注体系**：采样率为 **500 Hz**，记录时长为 **10-60 s**。全部记录由资深心电图医生依据美国心脏协会（AHA / ACC / HRS）标准化指南人工复核标注，涵盖 **44 类主要诊断语句**与 **15 类形态修饰词**，是评估心电基础模型在中国人群及 AHA 层级标签体系下泛化性能的重要开放基准。
- **获取方式**：遵循 CC BY 4.0 协议托管于 Springer Nature Figshare，直接开放下载。

### ZZU pECG

`纯 ECG` · 郑州大学第三附属医院 · 2025 · [主页/下载](https://doi.org/10.6084/m9.figshare.27078763) · [论文](https://doi.org/10.1038/s41597-025-05196-z)

ZZU pECG（Zhengzhou University Pediatric ECG Database）由郑州大学第三附属医院（河南省妇幼保健院）发布于 *Scientific Data*（2025），填补了开源心电生态中缺乏大规模儿科专用 12 导联数据集的空白。

- **样本规模**：包含 **11,643 名**儿童患者（年龄跨度 **0-14 岁**，涵盖新生儿、婴儿、幼儿与学龄儿童）的 **14,190 条**心电记录。
- **传感器与临床标签**：包含标准 12 导联或儿科 9 导联心电波形，采样率为 **500 Hz**，记录时长 **5-120 s**。数据集提供心电图医生审核的 **64 类 AHA 与中国儿科心电标准诊断语句**以及住院 **ICD-10 疾病编码**（涵盖室间隔缺损、房间隔缺损等先天性心脏病、川崎病、小儿心肌炎及各类小儿心律失常），在 [ECG-CPC](../models/ecg_models.md#ecg-cpc) 横评中被用于检验成人心电基础模型向儿科群体的跨年龄分布外（OOD）泛化表现。
- **获取方式**：遵循 CC BY 4.0 协议托管于 Figshare，直接开放下载。

### CPSC 2018 / CinC 2020

`纯 ECG` · 东南大学 / 中国生理信号挑战赛 / PhysioNet · 2018 / 2020 · [主页/下载](https://physionet.org/content/challenge-2020/1.0.2/) · [论文](https://doi.org/10.1088/1361-6579/ab2ef0)

CPSC 2018（China Physiological Signal Challenge 2018）由东南大学刘澄玉团队联合中国 11 家三甲医院构建，并作为核心子集并入 **PhysioNet / Computing in Cardiology Challenge 2020 & 2021** 跨国心电大库。

- **样本规模**：`CPSC 2018` 包含 **10,330 条**记录（公开训练集 6,877 条 + 测试集 3,453 条，后拆分为 `CPSC` 与 `CPSC-Extra`）；合并后的 `PhysioNet Challenge 2020/2021` 进一步整合了美国 **Georgia 12-Lead ECG Database（10,344 条）**、PTB-XL、Chapman-Shaoxing 与 Ningbo 等共 7 个数据源，总公开规模达 **88,253 条** 12 导联心电信号。
- **传感器与标注**：采样率主要为 **500 Hz**，时长 **6-60 s**。CPSC 2018 原始任务涵盖正常窦律与 8 类常见异常（房颤、一度房室阻滞、左/右束支阻滞、房早、室早、ST 段压低、ST 段抬高），PhysioNet 2020/2021 则统一映射为 **111 类 SNOMED-CT 临床编码（核心评测 26-30 类）**。
- **获取方式**：遵循 CC BY 4.0 协议托管于 PhysioNet，直接下载。

### SNUH LYDUS

`纯 ECG` · Seoul National University Hospital (SNUH) · 2025 · [主页/论文](https://doi.org/10.1038/s41597-025-04520-z)

SNUH LYDUS 12-Lead ECG Dataset 由首尔大学医院发布于 *Scientific Data*（2025），是面向韩国及东亚人群心电表征学习与间期回归的大型临床数据集。

- **样本规模**：完整受限版包含 128,152 名患者的 **167,000 条**记录；其中向全球研究者直接开放获取的公开子集包含 **38,402 名**患者的 **50,000 条** 12 导联心电记录。
- **传感器与标注**：每条记录时长 **10 s**，提供 **500 Hz** 与 **100 Hz** 采样率版本，附带心脏病学专家复核的结构化诊断分类标签及心率、PR、QRS、QT/QTc 等心电图机测量间期。
- **获取方式**：50,000 条公开子集遵循开放协议直接下载；167,000 条完整队列可通过首尔大学医院数据平台申请。

### Icentia11k

`纯 ECG` · University of Montreal / Mila / Icentia Inc. · 2019 / 2022 · [主页/下载](https://physionet.org/content/icentia11k-continuous-ecg/1.0/) · [论文](https://arxiv.org/abs/1910.09570)

Icentia11k Continuous ECG Database 由蒙特利尔大学、Mila 研究所与医疗器械公司 Icentia 联合发布于 PhysioNet，是目前规模最大的公开单导联长时程动态心电（Ambulatory Patch ECG）数据集，被 [HeartGPT](../models/ecg_models.md#heartgpt) 与 [HeartBERT](../models/ecg_models.md#heartbert) 等单导联基础模型选为核心预训练语料。

- **样本规模**：采集自加拿大 **11,000 名**门诊与居家受试者，提取出 **54,000 段**长约 70 分钟的连续波形切片（总时长约 **63,000 小时**），累计包含超过 **2.77 亿次（277M）** 经专业心脏技师逐拍核验的心跳标注。
- **传感器与标注体系**：受试者胸部佩戴 **CardioSTAT 单导联无导线动态心电贴片**（连续佩戴 3-14 天，采样率 **250 Hz**，16-bit 分辨率）。每段记录均附带逐拍心搏类型标签（正常搏动、室性早搏 PVC、室上性早搏 PAC）以及连续节律状态分段标签（正常窦律、心房颤动 AFib、心房扑动 AFL）。
- **获取方式**：遵循 CC BY 4.0 协议托管于 PhysioNet，直接下载。

---

## 6. 数据集全景汇总表

<!-- BEGIN:DATASETS_DOC_TABLE -->

| 数⁠据⁠集⁠与⁠领⁠域 | 机⁠构 / 来⁠源 | 传⁠感⁠器⁠与⁠采⁠样⁠率 | 样⁠本⁠规⁠模⁠与⁠适⁠用⁠任⁠务 | 获⁠取⁠方⁠式 |
| :--- | :--- | :--- | :--- | :---: |
| [**OpenMHC**](#openmhc)<br>手⁠腕⁠多⁠模⁠态 · `含 PPG` | Stanford / Imperial / Charité | 19 维⁠分⁠钟⁠级⁠时序 + 169 维⁠自⁠述⁠健⁠康变⁠量 | 11,894 人 (>6,700 万小时)<br>任务：基础模型预训练、多变量缺失插补、未来预测 | [开⁠放⁠获⁠取](https://github.com/AshleyLab/OpenMHC) |
| [**PhysioNet Wrist PPG**](#physionet-wrist-ppg)<br>手⁠腕⁠多⁠模⁠态 · `PPG+ECG 配对` | Univ. of Manchester | 腕⁠部 PPG/ACC/Gyro 256 Hz + 胸 ECG 256 Hz | 8 人 (走、跑、骑行 4 类运动)<br>任务：运动伪影消除、运动心率回归 | [开⁠放⁠获⁠取](https://physionet.org/content/wrist/1.0.0/) |
| [**PPG-DaLiA**](#ppg-dalia)<br>手⁠腕⁠多⁠模⁠态 · `PPG+ECG 配对` | Univ. of Siegen / Bosch | 腕⁠部 PPG 64 Hz、ACC 32 Hz、EDA 4 Hz + 胸 ECG 700 Hz | 15 人 (每人 2.5 小时，共 36 小时)<br>任务：日常活动心率监测与去噪 | [开⁠放⁠获⁠取](https://archive.ics.uci.edu/dataset/495/ppg+dalia) |
| [**WESAD**](#wesad)<br>手⁠腕⁠多⁠模⁠态 · `PPG+ECG 配对` | Univ. of Siegen / Bosch | 腕⁠部 E4 (PPG/ACC/EDA/Temp) + 胸 RespiBAN 700 Hz | 15 人 (实验室压力与情绪诱发)<br>任务：心理压力检测、情绪状态分类 | [开⁠放⁠获⁠取](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection) |
| [**TROIKA**](#troika)<br>手⁠腕⁠多⁠模⁠态 · `PPG+ECG 配对` | Samsung Research / 国⁠防⁠科⁠大 | 腕⁠部⁠双⁠通⁠道 PPG 125 Hz、ACC 125 Hz + 胸 ECG 125 Hz | 12 人 (跑步机最高 15 km/h)<br>任务：剧烈跑步心率跟踪基准 | [开⁠放⁠获⁠取](https://zenodo.org/records/3902660) |
| [**CAS-BP**](#cas-bp)<br>手⁠腕⁠多⁠模⁠态 · `PPG+ECG 配对` | 中⁠科⁠院⁠深圳⁠先⁠进⁠院 / 华⁠为 | 手⁠表⁠多⁠波⁠长 PPG 1000 Hz + ECG 1000 Hz + 听⁠诊⁠血⁠压 | 1,272 人 (4 周随访，12,624 次测量)<br>任务：无袖带连续血压回归与校准 | [学⁠术⁠申⁠请](https://github.com/zdzdliu/CAS-BP) |
| [**WildPPG**](#wildppg)<br>手⁠腕⁠多⁠模⁠态 · `PPG+ECG 配对` | ETH Zürich (SIPLab) | 4 部⁠位⁠多⁠波⁠长 PPG/ACC 128 Hz + 皮⁠温 + 气⁠压 + 胸 ECG | 16 人 (高山徒步与日常交通，共 216 小时)<br>任务：野外高噪/温差/海拔变化下 PPG 心率估计与去噪 | [开⁠放⁠获⁠取](https://github.com/eth-siplab/WildPPG) |
| [**BUT-PPG**](#but-ppg)<br>手⁠腕⁠多⁠模⁠态 · `PPG+ECG 配对` | 布⁠尔⁠诺⁠理工⁠大⁠学 (BUT) | 手⁠机⁠镜⁠头⁠指端/耳⁠部 PPG 30 Hz + 胸 ECG 1000 Hz + ACC | 48 人 (3,888 段 10 s 记录，含血压/血氧/血糖)<br>任务：PPG 信号质量评估 (SQI)、伪影判别与心率估计 | [开⁠放⁠获⁠取](https://physionet.org/content/butppg/2.0.0/) |
| [**RealWorldPPG**](#realworldppg)<br>手⁠腕⁠多⁠模⁠态 · `含 PPG` | 曼⁠努⁠菲⁠亚大⁠学 (Menoufia Univ.) | 指⁠端⁠光⁠电 PPG 50 Hz (单⁠段 6 s，300 采⁠样⁠点) | 35 人 (共 2,074 段记录，年龄 10-75 岁)<br>任务：脉搏波形特征提取、心率估计与生物身份识别 | [开⁠放⁠获⁠取](https://data.mendeley.com/datasets/yynb8t9x3d/3) |
| [**ECSMP**](#ecsmp)<br>手⁠腕⁠多⁠模⁠态 · `PPG+ECG 配对` | 重⁠庆⁠邮⁠电大⁠学 / 西⁠南⁠大⁠学 | 腕⁠部 E4 (PPG 64 Hz/ACC/EDA/Temp) + 脑⁠电 EEG + 胸 ECG | 89 人 (静息、情绪诱发、睡眠与抑郁量表评估)<br>任务：多模态情绪识别、抑郁筛查与自主神经状态评估 | [开⁠放⁠获⁠取](https://data.mendeley.com/datasets/f9bf7b6264/2) |
| [**ShanghaiT1DM / T2DM**](#shanghait1dm--shanghait2dm)<br>CGM 连⁠续⁠血⁠糖 | 上⁠海⁠市⁠第六⁠人⁠民⁠医院 | Abbott Libre (15 min 间⁠质⁠液⁠血⁠糖) + 入⁠院⁠血⁠检⁠生化 | 112 人 (12 例 T1D + 100 例 T2D，3-14 天)<br>任务：连续血糖预测、缺失插补、生化指标探针 | [开⁠放⁠获⁠取](https://figshare.com/articles/dataset/ShanghaiT1DM_T2DM_data/20444397) |
| [**Hall Glucotypes**](#hall-glucotypes)<br>CGM 连⁠续⁠血⁠糖 | 斯⁠坦⁠福⁠大学 (Michael Snyder 实⁠验⁠室) | Dexcom G4 (5 min 血⁠糖) + 标⁠准⁠化⁠早⁠餐记⁠录 | 57 人 (105,426 个采样点)<br>任务：血糖动态波动分型、前驱糖尿病表征 | [开⁠放⁠获⁠取](https://doi.org/10.1371/journal.pbio.2005143.s010) |
| [**BIG IDEAs**](#big-ideas)<br>CGM 连⁠续⁠血⁠糖 · `含 PPG` | 杜⁠克⁠大⁠学 (Jessilyn Dunn 实⁠验⁠室) | Dexcom G6 (5 min) + Empatica E4 手⁠环 | 16 人 (连续配对 8-10 天)<br>任务：腕戴光电与皮下间质血糖跨模态建模 | [开⁠放⁠获⁠取](https://physionet.org/content/big-ideas-glycemic-wearable/1.1.2/) |
| [**CGMacros**](#cgmacros)<br>CGM 连⁠续⁠血⁠糖 · `含 PPG` | 德⁠州⁠农⁠工大⁠学 / Sansum | 双 CGM (5/15 min) + Fitbit + 餐⁠食⁠照⁠片⁠与宏⁠量⁠营⁠养⁠素 | 45 人 (每人约 10 天，含生化与菌群)<br>任务：个性化营养餐后血糖响应建模 | [开⁠放⁠获⁠取](https://physionet.org/content/cgmacros/1.0.0/) |
| [**Colas DFA**](#colas-dfa)<br>CGM 连⁠续⁠血⁠糖 | Hospital Universitario de Móstoles | Medtronic iPro (5 min 血⁠糖) | 208 人 (累计约 9,500 小时)<br>任务：2 型糖尿病发病风险预测、时序预训练 | [开⁠放⁠获⁠取](https://doi.org/10.1371/journal.pone.0225817.s001) |
| [**OhioT1DM**](#ohiot1dm)<br>CGM 连⁠续⁠血⁠糖 · `含 PPG` | 俄⁠亥⁠俄⁠大学 | Medtronic Enlite (5 min) + 胰⁠岛⁠素⁠泵 + 碳⁠水 + 手⁠环 | 12 人 (每人 8 周连续自由生活)<br>任务：30-60 min 短程血糖自回归预测基准 | [学⁠术⁠申⁠请](http://smarthealth.cs.ohio.edu/OhioT1DM-dataset.html) |
| [**Weinstock 2016**](#weinstock-2016)<br>CGM 连⁠续⁠血⁠糖 | T1D Exchange / Jaeb Center | 盲⁠态 Dexcom G4 (5 min) + 严⁠重⁠低⁠血⁠糖发⁠作⁠史 | 201 人 (高龄 T1D，每人约 14 天)<br>任务：老年高危人群严重低血糖风险分析 | [开⁠放⁠获⁠取](https://public.jaeb.org/) |
| [**GlucoBench**](#glucobench)<br>CGM 连⁠续⁠血⁠糖 | 德⁠州⁠农⁠工大⁠学 (Irina Gaynanova 实⁠验⁠室) | 整⁠合 5 个⁠公⁠开 CGM 队⁠列 (统⁠一 5 min 网⁠格) | >500 人<br>任务：跨队列标准化血糖预测与零样本评测 | [开⁠放⁠获⁠取](https://github.com/IrinaStatsLab/GlucoBench) |
| [**Glucose-ML**](#glucose-ml)<br>CGM 连⁠续⁠血⁠糖 | Emory / Dartmouth (Prioleau 团⁠队) | 自⁠动⁠化⁠整⁠合 20 余⁠个⁠公⁠开 CGM 数⁠据⁠集 | 4,300+ 人 (约 4,490 万采样点)<br>任务：大规模 CGM 基础模型预训练与标准化转换 | [开⁠放⁠获⁠取](https://github.com/Augmented-Health-Lab/Glucose-ML-Project) |
| [**MC-MED**](#mc-med)<br>临⁠床⁠连⁠续⁠监⁠护 · `PPG+ECG 配对` | 斯⁠坦⁠福⁠大学⁠医⁠学⁠院 | 床⁠旁 ECG 500 Hz、PPG 125 Hz、Resp 125 Hz + 急⁠诊 EHR | 118,385 次就诊 (70,545 人)<br>任务：脉搏与心电基础模型预训练、急诊表型预测 | [凭⁠证⁠申⁠请](https://physionet.org/content/mc-med/) |
| [**VitalDB**](#vitaldb)<br>临⁠床⁠连⁠续⁠监⁠护 · `PPG+ECG 配对` | 首⁠尔⁠大⁠学医⁠院 | 桡⁠动⁠脉⁠波 ABP 500 Hz、PPG 500 Hz、ECG 500 Hz | 6,388 例非心脏手术全程监护<br>任务：PPG 转血压波形重建、围术期多参数建模 | [开⁠放⁠获⁠取](https://vitaldb.net/) |
| [**BIDMC PPG**](#bidmc-ppg)<br>临⁠床⁠连⁠续⁠监⁠护 · `PPG+ECG 配对` | 牛⁠津⁠大⁠学 / BIDMC | 指⁠尖 PPG 125 Hz、ECG 125 Hz、阻⁠抗⁠呼⁠吸⁠波 125 Hz | 53 人 (每人 8 min 重症监护记录)<br>任务：呼吸率估计、脉搏波微观形态分析基准 | [开⁠放⁠获⁠取](https://physionet.org/content/bidmc/1.0.0/) |
| [**PulseDB**](#pulsedb)<br>临⁠床⁠连⁠续⁠监⁠护 · `PPG+ECG 配对` | 德⁠州⁠农⁠工大⁠学 (PulseLab) | 同⁠步 ECG + 指⁠尖 PPG + 有⁠创⁠动⁠脉⁠波 ABP 125 Hz (10 s) | 5,361 人 (524.5 万段清洗切片，源自 MIMIC-III 与 VitalDB)<br>任务：无袖带连续血压回归基准、血管生物年龄跨域评测 | [开⁠放⁠获⁠取](https://github.com/pulselabteam/PulseDB) |
| [**PPG-BP**](#ppg-bp)<br>临⁠床⁠连⁠续⁠监⁠护 · `含 PPG` | 桂⁠林⁠电⁠子科⁠技⁠大⁠学 / 英⁠属⁠哥⁠伦比⁠亚⁠大⁠学 | 短⁠时⁠指⁠尖 PPG 1000 Hz (单⁠段 2.1 s) + 袖⁠带⁠血⁠压⁠与临⁠床⁠诊⁠断 | 219 人 (657 段波形，年龄 20-89 岁，含高血压与糖尿病标签)<br>任务：短时 PPG 无袖带血压估计、高血压与慢病筛查 | [开⁠放⁠获⁠取](https://figshare.com/articles/dataset/PPG-BP_Database_zip/5459299) |
| [**MIMIC-III Waveform**](#mimic-iii-waveform)<br>临⁠床⁠连⁠续⁠监⁠护 · `PPG+ECG 配对` | MIT LCP / BIDMC | ICU 多⁠导⁠联 ECG、指⁠尖 PPG、有⁠创⁠动⁠脉⁠波 ABP、呼⁠吸 125 Hz | 67,830 条记录 (约 3 万人；匹配 EHR 子集 10,282 人 / 22,317 条)<br>任务：脉搏与心电连续波形预训练、无创血压估计与危重症预警 | [开⁠放⁠获⁠取](https://physionet.org/content/mimic3wdb/1.0/) |
| [**WAVES**](#waves)<br>临⁠床⁠连⁠续⁠监⁠护 · `PPG+ECG 配对` | 斯⁠坦⁠福⁠大学⁠卢⁠西⁠尔·帕⁠卡⁠德⁠儿童⁠医⁠院 | 儿⁠科⁠床⁠旁⁠连续 ECG 125 Hz、指⁠尖 PPG 125 Hz (平⁠均 7 小⁠时) | 50,400 名儿科患者 (55 万条心电 + 51 万条脉搏波记录)<br>任务：儿科重症连续生命体征预训练与跨年龄生理泛化评估 | [受⁠限⁠申⁠请](https://stanfordmedicine.box.com/v/waves-dataset) |
| [**MOVER**](#mover)<br>临⁠床⁠连⁠续⁠监⁠护 · `PPG+ECG 配对` | 加⁠州⁠大⁠学欧⁠文⁠分⁠校医⁠疗⁠中⁠心 (UCI) | 术⁠中 ECG 180/300 Hz、PPG 100 Hz、有⁠创⁠动⁠脉⁠波 ABP + 围⁠术⁠期 EHR | 58,799 名手术患者 (83,468 台次手术全程监护与电子病历)<br>任务：围术期血流动力学建模、不良事件预测与跨模态预训练 | [学⁠术⁠申⁠请](https://mover.ics.uci.edu/) |
| [**SHHS**](#shhs)<br>睡⁠眠 PSG | 美⁠国 NIH / NHLBI (NSRR) | 整⁠夜⁠家⁠庭 PSG (EEG/EOG/EMG/ECG/Resp/SpO2) | 5,804 人 (>15 年心血管随访)<br>任务：睡眠分期、心肺耦合与慢病发病预测 | [学⁠术⁠申⁠请](https://sleepdata.org/datasets/shhs) |
| [**MESA**](#mesa)<br>睡⁠眠 PSG · `PPG+ECG 配对` | 美⁠国 NIH / NHLBI (NSRR) | 整⁠夜 PSG (256 Hz) + 7 天⁠腕⁠部⁠体⁠动仪 Actigraphy | 2,237 人 (四族裔中老年队列)<br>任务：临床 PSG 与腕戴睡眠/脉搏跨模态对齐 | [学⁠术⁠申⁠请](https://sleepdata.org/datasets/mesa) |
| [**Sleep-EDF**](#sleep-edf)<br>睡⁠眠 PSG | St. Antonius Hospital / LUMC | 双⁠通⁠道 EEG 100 Hz、EOG 100 Hz、EMG、呼⁠吸 | 197 份整夜 PSG 记录<br>任务：睡眠分期算法快速原型验证基准 | [开⁠放⁠获⁠取](https://physionet.org/content/sleep-edfx/1.0.0/) |
| [**DREAMT**](#dreamt)<br>睡⁠眠 PSG · `PPG+ECG 配对` | 杜⁠克⁠大⁠学 (BIG IDEAs Lab) | 左⁠腕 Empatica E4 (PPG 64 Hz/ACC 32 Hz/EDA/Temp) + 整⁠夜⁠临⁠床 PSG | 100 名睡眠障碍门诊患者 (整夜同步记录，30 s 睡眠分期标注)<br>任务：腕戴 PPG/ACC 实时睡眠分期、睡眠呼吸暂停与肥胖筛查 | [注⁠册⁠获⁠取](https://physionet.org/content/dreamt/2.1.0/) |
| [**UK Biobank**](#uk-biobank)<br>人⁠群⁠队⁠列 | UK Biobank / 牛⁠津⁠大⁠学 | 腕⁠部⁠三⁠轴 ACC 100 Hz (7 天) + 全⁠基⁠因⁠组 + MRI + EHR | 103,712 人体动 (总队列 50 万人)<br>任务：运动数字生物标志物、全谱系慢病风险建模 | [受⁠限⁠申⁠请](https://www.ukbiobank.ac.uk/enable-your-research/apply-for-access) |
| [**All of Us**](#all-of-us)<br>人⁠群⁠队⁠列 · `含 PPG` | 美⁠国 NIH All of Us Research Program | Fitbit 多⁠年⁠连⁠续⁠序列 + 全⁠基⁠因⁠组 + OMOP EHR | 40 万+ 人 EHR (数万人含 Fitbit)<br>任务：跨族裔精准医学与长程可穿戴健康模型验证 | [受⁠限⁠申⁠请](https://www.researchallofus.org/) |
| [**MIMIC-IV-ECG**](#mimic-iv-ecg)<br>心⁠电⁠信⁠号 · `纯 ECG` | MIT / BIDMC | 12 导⁠联⁠诊⁠断⁠级 ECG 500 Hz (10 s) + 住⁠院 EHR | 800,035 条记录 (161,352 人)<br>任务：大规模心电自监督预训练与多病种筛查 | [开⁠放⁠获⁠取](https://physionet.org/content/mimic-iv-ecg/1.0/) |
| [**PTB-XL**](#ptb-xl)<br>心⁠电⁠信⁠号 · `纯 ECG` | 德⁠国 PTB / Fraunhofer HHI | 12 导⁠联 ECG 500/100 Hz (10 s，71 类 SCP 标⁠签) | 21,837 条记录 (18,885 人)<br>任务：多标签心电分类与基础模型微调基准 | [开⁠放⁠获⁠取](https://physionet.org/content/ptb-xl/1.0.3/) |
| [**CODE-15%**](#code-15)<br>心⁠电⁠信⁠号 · `纯 ECG` | 巴⁠西 TNMG / Uppsala Univ. | 12 导⁠联 ECG 400 Hz (7-10 s) + 死⁠亡⁠率⁠随⁠访 | 345,779 条记录 (233,770 人)<br>任务：心电预训练缩放定律与长程生存分析 | [开⁠放⁠获⁠取](https://zenodo.org/records/4916206) |
| [**Chapman-Shaoxing / Ningbo**](#chapman-shaoxing--ningbo)<br>心⁠电⁠信⁠号 · `纯 ECG` | Chapman Univ. / 绍⁠兴⁠与⁠宁波⁠医⁠院 | 12 导⁠联 ECG 500 Hz (10 s，SNOMED-CT 标⁠签) | 45,152 条临床记录<br>任务：东亚人群心律失常分类与跨中心泛化验证 | [开⁠放⁠获⁠取](https://physionet.org/content/ecg-arrhythmia/1.0.0/) |
| [**HEEDB**](#heedb)<br>心⁠电⁠信⁠号 · `纯 ECG` | 哈⁠佛⁠医⁠学院 (MGH / BWH) / Emory Univ. | 12 导⁠联⁠诊⁠断⁠级 ECG 250/500 Hz (10 s) + Marquette 12SL 诊⁠断⁠码 | 1,170 万条记录 (210 万名患者，大规模临床心电库)<br>任务：千万级心电基础模型预训练 (ECGFounder, ECG-CPC 底座) 与全谱系筛查 | [凭⁠证⁠申⁠请](https://bdsp.io/content/heedb/) |
| [**EchoNext**](#echonext)<br>心⁠电⁠信⁠号 · `纯 ECG` | 哥⁠伦⁠比⁠亚大⁠学⁠欧⁠文医⁠学⁠中⁠心 | 12 导⁠联 ECG 250 Hz (10 s) + 配⁠对⁠超⁠声⁠心动⁠图⁠结⁠构⁠性心⁠脏⁠病⁠标⁠签 | 100,000 条心电-超声配对记录 (36,286 名患者，11 类结构性病变标签)<br>任务：隐匿性结构性心脏病筛查 (LVEF 降低、瓣膜狭窄/反流、心室肥厚) | [凭⁠证⁠申⁠请](https://physionet.org/content/echonext/1.0.0/) |
| [**SPH**](#sph)<br>心⁠电⁠信⁠号 · `纯 ECG` | 山⁠东⁠省⁠立医⁠院 | 12 导⁠联 ECG 500 Hz (10-60 s，44 类 AHA 标⁠准⁠诊⁠断⁠语句) | 25,770 条记录 (24,666 名中国成人患者)<br>任务：多标签心电分类、AHA 标准层级诊断与跨中心泛化评测 | [开⁠放⁠获⁠取](https://springernature.figshare.com/collections/A_large-scale_multi-label_12-lead_electrocardiogram_database_with_standardized_diagnostic_statements/5779802) |
| [**ZZU pECG**](#zzu-pecg)<br>心⁠电⁠信⁠号 · `纯 ECG` | 郑⁠州⁠大⁠学第⁠三⁠附⁠属医⁠院 | 儿⁠童 12/9 导⁠联 ECG 500 Hz (5-120 s，64 类⁠心⁠电⁠语⁠句 + ICD-10) | 14,190 条记录 (11,643 名 0-14 岁儿科患者)<br>任务：儿科心律失常与先心病筛查、成人心电基模跨年龄泛化基准 | [开⁠放⁠获⁠取](https://doi.org/10.6084/m9.figshare.27078763) |
| [**CPSC 2018 / CinC 2020**](#cpsc-2018--cinc-2020)<br>心⁠电⁠信⁠号 · `纯 ECG` | 东⁠南⁠大⁠学 / 中⁠国⁠生⁠理信⁠号⁠挑⁠战赛 / PhysioNet | 12 导⁠联 ECG 500 Hz (6-60 s，9 类⁠心⁠律⁠与⁠传导⁠异⁠常⁠及 SNOMED-CT 标⁠签) | CPSC 10,330 条 (11 家医院)；CinC 2020/2021 跨国总库达 88,253 条<br>任务：多中心心律失常分类与心电基础模型标准微调评测 | [开⁠放⁠获⁠取](https://physionet.org/content/challenge-2020/1.0.2/) |
| [**SNUH LYDUS**](#snuh-lydus)<br>心⁠电⁠信⁠号 · `纯 ECG` | 首⁠尔⁠大⁠学医⁠院 (SNUH) | 12 导⁠联 ECG 500/100 Hz (10 s) + 专⁠家⁠复⁠核⁠诊断⁠与⁠测⁠量⁠间期 | 167,000 条受限访问版 + 50,000 条完全开放版 (38,402 人)<br>任务：东亚人群心电预训练、传导间期回归与心律失常诊断 | [开⁠放⁠获⁠取](https://doi.org/10.1038/s41597-025-04520-z) |
| [**Icentia11k**](#icentia11k)<br>心⁠电⁠信⁠号 · `纯 ECG` | 蒙⁠特⁠利⁠尔大⁠学 / Mila / Icentia | CardioSTAT 单⁠导⁠联⁠动⁠态胸⁠贴 ECG 250 Hz (长⁠达 3-14 天⁠连⁠续⁠监⁠测) | 11,000 名患者 (54,000 段长时切片，累计约 2.77 亿次专家标注心拍)<br>任务：单导联长程心电预训练 (HeartGPT, HeartBERT 底座) 与逐拍心律失常识别 | [开⁠放⁠获⁠取](https://physionet.org/content/icentia11k-continuous-ecg/1.0/) |

<!-- END:DATASETS_DOC_TABLE -->

---

## 7. 数据清洗与信号质量控制

### 7.1 连续血糖标准化流程

不同厂家的 CGM 设备在采样间隔（Dexcom 5 min vs. Abbott Libre 15 min）、血糖单位（$\text{mg/dL}$ vs. $\text{mmol/L}$）及导出表头命名上存在差异。在构建跨队列预训练语料时，通常采用 [Glucose-ML](#glucose-ml) 与 [GlucoBench](#glucobench) 的标准化流程：

1. **自动化拉取与列名对齐**：通过 `Glucose-ML-Project` 提供的 `auto-download-open-datasets.py` 与 `auto-harmonize-CGML-datasets.py` 脚本，将异构原始表统一重映射为标准三列结构（受试者 ID、ISO 时间戳、以 $\text{mg/dL}$ 为单位的血糖值，其中单位换算关系为 $1\text{ mmol/L} = 18.018\text{ mg/dL}$）：

```csv
subject_id,timestamp,glucose_value
shanghai_01,2021-05-10 08:00:00,112.5
shanghai_01,2021-05-10 08:05:00,115.0
```

2. **时间网格重采样与缺失处理**：将时间戳对齐至固定 5 min 网格；对短于 30-45 min 的短暂传感器丢包采用线性或样条插值，对超过阈值的长段脱落则切分为独立连续子序列，避免在自回归或掩码预训练中引入伪造趋势。

### 7.2 腕部生理信号质量评估与窗口对齐

将高频原始 PPG/ECG 波形送入编码器前，需过滤佩戴松动、传感器悬空与重度运动伪影片段，并统一多通道时间片（Patch）粒度：

1. **时域偏度与峭度指标**（Skewness & Kurtosis SQI）：正常动脉脉搏充盈波呈现收缩期快速上升、舒张期缓慢下降的正偏斜形态（偏度 $S_{\text{sqi}} > 0$）。当剧烈运动伪影导致波形削顶或随机震荡时，偏度与峭度会偏离正常生理范围，可据此在滑窗阶段剔除低质量片段。
2. **灌注指数与接触检测**（Perfusion Index, PI）：灌注指数定义为交流脉动分量与直流基线分量之比 $\text{PI} = (\text{AC} / \text{DC}) \times 100\%$。当 $\text{PI} < 0.1\%$ 时，通常表明光电探头与手腕皮肤脱离或局部外周灌注过低。
3. **多模态时间片对齐**：针对采样率不同的同步通道，按固定物理时间窗 $\tau$（例如 $\tau = 0.25\text{ s}$）切分 1D Patch——此时 64 Hz 的 PPG 每片包含 16 个采样点，32 Hz 的三轴 ACC 每片包含 8 个采样点，从而保证不同模态投影后的 Token 序列在时间轴上严格对齐。

---

> **欢迎交流与共建**  
> 本文由开源知识库项目 [Wearable-Physiological-Foundation-Models](https://github.com/tujy859/Wearable-Physiological-Foundation-Models) 整理策划。该项目持续追踪全球在智能手表、连续血糖监测（CGM）以及多模态生理健康基础模型领域的最新突破。如果您对可穿戴生理计算、时序健康基模与数字生物标志物有任何想法，欢迎在仓库提 Issue 交流！
