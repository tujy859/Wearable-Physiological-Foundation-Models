# 智能手表与腕戴生理基础模型深度解析

腕戴设备（智能手表、智能手环）是普及度最高的消费级生理监测终端。本专题深入解构针对光电容积脉搏波（PPG）、三轴加速度计（ACC）及体表心电（ECG）研发的专用时序基础模型，梳理其对抗运动伪影、多速率对齐、跨模态生成以及端侧轻量化部署的底层技术演进。

## 目录

- [1. 腕戴生理时序的独特性与核心技术挑战](#1-腕戴生理时序的独特性与核心技术挑战)
- [2. 经典多模态与波形形态基础模型](#2-经典多模态与波形形态基础模型)
  - [SensorFM](#sensorfm)
  - [PaPaGei](#papagei)
  - [Pulse-PPG](#pulse-ppg)
  - [xMAE](#xmae)
  - [HiMAE](#himae)
  - [UniCardio](#unicardio)
  - [PPGFlowECG](#ppgflowecg)
  - [AnyPPG](#anyppg)
- [3. 端侧轻量化、生物年龄与大一统评测基准](#3-端侧轻量化生物年龄与大一统评测基准)
  - [PPG-Distill](#ppg-distill)
  - [PpgAge](#ppgage)
  - [开源脉搏基础模型生物年龄横评](#开源脉搏基础模型生物年龄横评)
  - [PPG-Bench](#ppg-bench)
- [4. 空间运动与跨专题基模索引](#4-空间运动与跨专题基模索引)
  - [LIMU-BERT](#limu-bert)
  - [Mantis](#mantis)
  - [SleepFM](#sleepfm)
- [5. 大规模开源生态与时序语言多模态基座](#5-大规模开源生态与时序语言多模态基座)
  - [OpenMHC](#openmhc)
  - [PPG-LM](#ppg-lm)
  - [SensorLM](#sensorlm)
- [6. 核心基础模型横向对比矩阵](#6-核心基础模型横向对比矩阵)
- [7. 对可穿戴多模态大模型的工程架构启示](#7-对可穿戴多模态大模型的工程架构启示)
- [8. 核心文献与资源汇总](#8-核心文献与资源汇总)

---

## 1. 腕戴生理时序的独特性与核心技术挑战

智能手表采集的生理数据具备较强的物理与生物学先验约束：

1. **异构采样率与多模态物理对齐**：
   - **PPG（光电容积脉搏波）**：常见采样率为 25 Hz、50 Hz、64 Hz 或 100 Hz，波形由微血管收缩期充血与舒张期弹性回缩构成，包含收缩峰、重搏切迹及舒张峰；
   - **ACC（三轴加速度计）**：常见采样率为 25 Hz 或 50 Hz，记录人体手腕在空间三维的正交加速度矢量 $(a_x, a_y, a_z)$；
   - **ECG（偶发单导联心电）**：250～1000 Hz，由用户主动按压表冠或背部电极触发；
   - **低频慢速指标**：皮肤温度（0.1 Hz）、皮电反应 EDA（4 Hz）、气压高度计（1 Hz）；
2. **运动伪影与频域碰撞**：
   - 手臂摆动、跑步落足引起的传感器相对位移与局部静脉血液惯性流动，使光学传感器接收到的漫反射光被剧烈调制；
   - 日常跑步步频谐波（1.0～3.5 Hz，对应 60～210 SPM）与运动心率主频（1.2～3.2 Hz，对应 70～190 BPM）高度重叠，简单的频域滤波易误将步频识别为心率；
3. **端侧计算与功耗约束**：
   - 智能手表电池容量通常在 200～500 mAh 之间，运行在低功耗 ARM 核心上，对模型参数量、内存占用和推理延迟有严格要求。

---

## 2. 经典多模态与波形形态基础模型

### SensorFM

Google Research / Fitbit Health AI · 2026 · [论文](https://arxiv.org/abs/2605.22759)

![SensorFM 架构与评测流程](../figures/sensorfm/overview.png)

图源：SensorFM 论文 Fig. 1（arXiv:2605.22759）

SensorFM 是面向穿戴传感器时序构建的大规模基础模型，探索了海量多模态数据下的 Scaling 效应。

#### 1. 数据语料与模态构成

- **数据体量**：超过 1 万亿分钟（>1,000,000,000,000 分钟）真实可穿戴多模态时序；
- **用户覆盖**：涵盖 500 余万名知情同意的 Fitbit 与 Google Pixel Watch 全球用户；
- **传感器覆盖**：绿光 / 红外 PPG、三轴 ACC、皮电反应（EDA）、皮肤温度与气压高度计。

#### 2. 核心架构与自监督算法

- **统一时间分块（Multi-Rate Patch Tokenization）**：采用固定的物理时间窗口切片，避免单纯插值重采样带来的相位失真。为不同速率的传感器配备独立的投影层，附加时间位置编码后输入统一的双向 Transformer 编码器；
- **复合预训练目标**：
  - **掩码多模态建模（Masked Multimodal Modeling）**：在时域和传感器通道域进行随机掩码，联合重构被遮蔽的生理信号；
  - **跨模态填补（Cross-Modal Infilling）**：模拟部分传感器离线或节能关断场景，由在线传感器在隐空间推断缺失模态表征。

#### 3. 下游任务评测

- 在涵盖心血管（静息心率、HRV、VO2 Max）、代谢与活动（能耗、运动识别）、睡眠与神经（睡眠分期、呼吸暂停）以及压力评估等 35 项下游健康任务中进行了验证，证实了在大规模无标注穿戴数据上进行通用预训练的表征迁移价值。

---

### PaPaGei

Nokia Bell Labs / 剑桥大学 / 达特茅斯学院 · ICLR 2025 · [论文](https://arxiv.org/abs/2410.20542) · [代码](https://github.com/Nokia-Bell-Labs/papagei-foundation-model) · [权重](https://zenodo.org/records/13983110)

![PaPaGei 架构总览](../figures/papagei/overview.png)

图源：PaPaGei 论文 Fig. 1（ICLR 2025）

PaPaGei 针对光学脉搏波（PPG）的微观形态特点，提出了形态感知的开源生理基础模型。

#### 1. ResNet1D-MoE 架构

- **输入格式**：10 秒单通道 PPG 信号，统一重采样至 125 Hz（1,250 采样点）；
- **网络设计**：包含 18 个 1D 残差卷积块，引入 3 个混合专家（MoE）路由分支，基于瞬态脉搏形态动态分配特征通路，输出 512 维隐空间表征，编码器参数量约 1.5M。

#### 2. 形态感知对比学习 (Morphology-Aware Pretraining)

- 传统对比学习若将同一用户不同生理状态视作正样本，容易消除血管阻抗与搏动微形态差异；
- PaPaGei 提取收缩期上升斜率、舒张期凹槽等生理波形先验，构建连续动态对比正负样本对，促使隐空间保留心血管动力学特异性；
- 在心率回归、血压估计（收缩压 / 舒张压 MAE 约 5～8 mmHg）、血管弹性生理年龄与呼吸暂停检测等任务上验证了特征迁移性能。

---

### Pulse-PPG

孟菲斯大学 / UIUC · UbiComp 2025 · [代码](https://github.com/maxxu05/pulseppg)

![Pulse-PPG 架构总览](../figures/pulse-ppg/overview.png)

图源：Pulse-PPG 论文 Fig. 1（UbiComp 2025）

Pulse-PPG 聚焦于真实野外非受限生活场景下的手腕脉搏波建模，解决了临床平躺数据向真实手腕迁移时的领域漂移问题。

#### 1. 真实生活野外语料与域漂移

- 临床数据库（如 MIMIC、VitalDB）多为静卧麻醉状态下的指夹透射式 PPG，波形规则且几乎无运动伪影；而真实生活手腕由于手臂摆动和接触变化，信噪比容易大幅波动；
- 采集了 120 名受试者连续 100 天自由生活的腕戴数据，沉淀出约 5.5 万小时有效野外 PPG 语料。

#### 2. 相对对比学习 (Relative Contrastive Learning, RelCon)

- 摒弃离散的正负二元划分，将时序物理时间间隔与三轴加速度计估算的瞬时运动量联合编码为连续权重，构建自适应温度调节的相对对比损失；
- 骨干网络采用包含大卷积核（Kernel Size = 11）的 12 层 1D ResNet，在长程心动周期依赖建模与高噪抗伪影上展现出鲁棒性。

---

### xMAE

Samsung Research America / UCLA · ICML 2024

![xMAE 跨模态预训练流程](../figures/xmae/overview.png)

图源：xMAE 论文 Fig. 2（ICML 2024）

xMAE 探索了将全天候手腕 PPG 映射为高临床价值心电图（ECG）表征的跨模态虚拟传感方案。

- **动机与设定**：心电图具有明确的心律失常与传导阻滞诊断价值，但需双手形成回路测量；背部光学 PPG 可实现 24 小时被动采集；
- **跨模态掩码自编码**：在 9,400 小时配对同步 ECG-PPG 数据上训练，将高精度 ECG 大比例掩码（75%），以配对的腕部 PPG 为上下文条件，训练 Cross-Modal Transformer 重建被掩码的 ECG 潜表征；
- **推断优势**：在下游推断时无需触碰电极，利用手腕日常 PPG 即可连续提取与心肌电生理相关的特征。

---

### HiMAE

Samsung Research America / UCLA · ICLR 2026 · [论文](https://arxiv.org/abs/2510.25785) · [代码](https://github.com/Simonlee711/HiMAE)

![HiMAE 分层掩码自编码与多尺度流程](../figures/himae/overview.png)

图源：HiMAE 论文 Fig. 1（ICLR 2026）

HiMAE 针对通用大模型在智能手表 CPU 上计算开销过大的痛点，提出了专为腕上端侧实时推断设计的分层 1D 卷积掩码自编码器。

#### 1. 时间分辨率假说 (The Resolution Hypothesis)

- 腕戴生理信号具有强局部性与周期节律性。时间感受野不仅是一个超参数，更是生理表征的基本维度；
- 不同的健康指标对应不同的物理时间尺度：从毫秒级单心拍形态（早搏、高血压切迹），到秒级多心跳心率变异性（HRV），再到多秒级的呼吸性节律与生化指标，模型应显式分层建模。

#### 2. 1D 卷积 U-Net 与物理时间尺度映射

输入固定长度 10 秒（100 Hz，1,000 采样点）单通道绿光 PPG，采用 5 级残差卷积 U-Net 逐步下采样并恢复：

| 网络层级 | 特征点数 | 感受野点数 | 物理时间跨度 | 对应时间尺度 | 生理学物理意义 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Enc 1** | 500 点 | 5～13 点 | **50～130 ms** | 毫秒级 | 单心拍瞬态形态：收缩期上升支、收缩峰、重搏切迹 |
| **Enc 2** | 250 点 | 21～37 点 | **210～370 ms** | 亚秒级 | 单次心室射血与充盈期：外周阻力与顺应性响应 |
| **Enc 3** | 125 点 | 53～85 点 | **530～850 ms** | 约 1 秒级 | 完整单心动周期：对应常规心率（60～100 bpm） |
| **Enc 4** | 62 点 | 117～181 点 | **1.17～1.81 s** | 秒级 | 跨心拍间期交互：相邻搏动变异、代偿间歇 |
| **Enc 5** | 31 点 | 245～373+ 点 | **2.45～3.73 s+** | 多秒级 | 宏观血流动力学包络：呼吸性窦性心律不齐、基线波动 |

#### 3. 预训练语料与端侧部署表现

- 在涵盖 47,644 名用户、约 80,000 小时的三星智能手表绿光 PPG 数据上预训练，配合偏度截断、自相关周期离散度与模板契合度三道信号质控（SQI）；
- 在 Galaxy Watch 8 实机测试中，编码器参数量仅 0.31M，内存占用 3.6 MB，手表 CPU 上的单次推理延迟为 **0.99 ms**，满足端侧超低功耗常驻运行需求。

---

### UniCardio

清华大学 / 北京安贞医院 / 帝国理工学院 · Nature Machine Intelligence 2025/2026 · [论文](https://doi.org/10.1038/s42256-025-01147-y) · [代码](https://github.com/chen-zehua/UniCardio)

![UniCardio 扩散模型架构与训练范式](../figures/unicardio/overview.png)

图源：UniCardio 论文 Fig. 2（Nature Machine Intelligence 2025/2026）

UniCardio 提出了统一的心血管多模态扩散基础模型（Diffusion Transformer, DiT），将去噪、插补与跨模态生成整合进同一框架。

- **统一生成范式**：联合建模 PPG、ECG 与连续动脉血压波（BP）的多模态条件分布，实现多通道信号去噪、连续缺失插补以及“条件模态组 → 目标模态组”的连续波形生成（如腕部 PPG 生成虚拟心电图，或由 PPG+ECG 无创估算连续血压波）；
- **任务特异性注意力掩码**：在共享 DiT 骨干中约束条件 Token 到目标 Token 的信息流向，避免跨模态负迁移；
- **增量式持续学习**：采用轻量模态投影器（单模态新增参数约 0.3M），配合按“单条件 → 双条件 → 多条件”的递进持续学习策略，允许设备灵活接入新传感器而不引发灾难性遗忘。

---

### PPGFlowECG

北京大学数字健康实验室 · arXiv:2502.14856 / 2025-2026 · [代码](https://github.com/PKUDigitalHealth/PPGFlowECG)

![PPGFlowECG 潜空间整流流生成架构](../figures/ppgflowecg/overview.png)

图源：PPGFlowECG 论文 Fig. 2（arXiv:2502.14856）

PPGFlowECG 针对 PPG 转 ECG 跨模态生成中传统 GAN 模式崩溃与扩散模型采样步数过多的问题，提出了基于潜空间整流流（Latent Rectified Flow）的生成框架。

#### 1. 两阶段生成架构

1. **Stage 1: CardioAlign 潜空间生理对齐**：
   - 采用参数绑定的双分支编码器将 PPG 与 ECG 映射至低维生理潜空间，联合优化欧氏距离与对称 KL 散度，滤除外周光学伪影并保留心肌电动力学不变特征；
2. **Stage 2: 潜空间整流流生成（Latent Rectified Flow）**：
   - 沿直线轨迹（Straight-line Trajectory）将高斯噪声传输至 ECG 目标潜分布，以 PPG 潜向量作为条件引导；
   - 借助常微分方程（ODE）求解器，仅需 1～4 步数值积分即可完成高质量潜表征生成，推断速度显著快于多步反向扩散；
3. **ECG 波形还原**：由专用 1D 卷积解码器输出诊断级体表心电波形。

#### 2. 急诊队列与下游疾病筛查

- 在包含 11.8 万名急诊患者、超 1,000 万对真实样本的 MC-MED 数据集及 VitalDB 上评测；
- 生成波形在房颤（AFib）、室性期前收缩（PVC）及心肌缺血检测中表现出较高的病理形态一致性。

---

### AnyPPG

北京大学数字健康实验室 · ACM SIGKDD 2026 · [论文](https://arxiv.org/abs/2511.01747) · [代码](https://github.com/Ngk03/AnyPPG)

![AnyPPG 心电引导预训练框架](../figures/anyppg/overview.png)

图源：AnyPPG 论文 Fig. 2（ACM SIGKDD 2026）

AnyPPG 探讨了将单模态 PPG 从传统心血管任务拓展至全身多器官系统性疾病早筛的通用表征潜力。

- **心电引导的跨模态对比预训练**：
  - 聚合超过 100,000 小时同步 PPG-ECG 配对时序数据；
  - 单模态自监督易受肤色深浅、传感器接触压力及运动高频伪影影响；AnyPPG 引入中枢心电作为无运动伪影的物理参考，通过跨模态 InfoNCE 对比损失拉近同步 PPG 与 ECG 的隐空间表征，引导 PPG 编码器提取深层血流动力学特征；
- **网络结构**：双分支 1D-ResNet 骨干（单分支约 5.85M 参数），全局平均池化后输出 1024 维特征向量，端侧部署时仅需保留 PPG 单分支；
- **多器官表型拓展**：除心力衰竭、高血压等心血管任务外，在慢性肾脏病（CKD，感知微血管硬化与水钠潴留引起的波形微调）与帕金森病（感知自主神经紊乱引发的外周舒缩变化）分类测试中展现了表征泛化能力。

---

## 3. 端侧轻量化、生物年龄与大一统评测基准

### PPG-Distill

Emory University · 2025 · [代码](https://github.com/LingFengGold/PPG-Distill)

![PPG-Distill 框架流程图](../figures/ppg-distill/overview.png)

图源：PPG-Distill 论文 Fig. 3（2025）

PPG-Distill 针对大型脉搏基础模型难以常驻嵌入式硬件的问题，提出了面向脉搏时序的多层次知识蒸馏体系。

- **三级知识蒸馏**：
  1. **预测级蒸馏**：迁移下游多任务软标签分布；
  2. **特征级蒸馏**：在潜表征空间对齐紧凑学生模型与教师模型的特征分布；
  3. **形态与节律蒸馏**：显式约束局部收缩峰微形态与长程心跳周期性结构；
- **实验表现**：学生模型内存占用降低约 19 倍，推理吞吐量显著提升，在心率估计与房颤筛查中保持了稳健的性能。

---

### PpgAge

Apple Health · Nature Medicine 2025/2026

- **语料规模**：依托 Apple Health Study 中 213,593 名真实佩戴者的手腕 PPG 数据；
- **临床发现**：
  - 训练深度时序网络根据日常手腕脉搏波预测个体生理年龄（测试集 MAE = 2.43 年）；
  - **血管年龄差（PPG Age Gap）**：预测年龄显著高于实际日历年龄（反映微血管顺应性降低与动脉硬化）的受试者，未来罹患重大心血管不良事件的风险明显增高（Hazard Ratio = 1.46），为日常慢病筛查提供了数字化参考标志物。

---

### 开源脉搏基础模型生物年龄横评

Nabil Brag · arXiv 2026 · [论文](https://arxiv.org/abs/2603.14030) · [代码](https://github.com/Misterbra/ppg-age-benchmark)

![生物年龄预测曲线对照图](../figures/ppg-age-benchmark/overview.png)

图源：PPG 生物年龄横评论文 Fig. 1（arXiv:2603.14030）

该工作首次在独立公开基准（PulseDB，906 例受试者，41,104 个 10 秒片段）上横向对比了开源自监督基础模型（Pulse-PPG、PaPaGei-S）与监督学习模型在血管生物年龄预测上的表现。

- **关键结论**：
  - 在大型单中心队列（UK Biobank 21 万人）上强监督训练的模型在跨域迁移至术中受试者时出现泛化性能下降；
  - 采用无监督预训练的 Pulse-PPG 配合线性探针取得了更具鲁棒性的表现（MAE = 9.28 年，结合人口统计学特征后降至 8.22 年）；
  - 单心拍固定点数重采样会破坏主动脉脉搏波传导时间（PWTT）的绝对物理量纲并抹杀心率变异性（HRV）；保留连续 10 秒自然节律波形更有利于保留心血管衰老的多尺度信息。

---

### PPG-Bench

Mannas AI · arXiv 2026 · [论文](https://arxiv.org/abs/2610.13435) · [代码](https://github.com/mannas-ai/ppg-bench)

针对以往各个 PPG 基础模型采用各自碎片化的私有/公有测试集、不同的数据清洗过滤标准与微调协议导致无法公允对比的痛点，该工作构建了首个标准化、严谨防数据泄露的脉搏基础模型大一统评测套件。

- **评测体系与覆盖面**：
  - **参评基模**：AnyPPG、PaPaGei-S、PaPaGei-P、PulsePPG 与 SigmaPPG；
  - **评测规模**：跨越 9 个公开数据集（PPG-DaLiA、WildPPG、DREAMT、WESAD、ECSMP、PPG-BP、TROIKA、BAMI-1/2、Apple Heart Study），涵盖 26 项下游任务（心率回归、体动识别、睡眠分期、压力情绪、抑郁筛查、年龄/血压估计及身份识别）；
  - **4 级递进适配协议**：线性探测（Linear Probing）、随机森林（Random Forest）、冻结 MLP 探针、端到端全量微调（End-to-End Finetuning）；
  - **严苛对照组**：随机/常数基线、无训练参数的脉搏率变异性特征集（PRV）、通用时序大模型（MOMENT）。
- **核心结论（为基模“挤水分”）**：
  - 在全部 68 组“任务-适配策略”配置中，有 35.3% 的组合相比最简单的随机/常数基线未展现出统计学优势；
  - 在高达 83.8%（57/68）的任务组合中，无任何一个专用 PPG 基础模型能同时战胜“确定性 PRV 特征”与“通用时序大模型 MOMENT”；
  - 下游探测头容量越复杂，在小规模临床稀疏数据集上过拟合风险越显著，呼吁未来脉搏表征学习必须始终引入强物理先验的对照组。

---

## 4. 空间运动与跨专题基模索引

### LIMU-BERT

南洋理工大学 (NTU) · ACM SenSys 2021 · [代码](https://github.com/dapowan/LIMU-BERT-Public)

- **定位**：面向手腕三轴加速度计与陀螺仪的空间 IMU 自监督表征模型；
- **机制**：借鉴 BERT 的掩码传感器建模（Masked Sensor Modeling），将三轴运动时序分片掩码后重建，剥离个体特定步态特征，提炼通用的运动动力学表征；在人体活动识别（HAR）中表现良好。

### Mantis

华为诺亚方舟实验室 / Inria · ICML 2026 · [代码](https://github.com/vfeofanov/mantis)

华为诺亚方舟实验室研发的 8M 轻量级时序判别基础模型，采用自适应分块 TGU 与轻量 ViT-1D 架构，在智能手环端侧活动识别（HAR）与事件分类中具备高吞吐低延迟优势。完整架构剖析、设计原理与实验对比请参阅专题文档：[通用时间序列基础模型 (General TSFM)](general_tsfm.md#mantis)。

### SleepFM

斯坦福大学医学院 / DTU · Nature Medicine 2024 / arXiv 2026 · [代码](https://github.com/zou-group/sleepfm-v2-public)

Stanford Medicine 研发的多模态生理基座家族（涵盖 200 余万小时临床多导睡眠监测 PSG），通过留一对比学习（LOO-CL）与掩码局部重构（MAE）捕获脑心肺多器官生物物理耦合。其提出的手腕光电脉搏波（PPG）零样本迁移与手腕三轴加速度计（ACC）物理生理代理通道桥，为智能手表算法提供了重要范式参考。详细数学原理、架构消融对照及微觉醒机制问答请参阅专题文档：[睡眠与心肺多模态基础模型](sleep_models.md#sleepfm)。

---

## 5. 大规模开源生态与时序语言多模态基座

### OpenMHC

斯坦福大学医学院 (Euan Ashley 实验室) / 帝国理工学院 · arXiv 2026 · [论文](https://arxiv.org/abs/2607.16235) · [代码](https://github.com/AshleyLab/OpenMHC) · [数据集](https://huggingface.co/MyHeartCounts)

![OpenMHC 数据集画像](../figures/openmhc/overview.png)

图源：OpenMHC 论文 Fig. 1（arXiv:2607.16235）

OpenMHC（Open MyHeartCounts）是向科研界开放的大规模真实世界可穿戴基础模型基准数据集与开源实现套件。

- **数据规模**：涵盖 11,894 名受试者长达 13 年的纵向记录，包含超过 6,700 万小时的分钟级连续多传感器时序；
- **全景传感器与表型关联**：覆盖 19 个连续生理与运动通道（步数、心率、HRV、睡眠分期、运动能耗等），并与 169 维健康问卷、生活方式及 HealthKit 长期档案相对应；
- **统一评测协议**：规范化了三大基准赛道：32 项下游预测任务（Track 1）、生成式缺失值插补（Track 2A）以及自回归未来预测（Track 2B）；
- **开源复现**：在统一基准下开源复现了 Apple WBM（Wearable Base Model）与 LSM-2，为穿戴领域的模型对标提供了统一环境。

---

### PPG-LM

Emory University / UCLA / Google / Stanford · arXiv 2026 · [论文](https://arxiv.org/abs/2609.33516)

![PPG-LM 架构设计与流程](../figures/ppg-lm/overview.png)

图源：PPG-LM 论文 Fig. 1（arXiv:2609.33516）

PPG-LM 是通过多层级临床自然语言弱监督训练的脉搏波-语言基础模型家族。

#### 1. 核心动机

- 传统脉搏模型多局限于单一波形表征与固定分类头，难以进行跨模态语义检索、健康问答或报告生成；同时临床监护具有多时间尺度特性（30 秒局部波形形态、分钟级给药反应、数小时至数天的整场就诊演变）。

#### 2. 两阶段多尺度架构

- **Stage 1: 片段-语言预训练（30 秒波形）**：
  - 波形端采用轻量 1D 卷积 + 6 层 Transformer（12.7M 参数）；
  - 文本端采用从零初始化的 8 层因果 Transformer（避免通用大模型的数值幻觉），利用对齐损失与条件自回归 Captioning 损失联合预训练；
- **Stage 2: 宏观时空与长程时序聚合**：
  - **连续时间旋转位置编码（Continuous-time RoPE）**：将旋转角度直接绑定在真实物理时间差上，原生支持野外佩戴中的非规则断流与间断采样；
  - **Perceiver 时间探针**：利用预设时间锚点的潜变量 Query 聚合整场长程波形。

#### 3. 评测表现

- 在 MC-MED 与 MIMIC 队列的 32 项临床预测任务中展现了表征优势，并支持高保真双向跨模态检索（波形检索病历 / 病历检索波形）。

---

### SensorLM

Google Research / Google DeepMind · NeurIPS 2025 · [论文](https://arxiv.org/abs/2506.09108) · [代码](https://github.com/Google-Health/consumer-health-research/tree/main/sensorlm)

![SensorLM 架构图](../figures/sensorlm/overview.png)

图源：SensorLM 论文 Fig. 1（NeurIPS 2025）

SensorLM 是在近 6,000 万小时真实世界生活穿戴数据上训练的传感器-语言多模态基础模型。

#### 1. ViT-CoCa 统一框架

- **传感器编码器**：将 PPG、三轴 ACC、EDA、皮肤温度与气压计在时间和通道域划分为二维 Patch，经投影输入 Transformer 编码器；
- **文本编码与多模态解码**：结合文本编码器与因果多模态解码器，通过 CoCa（对比学习 + 自回归 Captioning）联合优化，打通波形与自然语言语义对齐。

#### 2. 自底向上三层自动标注语料管线

- **统计层（Statistical）**：解算均值、方差、能量谱密度与局部振幅差；
- **结构层（Structural）**：分析波形周期性、频域主频、波峰间期（IBI）趋势与多传感器时延互相关；
- **语义层（Semantic）**：整合高阶活动事件（如跑步、爬楼、睡眠、压力状态）生成结构化健康描述。

---

## 6. 核心基础模型横向对比矩阵

为清晰展现各模型的技术特点，将代表性模型分为两组进行横向对比：

### 表 1：脉搏波与心血管多模态基础模型

| 维度 / 特征 | PaPaGei | Pulse-PPG | AnyPPG | PPGFlowECG | UniCardio | HiMAE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **机构团队** | Nokia Bell Labs | UIUC / 孟菲斯大学 | 北京大学数字健康实验室 | 北京大学数字健康实验室 | 清华大学 / 安贞医院 | Samsung Research / UCLA |
| **核心模态** | 单通道 PPG | 腕部高噪 PPG | PPG (ECG 引导) | PPG → 诊断级 ECG | PPG + ECG + 动脉血压 (BP) | 单通道手腕绿光 PPG |
| **预训练规模** | 5.7 万小时 (10 个公开集) | 5.5 万小时 (野外 100 天) | 10+ 万小时配对数据 | 千万对 (MC-MED 11.8 万人) | 339 小时多模态时序 | 8 万小时 (4.7 万人) |
| **网络骨干** | ResNet1D-MoE | 12 层 1D-ResNet | 双分支 1D-ResNet | CardioAlign + 整流流 | 多模态扩散 DiT | 5 级 1D 卷积 U-Net |
| **自监督机制** | 形态引导对比学习 | 相对对比学习 (RelCon) | 跨模态 InfoNCE 对比 | 潜空间分布对齐 + 整流流 | 条件扩散去噪 + 持续学习 | 混合分块掩码自编码 (80% MAE) |
| **部署延迟** | 端侧轻量 (约 1.5M) | 端侧中等 (约 2M) | 单分支推理 (约 5.85M) | 1～4 步 ODE 采样 | 增量参数约 0.3M/模态 | 手表 CPU 约 0.99 ms (0.31M) |
| **开源状态** | 代码与权重开源 (Zenodo) | 代码与权重开源 (Zenodo) | 代码开源 (GitHub) | 代码开源 (GitHub) | 论文公开 / 代码维护中 | 代码开源 (GitHub) |

### 表 2：全传感器与时序语言多模态基础模型

| 维度 / 特征 | SensorFM | SensorLM | OpenMHC | PPG-LM | PPG-Distill | LIMU-BERT |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **机构团队** | Google / Fitbit Health AI | Google Research / DeepMind | 斯坦福大学医学院 | Emory / UCLA / Google | Emory University | 南洋理工大学 (NTU) |
| **核心模态** | PPG+ACC+EDA+Temp | 5 维传感器 (含气压计) | 19 通道穿戴 + 169 维变量 | PPG + 结构化临床文本 | 腕部 PPG | 三轴 ACC + 陀螺仪 |
| **预训练规模** | 1 万亿分钟 (500 万人) | 5,970 万小时 (10.3 万人) | 6,700+ 万小时 (1.2 万人) | 73,700 小时 (MC-MED) | 多任务穿戴语料 | 公开 IMU 运动数据集 |
| **网络骨干** | Patch Transformer | ViT-2D + CoCa 解码器 | WBM / LSM-2 统一复现 | 1D-CNN + 6 层 Transformer | 紧凑 1D 学生网络 | 1D-CNN + Transformer |
| **核心机制** | 掩码建模 + 跨模态填补 | CoCa 对比 + 自回归生成 | 多任务预测 + 插补基准 | 对称对比 + 连续物理 RoPE | 三级多尺度知识蒸馏 | 掩码传感器建模 (MSM) |
| **抗伪影机制** | 超大规模数据 Scaling | 多传感器交叉互补 | 真实生活全周期滤波 | 严格 SQI + Content-Key 过滤 | 结构化节律形态迁移 | 频域滤波 + 步态去噪 |
| **开源状态** | 闭源 (仅论文) | 代码开源 (GitHub) | 数据集与代码开源 (HF) | 论文公开 (arXiv) | 代码开源 (GitHub) | 代码开源 (GitHub) |

---

## 7. 对可穿戴多模态大模型的工程架构启示

结合前沿模型的技术演进，在智能手表端构建多模态生理健康系统具有以下工程参考价值：

1. **物理时间等长分块（Physical-Time Patch Tokenization）**：
   - 避免采用无先验的重采样平滑采样率差异。设定固定物理时长窗口（如 0.25 秒），各模态通过专用投影层映射为等长的 Patch Token 序列；
2. **多传感器互补消除运动伪影**：
   - 将加速度计（ACC）作为运动参考流，通过交叉注意力或滤波代理桥抑制 PPG 中由手臂摆动引起的同频步频谐波；
3. **分层时间尺度解耦**：
   - 借鉴 HiMAE 的设计，采用轻量级 1D 卷积作为秒级底层生理 Tokenizer（提取微观心拍形态），再在外层级联状态空间模型或 Transformer 建模昼夜节律；
4. **利用连续物理时间 RoPE 适应非规则佩戴**：
   - 佩戴中由于充电、沐浴等操作存在天然断流。借鉴 PPG-LM 的连续时间 RoPE，将旋转角度绑定在真实物理分钟差上，可在无需人工插补的前提下处理非均匀采样；
5. **弱监督分层数据构建**：
   - 借鉴 SensorLM 与 PPG-LM 的三层自动打标流水线（统计级 → 结构级 → 语义级），利用成熟数字信号处理算法（如 `pyPPG`、`NeuroKit2`）在未标注穿戴数据上低成本构建跨模态微调语料；
6. **判别与生成解耦**：
   - 端侧日常活动分类与心律失常筛查可优先采用轻量判别骨干（如 Mantis、HiMAE）；高保真波形重构、插补或虚拟心电生成则可采用整流流或扩散架构。

---

## 8. 核心文献与资源汇总

| 项目 / 论文名称 | 主要机构 | 发表情况 | 核心定位与资源 |
| :--- | :--- | :--- | :--- |
| **SensorFM** | Google / Fitbit Health AI | arXiv 2026 | 1 万亿分钟消费级穿戴多模态基座 · [论文](https://arxiv.org/abs/2605.22759) |
| **PaPaGei** | Nokia Bell Labs | ICLR 2025 | 开源形态感知脉搏波基础模型 · [论文](https://arxiv.org/abs/2410.20542) · [代码](https://github.com/Nokia-Bell-Labs/papagei-foundation-model) |
| **Pulse-PPG** | 孟菲斯大学 / UIUC | UbiComp 2025 | 野外非受限生活场景脉搏基座 · [代码](https://github.com/maxxu05/pulseppg) |
| **HiMAE** | Samsung Research / UCLA | ICLR 2026 | 手表端侧亚毫秒级分层掩码自编码基座 · [论文](https://arxiv.org/abs/2510.25785) · [代码](https://github.com/Simonlee711/HiMAE) |
| **UniCardio** | 清华大学 / 安贞医院 | Nat Mach Intell 2025/2026 | 统一心血管多模态扩散基础模型 · [论文](https://doi.org/10.1038/s42256-025-01147-y) · [代码](https://github.com/chen-zehua/UniCardio) |
| **PPGFlowECG** | 北京大学健康医疗大数据国家研究院 | arXiv 2025-2026 | 潜空间整流流 PPG 转 ECG 跨模态生成 · [论文](https://arxiv.org/abs/2502.14856) · [代码](https://github.com/PKUDigitalHealth/PPGFlowECG) |
| **AnyPPG** | 北京大学健康医疗大数据国家研究院 | ACM SIGKDD 2026 | 心电引导预训练的光电多器官基座大模型 · [论文](https://arxiv.org/abs/2511.01747) · [代码](https://github.com/Ngk03/AnyPPG) |
| **PPG-Distill** | Emory University | 2025 | 脉搏波基础模型多层次端侧知识蒸馏 · [代码](https://github.com/LingFengGold/PPG-Distill) |
| **OpenMHC** | 斯坦福大学医学院 | arXiv 2026 | 6,700 万小时真实世界穿戴基础模型基准与生态 · [论文](https://arxiv.org/abs/2607.16235) · [代码](https://github.com/AshleyLab/OpenMHC) |
| **PPG-LM** | Emory / UCLA / Google | arXiv 2026 | 多时间尺度临床对齐脉搏-语言基础模型 · [论文](https://arxiv.org/abs/2609.33516) |
| **SensorLM** | Google Research / DeepMind | NeurIPS 2025 | 5,970 万小时智能手表传感器-语言多模态基座 · [论文](https://arxiv.org/abs/2506.09108) · [代码](https://github.com/Google-Health/consumer-health-research/tree/main/sensorlm) |
| **LIMU-BERT** | 南洋理工大学 (NTU) | ACM SenSys 2021 | 手腕三轴 IMU 空间运动表征模型 · [代码](https://github.com/dapowan/LIMU-BERT-Public) |
| **PPG-Bench** | Mannas AI | arXiv 2026 | 9 数据集 26 任务脉搏大模型大一统评测套件 · [论文](https://arxiv.org/abs/2610.13435) · [代码](https://github.com/mannas-ai/ppg-bench) |
| **PPG 生物年龄横评** | Nabil Brag | arXiv 2026 | 脉搏基础模型血管生物年龄跨域评测基准 · [论文](https://arxiv.org/abs/2603.14030) · [代码](https://github.com/Misterbra/ppg-age-benchmark) |

---

> **欢迎交流与共建**  
> 本文由开源知识库项目 [Wearable-Physiological-Foundation-Models](https://github.com/tujy859/Wearable-Physiological-Foundation-Models) 整理策划。该项目持续追踪全球在智能手表、连续血糖监测（CGM）以及多模态生理健康基础模型领域的最新突破。如果您对可穿戴生理计算、时序健康基模与数字生物标志物有任何想法，欢迎在仓库提 Issue 交流！
