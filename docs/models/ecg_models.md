# 心电图（ECG）与心血管电生理基础模型

本专题基于 2026 年发表于 *npj Digital Medicine* 的综述论文《*Review of open foundation models and datasets for ECG and PPG waveforms*》（MIT 计算生理学实验室 Leo Anthony Celi 等著），梳理多导联心电自监督预训练、跨模态波形转换（PPG 与 ECG）及心血管临床下游任务迁移的技术路线。

## 目录

- [1. 领域综述导读：心电与脉搏波开源基础模型及数据集全景](#1-领域综述导读心电与脉搏波开源基础模型及数据集全景)
- [2. 心电时序的生理学特异性与空间几何先验](#2-心电时序的生理学特异性与空间几何先验)
- [3. 经典心电开源基础模型与核心预训练范式](#3-经典心电开源基础模型与核心预训练范式)
  - [ECG-FM](#ecg-fm)
  - [ST-MEM](#st-mem)
  - [HeartBEiT](#heartbeit)
  - [CLOCS](#clocs)
- [4. 消费级脉搏（PPG）与临床心电（ECG）的跨模态桥梁](#4-消费级脉搏ppg与临床心电ecg的跨模态桥梁)
- [5. 下游心血管临床任务迁移全景](#5-下游心血管临床任务迁移全景)
- [6. 行业结构性瓶颈与演进路线](#6-行业结构性瓶颈与演进路线)

---

## 1. 领域综述导读：心电与脉搏波开源基础模型及数据集全景

- **论文题目**：*Review of open foundation models and datasets for ECG and PPG waveforms*
- **发表期刊**：*npj Digital Medicine*（2026 年）
- **DOI 链接**：[10.1038/s41746-026-03101-7](https://doi.org/10.1038/s41746-026-03101-7)
- **作者团队**：Ahram Han（MIT / 首尔大学）、Takeshi Tohyama（MIT / 国际医疗福祉大学）、Dukyong Yoon（延世大学 / MIT）、Kenneth Paik（MIT）、Brian Gow（MIT）、Leo Anthony Celi（通讯作者，MIT 计算生理学实验室主任 / 哈佛大学）、Hyeonhoon Lee 与 Hyung-Chul Lee（首尔大学医学院）。

### 综述核心要点

1. **学习范式转变**：传统深度学习依赖稀缺的专家逐波标注，且局限于单一特定任务；生理基础模型通过自监督学习（SSL）在大规模无标注波形上预训练通用表征，具备较好的数据利用效率与少样本迁移能力。
2. **开源生态整理**：系统性收录并横向对比了公开的心电与脉搏波基础模型（如 ECG-FM、ST-MEM、HeartBEiT、PaPaGei 等）及其依赖的开源数据集（MIMIC-IV-ECG、PTB-XL、CODE-15%、VitalDB 等）。
3. **行业挑战分析**：指出了当前研究中的五项主要瓶颈，包括预训练数据人群同质性与偏倚、缺乏统一划分标准导致的切分泄漏、真实穿戴高噪伪影、黑盒表征缺乏电生理波形可解释性、以及移动端与微控制器轻量化部署壁垒。

---

## 2. 心电时序的生理学特异性与空间几何先验

心电信号具有明确的心脏去极化与复极化生物电向量投射规律：

```
单个心动周期波形：
        R
       / \
      /   \
 P   /     \     T
_/\_/       \_/\_/\_
    Q       S
 |◄- PR 间期 -►|
 |◄---- QRS 宽度 ----►|
 |◄------- QT 间期 -------►|
```

1. **时域形态与传导先验**：
   - **P 波**：心房去极化（窦房结激动）；
   - **PR 间期**：房室传导时间（120～200 ms）；
   - **QRS 波群**：左/右心室快速去极化（60～100 ms，宽度过大提示束支传导阻滞或室性异位）；
   - **ST 段与 T 波**：心室复极化过程（抬高或压低是心肌缺血或心梗的重要诊断依据）。
2. **12 导联空间立体投影**：
   - **肢体导联（I, II, III, aVR, aVL, aVF）**：捕获额面（Frontal Plane）的心电综合向量；
   - **胸前导联（V1～V6）**：贴附于胸壁，捕获水平面（Horizontal Plane）从右心室前壁到左心室侧壁的电位演变；
   - **通道约束关系**：如爱因托芬定律（$\text{Lead II} = \text{Lead I} + \text{Lead III}$）以及 $\text{aVR} + \text{aVL} + \text{aVF} = 0$。模型需建模导联间的空间几何相关性，而非仅视为 12 条独立的单变量时序。

---

## 3. 经典心电开源基础模型与核心预训练范式

### ECG-FM

多伦多大学 / Vector Institute / UHN · 2024 · [论文](https://arxiv.org/abs/2407.13576) · [代码](https://github.com/kadenmckeen/ecg-fm) · [权重](https://huggingface.co/kadenmckeen/ecg-fm)

![ECG-FM 架构与预训练流程](../figures/ecg-fm/overview.png)

图源：ECG-FM 论文 Fig. 1（arXiv:2407.13576）

ECG-FM 是面向 12 导联临床心电的开源基础模型，旨在解决心电领域缺乏公开可复现高容量基座模型的问题。

**方法**

- 采用类似 wav2vec 2.0 的结构，由一维卷积特征提取器与多层 Transformer 编码器组成。
- 结合掩码重构（Masked Latent Reconstruction）与对比学习目标，促使模型同时捕捉局部形态细节与跨时间步的上下文表征。
- 在包含 MIMIC-IV-ECG 与加拿大大学健康网络（UHN）私有队列的约 150 万条 12 导联心电记录上进行预训练。

**结果**

- 在射血分数降低（LVEF ≤ 40%）、心律失常多标签分类等任务中进行评估，在线性探测与少样本微调场景下性能优于从头训练的基线网络。
- 提出了基于 MIMIC-IV-ECG 的公开标准化评估基准。

---

### ST-MEM

VUNO Inc. · ICLR 2024 · [论文](https://arxiv.org/abs/2402.04631) · [代码](https://github.com/vuno-inc/ST-MEM)

![ST-MEM 时空掩码架构](../figures/st-mem/overview.png)

图源：ST-MEM 论文 Fig. 1（ICLR 2024）

ST-MEM（Spatio-Temporal Masked Electrocardiogram Modeling）针对心电多导联空间与时间维度的双重相关性，提出了时空解耦的掩码自编码器框架。

**方法**

- 将 12 导联连续波形切块为时空补丁（Spatio-Temporal Patches）。
- **空间导联掩码**：随机掩码部分导联，迫使模型利用剩余导联推断全导联表征，增强对减少导联（如可穿戴单导联或肢体导联）场景的适应性。
- **时间片段掩码**：在保留的导联上对时间片段进行随机掩码。
- 使用非对称编码器-解码器结构，较深的主干编码器仅处理未被掩码的补丁，轻量解码器重构全部补丁。

**结果**

- 在 PTB-XL、Chapman 等公开数据集的多标签心电诊断任务中进行了验证，在少样本和全样本微调设定下均展现出稳定表现。

---

### HeartBEiT

Mila / 蒙特利尔大学 · 2023 · [论文](https://arxiv.org/abs/2306.07920) · [代码](https://github.com/hsd1503/HeartBEiT)

![HeartBEiT 离散掩码架构](../figures/heartbeit/overview.png)

图源：HeartBEiT 论文架构图

HeartBEiT 借鉴计算机视觉中的 BEiT 思想，探索将连续电压波形量化为有限离散标记，用于视觉-时序转换与掩码预测。

**方法**

- 利用离散代码本（Discrete Codebook Tokenizer）将 12 导联心电波形映射为离散词元序列。
- 将连续电压回归任务转换为离散词元的掩码分类预测，降低均方误差（MSE）回归对高频噪声和基线漂移的敏感性。
- 引入多导联图纸视觉上下文对齐，支持对数字化心电图纸图像的处理。

---

### CLOCS

牛津大学 · NeurIPS 2020 · [论文](https://proceedings.neurips.cc/paper/2020/hash/da65ae7d09f25745b85e0546cb7ee3b4-Abstract.html) · [代码](https://github.com/danikiyasseh/CLOCS)

![CLOCS 对比学习框架](../figures/clocs/overview.png)

图源：CLOCS 论文 Fig. 1（NeurIPS 2020）

CLOCS（Contrastive Learning on Clinical Sequences）是将对比自监督学习引入多导联临床时序的代表性先驱工作。

**方法**

- **CMSC（Contrastive Multi-Segment Coding）**：同一患者在相邻时间段内的心电片段作为正样本，不同患者作为负样本，学习个体节律的一致性。
- **CMLC（Contrastive Multi-Lead Coding）**：同一时刻不同空间导联的投影作为正样本，学习空间多导联间的几何协同性。
- **CMSMLC**：联合时空双维度的复合对比学习策略。

**结果**

- 在 PhysioNet 2017 与 Chapman 数据集上验证了在极少标注样本下的表征迁移能力。

---

## 4. 消费级脉搏（PPG）与临床心电（ECG）的跨模态桥梁

### 4.1 为什么要打通 PPG 与 ECG？

| 维度 | 光电容积脉搏波（PPG） | 心电图（ECG） |
| :--- | :--- | :--- |
| **物理本质** | 外周微血管充盈引起的**光吸收阻抗变化** | 心脏激动传导系统产生的**微伏级生物电** |
| **采集形态** | 智能手表、手环、智能戒指（长程无感连续佩戴） | 临床 12 导联湿电极、胸部贴片（短程或佩戴受限） |
| **信息密度** | 反映机械搏动与外周顺应性，缺乏心室除极微细节 | 临床诊断金标准：能诊断心梗部位、传导阻滞与缺血 |
| **转换目标** | **利用日常低成本穿戴 PPG，跨模态重构或合成诊断级 ECG 波形。** |

### 4.2 跨模态生成代表性工作

1. **[PPGFlowECG](smartwatch_models.md#ppgflowecg)**（北京大学，2025/2026）：基于潜空间整流流（Latent Rectified Flow），以直线速度场在 1～4 步 ODE 求解内由手腕 PPG 合成高保真 12 导联心电，并在急诊队列上完成了医生双盲测试。
2. **[UniCardio](smartwatch_models.md#unicardio)**（清华大学 / 安贞医院，Nature MI 2025）：采用统一扩散 Transformer（DiT）框架建模 PPG、ECG 与连续血压，以 0.3M/模态增量参数支持多传感器热插拔与持续学习。
3. **[AnyPPG](smartwatch_models.md#anyppg)**（北京大学，KDD 2026）：利用冻结的高阶心电基模作为教师进行对比预训练，引导手腕脉搏波编码器学习跨模态生理表征，拓展至全身多系统疾病筛查。

详细模型架构与实验对比请查阅对应智能手表专题文档。

---

## 5. 下游心血管临床任务迁移全景

心血管基础模型下游任务通常划分为三个层次：

```
Level 3: 隐匿性与结构性心脏病筛查
  └── 左室射血分数降低 (LVEF ≤ 40%)、主动脉瓣狭窄、肥厚型心肌病、心肌淀粉样变

Level 2: 常见临床节律与形态学诊断
  └── 心房颤动 / 扑动、室性早搏 (PVC)、房性早搏 (PAC)、房室传导阻滞 (AVB)、心肌梗死 (STEMI/NSTEMI)

Level 1: 连续血流动力学与生理参数推断
  └── 连续无袖带血压估计、心率变异性 (HRV)、血管生物年龄估计、ICU 恶化风险预警
```

---

## 6. 行业结构性瓶颈与演进路线

依据 *npj Digital Medicine* 综述讨论，未来研究面临以下几项关键挑战：

1. **预训练人群分布偏倚**：现有开源模型多基于单中心或欧美重症监护群体（如 MIMIC-IV），亚裔、非裔及低收入地区人群代表性不足；穿戴式光学 PPG 对不同肤色群体的透光特性差异仍需校正。
2. **严格的“受试者级”划分标准**：切分训练、验证与测试集必须严格以**独立患者 ID** 为边界，防止同一患者的不同时间切片泄露导致指标虚高；在类别不平衡任务中需同时报告 AUPRC 与 F1 分数。
3. **端侧轻量化推理与边缘部署**：智能手表等可穿戴设备受功耗与电池容量限制，大参数量网络难以常驻端侧内存；需结合结构化剪枝、INT8/INT4 量化与知识蒸馏，降低计算与内存开销。
4. **波形级生理归因与可解释性**：临床决策依赖 P 波、PR 间期、QRS 宽度等物理依据，融合规则约束与可解释归因是进入临床应用的前提。
