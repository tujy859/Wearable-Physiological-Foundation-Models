# 连续血糖监测 (CGM) 基础模型深度解析 (Continuous Glucose Monitoring Foundation Models)

> 连续血糖监测（CGM）通过皮下微创传感器每 5 分钟连续测定组织间液葡萄糖浓度，是代谢与内分泌疾病数字化管理的核心手段。本专题深入对比以血糖时序为核心表征对象的四大顶会/顶刊基础模型，并结合作者在因子实验中的实证发现，系统总结 CGM 表征学习的设计空间与范式分野。

---

## 📑 目录

- [1. CGM 信号生理特性与时序表征痛点](#1-cgm-信号生理特性与时序表征痛点)
- [2. Google GlucoFM (2026) —— 轻量双流 JEPA 潜空间预测基座](#2-google-glucofm-2026--轻量双流-jepa-潜空间预测基座)
- [3. Pheno.AI / NVIDIA GluFormer (Nature 2026) —— 长程自回归生成大模型](#3-phenoai--nvidia-gluformer-nature-2026--长程自回归生成大模型)
- [4. 中科院 / 上海六院 CGMformer (NSR 2025) —— 大规模真实世界掩码重构基座](#4-中科院--上海六院-cgmformer-nsr-2025--大规模真实世界掩码重构基座)
- [5. JHU CDHAI CGM-LSM (2024) —— 短程预测语言模型](#5-jhu-cdhai-cgm-lsm-2024--短程预测语言模型)
- [6. 四大 CGM 基础模型横向对比全景表](#6-四大-cgm-基础模型横向对比全景表)
- [7. CGM 基础模型设计空间实证洞察 (Empirical Design Space)](#7-cgm-基础模型设计空间实证洞察-empirical-design-space)

---

## 1. CGM 信号生理特性与时序表征痛点

相比高频波形信号（如 100Hz PPG），CGM 呈现出低采样率、强生物节律及独特的生化动力学特性：

1. **昼夜节律与 288 点固定时间网格**:
   - 绝大多数商用 CGM（如 Dexcom、Abbott Libre、Medtronic）采样间隔为 5 分钟（部分型号如旧版 Libre 为 15 分钟）。
   - 单日完整时程恰好包含 $24 \times 60 / 5 = 288$ 个读数点。血糖水平受人体皮质醇分泌、胰岛素敏感性周期及睡眠节律支配，具有极其鲜明的 24 小时昼夜波动。
2. **皮下组织间液生化延迟与非线性突变**:
   - 传感器测定的是组织间隙液葡萄糖，相比静脉/毛细血管全血存在 5～15 分钟的生理滞后。
   - 信号由两部分叠加：**慢速生理基线漂移**（空腹血糖稳态、长效胰岛素基础吸收）与**高频餐后冲击/外源干扰**（进食碳水化合物峰值、胰岛素大剂量注射、夜间受压骤降 Compression Artifact）。
3. **严重的观测缺失（Observational Missingness）**:
   - 传感器脱落、蓝牙断连、探针初始化校准会导致时序碎片化。传统时序模型习惯使用线性插值或多项式填补，但在 CGM 领域，稠密插值会引入虚假的生理先验（消融证实插值明显劣化下游表征迁移质量）。

---

## 2. Google GlucoFM (2026) —— 轻量双流 JEPA 潜空间预测基座

- **论文**: *GlucoFM: A Dual-Stream Foundation Model for Continuous Glucose Monitoring* (arXiv:2605.30865)
- **研发团队**: Google Research & 澳大利亚新南威尔士大学 (UNSW)
- **发表时间**: 2026 年 5 月
- **开源状态**: 官方代码已在论文中承诺开放；基准测试论文配套 *GlucoFM-Bench* (arXiv:2606.06881)

### 2.1 核心创新：双流分解 (Dual-Stream Decomposition)
GlucoFM 首次将血糖动力学解耦为两个并行的生理物理流：
1. **状态流 (State Stream - 慢生理稳态)**:
   - 引入**可学习因果高斯滤波器 (Learnable Causal Gaussian Filter)** 提取低频基线。
   - 滤波带宽 $\sigma \in [2, 12]$ 网格步（约 10～60 分钟），通过 sigmoid 重参数化实现反向传播梯度端到端联合学习。
2. **事件流 (Event Stream - 瞬态偏移)**:
   - 残差项定义为：$\text{Event}(t) = \text{Signal}(t) - \text{State}(t)$。
   - 专门捕捉餐后剧烈血糖脉冲、剧烈运动下降及传感器受压跌落等突变成分。

### 2.2 24h 时间网格与观测掩码显式保留
- 不规则记录对齐至统一的 288 点昼夜网格，**坚决分离对齐与插值**。
- 保留二值观测掩码 $M \in \{0, 1\}$，$\le 1\text{h}$ 的短时缺失在网格内标注 $M=0$ 并贯通至损失函数；$> 1\text{h}$ 的长时间断连直接进行物理切段。

### 2.3 双预训练目标 (JEPA 式潜空间表征预测)
GlucoFM 摒弃了重建像素/原值的 MAE 路线，转向非生成式的联合嵌入预测架构 (JEPA)：
1. **MCR (Masked Contextual Latent Prediction)**:
   - 随机掩码 50%～60% 的 Patch，利用双向编码器由可见 Patch 推测被掩码 Patch 在 EMA 目标编码器（动量 0.997）中的潜空间表征向量。
   - 采用 Smooth L1 损失，并根据 Patch 内部的观测点密度进行加权衰减。
2. **TD (Temporal Dynamics Residual Prediction)**:
   - 引入时序动力学残差转移头：$S_{\tau+1} = S_\tau + g(S_\tau, E_\tau, \tau)$。
   - 迫使模型在隐空间建模血糖随时间的物理递推演变过程。

### 2.4 极简规模以小博大
- **仅 0.72M 可训练参数**（3 层 Transformer、4 头、特征维度 $D=128$、FFN=256）。
- 在单张 H100 GPU 上即可快速完成全量预训练。
- 在 7 项临床代谢下游任务（糖尿病风险、胰岛素抵抗 HOMA-IR、$\beta$ 细胞功能障碍、Glucotype 分型、高脂血症、低血糖风险、肥胖）中，平均 PR-AUC 相比最强基线提升 **+4.1 点**，参数量仅为百兆大模型的 1/200。

---

## 3. Pheno.AI / NVIDIA GluFormer (Nature 2026) —— 长程自回归生成大模型

- **论文**: *A Foundation Model for Continuous Glucose Monitoring and Metabolic Health* (Nature 2026 / arXiv:2408.11876)
- **研发团队**: Pheno.AI, 魏茨曼科学研究所 (Weizmann Institute), NVIDIA
- **开源代码**: [GitHub: Guylu/GluFormer](https://github.com/Guylu/GluFormer)（包含上海队列特征评估脚本；预训练权重受制于 HPP 伦理限制未完全公开）

### 核心设计与自回归路线
- **分箱离散化 (Quantization Tokenization)**:
  - 将连续血糖值映射到 460 个离散数值分箱中，将时序问题转化为语言建模（Next-Token Prediction）。
- **超长上下文能力**:
  - 拥有 **1200 个 token（对应约 12.5 天连续监测）** 的超长上下文窗口。
  - 基于 GPT 式单向因果 Transformer 架构，参数量达到 **135M**（16 层、特征维度 1024）。
- **海量纵向预训练语料**:
  - 基于人类表型组计划 (HPP) 10,812 名健康及前驱糖尿病受试者的逾 1000 万条连续血糖读数。

### 临床长程预后突破
- 得益于因果自回归生成能力，GluFormer 在**长程心血管代谢结局预测**上表现优异：
  - 能够跨越 2～12 年时间跨度，仅凭入组基线 CGM 预测受试者未来发生 2 型糖尿病、冠心病、视网膜病变等重大远期代谢风险。
  - 支持多模态饮食营养成分输入，模拟进食特定餐食后的餐后血糖响应曲线。

---

## 4. 中科院 / 上海六院 CGMformer (NSR 2025) —— 大规模真实世界掩码重构基座

- **论文**: *CGMformer: A Foundation Model for Continuous Glucose Monitoring in Real-World Clinical Practice* (National Science Review, 2025)
- **研发团队**: 中国科学院微系统所、上海交通大学附属第六人民医院
- **开源代码**: [GitHub: YurunLu/CGMformer](https://github.com/YurunLu/CGMformer)（提供官方模型 Checkpoint 与特征提取代码）

### 掩码语言建模 (BERT-style MLM)
- 采用双向 Transformer 编码器结构（参数量 0.85M～10M 多种规格）。
- 将单日 288 点血糖读数映射为 260 个离散 Token，引入 PAD 与 MASK 特殊标记。
- 在预训练阶段随机掩蔽部分血糖点，由上下文双向注意力重构掩码真实值。

### 规模化真实世界验证
- 预训练汇集了中国人群 **58,847 名受试者、131 万天** 的真实世界 CGM 监测记录。
- 重点服务于临床多任务闭环：院内血糖波动评估、糖尿病视网膜病变 (DR) 筛查、个性化口服降糖药与胰岛素处方推荐。

---

## 5. JHU CDHAI CGM-LSM (2024) —— 短程预测语言模型

- **论文**: *CGM-LSM: A Language Model-Based Foundation Model for Continuous Glucose Monitoring* (arXiv:2412.09727)
- **研发团队**: 约翰霍普金斯大学健康人工智能中心 (JHU CDHAI)
- **开源资源**: [GitHub: JHU-CDHAI/cgmlsm](https://github.com/JHU-CDHAI/cgmlsm)

### 核心定位与设计
- 基于 GPT-2 Small 架构（参数量约 124M），将血糖连续值划分为 400 个词元。
- 输入上下文设计为 26 小时（24 小时历史上下文 + 未来 2 小时自回归预测窗口）。
- **下游专长**: 在 OhioT1DM 基准上针对未来 30 分钟、60 分钟及 120 分钟的瞬时短程血糖预测，均方根误差 (RMSE) 相比传统 RNN/ARIMA 降低达 48.51%。

---

## 6. 四大 CGM 基础模型横向对比全景表

| 维度 | **Google GlucoFM** | **Pheno.AI GluFormer** | **中科院 CGMformer** | **JHU CGM-LSM** |
| :--- | :--- | :--- | :--- | :--- |
| **发表载体** | arXiv 2026 (Google Research) | **Nature 2026** (Pheno.AI) | NSR 2025 (中科院/上海六院) | arXiv 2024 (JHU CDHAI) |
| **学习范式** | **JEPA 潜空间预测 (非生成)** | GPT 式因果自回归 (生成) | BERT 式 MLM 掩码重构 | GPT-2 式因果自回归 |
| **词元化方式** | 连续 Conv1d + 双流分解 | 460 Bins 离散化 | 260 Bins 离散化 | 400 Bins 离散化 |
| **参数量** | **0.72M (轻量极致)** | \~135M | 0.85M～10M | \~124M |
| **预训练规模** | 477 人 (10.9 万小时) | 10,812 人 (>1000 万读数) | 58,847 人 (131 万天) | 592 人 (1600 万读数) |
| **时序上下文** | 单日 24 小时 (288 点) | **1200 点 (\~12.5 天)** | 单日 24 小时 (288 点) | 26 小时 (24h 历史 + 2h 预测)|
| **缺失值策略** | **观测掩码显式保留 (拒绝插值)** | 线性插值 | PAD Token 填充 | 连续完整性过滤 |
| **核心优势任务**| 临床代谢综合表型判别探针 | 长程结局 (2～12年) 与饮食响应 | 临床筛查、分型、并发症预测 | 30～120 分钟超短程连续预测 |
| **代码与权重** | 🟡 代码承诺开源 | 🟢 官方代码 / 🔴 权重受限 | 🟢 官方代码 + 权重公开 | 🟢 官方代码 / 🔴 数据私有 |

---

## 7. CGM 基础模型设计空间实证洞察 (Empirical Design Space)

在本项目（[`CGM_FM`](https://github.com/tujy859/CGM_FM) 子工程）进行的 3 目标（JEPA vs 重建 vs 因果）$\times$ 3 架构（普通 Transformer vs 双流 GlucoFM vs 纯卷积）全因子矩阵（30 组预训练、三轨综合评测）中，我们获得了以下核心结论：

### 洞察 1：目标函数决定表征天花板（Q1）
- **连续值掩码重建 (Recon) 在高阶下游最差**: 单纯迫使 Decoder 像素级拟合数值，模型会过度学习局部的“数值平滑先验”，而丢失全局生理突变特征。
- **JEPA 潜空间预测与因果预测最优**: JEPA (MCR + TD) 迫使表征在高维隐空间捕捉动态变动趋势，在 Hall Glucotype 严重变异度分型中 AUROC 达到 **0.984**（相比随机初始化 0.841 提升显著）。

### 洞察 2：双流分解的必要性与增益（Q2）
- 无论是在 Transformer 还是 CNN 骨干中，引入因果高斯滤波将慢生理状态流（$\sigma \in [2, 12]$）与快事件流剥离，能为插补与预测提供明确的归纳偏置。
- 在未观测时序插补探针中，双流架构将平均绝对误差 (MAE) 从基线 23.8 mg/dL 降低至 **16.6 mg/dL（误差下降 30%）**。

### 洞察 3：观测掩码优于盲目插值
- 血糖记录的不规则稀疏性是物理传感器脱落与低功耗采样的客观反映。显式将二值掩码 $M$ 传递给 Attention 与损失函数，使模型学会在“高置信度观测区”建立时序动力学，比线性插值方案更具跨设备鲁棒性。
