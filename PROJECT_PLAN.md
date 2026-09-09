# Wearable Physiological Foundation Models 项目建设与学习推进计划

> **定位**：聚焦于**智能手表、CGM 等可穿戴生理信号基础模型**（兼顾通用时序基础模型）的系统性开源调研、深度技术拆解与实战评测知识库。
> **双重价值**：对内作为个人系统性学习、技术沉淀与实验追踪笔记；对外作为 GitHub 上垂直、高信噪比的标杆级 Awesome/Survey 项目。

---

## 一、整体架构规划（Hub-and-Spoke 体系）

```
[顶层调研仓: Wearable-Physiological-Foundation-Models] (当前仓)
  ├── 核心门面: README.md (全景分类树、模型横评大表、最新前沿追踪)
  ├── 深度专题: docs/ (模型深度精读、底层技术原理、数据集与评测协议)
  ├── 极简代码: notebooks/ (公开开源权重 10 行开箱即用极简 Demo)
  └── 静态图表: assets/ (高清架构图、分类树、雷达图)
         │
         ├── 链接到子实验仓 1: [Watch_LSM](https://github.com/tujy859/Watch_LSM) (手表端到端自建模型、多模态融合、真实野外伪影实证)
         └── 链接到子实验仓 2: [CGM_FM](https://github.com/tujy859/CGM_FM) (CGM 血糖模型对比、GlucoFM/GluFormer 复现分析)
```

---

## 二、分阶段建设路线图 (Roadmap)

### 阶段 1：知识库骨架与全景大表建立（已完成）
- [x] 创建顶层独立仓库目录与轻量化规范配置（`.gitignore`）
- [x] 确立仓库定位、内容边界与组织规范
- [x] 编写顶层 `README.md` 门面：
  - [x] 撰写引言、研究动机与穿戴生理信号的特殊挑战
  - [x] 绘制并组织全景分类脉络（Taxonomy Tree）
  - [x] 制作【全景基础模型横向对比大表】（涵盖通用时序、智能手表、CGM、睡眠多模态）
  - [x] 建立公开数据集与评测基准索引表
- [x] 编写开源协作与收录规范（`CONTRIBUTING.md`）

### 阶段 2：资产聚合与专题精读整理（核心文档就绪）
- [x] **手表专题整理**：将 `Watch_LSM` 中的 7 篇核心模型分析（SensorFM, PaPaGei, Pulse-PPG, xMAE, HiMAE, SleepFM, LIMU-BERT, Mantis）格式化整理至 `docs/models/smartwatch_models.md`
- [x] **CGM 专题整理**：将 `cgm_fm` 中的成果（GlucoFM, GluFormer, CGMformer, CGM-LSM 对比及实证设计空间）整理至 `docs/models/cgm_models.md`
- [x] **通用时序模型专题**：整理 TimesFM, Chronos, MOMENT, MANTIS, MOIRAI 的技术演变逻辑与域鸿沟至 `docs/models/general_tsfm.md`
- [x] **睡眠心肺专题整理**：解构 SleepFM、SleepMaMi、U-Sleep 跨模态耦合与 NSRR 数据集至 `docs/models/sleep_models.md`
- [x] **公开数据集专题**：补全智能手表与连续血糖公开数据集的获取与授权指引至 `docs/datasets/wearable_datasets.md`
- [ ] **核心技术纵向对比**：撰写《穿戴时序特定归纳偏置：多采样率对齐、频域碰撞消除与潜空间预测》至 `docs/principles/`

### 阶段 3：实操体验增强（极简 Notebooks）
- [ ] 编写 `notebooks/01_timesfm_quickstart.ipynb`：演示从 Hugging Face 加载 TimesFM 进行穿戴时序预测
- [ ] 编写 `notebooks/02_papagei_ppg_probing.ipynb`：演示加载 PaPaGei 官方权重提取 1D-ResNet 生理表征
- [ ] 编写 `notebooks/03_cgm_standard_pipeline.ipynb`：演示 CGM 不规则 5 分钟网格到昼夜 288 点的标准处理流水线

### 阶段 4：GitHub 开源发布与社区维护
- [ ] 补充精美的 Banner 封面图与 Mermaid 架构图（存入 `assets/`）
- [ ] 在 GitHub 创建公共仓库并发布 v1.0 版本
- [ ] 在相关领域社区（arXiv 追踪、PaperReading、知乎、推特、Awesome 列表）分享并持续收录最新顶会进展（NeurIPS / ICLR / KDD / UbiComp / Nature 等）

---

## 三、各模块内容填写规范

1. **论文收录标准**：
   - 必须提供官方论文链接（arXiv/DOI/Publisher）与发表年份、会议/期刊名称；
   - 必须注明代码是否开源、模型权重是否公开（提供 Hugging Face/Zenodo 链接）；
   - 用 2～3 句精炼语言指出其“核心创新点”与“针对穿戴/生理信号的特定设计”。
2. **严禁在 Git 历史中提交大文件**：
   - 权重一律使用 Hugging Face / Zenodo 链接；
   - 数据集一律使用 PhysioNet / Kaggle / 官方申请入口链接；
   - 论文 PDF 原件一律不上传，本地如有需要放入 `papers/`（已被 gitignore）。
