# 参与贡献

欢迎通过 Issue 或 Pull Request 补充与修正本仓库收录的论文、开源模型权重及基准数据集。

## 收录规范

提交新的论文或基础模型时，请遵循 [写作与格式规范](docs/STYLE.md)，并在 Pull Request 中提供以下信息：

1. **基本元数据**：论文标题、发表年份、收录会议或期刊（或 arXiv 编号）、主要研究机构。
2. **官方链接**：arXiv 或 DOI 论文链接、官方代码库链接（GitHub / GitLab）、预训练权重链接（Hugging Face / Zenodo）。
3. **方法与结果摘要**：说明其针对可穿戴或生理时序信号的具体设计（如异构采样率对齐、运动伪影抑制、跨模态对齐等），并给出可查证的实验指标。
4. **结构化索引更新**：
   - 在 `data/models.yaml`（或 `data/datasets.yaml`）中添加对应条目，`added` 字段填写收录日期（`YYYY-MM-DD`）；
   - 在 `docs/models/`（或 `docs/datasets/`）对应专题文档中补充小节与论文架构图（放置在 `docs/figures/<id>/`）；
   - 运行 `python scripts/build_tables.py` 更新 README 索引表，并运行 `python scripts/lint_md.py` 确保无格式与文风错误。

## 注意事项

- 请勿在 Git 仓库中提交模型权重文件（`.pth`、`.bin`、`.safetensors` 等）、原始数据集文件或论文 PDF 原件。
- 论文插图请标注来源（如 `图源：PaPaGei 论文 Fig. 1（arXiv:2410.17260）`），单张图片控制在 500 KB 以内。
