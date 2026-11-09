# 参与贡献

欢迎通过 Issue 或 Pull Request 补充与修正本仓库收录的论文、开源模型权重及基准数据集。

## 文档组织与目录边界

本仓库以持续维护的研究专题为主体。目录用于表达稳定的内容职责；新增一次检索、一次更新或一篇文档时，先选择现有归属。

| 位置 | 长期职责与维护方式 |
| :--- | :--- |
| `README.md` | 提供阅读导航与自动生成索引，不复制完整模型分析 |
| `data/models.yaml`、`data/datasets.yaml` | 模型和数据集元数据的统一来源；更新后重新生成索引 |
| `docs/models/` | 按输入表示、模态或研究任务维护模型、方法与评测专题；同一主题在原文件中持续更新，候选与收录判断随专题维护 |
| `docs/datasets/` | 维护数据集、队列、访问条件和使用协议说明；结构化索引与 YAML 同步 |
| `docs/figures/<model-id>/` | 存放对应研究的论文图与说明图，正文标明来源；可复用图片不重复存放 |
| `assets/` | 首页和全仓通用图片；自绘图的声明式源码与渲染结果一起维护；`catalog.html` 与 `article.html` 分别为索引首页和文章模板 |
| `scripts/` | 索引生成、网页构建、绘图与文档检查工具，不存放研究正文 |
| `.github/workflows/` | GitHub Actions 配置；`pages.yml` 在 PR 中检查内容一致性，在推送至 `main` 后检查并发布网页，也支持手动触发 |
| `_site/` | 索引首页、站内文章及引用图片的构建产物；不纳入 Git，由现有元数据、Markdown 与模板生成，不作为研究内容维护位置 |
| `blogs/` | 面向对外阅读的完整技术文章，注明资料截止日期，并链接持续更新的专题 |
| `CONTRIBUTING.md` | 维护收录标准、贡献流程、目录职责与资料维护规则 |
| `docs/STYLE.md` | 维护写作、排版、图表及引用格式，不另存一套收录或目录规则 |
| `AGENTS.md` | 约束 Agent 的项目操作，并引用上述具体规范 |
| `internal_strategy/` | 本地内部讨论、临时检索和更新快照；已被 Git 忽略，不作为公开文档或索引的依赖 |

跨模态综述、方法原理和评测协议可以形成 `docs/models/` 中的主题文档。先在相关专题内补充小节；只有内容形成独立阅读问题时再拆分文件。模型与基准保留一个主要分析位置，其他专题用简要说明和链接引用，避免多份结论逐渐不一致。

当前不设 `docs/surveys/` 与 `docs/principles/`。日期命名的检索更新记录属于本地维护过程；项目原则由已有规范文件承载。目录名称本身不构成新增理由。

网页构建使用独立的 `_site/`，便于仅发布网页所需文件，避免把工作目录和内部记录带入站点；研究内容仍保存在 `data/`、`docs/` 和 `blogs/`。`.github/workflows/` 是 GitHub 识别的自动化配置位置，不能由 `scripts/` 替代。两者的入口与维护步骤见下文“网页浏览与发布”。

### 新增目录的条件

新增目录必须在同次修改中说明以下事项，并更新本节及相关导航：

1. **服务对象与内容**：读者要解决什么问题，已有或本次实际交付哪些可持续维护的内容。
2. **归属边界**：现有目录为什么无法容纳，哪些内容应进入新目录，哪些仍留在原位置。
3. **维护方式**：如何命名、何时更新，如何处理过时内容，以及是否纳入 Git。
4. **阅读与迁移入口**：读者从哪里进入，已有内容如何迁移，引用如何同步；公开文档不能依赖本地忽略文件。

缺少实际内容或独立职责时，不创建目录，也不预留空目录。合并或移除目录前，将有效结论归回对应专题和规范，更新全部引用；仅有过程价值的记录可保留在本地工作目录。

## 收录规范

提交新的论文或基础模型时，请遵循 [写作与格式规范](docs/STYLE.md)，并在 Pull Request 中提供以下信息：

1. **基本元数据**：论文标题、发表年份、收录会议或期刊（或 arXiv 编号）、主要研究机构。
2. **官方链接**：arXiv 或 DOI 论文链接、官方代码库链接（GitHub / GitLab）、预训练权重链接（Hugging Face / Zenodo）。
3. **方法与结果摘要**：说明其针对可穿戴或生理时序信号的具体设计（如异构采样率对齐、运动伪影抑制、跨模态对齐等），并给出可查证的实验指标。
4. **结构化索引更新**：
   - 需要独立索引时，在 `data/models.yaml`（或 `data/datasets.yaml`）中添加对应条目，`added` 字段填写收录日期（`YYYY-MM-DD`）；修改已有条目时保留 `added`，填写 `updated`，让最近更新表准确反映修订；
   - 在 `docs/models/`（或 `docs/datasets/`）对应专题文档中补充小节与论文架构图（放置在 `docs/figures/<id>/`）；
   - 运行 `python scripts/build_tables.py` 更新 README 和数据集专题的索引表，将生成结果与元数据一起提交；按下方“本地预览”步骤检查网页和文档。

### 检索证据与收录判断

检索以主题词和模型完整名称定位候选，再回查论文、正式会议或期刊记录、作者仓库与模型卡。搜索摘要用于发现线索，方法、数据和实验结论应能追溯至一手来源；注明版本与核查日期，无法取得的证据明确保留为待核查项。

优先收录与可穿戴或生理计算相关、具有可迁移预训练或标准化评测价值的工作。单任务适配和观点文章按其实际用途作为对照或延伸，不重复计为新基础模型。同一论文的预印本和正式版本合并维护；实质不同的系列模型分开说明。

`data/models.yaml` 的 `type` 按主要研究产物填写，不表示与仓库主线的相关程度：

| 类型 | 含义 |
| :--- | :--- |
| `model`（模型） | 以训练后的模型及其表征或预测能力为主要产物，包括生理信号、多模态和扩展参考模型 |
| `method`（方法） | 既有模型的任务适配、重编程或 Agent 组合方法；不作为新的生理基础模型统计 |
| `benchmark`（基准） | 以数据、任务和评测协议为主要产物，即使配有 Agent 或其他参考实现，仍按基准收录 |
| `survey`（综述） | 以领域梳理与方法比较为主要产物 |

仅作应用背景或延伸阅读的轻量案例（如 PHA）可保留在相关专题，说明来源、用途与实验边界，不必建立独立 YAML 条目。需要在首页检索和比较的工作再按上述流程建立索引；同一工作保留一个主要分析位置，其他专题使用摘要与链接。

收录理由和暂缓候选写在相关专题中，至少保留来源、当前判断和待补证据。候选完成核查后更新原位置；已建立索引的条目同步更新元数据与生成结果。不另建按日期增长的公开检索日志。一次更新的增删清单由提交或 PR 说明记录，临时快照可留在 `internal_strategy/`。

## 注意事项

- 区分原始波形、派生特征与行为指标；人数、记录数、采样点数、传感器累计时长与实际预训练划分分别标注。
- `code`、`weights` 与 `dataset` 分别填写实现、检查点和数据入口；论文公开不等于权重公开。官方需申请的检查点使用 `access: gated`，许可限制在专题说明。
- 首页可用 `org_short`、`venue_short`、`modality_short` 提供有依据的简写。机构按分隔符换行，长刊名允许词间换行，机制与数据按行展示；不使用 CSS 强锁 GitHub 表格列宽。
- 首页研究地图源文件为 `assets/taxonomy_overview.mmd`；运行 `python scripts/render_diagrams.py --only taxonomy` 同步生成 PNG/SVG，并在首页实际阅读宽度检查文字。
- 请勿在 Git 仓库中提交模型权重文件（`.pth`、`.bin`、`.safetensors` 等）、原始数据集文件或论文 PDF 原件。
- 论文插图请标注来源（如 `图源：PaPaGei 论文 Fig. 1（arXiv:2410.20542）`），单张图片控制在 500 KB 以内。

## 网页浏览与发布

README 保留完整索引，网页首页提供关键词搜索、分类筛选和折叠详情；专题与技术长文在同一站点内阅读。两种索引都读取 `data/models.yaml` 和 `data/datasets.yaml`，资源标签共用 `scripts/build_tables.py` 中的 `model_resources`。最近更新按 `updated` 排序，缺省时使用 `added`。

模型的详细分析仍写在所属专题的 Markdown 小节中，技术长文仍写在 `blogs/`，不为网页另存一份正文或拆成每个模型一个文件。构建时将公开 Markdown 转为对应的 HTML 页面，转换文章互链和小节锚点，并复制实际引用的图片；模型卡片直接跳到站内专题的小节。文章提供页内目录、返回索引与 GitHub 源文件入口。

首页的“所属专题”按模型记录中 `doc` 指向的主要维护位置筛选，用于定位专题内的条目；它不是完整的模态或任务分类。跨专题寻找评测基准时使用“条目类型”，寻找某种信号或任务时使用关键词搜索。修改主维护位置时同步更新 `doc`，保留其他专题的引用链接。

### 内容同步

- **条目信息**：模型、基准和数据集统一在 `data/` 更新。README 表格和网页卡片分别由同一份元数据生成，资源标签共用规则。
- **专题正文**：直接修改 `docs/` 或 `blogs/` 的 Markdown；网页在发布时转换这些原文，不维护第二份正文。
- **首页呈现**：README 的介绍、导航与网页首页的布局文案分别维护，不要求逐字相同；数量和条目等数据由生成器提供。

每次 Pull Request 和发布前运行 `python scripts/build_tables.py --check`。该命令不修改文件，会检查 README 的最近更新、模型索引、数据集索引，以及数据集专题中的生成表格是否与当前元数据一致；过期或缺少生成标记时检查失败。先运行 `python scripts/build_tables.py`，再提交更新后的 Markdown 即可修复。

网页检查还会核对卡片 ID 与元数据是否一一对应，并验证专题链接、小节锚点、图片和站点地图。任何检查失败都会停止此次发布，线上继续保留上次成功版本。检查能发现生成内容的遗漏；研究结论与手写摘要仍需维护者审核。

### 本地预览

使用 Python 3.10 或更新版本，按以下步骤安装构建依赖、生成页面并检查站内链接：

```bash
python -m pip install -r scripts/requirements-site.txt
python scripts/build_tables.py
python scripts/build_tables.py --check
python scripts/build_site.py
python scripts/check_site.py
python scripts/lint_md.py --errors-only
python -m http.server 8000 --directory _site --bind 127.0.0.1
```

在浏览器打开 `http://127.0.0.1:8000`；也可直接打开本地 `_site/index.html`。页面不需要外部脚本、字体或网络接口即可搜索和阅读，公式在构建时转换为 MathML。构建依赖统一维护在 `scripts/requirements-site.txt`，GitHub Actions 使用同一份依赖清单。

修改元数据、Markdown 正文或 `assets/catalog.html`、`assets/article.html` 后重新构建；更新地图后也需构建，以复制最新图片。提交前运行上述检查，并在浏览器检查正文表格、公式和手机宽度下的排版。不手动编辑 `_site/` 内的 HTML。

### GitHub Pages

站点地址：[可穿戴与生理基础模型研究索引](https://tujy859.github.io/Wearable-Physiological-Foundation-Models/)。README 顶部和仓库 About 的 Website 字段提供入口。

此项目只依赖静态 HTML、CSS 和 JavaScript，通过 [GitHub Pages 自定义工作流](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)部署。向 `main` 推送后自动构建、检查并发布；也可在 **Actions → Publish research catalog → Run workflow** 中手动触发。首次启用或部署 Fork 时，在 **Settings → Pages → Build and deployment** 中将 **Source** 设为 **GitHub Actions**，并按实际默认分支调整工作流中的 `push.branches`。

首页、专题和技术长文都发布到同一个站点。工作流仅上传 `_site/`，不上传源码仓库、论文原件或本地工作记录。构建和发布失败可在同一工作流查看日志。工作流读取 Pages 的实际地址，通过 `CATALOG_SITE_URL` 生成页面规范地址与站点地图；Fork 使用自身仓库与默认分支生成 GitHub 源文件和编辑链接，同时需更新 README 和仓库 About 中的网页入口。

### 搜索引擎收录

页面正文在构建时生成，搜索引擎无需执行筛选脚本即可读取索引和文章。构建同时生成页面摘要、规范地址（canonical）和 [sitemap.xml](https://tujy859.github.io/Wearable-Physiological-Foundation-Models/sitemap.xml)，后续新增公开文章会随构建更新站点地图。

Google 可以通过公开链接自动发现网站，手动提交并非收录前提。维护者可使用 [Google Search Console](https://search.google.com/search-console/) 查看抓取、收录和搜索表现：

1. 添加“网址前缀”资源，填写完整站点地址 `https://tujy859.github.io/Wearable-Physiological-Foundation-Models/`。
2. 选择 HTML 标记验证，将该账号获得的 `google-site-verification` 标记加入 `assets/catalog.html` 的 `<head>`，提交并等待 Pages 发布成功，再回到 Search Console 验证。保留该标记，以便后续验证持续有效。
3. 在“站点地图”中提交 `sitemap.xml`；可用“网址检查”查看首页与专题页并请求编入索引。

提交站点地图有助于发现页面，是否收录及何时展示由搜索引擎决定。参见 Google 的[搜索工作原理](https://developers.google.com/search/docs/fundamentals/how-search-works)、[所有权验证](https://support.google.com/webmasters/answer/9008080)和[站点地图说明](https://developers.google.com/search/docs/crawling-indexing/sitemaps/build-sitemap)。
