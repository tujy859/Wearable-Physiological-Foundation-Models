"""根据 data/models.yaml 与 data/datasets.yaml 生成 README.md 及数据集专题中的索引表。

用法：
    python scripts/build_tables.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
MODELS_YAML = ROOT / "data" / "models.yaml"
DATASETS_YAML = ROOT / "data" / "datasets.yaml"
README_PATH = ROOT / "README.md"
DATASETS_DOC_PATH = ROOT / "docs" / "datasets" / "wearable_datasets.md"

WJ = "\u2060"        # Unicode Word Joiner (零宽不换行符，仅插入汉字之间防止逐字竖排)
NBSP = "\u00a0"      # Non-Breaking Space (不换行空格)
NBHYPHEN = "\u2011"  # Non-Breaking Hyphen (不换行连字符)

CATEGORY_LABELS = {
    "wrist": "手腕多模态",
    "cgm": "CGM 连续血糖",
    "clinical": "临床多模态",
    "ecg": "临床心电",
    "sleep": "睡眠 PSG",
    "cohort": "人群队列",
    "benchmark": "推理基准",
}

ACCESS_LABELS = {
    "open": "开放获取",
    "registration": "注册获取",
    "credentialed": "凭证申请",
    "application": "学术申请",
    "restricted": "受限申请",
    "gated": "申请获取",
    "closed": "未公开",
}


def nowrap(text: str) -> str:
    """将短标签或表头设为整词不换行（仅在相邻汉字间插入 WJ，空格替换为 NBSP，连字符替换为 NBHYPHEN）。"""
    text = str(text).strip().replace("-", NBHYPHEN).replace(" ", NBSP)
    return re.sub(r"(?<=[\u4e00-\u9fff])(?=[\u4e00-\u9fff])", WJ, text)


def protect_cjk_chunks(text: str, chunk_size: int = 5) -> str:
    """为中文短语按 4-5 字词块注入零宽连接符，使每列在 GitHub 网页端拥有合理的最小列宽。"""
    text = str(text).strip()
    # 保护“中文 + 空格 + 短英文缩写”（如 "单通道 PPG"、"12 导联"）不在中间断行
    text = re.sub(
        r"([\u4e00-\u9fff])\s+([A-Za-z0-9]{1,5})(?![A-Za-z0-9])",
        rf"\1{NBSP}\2",
        text,
    )
    text = re.sub(
        r"(\d+)\s+([\u4e00-\u9fff])",
        rf"\1{NBSP}\2",
        text,
    )

    def _join_run(match: re.Match) -> str:
        s = match.group(0)
        if len(s) <= chunk_size:
            return WJ.join(list(s))
        chunks = [WJ.join(list(s[i : i + chunk_size])) for i in range(0, len(s), chunk_size)]
        return "".join(chunks)

    return re.sub(r"[\u4e00-\u9fff]{2,}", _join_run, text)


def replace_block(text: str, marker: str, content: str) -> str:
    pattern = re.compile(
        rf"(<!-- BEGIN:{re.escape(marker)} -->)(.*?)(<!-- END:{re.escape(marker)} -->)",
        re.DOTALL,
    )
    if not pattern.search(text):
        raise RuntimeError(f"未找到标记块 {marker}")
    return pattern.sub(rf"\1\n\n{content.strip()}\n\n\3", text)


def format_model_resources(m: dict, inline: bool = False) -> str:
    links: list[str] = []
    if m.get("paper"):
        paper_label = (
            "报告"
            if any(k in m["paper"] for k in ["mistral.ai", "developer.nvidia.com"])
            else "论文"
        )
        links.append(f"[{nowrap(paper_label)}]({m['paper']})")

    open_parts: list[str] = []
    if m.get("code"):
        open_parts.append(f"[{nowrap('代码')}]({m['code']})")
    if m.get("weights"):
        open_parts.append(f"[{nowrap('权重')}]({m['weights']})")
    elif m.get("access") == "gated":
        open_parts.append(nowrap("权重需申请"))

    if open_parts:
        links.extend(open_parts)
    else:
        if m.get("access") == "gated":
            links.append(nowrap("需申请"))
        else:
            links.append(nowrap("闭源"))

    sep = f"{NBSP}·{NBSP}"
    if inline or len(links) <= 2:
        return sep.join(links)
    return sep.join(links[:2]) + "<br>" + sep.join(links[2:])


def format_model_name(m: dict) -> str:
    tag = "`基准` " if m.get("type") == "benchmark" else ""
    if m.get("type") == "survey":
        tag = "`综述` "
    name = str(m.get("name", "")).strip()
    if m.get("doc"):
        return f"{tag}[**{name}**]({m['doc']})"
    return f"{tag}**{name}**"


def build_recent_table(models: list[dict], limit: int = 10) -> str:
    items = sorted(
        models,
        key=lambda x: (str(x.get("added", "")), str(x.get("date", ""))),
        reverse=True,
    )[:limit]
    headers = [
        nowrap("收录日期"),
        nowrap("模型 / 基准"),
        nowrap("机构与发表"),
        nowrap("传感模态"),
        nowrap("核心机制与开源资源"),
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| :---: | :--- | :--- | :--- | :--- |",
    ]
    for m in items:
        added_cell = nowrap(str(m.get("added", "")))
        name_cell = format_model_name(m)
        org = protect_cjk_chunks(str(m.get("org", "")), chunk_size=4)
        venue = nowrap(str(m.get("venue", "")))
        org_venue_cell = f"{org}<br>{venue}" if org and venue else (org or venue)
        modality_cell = protect_cjk_chunks(str(m.get("modality", "")), chunk_size=5)
        res = format_model_resources(m, inline=True)
        summary = str(m.get("method", "")).strip()
        if m.get("data"):
            summary = f"{summary}（{m['data']}）" if summary else str(m["data"])
        lines.append(
            f"| {added_cell} | {name_cell} | {org_venue_cell} | {modality_cell} | {summary}<br>{res} |"
        )
    return "\n".join(lines)


def build_models_table(models: list[dict]) -> str:
    headers = [
        nowrap("模型 / 基准"),
        nowrap("机构与发表"),
        nowrap("传感模态"),
        nowrap("核心机制与数据规模"),
        nowrap("开源资源"),
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| :--- | :--- | :--- | :--- | :---: |",
    ]
    for m in models:
        if m.get("type") == "survey":
            continue
        name_cell = format_model_name(m)
        org = protect_cjk_chunks(str(m.get("org", "")), chunk_size=4)
        venue = nowrap(str(m.get("venue", "")))
        org_venue_cell = f"{org}<br>{venue}" if org and venue else (org or venue)
        modality_cell = protect_cjk_chunks(str(m.get("modality", "")), chunk_size=5)

        method = str(m.get("method", "")).strip()
        params = str(m.get("params", "")).strip()
        if params and params not in ("-", "—") and m.get("type") != "benchmark":
            method_line = f"{method}（{params}）" if method else params
        else:
            method_line = method or "-"
        data_str = str(m.get("data", "")).strip()
        if data_str and data_str not in ("-", "—"):
            method_data_cell = f"{method_line}<br>规模：{data_str}"
        else:
            method_data_cell = method_line

        res = format_model_resources(m, inline=False)
        lines.append(
            f"| {name_cell} | {org_venue_cell} | {modality_cell} | {method_data_cell} | {res} |"
        )
    return "\n".join(lines)


def build_datasets_table(datasets: list[dict], local_doc_links: bool = False) -> str:
    headers = [
        nowrap("数据集与领域"),
        nowrap("机构 / 来源"),
        nowrap("传感器与采样率"),
        nowrap("样本规模与适用任务"),
        nowrap("获取方式"),
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| :--- | :--- | :--- | :--- | :---: |",
    ]
    for d in datasets:
        cat = nowrap(CATEGORY_LABELS.get(d.get("category", ""), d.get("category", "")))
        doc_link = str(d.get("doc", "")).strip()
        if local_doc_links and "#" in doc_link:
            doc_link = "#" + doc_link.split("#", 1)[1]
        name_link = f"[**{d['name']}**]({doc_link})" if doc_link else f"**{d['name']}**"
        name_cell = f"{name_link}<br>{cat}"

        org_cell = protect_cjk_chunks(str(d.get("org", "")), chunk_size=4)
        sensors_cell = protect_cjk_chunks(str(d.get("sensors", "")), chunk_size=5)

        scale_str = str(d.get("scale", "")).strip()
        tasks_str = str(d.get("tasks", "")).strip()
        scale_tasks_cell = f"{scale_str}<br>任务：{tasks_str}" if tasks_str else scale_str

        access_label = nowrap(ACCESS_LABELS.get(d.get("access", ""), d.get("access", "查看主页")))
        url = d.get("url") or d.get("paper") or ""
        access_cell = f"[{access_label}]({url})" if url else access_label
        lines.append(
            f"| {name_cell} | {org_cell} | {sensors_cell} | {scale_tasks_cell} | {access_cell} |"
        )
    return "\n".join(lines)


def main() -> int:
    if not MODELS_YAML.exists() or not DATASETS_YAML.exists():
        print("等待 data/models.yaml 与 data/datasets.yaml 生成...", file=sys.stderr)
        return 1

    models = yaml.safe_load(MODELS_YAML.read_text(encoding="utf-8")) or []
    datasets = yaml.safe_load(DATASETS_YAML.read_text(encoding="utf-8")) or []
    readme = README_PATH.read_text(encoding="utf-8")

    readme = replace_block(readme, "RECENT_UPDATES", build_recent_table(models))
    readme = replace_block(readme, "MODELS_TABLE", build_models_table(models))
    readme = replace_block(readme, "DATASETS_TABLE", build_datasets_table(datasets, local_doc_links=False))

    README_PATH.write_text(readme, encoding="utf-8")
    print(f"Updated README.md with {len(models)} model entries and {len(datasets)} dataset entries.")

    if DATASETS_DOC_PATH.exists():
        ds_doc = DATASETS_DOC_PATH.read_text(encoding="utf-8")
        if "<!-- BEGIN:DATASETS_DOC_TABLE -->" in ds_doc:
            ds_doc = replace_block(
                ds_doc,
                "DATASETS_DOC_TABLE",
                build_datasets_table(datasets, local_doc_links=True),
            )
            DATASETS_DOC_PATH.write_text(ds_doc, encoding="utf-8")
            print(f"Updated {DATASETS_DOC_PATH.relative_to(ROOT)} with {len(datasets)} dataset entries.")

    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
