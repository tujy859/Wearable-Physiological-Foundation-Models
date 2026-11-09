"""根据 data/models.yaml 与 data/datasets.yaml 生成 README.md 及数据集专题中的索引表。

用法：
    python scripts/build_tables.py
    python scripts/build_tables.py --check  # 只检查生成区块是否最新，不写文件
"""

from __future__ import annotations

import argparse
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
    "clinical": "临床连续监护",
    "ecg": "心电信号",
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
    "planned": "待发布",
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
    begin = f"<!-- BEGIN:{marker} -->"
    end = f"<!-- END:{marker} -->"
    if text.count(begin) != 1 or text.count(end) != 1:
        raise RuntimeError(f"标记块 {marker} 必须包含且仅包含一对 BEGIN / END 标记")
    pattern = re.compile(
        rf"(<!-- BEGIN:{re.escape(marker)} -->)(.*?)(<!-- END:{re.escape(marker)} -->)",
        re.DOTALL,
    )
    if not pattern.search(text):
        raise RuntimeError(f"未找到标记块 {marker}")
    return pattern.sub(lambda match: f"{match[1]}\n\n{content.strip()}\n\n{match[3]}", text)


def format_org_venue(m: dict) -> str:
    """按机构分行，长刊名保留词间换行机会，避免整列被不可断行文本撑宽。"""
    org = str(m.get("org_short") or m.get("org", "")).strip()
    org_lines = [protect_cjk_chunks(part, chunk_size=4) for part in re.split(r"\s+/\s+", org)]
    venue = str(m.get("venue_short") or m.get("venue", "")).strip()
    if len(venue) <= 12:
        venue = nowrap(venue)
    else:
        venue = re.sub(r"\s+(\d{4}(?:\s*[/\-\u2011\u2013]\s*\d{4})?)$", r"<br>\1", venue)
    return "<br>".join(part for part in [*org_lines, venue] if part)


def format_modality(m: dict) -> str:
    """把模态组合与括号说明分行，让机制列获得更多可用宽度。"""
    text = str(m.get("modality_short") or m.get("modality", "")).strip()
    text = text.replace(" (", "<br>(").replace(" + ", " +<br>")
    return protect_cjk_chunks(text, chunk_size=4).replace(NBSP, " ")


def model_resources(m: dict) -> list[tuple[str, str | None]]:
    """为 Markdown 与网页提供统一的资源标签、链接及明确的获取状态。"""
    resources: list[tuple[str, str | None]] = []
    if m.get("paper"):
        paper_label = (
            "报告"
            if any(k in m["paper"] for k in ["mistral.ai", "developer.nvidia.com"])
            else "论文"
        )
        resources.append((paper_label, m["paper"]))

    if m.get("code"):
        resources.append(("代码", m["code"]))
    dataset = str(m.get("dataset") or "").strip()
    weights = str(m.get("weights") or "").strip()
    # 兼容旧条目误放在 weights 中的 Hugging Face 数据集，避免生成“权重”标签。
    dataset_in_weights = "/datasets/" in weights
    if dataset:
        resources.append(("数据", dataset))
    if dataset_in_weights and weights != dataset:
        resources.append(("数据", weights))
    if weights and not dataset_in_weights:
        label = m.get("weights_label") or ("权重需申请" if m.get("access") == "gated" else "权重")
        resources.append((label, weights))
    elif m.get("access") == "gated":
        resources.append(("权重需申请", None))
    elif m.get("access") == "planned" and m.get("type") == "model":
        label = "权重待发布" if m.get("code") else "代码与权重待发布"
        resources.append((label, None))
    # 缺少链接不代表作者确认未公开；资源栏只展示已有入口或明确的获取状态。
    return resources


def format_model_resources(m: dict, inline: bool = False) -> str:
    links = [
        f"[{nowrap(label)}]({url})" if url else nowrap(label)
        for label, url in model_resources(m)
    ]
    return " · ".join(links) if inline else "<br>".join(links)


def format_model_name(m: dict) -> str:
    tag = "`基准` " if m.get("type") == "benchmark" else ""
    if m.get("type") == "survey":
        tag = "`综述` "
    elif m.get("type") == "method":
        tag = "`方法` "
    name = str(m.get("name", "")).strip()
    if m.get("doc"):
        return f"{tag}[**{name}**]({m['doc']})"
    return f"{tag}**{name}**"


def build_recent_table(models: list[dict], limit: int = 10) -> str:
    items = sorted(
        models,
        key=lambda x: (str(x.get("updated") or x.get("added", "")), str(x.get("date", ""))),
        reverse=True,
    )[:limit]
    headers = [
        nowrap("更新日期"),
        nowrap("研究条目"),
        nowrap("机构") + "<br>" + nowrap("与发表"),
        nowrap("传感模态"),
        nowrap("核心机制") + "<br>" + nowrap("与资源"),
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| :---: | :--- | :--- | :--- | :--- |",
    ]
    for m in items:
        added_cell = nowrap(str(m.get("updated") or m.get("added", "")))
        name_cell = format_model_name(m)
        org_venue_cell = format_org_venue(m)
        modality_cell = format_modality(m)
        res = format_model_resources(m, inline=True)
        summary = str(m.get("method", "")).strip()
        if m.get("data"):
            summary = f"{summary}<br>规模 {m['data']}" if summary else str(m["data"])
        lines.append(
            f"| {added_cell} | {name_cell} | {org_venue_cell} | {modality_cell} | {summary}<br>{res} |"
        )
    return "\n".join(lines)


def build_models_table(models: list[dict]) -> str:
    headers = [
        nowrap("研究条目"),
        nowrap("机构") + "<br>" + nowrap("与发表"),
        nowrap("传感模态"),
        nowrap("核心机制") + "<br>" + nowrap("与数据规模"),
        nowrap("论文") + "<br>" + nowrap("与资源"),
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| :--- | :--- | :--- | :--- | :---: |",
    ]
    for m in models:
        if m.get("type") == "survey":
            continue
        name_cell = format_model_name(m)
        org_venue_cell = format_org_venue(m)
        modality_cell = format_modality(m)

        method = str(m.get("method", "")).strip()
        params = str(m.get("params", "")).strip()
        param_pattern = r"\s*".join(re.escape(part) for part in params.split())
        params_in_method = bool(params) and re.search(
            rf"(?<![A-Za-z0-9.]){param_pattern}(?![A-Za-z0-9.])", method
        )
        if params and params not in ("-", "—") and m.get("type") != "benchmark" and not params_in_method:
            method_line = f"{method}<br>参数 {params}" if method else f"参数 {params}"
        else:
            method_line = method or "-"
        data_str = str(m.get("data", "")).strip()
        if data_str and data_str not in ("-", "—"):
            method_data_cell = f"{method_line}<br>规模 {data_str}"
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
        tag = str(d.get("tag", "")).strip()
        doc_link = str(d.get("doc", "")).strip()
        if local_doc_links and "#" in doc_link:
            doc_link = "#" + doc_link.split("#", 1)[1]
        name_link = f"[**{d['name']}**]({doc_link})" if doc_link else f"**{d['name']}**"
        sub_label = f"{cat}{NBSP}·{NBSP}`{tag}`" if tag else cat
        name_cell = f"{name_link}<br>{sub_label}"

        org_cell = protect_cjk_chunks(str(d.get("org", "")), chunk_size=4)
        sensors_cell = protect_cjk_chunks(str(d.get("sensors", "")), chunk_size=5)

        scale_str = str(d.get("scale", "")).strip()
        tasks_str = str(d.get("tasks", "")).strip()
        scale_tasks_cell = f"{scale_str}<br>任务：{tasks_str}" if tasks_str else scale_str

        access_label = nowrap(ACCESS_LABELS.get(d.get("access", ""), d.get("access", "查看主页")))
        url = d.get("url") or d.get("paper") or ""
        access_cell = f"[{access_label}]({url})" if url else access_label
        if d.get("access_note"):
            access_cell += "<br>" + protect_cjk_chunks(str(d["access_note"]), chunk_size=4)
        lines.append(
            f"| {name_cell} | {org_cell} | {sensors_cell} | {scale_tasks_cell} | {access_cell} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="检查索引生成区块是否最新，不写文件")
    args = parser.parse_args(argv)

    try:
        models = yaml.safe_load(MODELS_YAML.read_text(encoding="utf-8")) or []
        datasets = yaml.safe_load(DATASETS_YAML.read_text(encoding="utf-8")) or []
        # read_text normalizes CRLF/LF, so Windows checkouts compare identically.
        readme_source = README_PATH.read_text(encoding="utf-8")
        readme = replace_block(readme_source, "RECENT_UPDATES", build_recent_table(models))
        readme = replace_block(readme, "MODELS_TABLE", build_models_table(models))
        readme = replace_block(readme, "DATASETS_TABLE", build_datasets_table(datasets, local_doc_links=False))
        ds_source = DATASETS_DOC_PATH.read_text(encoding="utf-8")
        ds_doc = replace_block(ds_source, "DATASETS_DOC_TABLE", build_datasets_table(datasets, local_doc_links=True))
    except (OSError, RuntimeError, yaml.YAMLError) as error:
        print(f"索引生成失败：{error}", file=sys.stderr)
        return 1

    documents = [(README_PATH, readme_source, readme), (DATASETS_DOC_PATH, ds_source, ds_doc)]
    if args.check:
        stale = [path.relative_to(ROOT).as_posix() for path, source, generated in documents if source != generated]
        if stale:
            print("以下文档的索引生成区块尚未同步：" + "、".join(stale), file=sys.stderr)
            print("请运行 python scripts/build_tables.py，并提交更新后的文档。", file=sys.stderr)
            return 1
        print("Index tables are up to date: README.md and docs/datasets/wearable_datasets.md.")
        return 0

    # Validate both documents before writing either one.
    for path, _, generated in documents:
        path.write_text(generated, encoding="utf-8")
    print(f"Updated README.md with {len(models)} research entries and {len(datasets)} dataset entries.")
    print(f"Updated {DATASETS_DOC_PATH.relative_to(ROOT)} with {len(datasets)} dataset entries.")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
