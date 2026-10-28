"""根据 data/models.yaml 与 data/datasets.yaml 生成 README.md 中的索引表。

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


def replace_block(text: str, marker: str, content: str) -> str:
    pattern = re.compile(
        rf"(<!-- BEGIN:{re.escape(marker)} -->)(.*?)(<!-- END:{re.escape(marker)} -->)",
        re.DOTALL,
    )
    if not pattern.search(text):
        raise RuntimeError(f"README.md 中未找到标记块 {marker}")
    return pattern.sub(rf"\1\n\n{content.strip()}\n\n\3", text)


def format_model_resources(m: dict) -> str:
    links: list[str] = []
    if m.get("paper"):
        links.append(f"[论文]({m['paper']})")
    if m.get("code"):
        links.append(f"[代码]({m['code']})")
    if m.get("weights"):
        links.append(f"[权重]({m['weights']})")
    elif m.get("access") == "gated":
        links.append("权重需申请")
    return " · ".join(links) if links else "未公开"


def build_recent_table(models: list[dict], limit: int = 10) -> str:
    items = sorted(
        models,
        key=lambda x: (str(x.get("added", "")), str(x.get("date", ""))),
        reverse=True,
    )[:limit]
    lines = [
        "| 收录日期 | 名称 | 机构 | 发表 | 模态 | 简介与资源 |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for m in items:
        name_cell = f"[**{m['name']}**]({m['doc']})" if m.get("doc") else f"**{m['name']}**"
        res = format_model_resources(m)
        summary = m.get("method", "")
        if m.get("data"):
            summary = f"{summary}（{m['data']}）" if summary else m["data"]
        lines.append(
            f"| {m.get('added', '')} | {name_cell} | {m.get('org', '')} | {m.get('venue', '')} | {m.get('modality', '')} | {summary} · {res} |"
        )
    return "\n".join(lines)


def build_models_table(models: list[dict]) -> str:
    lines = [
        "| 模型 / 基准 | 机构 | 发表 | 主要模态 | 方法与参数规模 | 预训练 / 评测规模 | 开源资源 | 收录日期 |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for m in models:
        if m.get("type") == "survey":
            continue
        name_cell = f"[**{m['name']}**]({m['doc']})" if m.get("doc") else f"**{m['name']}**"
        method = str(m.get("method", "")).strip()
        params = str(m.get("params", "")).strip()
        if params and params not in ("-", "—"):
            method_cell = f"{method}（{params}）" if method else params
        else:
            method_cell = method or "-"
        res = format_model_resources(m)
        lines.append(
            f"| {name_cell} | {m.get('org', '')} | {m.get('venue', '')} | {m.get('modality', '')} | {method_cell} | {m.get('data', '-')} | {res} | {m.get('added', '')} |"
        )
    return "\n".join(lines)


def build_datasets_table(datasets: list[dict]) -> str:
    lines = [
        "| 领域 | 数据集 | 机构 / 来源 | 传感器与采样率 | 样本规模 | 适用任务 | 获取方式 |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for d in datasets:
        cat = CATEGORY_LABELS.get(d.get("category", ""), d.get("category", ""))
        name_cell = f"[**{d['name']}**]({d['doc']})" if d.get("doc") else f"**{d['name']}**"
        access_label = ACCESS_LABELS.get(d.get("access", ""), d.get("access", "查看主页"))
        url = d.get("url") or d.get("paper") or ""
        access_cell = f"[{access_label}]({url})" if url else access_label
        lines.append(
            f"| {cat} | {name_cell} | {d.get('org', '')} | {d.get('sensors', '')} | {d.get('scale', '')} | {d.get('tasks', '')} | {access_cell} |"
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
    readme = replace_block(readme, "DATASETS_TABLE", build_datasets_table(datasets))

    README_PATH.write_text(readme, encoding="utf-8")
    print(f"Updated README.md with {len(models)} model entries and {len(datasets)} dataset entries.")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
