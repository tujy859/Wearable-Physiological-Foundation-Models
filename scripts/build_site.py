"""Build a static catalogue and article pages from repository metadata and Markdown.

Run python scripts/build_site.py; preview _site/index.html directly or serve _site.
Only generated files belong in _site. Edit data/*.yaml or assets/catalog.html.
"""

from __future__ import annotations

import html
import os
import re
import shutil
from pathlib import Path
from urllib.parse import quote, urlsplit

import yaml

from build_tables import ACCESS_LABELS, CATEGORY_LABELS, model_resources
from site_articles import article_sources, build_articles
from site_metadata import canonical_url, site_base_url, write_sitemap

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "_site"
REPOSITORY_SLUG = os.environ.get("CATALOG_REPOSITORY", "tujy859/Wearable-Physiological-Foundation-Models")
if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", REPOSITORY_SLUG):
    raise ValueError("CATALOG_REPOSITORY must be a GitHub owner/repository name")
REPOSITORY = "https://github.com/" + REPOSITORY_SLUG
BRANCH = os.environ.get("CATALOG_BRANCH", "main")
BLOB_BASE = f"{REPOSITORY}/blob/{quote(BRANCH, safe='')}"
TOPICS = {
    "longitudinal_wearable_models": "纵向可穿戴与行为",
    "wearable_transfer_models": "腕戴波形与运动迁移",
    "smartwatch_models": "智能手表与腕戴生理",
    "ecg_models": "心电与心血管",
    "cgm_models": "连续血糖 CGM",
    "sleep_models": "睡眠与心肺",
    "eeg_models": "脑电与跨导联迁移",
    "physiological_benchmarks": "生理信号评测",
    "health_reasoning_llms": "健康推理与语言模型",
    "universal_health_models": "系统级健康模型",
    "general_tsfm": "通用时间序列",
}
TYPES = {"model": "模型", "method": "方法", "benchmark": "基准", "survey": "综述"}


def text(value: object) -> str:
    return str(value or "").strip()


def esc(value: object) -> str:
    return html.escape(text(value), quote=True)


def url(value: object) -> str:
    value = text(value)
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Expected an HTTP(S) resource URL: {value!r}")
    return esc(value)


def doc_url(value: object) -> str:
    value = text(value)
    if not value:
        return ""
    path, _, anchor = value.partition("#")
    target = (ROOT / path).resolve()
    if not any(target.is_relative_to(ROOT / directory) for directory in ("docs", "blogs")) or not target.is_file() or target.suffix != ".md":
        raise ValueError(f"Missing or invalid topic document: {path}")
    return esc(quote(Path(path).with_suffix('.html').as_posix(), safe='/') + (f"#{quote(anchor)}" if anchor else ""))


def link(label: object, target: object, css: str = "") -> str:
    return f'<a class="{css}" href="{url(target)}">{esc(label)}</a>'


def details(fields: list[tuple[str, object]], summary: str) -> str:
    rows = "".join(f"<div><dt>{esc(label)}</dt><dd>{esc(value)}</dd></div>" for label, value in fields if text(value) not in {"", "-", "—"})
    return f'<details class="entry-details"><summary>{esc(summary)}</summary><dl>{rows}</dl></details>' if rows else ""


def entry_title(item: dict) -> str:
    title = esc(item["name"])
    target = doc_url(item.get("doc"))
    return f'<a href="{target}">{title}</a>' if target else title


def attributes(item: dict, group: str, topic: str, kind: str, resources: list[str]) -> str:
    search = " ".join(text(value) for value in item.values()) + " " + topic
    return (f'data-group="{group}" data-topic="{esc(topic)}" data-kind="{esc(kind)}" '
            f'data-resources="{esc(" ".join(resources))}" data-search="{esc(search)}"')


def model_card(item: dict) -> str:
    topic_key = Path(text(item.get("doc")).split("#")[0]).stem
    if topic_key not in TOPICS:
        raise ValueError(f"Add a topic label for {topic_key!r}")
    topic = TOPICS[topic_key]
    kind = text(item.get("type")) or "model"
    resources = model_resources(item)
    rendered = "".join(link(label, target) if target else f'<span class="resource-note">{esc(label)}</span>' for label, target in resources)
    flags = []
    if item.get("code"):
        flags.append("code")
    if item.get("weights") and "/datasets/" not in str(item["weights"]):
        flags.append("weights")
    if item.get("dataset") or "/datasets/" in text(item.get("weights")):
        flags.append("data")
    attrs = attributes(item, "models", topic, kind, flags)
    info = " · ".join(filter(None, [text(item.get("org_short") or item.get("org")), text(item.get("venue"))]))
    expanded = details([("机制", item.get("method")), ("参数", item.get("params")), ("训练 / 评测规模", item.get("data")), ("获取说明", item.get("access_note")), ("收录更新", item.get("updated") or item.get("added"))], "方法与数据")
    return f'''<article class="entry" id="model-{esc(item['id'])}" {attrs}>
      <div class="entry-top"><span class="eyebrow">{esc(topic)}</span><span class="kind">{esc(TYPES[kind])}</span></div>
      <h3>{entry_title(item)}</h3><p class="entry-meta">{esc(info)}</p>
      <p class="modality">{esc(item.get('modality'))}</p>
      {expanded}<div class="resources">{rendered}</div>
    </article>'''


def dataset_card(item: dict) -> str:
    topic = CATEGORY_LABELS.get(item.get("category"), text(item.get("category")))
    access = text(item.get("access"))
    access_label = ACCESS_LABELS.get(access, "数据主页")
    flags = ["open"] if access == "open" else ["planned"] if access == "planned" else ["request"] if access in {"registration", "credentialed", "application", "restricted", "gated"} else []
    attrs = attributes(item, "datasets", topic, "dataset", flags)
    resources = []
    if item.get("url"):
        resources.append(link(access_label, item["url"]))
    if item.get("paper"):
        resources.append(link("论文", item["paper"]))
    expanded = details([("规模", item.get("scale")), ("适用任务", item.get("tasks")), ("获取方式", access_label), ("使用条件", item.get("access_note"))], "规模与获取方式")
    return f'''<article class="entry" id="dataset-{esc(item['id'])}" {attrs}>
      <div class="entry-top"><span class="eyebrow">{esc(topic)}</span><span class="kind">数据集</span></div>
      <h3>{entry_title(item)}</h3><p class="entry-meta">{esc(item.get('org'))}</p>
      <p class="modality">{esc(item.get('sensors'))}</p>
      {expanded}<div class="resources">{''.join(resources)}</div>
    </article>'''


def main() -> None:
    site_url = site_base_url()
    models = yaml.safe_load((ROOT / "data/models.yaml").read_text(encoding="utf-8")) or []
    datasets = yaml.safe_load((ROOT / "data/datasets.yaml").read_text(encoding="utf-8")) or []
    for items in (models, datasets):
        ids = [item["id"] for item in items]
        if len(set(ids)) != len(ids):
            raise ValueError("Catalogue IDs must be unique within each collection.")
    models.sort(key=lambda item: (text(item.get("updated") or item.get("added")), text(item.get("date"))), reverse=True)
    cards = '<section class="collection" data-collection="models" aria-labelledby="models-heading"><h2 class="collection-heading" id="models-heading">研究条目</h2><div class="entries">' + "\n".join(map(model_card, models)) + '</div></section>'
    cards += '<section class="collection" data-collection="datasets" aria-labelledby="datasets-heading"><h2 class="collection-heading" id="datasets-heading">数据集与队列</h2><div class="entries">' + "\n".join(map(dataset_card, datasets)) + '</div></section>'
    topics = "".join(f'<a data-topic="{esc(label)}" href="{doc_url(f"docs/models/{key}.md")}">{esc(label)}<span aria-hidden="true">→</span></a>' for key, label in TOPICS.items())
    topics += f'<a href="{doc_url("docs/datasets/wearable_datasets.md")}">数据集与队列<span aria-hidden="true">→</span></a>'
    blogs = []
    for path in sorted((ROOT / 'blogs').glob('*.md')):
        heading = re.search(r'^#\s+(.+)$', path.read_text(encoding='utf-8'), re.MULTILINE)
        title = heading.group(1).strip() if heading else path.stem
        blogs.append(f'<a href="{doc_url(path.relative_to(ROOT).as_posix())}">{esc(title)}<span aria-hidden="true">→</span></a>')
    substitutions = {
        "CARDS": cards, "TOPICS": topics, "BLOGS": ''.join(blogs), "MODEL_COUNT": str(len(models)),
        "DATASET_COUNT": str(len(datasets)), "TOTAL_COUNT": str(len(models) + len(datasets)),
        "UPDATED": esc(max((text(item.get("updated") or item.get("added")) for item in [*models, *datasets]), default="")),
        "REPOSITORY": url(REPOSITORY), "BLOB_BASE": url(BLOB_BASE),
        "CANONICAL_URL": esc(canonical_url(site_url, Path("index.html"))),
        "SOCIAL_IMAGE": esc(site_url + "assets/taxonomy_overview.png"),
    }
    document = (ROOT / "assets/catalog.html").read_text(encoding="utf-8")
    for key, value in substitutions.items():
        document = document.replace(f"@@{key}@@", value)
    if re.search(r"@@[A-Z_]+@@", document):
        raise ValueError("Unresolved catalogue template placeholder")
    (OUTPUT / "assets").mkdir(parents=True, exist_ok=True)
    (OUTPUT / "index.html").write_text(document, encoding="utf-8", newline="\n")
    (OUTPUT / ".nojekyll").write_text("", encoding="utf-8")
    shutil.copyfile(ROOT / "assets/taxonomy_overview.png", OUTPUT / "assets/taxonomy_overview.png")
    article_count = build_articles(ROOT, OUTPUT, REPOSITORY, BRANCH, TOPICS, site_url)
    write_sitemap(OUTPUT, site_url, [Path("index.html"), *(p.relative_to(ROOT).with_suffix(".html") for p in article_sources(ROOT))])
    print(f"Built _site/index.html: {len(models)} research entries, {len(datasets)} datasets.")
    print(f"Built {article_count} article pages.")


if __name__ == "__main__":
    main()
