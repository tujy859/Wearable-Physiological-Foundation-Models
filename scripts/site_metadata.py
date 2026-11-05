"""Search metadata shared by the static site builder and its checks."""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote, urlsplit

DEFAULT_REPOSITORY = "tujy859/Wearable-Physiological-Foundation-Models"
SITEMAP_NAMESPACE = "http://www.sitemaps.org/schemas/sitemap/0.9"


def site_base_url() -> str:
    owner, repository = os.environ.get("CATALOG_REPOSITORY", DEFAULT_REPOSITORY).split("/", 1)
    default = f"https://{owner.lower()}.github.io/"
    if repository.lower() != f"{owner.lower()}.github.io":
        default += repository + "/"
    value = os.environ.get("CATALOG_SITE_URL", "").strip() or default
    parsed = urlsplit(value)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("CATALOG_SITE_URL must be an absolute HTTP(S) site URL without credentials, query or fragment")
    return value.rstrip("/") + "/"


def canonical_url(base: str, relative: Path) -> str:
    """Use one homepage URL, including for filtered views and index.html."""
    path = relative.as_posix()
    return base if path == "index.html" else base + quote(path, safe="/")


class ParagraphText(HTMLParser):
    def __init__(self, body: str):
        super().__init__(convert_charrefs=True)
        self.paragraphs: list[str] = []
        self.current: list[str] | None = None
        self.feed(body)

    def handle_starttag(self, tag, attrs):
        if tag == "p":
            self.current = []
        elif tag == "br" and self.current is not None:
            self.current.append(" ")

    def handle_endtag(self, tag):
        if tag == "p" and self.current is not None:
            self.paragraphs.append("".join(self.current))
            self.current = None

    def handle_data(self, data):
        if self.current is not None:
            self.current.append(data)


def article_description(body: str, title: str) -> str:
    """Reuse the article introduction instead of maintaining a second summary."""
    paragraphs = [re.sub(r"\s+", " ", p).strip() for p in ParagraphText(body).paragraphs]
    description = next((p for p in paragraphs if len(p) >= 30), title)
    description = re.sub(r"^(?:摘要|导读)\s*[：:]\s*", "", description)
    description = re.sub(r"^检索更新日期[为：:]\s*\d{4}-\d{2}-\d{2}。\s*", "", description)
    if len(description) > 160:
        excerpt = description[:157]
        boundary = excerpt.rfind("。")
        description = excerpt[:boundary + 1] if boundary >= 70 else excerpt.rstrip("，、；： ") + "…"
    return description


def write_sitemap(output: Path, base: str, pages: list[Path]) -> None:
    ET.register_namespace("", SITEMAP_NAMESPACE)
    document = ET.Element(f"{{{SITEMAP_NAMESPACE}}}urlset")
    for page in pages:
        entry = ET.SubElement(document, f"{{{SITEMAP_NAMESPACE}}}url")
        ET.SubElement(entry, f"{{{SITEMAP_NAMESPACE}}}loc").text = canonical_url(base, page)
    ET.indent(document, space="  ")
    ET.ElementTree(document).write(output / "sitemap.xml", encoding="utf-8", xml_declaration=True)
