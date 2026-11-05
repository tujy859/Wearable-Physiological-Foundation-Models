"""Check generated pages, local links, heading anchors and copied resources.

Run after scripts/build_site.py. External links are checked for supported URL
schemes only; this is an offline build check, not a remote availability crawl.
"""

from __future__ import annotations

from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET

import yaml

from site_articles import article_sources, render_markdown
from site_metadata import SITEMAP_NAMESPACE, canonical_url, site_base_url

ROOT = Path(__file__).resolve().parent.parent


class Page(HTMLParser):
    def __init__(self, source: str):
        super().__init__(convert_charrefs=True)
        self.ids: list[str] = []
        self.links: list[str] = []
        self.canonicals: list[str] = []
        self.descriptions: list[str] = []
        self.math = 0
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get("id"):
            self.ids.append(attrs["id"])
        if tag == "link" and "canonical" in attrs.get("rel", "").split():
            self.canonicals.append(attrs.get("href", ""))
        if tag == "meta" and attrs.get("name") == "description":
            self.descriptions.append(attrs.get("content", ""))
        for key in ("href", "src"):
            if attrs.get(key):
                self.links.append(attrs[key])
        if tag == "math":
            self.math += 1

    handle_startendtag = handle_starttag


def check_site(output: Path) -> tuple[int, int, int]:
    output = output.resolve()
    pages = {path.resolve(): Page(path.read_text(encoding="utf-8")) for path in output.rglob("*.html")}
    problems = []
    base_url = site_base_url()
    checked_links = 0
    for path, page in pages.items():
        label = path.relative_to(output).as_posix()
        expected_canonical = canonical_url(base_url, path.relative_to(output))
        if page.canonicals != [expected_canonical]:
            problems.append(f"{label}: expected one canonical URL {expected_canonical}")
        if len(page.descriptions) != 1 or not page.descriptions[0].strip():
            problems.append(f"{label}: missing or duplicate description")
        duplicates = [anchor for anchor, count in Counter(page.ids).items() if count > 1]
        if duplicates:
            problems.append(f"{label}: duplicate IDs {duplicates}")
        for link in page.links:
            parsed = urlsplit(link)
            if parsed.scheme or parsed.netloc:
                if parsed.scheme not in {"", "http", "https", "mailto", "data"}:
                    problems.append(f"{label}: unsupported scheme {link}")
                continue
            checked_links += 1
            raw = unquote(parsed.path)
            target = (output / raw.lstrip("/") if raw.startswith("/") else path.parent / raw).resolve() if raw else path
            if target.is_dir():
                target /= "index.html"
            if not target.is_relative_to(output) or not target.is_file():
                problems.append(f"{label}: missing/escaping local target {link}")
                continue
            if target.suffix == ".md":
                problems.append(f"{label}: unconverted Markdown link {link}")
            if parsed.fragment and target in pages:
                fragment = unquote(parsed.fragment)
                if fragment not in pages[target].ids:
                    problems.append(f"{label}: missing anchor {link}")
    for source in article_sources(ROOT):
        expected = output / source.relative_to(ROOT).with_suffix(".html")
        if expected.resolve() not in pages:
            problems.append(f"Missing article {expected.relative_to(output)}")
    homepage = pages.get(output / "index.html")
    if homepage is None:
        problems.append("Missing index.html")
    for metadata, prefix in (("models.yaml", "model-"), ("datasets.yaml", "dataset-")):
        entries = yaml.safe_load((ROOT / "data" / metadata).read_text(encoding="utf-8"))
        expected_ids = {prefix + str(item["id"]) for item in entries}
        if homepage is not None:
            actual_ids = {value for value in homepage.ids if value.startswith(prefix)}
            if actual_ids != expected_ids:
                problems.append(f"{metadata}: homepage card mismatch; missing {sorted(expected_ids - actual_ids)}, extra {sorted(actual_ids - expected_ids)}")
        for item in entries:
            doc, _, anchor = item.get("doc", "").partition("#")
            if not doc:
                continue
            target = (output / Path(doc).with_suffix(".html")).resolve()
            if target not in pages or (anchor and anchor not in pages[target].ids):
                problems.append(f"{metadata}/{item['id']}: missing article or anchor {item['doc']}")
    if any(part.startswith(".") and part != ".nojekyll" or part == "internal_strategy" for path in output.rglob("*") for part in path.relative_to(output).parts):
        problems.append("Private or hidden files present in site output")
    sitemap_path = output / "sitemap.xml"
    try:
        sitemap = ET.parse(sitemap_path)
        locations = [node.text for node in sitemap.findall(f"{{{SITEMAP_NAMESPACE}}}url/{{{SITEMAP_NAMESPACE}}}loc")]
        expected_urls = {canonical_url(base_url, path.relative_to(output)) for path in pages}
        if len(locations) != len(set(locations)) or set(locations) != expected_urls:
            problems.append("Sitemap must list each public page's canonical URL exactly once")
    except (OSError, ET.ParseError) as error:
        problems.append(f"Missing or invalid sitemap.xml: {error}")
    if problems:
        raise ValueError("\n".join(problems))
    return len(pages), checked_links, sum(page.math for page in pages.values())


def check_renderer() -> None:
    """Small regression fixture for the source syntax used by this collection."""
    source = '# Example\n\n## Apple PPG / ECG\n\n## 重复\n\n## 重复\n\n<a id="alias"></a>\n\nInline $x_1$ and literal `$x$` plus escaped \\$5.\n\n| A | B |\n|---|---|\n| $x^2$ | line<br>two |\n\n1. Formula:\n\n   $$\n   x^2 = 4\n   $$\n'
    body, toc, title = render_markdown(source)
    page = Page(body)
    assert title == "Example"
    assert {"apple-ppg--ecg", "重复", "重复-1", "alias"} <= set(page.ids)
    assert page.math == 3, "Inline, table and nested display math must render"
    assert '<code>$x$</code>' in body and "$5" in body, "Code/currency must remain literal"
    assert "<br>" in body and "<table>" in body and 'href="#apple-ppg--ecg"' in toc


if __name__ == "__main__":
    check_renderer()
    count, links, formulas = check_site(ROOT / "_site")
    print(f"Checked {count} pages, {links} local links/resources and {formulas} rendered formulas.")
