"""Render the public Markdown collection into portable, linked article pages.

Markdown stays the only prose source. The output contains an explicit public
document set and its referenced images; no work logs or repository internals.
"""

from __future__ import annotations

import html
import posixpath
import re
import shutil
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote, unquote, urlencode, urlsplit, urlunsplit

from latex2mathml.converter import convert as latex_to_mathml
from markdown_it import MarkdownIt
from mdit_py_plugins.dollarmath import dollarmath_plugin

from lint_md import slugify
from site_metadata import article_description, canonical_url

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".avif", ".ico"}


def article_sources(root: Path) -> list[Path]:
    """Keep published prose explicit, excluding internal records and agent rules."""
    sources = [root / "README.md", root / "CONTRIBUTING.md", root / "docs/STYLE.md"]
    for directory in ("docs/models", "docs/datasets", "blogs"):
        sources.extend(sorted((root / directory).glob("*.md")))
    return sources


def math_html(content: str, options: dict) -> str:
    # Convert before publication so formulas work offline without a CDN or JS.
    mode = "block" if options["display_mode"] else "inline"
    return latex_to_mathml(content, display=mode)


def markdown_parser() -> MarkdownIt:
    parser = MarkdownIt("commonmark", {"html": True}).enable(["table", "strikethrough"])
    parser.use(dollarmath_plugin, renderer=math_html, allow_labels=False, double_inline=True)
    return parser


def heading_text(token) -> str:
    return html.unescape("".join(
        child.content for child in (token.children or [])
        if child.type in {"text", "code_inline", "math_inline", "image"}
    ))


def render_markdown(source: str) -> tuple[str, str, str]:
    """Return HTML, a heading TOC, and title with GitHub-compatible anchors."""
    parser = markdown_parser()
    tokens = parser.parse(source)
    used: set[str] = set(re.findall(r'<a\s[^>]*\bid=["\']([^"\']+)', source))
    headings = []
    title = ""
    for i, token in enumerate(tokens):
        if token.type != "heading_open":
            continue
        label = heading_text(tokens[i + 1])
        base = slugify(label)
        anchor = base
        suffix = 1
        while anchor in used:
            anchor = f"{base}-{suffix}"
            suffix += 1
        used.add(anchor)
        token.attrSet("id", anchor)
        level = int(token.tag[1:])
        if level == 1 and not title:
            title = label
        if 2 <= level <= 3:
            headings.append((level, label, anchor))
    toc = '<ul class="toc-list">' + "".join(
        f'<li class="toc-h{level}"><a href="#{quote(anchor)}">{html.escape(label)}</a></li>'
        for level, label, anchor in headings
    ) + "</ul>"
    body = parser.renderer.render(tokens, parser.options, {})
    # Keep the existing GitHub note readable without requiring a GitHub renderer.
    body = body.replace('<blockquote>\n<p>[!NOTE]', '<blockquote class="note">\n<p><strong>说明</strong><br>')
    return body, toc, title


class ArticleLinks(HTMLParser):
    """Rewrite rendered HTML links and collect only referenced local images."""

    def __init__(self, root: Path, source: Path, pages: set[Path], repository: str, branch: str):
        super().__init__(convert_charrefs=False)
        self.root, self.source, self.pages = root, source, pages
        self.repository, self.branch = repository, branch
        self.parts: list[str] = []
        self.images: set[Path] = set()

    def rewrite(self, value: str, image: bool = False) -> str:
        parsed = urlsplit(value)
        if parsed.scheme or parsed.netloc:
            if parsed.scheme not in {"", "http", "https", "mailto"}:
                raise ValueError(f"Unsupported link in {self.source}: {value}")
            return value
        if not parsed.path:
            return value
        path = unquote(parsed.path)
        target = (self.root / path.lstrip("/") if path.startswith("/") else self.source.parent / path).resolve()
        if not target.is_relative_to(self.root):
            raise ValueError(f"Link leaves repository in {self.source}: {value}")
        rel = target.relative_to(self.root)
        if any(part.startswith(".") or part == "internal_strategy" for part in rel.parts):
            raise ValueError(f"Private path in public document {self.source}: {value}")
        if not target.exists():
            raise ValueError(f"Missing local target in {self.source}: {value}")
        if target in self.pages:
            destination = rel.with_suffix(".html").as_posix()
        elif target.suffix.lower() in IMAGE_EXTENSIONS and rel.parts[0] in {"assets", "docs", "blogs"}:
            self.images.add(target)
            destination = rel.as_posix()
        elif image:
            raise ValueError(f"Unsupported local image in {self.source}: {value}")
        else:
            # Non-article public source files (e.g. LICENSE, YAML and scripts)
            # stay on GitHub; they are never copied into the published bundle.
            if target.suffix == ".md" or (rel.parts[0] not in {"assets", "docs", "blogs", "scripts", "data"} and rel.as_posix() != "LICENSE"):
                raise ValueError(f"Target is outside public site allowlist: {value}")
            kind = "tree" if target.is_dir() else "blob"
            return f"{self.repository}/{kind}/{quote(self.branch, safe='')}/{quote(rel.as_posix())}" + (f"#{quote(unquote(parsed.fragment))}" if parsed.fragment else "")
        relative = posixpath.relpath(destination, self.source.relative_to(self.root).parent.as_posix())
        return urlunsplit(("", "", quote(relative), parsed.query, quote(unquote(parsed.fragment))))

    def handle_starttag(self, tag, attrs):
        self.start(tag, attrs, False)

    def handle_startendtag(self, tag, attrs):
        self.start(tag, attrs, True)

    def start(self, tag: str, attrs: list[tuple[str, str | None]], closed: bool):
        if tag == "table":
            self.parts.append('<div class="table-scroll" tabindex="0" role="region" aria-label="数据对照表">')
        attributes = []
        for name, value in attrs:
            if value is not None and name in {"href", "src"}:
                value = self.rewrite(value, image=(name == "src"))
            attributes.append(name if value is None else f'{name}="{html.escape(value, quote=True)}"')
        if tag == "img" and not any(name == "loading" for name, _ in attrs):
            attributes.append('loading="lazy"')
        self.parts.append("<" + tag + (" " + " ".join(attributes) if attributes else "") + (" />" if closed else ">"))

    def handle_endtag(self, tag):
        self.parts.append(f"</{tag}>")
        if tag == "table":
            self.parts.append("</div>")

    def handle_data(self, data):
        self.parts.append(data)

    def handle_entityref(self, name):
        self.parts.append(f"&{name};")

    def handle_charref(self, name):
        self.parts.append(f"&#{name};")

    def handle_comment(self, data):
        # Generated table delimiters and other maintenance comments are not
        # necessary in the reader-facing artifact.
        pass


def build_articles(root: Path, output: Path, repository: str, branch: str, topics: dict, site_url: str) -> int:
    root, output = root.resolve(), output.resolve()
    sources = article_sources(root)
    pages = set(sources)
    template = (root / "assets/article.html").read_text(encoding="utf-8")
    images: set[Path] = set()
    for source in sources:
        body, toc, title = render_markdown(source.read_text(encoding="utf-8"))
        rewritten = ArticleLinks(root, source, pages, repository, branch)
        rewritten.feed(body)
        rewritten.close()
        images.update(rewritten.images)
        relative = source.relative_to(root)
        prefix = "../" * (len(relative.parts) - 1)
        section, label = "阅读指南", "浏览研究索引"
        catalog = prefix + "index.html#catalog"
        if relative.parent.as_posix() == "docs/models":
            section, label = "专题阅读", "查看本专题条目"
            catalog = prefix + "index.html?" + urlencode({"topic": topics[source.stem]}) + "#catalog"
        elif relative.parent.as_posix() == "docs/datasets":
            section, label = "数据集与队列", "查看数据集索引"
            catalog = prefix + "index.html?view=datasets#catalog"
        elif relative.parts[0] == "blogs":
            section = "技术长文"
        elif relative.as_posix() == "CONTRIBUTING.md":
            section = "参与共建"
        values = {
            "TITLE": html.escape(title or source.stem), "CONTENT": "".join(rewritten.parts),
            "TOC": toc, "ROOT_PREFIX": prefix, "SECTION": html.escape(section),
            "SOURCE_URL": html.escape(f"{repository}/blob/{quote(branch, safe='')}/{quote(relative.as_posix())}", quote=True),
            "EDIT_URL": html.escape(f"{repository}/edit/{quote(branch, safe='')}/{quote(relative.as_posix())}", quote=True),
            "REPOSITORY": html.escape(repository, quote=True),
            "CATALOG_URL": html.escape(catalog, quote=True), "CATALOG_LABEL": label,
            "DESCRIPTION": html.escape(article_description(body, title), quote=True),
            "CANONICAL_URL": html.escape(canonical_url(site_url, relative.with_suffix(".html")), quote=True),
            "SOCIAL_IMAGE": html.escape(site_url + "assets/taxonomy_overview.png", quote=True),
        }
        document = template
        for key, value in values.items():
            document = document.replace(f"@@{key}@@", value)
        if re.search(r"@@[A-Z_]+@@", document):
            raise ValueError(f"Unresolved article template placeholder: {source}")
        destination = output / relative.with_suffix(".html")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(document, encoding="utf-8", newline="\n")
    for image in images:
        destination = output / image.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(image, destination)
    return len(sources)
