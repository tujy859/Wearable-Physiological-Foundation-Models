"""Markdown 格式与文风检查，规则见 docs/STYLE.md。

用法：
    python scripts/lint_md.py              # 检查 README、CONTRIBUTING、docs/、blogs/
    python scripts/lint_md.py docs/models/cgm_models.md
    python scripts/lint_md.py --errors-only

退出码：存在 error 时为 1。
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TARGETS = ["README.md", "CONTRIBUTING.md", "docs", "blogs"]

# 只做格式检查、不做用词检查的文件（规范本身含有示例词；博客保留作者原文风格）
WORD_CHECK_EXEMPT = {"docs/STYLE.md"}
WORD_CHECK_EXEMPT_DIRS = ("blogs/",)

BANNED_WORDS = [
    "神级", "巨座", "颠覆", "革命性", "降维打击", "解毒", "完美", "极致", "终局", "天花板",
    "旗舰", "黄金底座", "黄金标准", "黄金基准", "标杆", "里程碑", "重磅",
    "已核实", "已验证", "经核实",
]
CAUTION_WORDS = ["首创", "全面超越", "彻底"]

SIMPLE_MATH = re.compile(
    r"\$\s*\\(to|rightarrow|leftarrow|longrightarrow|longleftrightarrow|leftrightarrow|le|ge|leq|geq|times|pm|approx)\s*\$"
)
EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\uFE0F\u203C\u2049\u2122\u2139\u231A-\u23FF]"
)


def is_punct(ch: str) -> bool:
    return unicodedata.category(ch)[0] in "PS"


def is_space(ch: str) -> bool:
    return ch == "" or ch.isspace()


def slugify(text: str) -> str:
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("`", "").replace("*", "")
    text = text.strip().lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


class Report:
    def __init__(self) -> None:
        self.items: list[tuple[str, str, int, str, str]] = []

    def add(self, level: str, path: Path, line: int, code: str, msg: str) -> None:
        self.items.append((level, rel(path), line, code, msg))


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def strip_inline_code(line: str) -> str:
    """把行内代码替换为等长占位符，保留列位置。"""
    return re.sub(r"(`+)(.+?)\1", lambda m: "\x00" * len(m.group(0)), line)


def iter_content_lines(text: str):
    """逐行返回 (行号, 原始行, 去掉代码后的行)，跳过代码块与 HTML 注释。"""
    in_fence = False
    fence = ""
    in_comment = False
    for no, line in enumerate(text.splitlines(), 1):
        stripped = line.lstrip()
        if in_comment:
            if "-->" in line:
                in_comment = False
            continue
        m = re.match(r"(```+|~~~+)", stripped)
        if m:
            if not in_fence:
                in_fence, fence = True, m.group(1)[:3]
            elif stripped.startswith(fence):
                in_fence = False
            continue
        if in_fence:
            continue
        if stripped.startswith("<!--") and "-->" not in stripped:
            in_comment = True
            continue
        clean = re.sub(r"<!--.*?-->", "", line)
        yield no, line, strip_inline_code(clean)


def headings_of(path: Path) -> set[str]:
    slugs: set[str] = set()
    counts: dict[str, int] = {}
    if not path.exists():
        return slugs
    for _, _, clean in iter_content_lines(path.read_text(encoding="utf-8")):
        m = re.match(r"\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$", clean)
        if not m:
            continue
        base = slugify(m.group(2))
        n = counts.get(base, 0)
        slugs.add(base if n == 0 else f"{base}-{n}")
        counts[base] = n + 1
    for m in re.finditer(r'<a\s+(?:name|id)="([^"]+)"', path.read_text(encoding="utf-8")):
        slugs.add(m.group(1))
    return slugs


def check_bold(path: Path, no: int, clean: str, rep: Report) -> None:
    positions = [m.start() for m in re.finditer(r"(?<![\\*])\*\*(?!\*)", clean)]
    if len(positions) % 2:
        rep.add("error", path, no, "bold-unpaired", "本行 ** 数量为奇数，加粗可能未闭合")
        return
    for i in range(0, len(positions), 2):
        o, c = positions[i], positions[i + 1]
        inner = clean[o + 2 : c]
        if not inner:
            continue
        before = clean[o - 1] if o > 0 else ""
        after = clean[c + 2] if c + 2 < len(clean) else ""
        first, last = inner[0], inner[-1]
        if first.isspace() or last.isspace():
            rep.add("error", path, no, "bold-space", f"加粗内容首尾不能是空格：**{inner}**")
            continue
        if is_punct(first) and not (is_space(before) or is_punct(before)):
            rep.add("error", path, no, "bold-flank", f"加粗以标点开头且前面紧挨文字，GitHub 不会渲染：**{inner}**")
        if is_punct(last) and not (is_space(after) or is_punct(after)):
            rep.add("error", path, no, "bold-flank", f"加粗以标点结尾且后面紧挨文字，GitHub 不会渲染：**{inner}**")


def check_links(path: Path, no: int, clean: str, rep: Report, cache: dict) -> None:
    for m in re.finditer(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)", clean):
        target = m.group(1)
        if target.startswith("file:"):
            rep.add("error", path, no, "link-file", f"不要使用 file:/// 链接：{target}")
            continue
        if re.match(r"[a-z]+:", target) or target.startswith("//"):
            continue
        file_part, _, anchor = target.partition("#")
        dest = path if not file_part else (path.parent / file_part)
        if file_part and re.match(r"^[A-Za-z]:[\\/]", file_part):
            rep.add("error", path, no, "link-abs", f"不要使用本机绝对路径：{target}")
            continue
        if not dest.exists():
            rep.add("error", path, no, "link-missing", f"链接目标不存在：{target}")
            continue
        if m.group(0).startswith("!") and dest.is_file() and dest.stat().st_size > 500 * 1024:
            rep.add("warn", path, no, "image-size", f"图片超过 500 KB：{target}（{dest.stat().st_size // 1024} KB）")
        if anchor and dest.suffix == ".md":
            key = dest.resolve()
            if key not in cache:
                cache[key] = headings_of(dest)
            if anchor.lower() not in cache[key]:
                rep.add("error", path, no, "link-anchor", f"锚点不存在：{target}")


def lint_file(path: Path, rep: Report, cache: dict) -> None:
    text = path.read_text(encoding="utf-8")
    r = rel(path)
    check_words = r not in WORD_CHECK_EXEMPT and not r.startswith(WORD_CHECK_EXEMPT_DIRS)
    prev_blank = True
    for no, raw, clean in iter_content_lines(text):
        heading = re.match(r"\s{0,3}#{1,6}\s+(.*)", clean)
        if heading:
            if EMOJI.search(heading.group(1)):
                rep.add("error", path, no, "heading-emoji", "标题中不要使用 emoji")
            if "**" in heading.group(1):
                rep.add("error", path, no, "heading-bold", "标题中不要加粗")
        elif EMOJI.search(clean):
            rep.add("error", path, no, "emoji", f"不要使用 emoji：{''.join(sorted(set(EMOJI.findall(clean))))}")

        check_bold(path, no, clean, rep)

        for m in re.finditer(r"(?<![~\\])~(?!~)", clean):
            rep.add("error", path, no, "tilde", "半角 ~ 可能被渲染为删除线，范围请用 - 或 ～，近似请写“约”")
            break
        if "\\~" in clean:
            rep.add("warn", path, no, "tilde-escaped", "近似值建议写“约”，范围建议用 - 或 ～")

        if SIMPLE_MATH.search(clean):
            rep.add("warn", path, no, "simple-math", "简单符号请用 Unicode（→ ≤ ≥ × ±），不要用 LaTeX")
        if re.search(r"(?<!\$)\$\$(?!\$).+\$\$", clean):
            rep.add("error", path, no, "display-math-inline", "独立公式的 $$ 需要各占一行")

        if re.match(r"\s+>\s*\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]", raw):
            rep.add("error", path, no, "alert-nested", "GitHub 提示框只能放在顶层，不能嵌在列表里")

        check_links(path, no, clean, rep, cache)

        if check_words:
            for w in BANNED_WORDS:
                if w in clean:
                    rep.add("error", path, no, "word-banned", f"禁用词：{w}")
            for w in CAUTION_WORDS:
                if w in clean:
                    rep.add("warn", path, no, "word-caution", f"慎用词：{w}")
        prev_blank = not raw.strip()


def collect(targets: list[str]) -> list[Path]:
    files: list[Path] = []
    for t in targets:
        p = (ROOT / t) if not Path(t).is_absolute() else Path(t)
        if p.is_dir():
            files.extend(sorted(p.rglob("*.md")))
        elif p.exists():
            files.append(p)
    return files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="要检查的文件或目录")
    ap.add_argument("--errors-only", action="store_true", help="只显示 error")
    args = ap.parse_args()

    rep = Report()
    cache: dict = {}
    for f in collect(args.paths or DEFAULT_TARGETS):
        lint_file(f, rep, cache)

    errors = [i for i in rep.items if i[0] == "error"]
    warns = [i for i in rep.items if i[0] == "warn"]
    for level, p, line, code, msg in rep.items:
        if args.errors_only and level != "error":
            continue
        print(f"{p}:{line}: {level} [{code}] {msg}")
    print(f"\n{len(errors)} error(s), {len(warns)} warning(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
