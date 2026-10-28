"""从论文 PDF 裁取插图，规则见 docs/STYLE.md 第三节。

论文 PDF 统一放在本地 papers/ 目录（已被 .gitignore 忽略）。

常用流程：
    # 1. 下载 arXiv 论文
    python scripts/figures.py download 2506.09108 --name sensorlm

    # 2. 列出所有图注及其所在页和坐标
    python scripts/figures.py find papers/sensorlm.pdf

    # 3a. 按图注自动裁图（适合大多数单图页面）
    python scripts/figures.py auto papers/sensorlm.pdf "Figure 1" docs/figures/sensorlm/overview.png

    # 3b. 自动结果不理想时，先渲染整页预览，再按坐标手动裁
    python scripts/figures.py page papers/sensorlm.pdf 2 scratch/page2.png
    python scripts/figures.py crop papers/sensorlm.pdf 2 50 45 545 395 docs/figures/sensorlm/overview.png

页码从 1 开始；坐标单位为 PDF point（72 point = 1 inch），原点在页面左上角。
输出图片会缩放到最大宽度 1600 px 并压缩，尽量控制在 500 KB 以内。
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import urllib.request
from pathlib import Path

import pymupdf
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
PAPERS = ROOT / "papers"
MAX_WIDTH = 1600
MAX_BYTES = 500 * 1024
CAPTION = re.compile(r"^\s*(Figure|Fig\.?)\s*(\d+)", re.IGNORECASE)


def save_image(png_bytes: bytes, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    if img.width > MAX_WIDTH:
        img = img.resize((MAX_WIDTH, round(img.height * MAX_WIDTH / img.width)), Image.LANCZOS)

    def encode(im: Image.Image, palette: bool) -> bytes:
        buf = io.BytesIO()
        if out.suffix.lower() in (".jpg", ".jpeg"):
            im.save(buf, "JPEG", quality=85, optimize=True)
        elif palette:
            im.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(
                buf, "PNG", optimize=True
            )
        else:
            im.save(buf, "PNG", optimize=True)
        return buf.getvalue()

    data = encode(img, palette=False)
    if len(data) > MAX_BYTES:
        data = encode(img, palette=True)
    while len(data) > MAX_BYTES and img.width > 900:
        img = img.resize((int(img.width * 0.85), int(img.height * 0.85)), Image.LANCZOS)
        data = encode(img, palette=True)
    out.write_bytes(data)
    print(f"saved {out} ({img.width}x{img.height}, {len(data) // 1024} KB)")


def render(pdf: Path, page_no: int, rect: pymupdf.Rect | None, dpi: int) -> bytes:
    doc = pymupdf.open(pdf)
    page = doc[page_no - 1]
    pix = page.get_pixmap(clip=rect, dpi=dpi)
    return pix.tobytes("png")


def cmd_download(args: argparse.Namespace) -> None:
    PAPERS.mkdir(exist_ok=True)
    arxiv_id = args.arxiv_id.replace("arXiv:", "").strip()
    dest = PAPERS / f"{args.name or arxiv_id}.pdf"
    if dest.exists() and not args.force:
        print(f"exists {dest}")
        return
    req = urllib.request.Request(
        f"https://arxiv.org/pdf/{arxiv_id}", headers={"User-Agent": "Mozilla/5.0 (figures.py)"}
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    if not data.startswith(b"%PDF"):
        sys.exit(f"not a PDF: {arxiv_id}")
    dest.write_bytes(data)
    print(f"saved {dest} ({len(data) // 1024} KB)")


def captions(doc: pymupdf.Document):
    for pno, page in enumerate(doc, 1):
        for b in page.get_text("blocks"):
            x0, y0, x1, y1, text = b[:5]
            m = CAPTION.match(text)
            if m:
                yield pno, page, pymupdf.Rect(x0, y0, x1, y1), int(m.group(2)), " ".join(text.split())


def cmd_find(args: argparse.Namespace) -> None:
    doc = pymupdf.open(args.pdf)
    for pno, page, r, num, text in captions(doc):
        print(f"p{pno:<3} Fig {num:<3} caption=({r.x0:.0f},{r.y0:.0f},{r.x1:.0f},{r.y1:.0f})  {text[:90]}")


def figure_rect(page: pymupdf.Page, cap: pymupdf.Rect) -> pymupdf.Rect | None:
    """估计图注上方插图的范围：图片 + 矢量图形 + 落在其中的文字标签。"""
    W = page.rect.width
    full_width = cap.width > 0.55 * W
    xmin, xmax = (0, W) if full_width else (cap.x0 - 15, cap.x1 + 15)

    graphics = [pymupdf.Rect(i["bbox"]) for i in page.get_image_info()]
    graphics += [d["rect"] for d in page.get_drawings() if d["rect"].width > 2 or d["rect"].height > 2]
    graphics = [g for g in graphics if g.y1 <= cap.y0 + 4 and g.x1 >= xmin and g.x0 <= xmax and g.height < page.rect.height * 0.95]
    if not graphics:
        return None

    region = None
    top = cap.y0
    for g in sorted(graphics, key=lambda r: -r.y1):
        if g.y1 >= top - 25:
            region = pymupdf.Rect(g) if region is None else region | g
            top = min(top, g.y0)
    if region is None:
        return None
    for b in page.get_text("blocks"):
        tb = pymupdf.Rect(b[:4])
        if tb.y0 >= region.y0 - 2 and tb.y1 <= cap.y0 + 1 and tb.x0 >= region.x0 - 30 and tb.x1 <= region.x1 + 30:
            region |= tb
    return pymupdf.Rect(region.x0 - 4, region.y0 - 4, region.x1 + 4, min(region.y1 + 4, cap.y0)) & page.rect


def cmd_auto(args: argparse.Namespace) -> None:
    doc = pymupdf.open(args.pdf)
    want = int(re.search(r"\d+", args.caption).group())
    for pno, page, cap, num, _ in captions(doc):
        if num != want:
            continue
        rect = figure_rect(page, cap)
        if rect is None or rect.height < 40:
            continue
        print(f"p{pno} rect=({rect.x0:.0f},{rect.y0:.0f},{rect.x1:.0f},{rect.y1:.0f})")
        save_image(render(args.pdf, pno, rect, args.dpi), args.out)
        return
    sys.exit(f"could not locate {args.caption}; use `page` + `crop` instead")


def cmd_page(args: argparse.Namespace) -> None:
    save_image(render(args.pdf, args.page, None, args.dpi), args.out)


def cmd_crop(args: argparse.Namespace) -> None:
    rect = pymupdf.Rect(args.x0, args.y0, args.x1, args.y1)
    save_image(render(args.pdf, args.page, rect, args.dpi), args.out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("download", help="下载 arXiv 论文到 papers/")
    p.add_argument("arxiv_id")
    p.add_argument("--name")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("find", help="列出图注位置")
    p.add_argument("pdf", type=Path)
    p.set_defaults(func=cmd_find)

    p = sub.add_parser("auto", help="按图注自动裁图")
    p.add_argument("pdf", type=Path)
    p.add_argument("caption", help='如 "Figure 1"')
    p.add_argument("out", type=Path)
    p.add_argument("--dpi", type=int, default=220)
    p.set_defaults(func=cmd_auto)

    p = sub.add_parser("page", help="渲染整页预览")
    p.add_argument("pdf", type=Path)
    p.add_argument("page", type=int)
    p.add_argument("out", type=Path)
    p.add_argument("--dpi", type=int, default=110)
    p.set_defaults(func=cmd_page)

    p = sub.add_parser("crop", help="按坐标裁图")
    p.add_argument("pdf", type=Path)
    p.add_argument("page", type=int)
    for k in ("x0", "y0", "x1", "y1"):
        p.add_argument(k, type=float)
    p.add_argument("out", type=Path)
    p.add_argument("--dpi", type=int, default=220)
    p.set_defaults(func=cmd_crop)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
