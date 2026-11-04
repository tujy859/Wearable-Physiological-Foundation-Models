"""使用本地 Mermaid CLI 和 Chromium 渲染 2x Retina 图。

运行 python scripts/render_diagrams.py 渲染全部图。
运行 python scripts/render_diagrams.py --only taxonomy 仅更新首页 PNG/SVG。
可通过 PUPPETEER_EXECUTABLE_PATH 或 CHROME_PATH 指定浏览器。

遵循学术简明、扁平化与高保真规范：
- 首页采用低饱和绿、橙、蓝与板岩灰，分别表示输入、模型、能力与支撑资源；
- 统一样式：rx:10px, ry:10px, stroke-width:2px；
- 拒绝垂直过长布局，保持 4:3 至 16:9 对称宽幅。
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
FONT_FAMILY = "system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif"


def browser_options(work: Path) -> dict:
    browser_paths = [
        os.environ.get("PUPPETEER_EXECUTABLE_PATH", ""),
        os.environ.get("CHROME_PATH", ""),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    ]
    browser = next((path for path in browser_paths if path and Path(path).is_file()), None)
    options = {
        "pipe": True,
        "userDataDir": str(work / "chrome"),
        "args": [
            "--no-sandbox", "--disable-gpu",
            "--disable-crash-reporter", "--disable-features=Crashpad",
        ],
    }
    if browser:
        options["executablePath"] = browser
    return options


def render_mermaid_string(source: str, destination: str, scale: int = 2) -> Path:
    """Render with the installed Mermaid CLI and a local Chromium browser."""
    executable = shutil.which("mmdc")
    if not executable:
        raise RuntimeError("Install @mermaid-js/mermaid-cli and add mmdc to PATH.")
    command = [executable]
    if os.name == "nt":
        cli = Path(executable).parent / "node_modules/@mermaid-js/mermaid-cli/src/cli.js"
        node = shutil.which("node")
        if cli.exists() and node:
            command = [node, str(cli)]

    output = ROOT / destination
    output.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "theme": "base",
        "look": "classic",
        "securityLevel": "loose",
        "fontFamily": FONT_FAMILY,
        "themeVariables": {
            "fontFamily": FONT_FAMILY,
            "fontSize": "20px",
            "primaryTextColor": "#0F172A",
            "lineColor": "#64748B",
            "clusterBkg": "#FFFFFF",
            "clusterBorder": "#FFFFFF",
        },
        "flowchart": {
            "htmlLabels": True,
            "curve": "linear",
            "nodeSpacing": 18,
            "rankSpacing": 18,
            "padding": 10,
            "diagramPadding": 16,
        },
        "themeCSS": (
            ".node rect { filter: none; } "
            ".cluster-label { font-weight: 700; } "
            ".flowchartTitleText { font-size: 28px; font-weight: 700; }"
        ),
    }
    with tempfile.TemporaryDirectory(prefix=".mermaid-", dir=output.parent) as directory:
        work = Path(directory)
        source_path = work / "diagram.mmd"
        source_path.write_text(source, encoding="utf-8")
        config_path = work / "mermaid.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        args = command + [
            "-i", str(source_path), "-o", str(output),
            "-c", str(config_path), "-b", "white", "-s", str(scale),
        ]
        browser_config = work / "puppeteer.json"
        browser_config.write_text(json.dumps(browser_options(work)), encoding="utf-8")
        args += ["-p", str(browser_config)]
        environment = os.environ.copy()
        environment.update({"TEMP": str(work), "TMP": str(work)})
        subprocess.run(args, check=True, cwd=ROOT, env=environment)
    return output


def normalize_taxonomy_layout(svg: str) -> str:
    """Keep Mermaid labels/topology, then place its nodes on a compact 16:9 grid.

    Dagre inserts large, version-dependent gaps between nested subgraphs. This
    small post-layout pass fixes the publication canvas without shrinking text.
    The source remains assets/taxonomy_overview.mmd; unexpected topology fails
    explicitly instead of silently placing a new node in the wrong category.
    """
    namespace = "http://www.w3.org/2000/svg"
    ET.register_namespace("", namespace)
    ET.register_namespace("html", "http://www.w3.org/1999/xhtml")
    root = ET.fromstring(svg)
    q = lambda tag: f"{{{namespace}}}{tag}"
    positions = {
        "A1": (210, 157), "A2": (210, 333), "A3": (210, 509),
        "M1": (640, 157), "M2": (640, 333), "M3": (640, 509),
        "T": (1035, 333),
    }
    found = set()
    for element in root.iter(q("g")):
        if "node" in element.get("class", "").split():
            match = re.search(r"-flowchart-(.+)-\d+$", element.get("id", ""))
            if not match or match.group(1) not in positions:
                raise RuntimeError(f"Unexpected taxonomy node: {element.get('id')}")
            name = match.group(1)
            rectangle = element.find(q("rect"))
            expected_size = (270, 510) if name == "T" else (380, 158)
            if rectangle is None or (
                float(rectangle.get("width", "0")), float(rectangle.get("height", "0"))
            ) != expected_size:
                raise RuntimeError(f"Update the fixed taxonomy grid for resized node {name}.")
            x, y = positions[name]
            element.set("transform", f"translate({x}, {y})")
            found.add(name)
            if name == "A1":
                # Mermaid's image CSS makes a floated HTML image a full line.
                # Preserve the source glyph, placing it beside the title in SVG.
                for parent in element.iter():
                    for child in list(parent):
                        if child.tag == "{http://www.w3.org/1999/xhtml}img":
                            ET.SubElement(root, q("image"), {
                                "href": child.get("src", ""), "x": "285", "y": "88",
                                "width": "90", "height": "27",
                            })
                            parent.remove(child)
    if found != set(positions):
        raise RuntimeError(f"Unexpected taxonomy nodes: {sorted(found)}")
    expected_edges = {(f"A{i}", f"M{i}") for i in range(1, 4)} | {
        (f"M{i}", "T") for i in range(1, 4)
    }
    found_edges = set()
    for path in root.iter(q("path")):
        match = re.search(r"L_([AM][123])_([AM][123]|T)_\d+$", path.get("id", ""))
        if match:
            source, target = match.groups()
            found_edges.add((source, target))
            y = positions[source][1]
            start, end = (400, 446) if source.startswith("A") else (830, 896)
            path.set("d", f"M{start},{y} L{end},{y}")
            path.attrib.pop("data-points", None)
    if found_edges != expected_edges:
        raise RuntimeError(f"Unexpected taxonomy edges: {sorted(found_edges)}")
    # Transparent Mermaid clusters retain their semantic labels, aligned here.
    for parent in root.iter():
        for child in list(parent):
            if "clusters" in child.get("class", "").split():
                parent.remove(child)
    for element in root.iter(q("text")):
        if element.get("class") == "flowchartTitleText":
            element.set("x", "600")
            element.set("y", "31")
    for x, label in ((210, "01　输入数据"), (640, "02　基础表征"), (1035, "03　共享下游能力")):
        heading = ET.SubElement(root, q("text"), {
            "x": str(x), "y": "65", "text-anchor": "middle",
            "font-size": "22", "font-weight": "700", "fill": "#0F172A",
        })
        heading.text = label
    root.set("viewBox", "0 0 1200 675")
    root.set("style", "max-width:1200px; background-color:white;")
    root.set("width", "1200")
    root.set("height", "675")
    result = ET.tostring(root, encoding="unicode")
    # HTML parsers do not interpret prefixed XHTML tag names inside foreignObject.
    result = result.replace(' xmlns:html="http://www.w3.org/1999/xhtml"', "")
    result = result.replace("<html:", "<").replace("</html:", "</")
    return re.sub(
        r"(<foreignObject\b[^>]*>)<div\b",
        r'\1<div xmlns="http://www.w3.org/1999/xhtml"', result,
    )


def render_taxonomy() -> None:
    """Render a readable 1200 × 675 overview and export its 2x PNG."""
    svg_path = render_mermaid_string(mmd_taxonomy, "assets/taxonomy_overview.svg")
    svg = normalize_taxonomy_layout(svg_path.read_text(encoding="utf-8"))
    footer = (
        f'<g font-family="{FONT_FAMILY}" font-size="18" fill="#334155">'
        '<rect x="20" y="601" width="1160" height="62" '
        'rx="10" fill="#F8FAFC" stroke="#64748B" stroke-width="2"/>'
        '<text x="36" y="625">'
        '<tspan font-weight="700">语言对齐与健康推理</tspan>　SensorLM · GEM　　'
        '<tspan font-weight="700">数据与评测</tspan>　OpenMHC · WearableQA · HEARTS</text>'
        '<text x="36" y="650">'
        '<tspan font-weight="700">通用时序对照</tspan>　TimesFM · Chronos　　'
        '按输入表示组织研究路线，具体能力取决于训练与评测任务。</text></g>'
    )
    svg = svg.replace('</svg>', footer + '</svg>')
    svg_path.write_text(svg, encoding="utf-8")

    cli = Path(shutil.which("mmdc"))
    package = cli.parent / "node_modules/@mermaid-js/mermaid-cli/node_modules/puppeteer"
    if not package.is_dir():
        npm = shutil.which("npm")
        if not npm:
            raise RuntimeError("npm is required to locate Mermaid CLI's Puppeteer dependency.")
        global_modules = subprocess.check_output([npm, "root", "-g"], text=True).strip()
        package = Path(global_modules) / "@mermaid-js/mermaid-cli/node_modules/puppeteer"
    script = r"""
const fs = require('node:fs');
const puppeteer = require(process.argv[2]);
(async () => {
    const options = JSON.parse(process.argv[3]);
    const browser = await puppeteer.launch(options);
    try {
        const page = await browser.newPage();
        await page.setViewport({ width: Number(process.argv[6]), height: Number(process.argv[7]), deviceScaleFactor: 2 });
        const svg = fs.readFileSync(process.argv[4], 'utf8');
        await page.setContent('<html><body style="margin:0;background:white">' + svg + '</body></html>');
        await page.evaluate(() => document.fonts.ready);
        await page.$eval('svg', element => { element.style.maxWidth = 'none'; element.style.width = '100%'; });
        await page.screenshot({ path: process.argv[5], fullPage: true });
    } finally {
        await browser.close();
    }
})();
"""
    with tempfile.TemporaryDirectory(prefix=".mermaid-", dir=svg_path.parent) as directory:
        work = Path(directory)
        script_path = work / "rasterize.cjs"
        script_path.write_text(script, encoding="utf-8")
        environment = os.environ.copy()
        environment.update({"TEMP": str(work), "TMP": str(work)})
        subprocess.run([
            shutil.which("node"), str(script_path), str(package), json.dumps(browser_options(work)),
            str(svg_path), str(svg_path.with_suffix(".png")), "1200", "675",
        ], check=True, env=environment)
    print("Rendered", svg_path.with_suffix(".png"))
    print("Rendered", svg_path)


# 1. 首页研究地图。声明式源文件与 PNG/SVG 一起维护。
mmd_taxonomy = (ROOT / "assets/taxonomy_overview.mmd").read_text(encoding="utf-8")

# 2. 博客 2 图 10：端云分层协同的时序-语言健康 Agent 架构 (blogs/images/edge_cloud_health_agent.png)
mmd_edge_cloud = """flowchart TB
    classDef cloudBrain fill:#F8FAFC,stroke:#334155,stroke-width:1.5px,rx:8px,ry:8px;
    classDef edgeSensor fill:#FFFFFF,stroke:#475569,stroke-width:1.5px,rx:8px,ry:8px;
    classDef toolBox fill:#F8FAFC,stroke:#64748B,stroke-width:1.5px,rx:8px,ry:8px;

    BRAIN["<div style='width:580px; text-align:center; padding:8px 0; font-family:system-ui, -apple-system, Segoe UI, Roboto, sans-serif;'><strong style='font-size:14px; color:#0F172A;'>云端 / 手机推理中枢 (LLM Reasoning Brain)</strong><br/><span style='font-size:12px; color:#334155; line-height:1.6;'>• 医学知识检索 (EHR / 临床指南) 与连续因果链推导<br/>• 交互式反事实推演 (What-If) 与 Agent 任务规划调度</span></div>"]:::cloudBrain

    EDGE["<div style='width:270px; text-align:left; padding:8px 10px; font-family:system-ui, -apple-system, Segoe UI, Roboto, sans-serif;'><strong style='font-size:13px; color:#0F172A;'>端侧轻量生理时序基模</strong><br/><span style='font-size:11.5px; color:#334155; line-height:1.6;'>• 1D U-Net / Patch 编码器<br/>• 毫秒级波形特征提取与去噪<br/>• 连续物理时间 RoPE 锚定</span></div>"]:::edgeSensor

    TOOLS["<div style='width:270px; text-align:left; padding:8px 10px; font-family:system-ui, -apple-system, Segoe UI, Roboto, sans-serif;'><strong style='font-size:13px; color:#334155;'>CodeAct 与专业信号工具库</strong><br/><span style='font-size:11.5px; color:#475569; line-height:1.6;'>• pyPPG / SciPy 数字滤波与峰值检测<br/>• HRV 频域分析与 SQI 质量评估<br/>• 动态绘图与指南阈值核验</span></div>"]:::toolBox

    BRAIN ==>|"<span style='background:#F8FAFC; color:#475569; padding:2px 4px; font-size:11px; border:1px solid #E2E8F0; border-radius:4px;'>生理 Token 上传 / 高频重采样</span>"| EDGE
    BRAIN ==>|"<span style='background:#F8FAFC; color:#475569; padding:2px 4px; font-size:11px; border:1px solid #E2E8F0; border-radius:4px;'>执行 Python 脚本 / 返回精确指标</span>"| TOOLS

    linkStyle default stroke:#64748B,stroke-width:1.5px;
"""

# 3. RisQ 多模态共生示意图 (docs/figures/risq/multimodal_coexistence.png)
mmd_risq = """flowchart LR
    classDef edgeBox fill:#FFFFFF,stroke:#475569,stroke-width:1.5px,rx:8px,ry:8px;
    classDef centerBox fill:#F8FAFC,stroke:#334155,stroke-width:1.5px,rx:8px,ry:8px;
    classDef staticBox fill:#F8FAFC,stroke:#64748B,stroke-width:1.5px,rx:8px,ry:8px;

    EDGE["<div style='width:240px; text-align:left; padding:6px 8px; font-family:system-ui;'><strong style='font-size:13px; color:#0F172A;'>动态穿戴时序感知</strong><br/><span style='font-size:11.5px; color:#334155; line-height:1.6;'>• 智能手表与体动仪传感器<br/>• 提取连续生理特征与昼夜节律</span></div>"]:::edgeBox
    STATIC["<div style='width:240px; text-align:left; padding:6px 8px; font-family:system-ui;'><strong style='font-size:13px; color:#334155;'>静态组学与临床档案</strong><br/><span style='font-size:11.5px; color:#475569; line-height:1.6;'>• 全基因组变异与生化化验<br/>• 长期住院与就诊 ICD-10 记录</span></div>"]:::staticBox
    RISQ["<div style='width:240px; text-align:left; padding:6px 8px; font-family:system-ui;'><strong style='font-size:13px; color:#0F172A;'>系统级多模态健康模型</strong><br/><span style='font-size:11.5px; color:#334155; line-height:1.6;'>• 统一跨模态潜表征空间<br/>• 自然语言 Prompt 零样本风险推演</span></div>"]:::centerBox

    EDGE ==>|"<span style='background:#F8FAFC; color:#475569; padding:2px 4px; font-size:11px; border:1px solid #E2E8F0; border-radius:4px;'>高频动态生理输入</span>"| RISQ
    STATIC ==>|"<span style='background:#F8FAFC; color:#475569; padding:2px 4px; font-size:11px; border:1px solid #E2E8F0; border-radius:4px;'>先天遗传与临床基线</span>"| RISQ

    linkStyle default stroke:#64748B,stroke-width:1.5px;
"""

# 4. 健康推理三层分工体系 (docs/figures/health_reasoning/hierarchical_ai_stack.png)
mmd_hierarchical = """flowchart TB
    classDef l3Brain fill:#F8FAFC,stroke:#334155,stroke-width:1.5px,rx:8px,ry:8px;
    classDef l2Tools fill:#F8FAFC,stroke:#64748B,stroke-width:1.5px,rx:8px,ry:8px;
    classDef l1Edge fill:#FFFFFF,stroke:#475569,stroke-width:1.5px,rx:8px,ry:8px;

    L3["<div style='width:560px; text-align:center; padding:8px 0; font-family:system-ui;'><strong style='font-size:14px; color:#0F172A;'>顶层 · 认知与健康决策中枢 (Cognitive &amp; Reasoning Layer)</strong><br/><span style='font-size:12px; color:#334155; line-height:1.6;'>• 健康推理 Agent (WearableQA / 临床指南循证库)<br/>• 跨模态健康画像、纵向病理因果推演与前瞻干预建议</span></div>"]:::l3Brain

    L2["<div style='width:560px; text-align:center; padding:8px 0; font-family:system-ui;'><strong style='font-size:13px; color:#334155;'>中层 · 神经符号与工具调度执行层 (Neuro-Symbolic &amp; CodeAct Layer)</strong><br/><span style='font-size:12px; color:#475569; line-height:1.6;'>• Python 科学计算运行时 (SciPy / NeuroKit2 / BioSPPy)<br/>• 动态高精度数值积分、假设检验与频域功率谱解算</span></div>"]:::l2Tools

    L1["<div style='width:560px; text-align:center; padding:8px 0; font-family:system-ui;'><strong style='font-size:13px; color:#0F172A;'>底层 · 微观高频物理感知与生理基座 (Edge Sensor Foundation Models)</strong><br/><span style='font-size:12px; color:#334155; line-height:1.6;'>• 专用时序基座: SensorFM / PaPaGei / UniCardio / SleepFM / GlucoFM<br/>• 毫秒级滤波去噪、运动伪影抑制与连续潜表征抽取</span></div>"]:::l1Edge

    L3 ==>|"<span style='background:#F8FAFC; color:#475569; padding:2px 4px; font-size:11px; border:1px solid #E2E8F0; border-radius:4px;'>意图解析与计算指令派发</span>"| L2
    L2 ==>|"<span style='background:#F8FAFC; color:#475569; padding:2px 4px; font-size:11px; border:1px solid #E2E8F0; border-radius:4px;'>提取生理指标特征 / 调度波形计算</span>"| L1

    linkStyle default stroke:#64748B,stroke-width:1.5px;
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=["taxonomy"], help="Only render the README overview.")
    args = parser.parse_args()
    render_taxonomy()
    if args.only != "taxonomy":
        for source, destination in (
            (mmd_edge_cloud, "blogs/images/edge_cloud_health_agent.png"),
            (mmd_risq, "docs/figures/risq/multimodal_coexistence.png"),
            (mmd_hierarchical, "docs/figures/health_reasoning/hierarchical_ai_stack.png"),
        ):
            print("Rendered", render_mermaid_string(source, destination))
