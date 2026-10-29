"""渲染仓库与博客中的 Mermaid 架构图为 2x Retina PNG。

遵循学术简明、扁平化与高保真规范：
- 避免过于花哨的饱和色彩堆叠，采用冷灰色与中性学术配色；
- 统一样式：rx:8px, ry:8px, stroke-width:1.5px；
- 拒绝垂直过长布局，保持 4:3 至 16:9 对称宽幅。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, r"C:\Career\Media")
from src.renderers.diagram_renderer import render_mermaid_string

os.makedirs("assets", exist_ok=True)
os.makedirs("blogs/images", exist_ok=True)
os.makedirs("docs/figures/risq", exist_ok=True)
os.makedirs("docs/figures/health_reasoning", exist_ok=True)

# 1. README 顶层技术架构与模态全景图 (assets/taxonomy_overview.png)
top_html = (
    "<div style='width:820px; text-align:center; padding:10px 0; font-family:system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;'>"
    "<div style='font-size:14px; font-weight:700; color:#0F172A; letter-spacing:0.3px;'>健康认知推理与大语言模型 (Health Reasoning &amp; Multimodal AI)</div>"
    "<div style='font-size:12px; color:#334155; margin-top:5px; line-height:1.6;'>"
    "<b>时序-语言跨模态对齐</b>：SensorLM · PPG-LM · TimeBraid &nbsp;|&nbsp; "
    "<b>临床生理参数物理锚定</b>：GEM (ECG) · CBPM-LLaMA (BP)<br/>"
    "<b>全域认知评测与系统健康</b>：WearableQA · HEARTS · RisQ"
    "</div>"
    "</div>"
)

physio_html = (
    "<div style='width:510px; text-align:left; padding:10px 14px; font-family:system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;'>"
    "<div style='font-size:13px; font-weight:700; color:#0F172A; border-bottom:1px solid #CBD5E1; padding-bottom:4px; margin-bottom:8px;'>"
    "专用生理信号基础模型 (Specialized Physiological FMs)"
    "</div>"
    "<div style='font-size:11.5px; line-height:1.65; color:#334155;'>"
    "• <b>腕戴光电与运动 (PPG / Wearable)</b><br/>"
    "&nbsp;&nbsp;SensorFM · PaPaGei · Pulse-PPG · HiMAE · OpenMHC · LIMU-BERT<br/>"
    "• <b>心脏电生理 (ECG)</b><br/>"
    "&nbsp;&nbsp;ECG-FM · ST-MEM · HeartBEiT · CLOCS<br/>"
    "• <b>连续动态血糖 (CGM)</b><br/>"
    "&nbsp;&nbsp;GlucoFM · CGM-JEPA · GluFormer · CGMformer · CGM-LSM<br/>"
    "• <b>多导睡眠与心肺 (Sleep PSG)</b><br/>"
    "&nbsp;&nbsp;SleepFM-1/2 · SleepMaMi · U-Sleep<br/>"
    "• <b>跨模态生理生成与虚拟传感 (Cross-Modal Generative)</b><br/>"
    "&nbsp;&nbsp;PPGFlowECG (PPG→ECG) · UniCardio (多模态扩散) · xMAE"
    "</div>"
    "</div>"
)

tsfm_html = (
    "<div style='width:290px; text-align:left; padding:10px 14px; font-family:system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;'>"
    "<div style='font-size:13px; font-weight:700; color:#334155; border-bottom:1px solid #CBD5E1; padding-bottom:4px; margin-bottom:8px;'>"
    "通用时序基础模型 (General TSFM)"
    "</div>"
    "<div style='font-size:11.5px; line-height:1.65; color:#475569;'>"
    "<b>跨界对照与通用特征底座</b><br/>"
    "• <b>预测基模</b>：TimesFM · Chronos<br/>"
    "• <b>多任务掩码</b>：MOMENT · MOIRAI<br/>"
    "• <b>轻量判别</b>：Mantis (端侧分类)<br/>"
    "• <b>概率预测</b>：Timestral (DHF 解耦)<br/>"
    "• <b>检索增强</b>：Kumo-Forecast (时序 RAG)<br/>"
    "• <b>语言重编程</b>：Time-LLM (原型对齐)"
    "</div>"
    "</div>"
)

data_html = (
    "<div style='width:820px; text-align:left; padding:10px 14px; font-family:system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;'>"
    "<div style='font-size:13px; font-weight:700; color:#0F172A; border-bottom:1px solid #CBD5E1; padding-bottom:4px; margin-bottom:6px;'>"
    "开放数据集与临床队列底座 (Open Datasets &amp; Clinical Cohorts)"
    "</div>"
    "<div style='font-size:11.5px; color:#334155; line-height:1.65;'>"
    "<b>日常穿戴与动态血糖</b>：OpenMHC · PPG-DaLiA · WESAD · CAS-BP · TROIKA · ShanghaiT1D/T2D · CGMacros · GlucoBench<br/>"
    "<b>临床监护与人群队列</b>：MC-MED · VitalDB · MIMIC-IV-ECG · PTB-XL · CODE-15% · SHHS · MESA · UK Biobank · All of Us"
    "</div>"
    "</div>"
)

def make_label(text: str) -> str:
    return f"<span style='background:#F8FAFC; color:#475569; padding:2px 6px; border-radius:4px; font-size:11px; border:1px solid #E2E8F0; font-family:system-ui;'>{text}</span>"

mmd_taxonomy = f"""flowchart TB
    classDef topBrain fill:#F8FAFC,stroke:#334155,stroke-width:1.5px,rx:8px,ry:8px;
    classDef physioCol fill:#FFFFFF,stroke:#475569,stroke-width:1.5px,rx:8px,ry:8px;
    classDef tsfmCol fill:#F8FAFC,stroke:#64748B,stroke-width:1.5px,stroke-dasharray: 4 2,rx:8px,ry:8px;
    classDef bottomData fill:#F8FAFC,stroke:#334155,stroke-width:1.5px,rx:8px,ry:8px;

    REASONING["{top_html}"]:::topBrain
    PHYSIO["{physio_html}"]:::physioCol
    TSFM["{tsfm_html}"]:::tsfmCol
    DATA["{data_html}"]:::bottomData

    REASONING ==>|"{make_label('时序-语言跨模态对齐与临床决策')}"| PHYSIO
    REASONING ==>|"{make_label('通用推理与假设性情景推演')}"| TSFM
    PHYSIO ==>|"{make_label('自监督预训练与生理表征评测')}"| DATA
    TSFM ==>|"{make_label('通用时序模式预训练与跨域验证')}"| DATA

    linkStyle default stroke:#64748B,stroke-width:1.5px;
"""

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
    p1 = render_mermaid_string(mmd_taxonomy, "assets/taxonomy_overview.png", scale=2)
    print("Rendered:", p1)
    p2 = render_mermaid_string(mmd_edge_cloud, "blogs/images/edge_cloud_health_agent.png", scale=2)
    print("Rendered:", p2)
    p3 = render_mermaid_string(mmd_risq, "docs/figures/risq/multimodal_coexistence.png", scale=2)
    print("Rendered:", p3)
    p4 = render_mermaid_string(mmd_hierarchical, "docs/figures/health_reasoning/hierarchical_ai_stack.png", scale=2)
    print("Rendered:", p4)
