"""渲染仓库与博客中的 Mermaid 架构图为 2x Retina PNG。

遵循 AGENTS.md 中的视觉与布局规范：
- 认知推理 / 顶层层：fill:#EFF6FF, stroke:#3B82F6
- 端侧感知 / 传感器：fill:#F0FDF4, stroke:#10B981
- 时序基模 / 生成层：fill:#FFF7ED, stroke:#F97316
- 数据底座 / 基础设施：fill:#F8FAFC, stroke:#64748B
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, r"C:\Career\Media")
from src.renderers.diagram_renderer import render_mermaid_string

os.makedirs("assets", exist_ok=True)
os.makedirs("blogs/images", exist_ok=True)

# 1. README 顶层技术分层与模态全景图 (assets/taxonomy_overview.png)
mmd_taxonomy = """flowchart TB
    classDef l3Brain fill:#EFF6FF,stroke:#3B82F6,stroke-width:2px,rx:10px,ry:10px;
    classDef l2Model fill:#FFF7ED,stroke:#F97316,stroke-width:2px,rx:10px,ry:10px;
    classDef l1Edge fill:#F0FDF4,stroke:#10B981,stroke-width:2px,rx:10px,ry:10px;
    classDef l0Data fill:#F8FAFC,stroke:#64748B,stroke-width:2px,rx:10px,ry:10px;

    L3["<div style='width:600px; text-align:center; line-height:1.7; font-family:system-ui, -apple-system, Segoe UI, Roboto, sans-serif;'><strong style='font-size:15px; color:#1E3A8A;'>Layer 3 · 健康认知推理、跨模态对齐与全景风险模型</strong><br/><span style='font-size:12px; color:#1D4ED8;'><b>语言对齐与推理</b>：SensorLM · PPG-LM · GEM · CBPM-LLaMA · TimeBraid<br/><b>评测基准与系统健康</b>：WearableQA · HEARTS · RisQ</span></div>"]:::l3Brain

    L2_LEFT["<div style='width:280px; text-align:left; line-height:1.7; font-family:system-ui, -apple-system, Segoe UI, Roboto, sans-serif;'><strong style='font-size:14px; color:#065F46;'>Layer 1 · 腕戴、心电与睡眠基座</strong><br/><span style='font-size:12px; color:#047857;'>• <b>腕戴 PPG</b>：PaPaGei · Pulse-PPG · AnyPPG<br/>• <b>多模态与端侧</b>：SensorFM · HiMAE · OpenMHC<br/>• <b>12 导联心电</b>：ECG-FM · ST-MEM · HeartBEiT<br/>• <b>多导睡眠 PSG</b>：SleepFM-1/2 · SleepMaMi</span></div>"]:::l1Edge

    L2_RIGHT["<div style='width:280px; text-align:left; line-height:1.7; font-family:system-ui, -apple-system, Segoe UI, Roboto, sans-serif;'><strong style='font-size:14px; color:#9A3412;'>Layer 2 · 血糖、生成与通用时序</strong><br/><span style='font-size:12px; color:#C2410C;'>• <b>CGM 血糖</b>：GlucoFM · GluFormer · CGMformer<br/>• <b>跨模态生成</b>：PPGFlowECG · UniCardio · xMAE<br/>• <b>通用时序</b>：TimesFM · Chronos · MOMENT<br/>• <b>轻量与检索</b>：Timestral · Kumo-Forecast</span></div>"]:::l2Model

    L0["<div style='width:600px; text-align:center; line-height:1.7; font-family:system-ui, -apple-system, Segoe UI, Roboto, sans-serif;'><strong style='font-size:15px; color:#334155;'>Layer 0 · 开放数据集与临床队列底座</strong><br/><span style='font-size:12px; color:#475569;'><b>穿戴与血糖</b>：OpenMHC · PPG-DaLiA · WESAD · CAS-BP · CGMacros · GlucoBench<br/><b>临床与人群队列</b>：MC-MED · MIMIC-IV-ECG · PTB-XL · CODE-15% · SHHS · UK Biobank</span></div>"]:::l0Data

    L3 <==>|"生理语义锚定<br/>多尺度表征输入"| L2_LEFT
    L3 <==>|"动力学先验约束<br/>反事实情景推演"| L2_RIGHT
    L2_LEFT ==>|"自监督预训练<br/>下游探针评测"| L0
    L2_RIGHT ==>|"跨模态配对训练<br/>零样本基准验证"| L0
"""

# 2. 博客 2 图 10：端云分层协同的时序-语言健康 Agent 架构 (blogs/images/edge_cloud_health_agent.png)
mmd_edge_cloud = """flowchart TB
    classDef cloudBrain fill:#EFF6FF,stroke:#3B82F6,stroke-width:2px,rx:10px,ry:10px;
    classDef edgeSensor fill:#F0FDF4,stroke:#10B981,stroke-width:2px,rx:10px,ry:10px;
    classDef toolBox fill:#FFF7ED,stroke:#F97316,stroke-width:2px,rx:10px,ry:10px;

    BRAIN["<div style='width:580px; text-align:center; line-height:1.7; font-family:system-ui, -apple-system, Segoe UI, Roboto, sans-serif;'><strong style='font-size:15px; color:#1E3A8A;'>云端 / 手机推理中枢 (LLM Reasoning Brain)</strong><br/><span style='font-size:12px; color:#1D4ED8;'>• 医学知识检索 (EHR / 临床指南) 与连续因果链推导<br/>• 交互式反事实推演 (What-If) 与 Agent 任务规划调度</span></div>"]:::cloudBrain

    EDGE["<div style='width:270px; text-align:left; line-height:1.7; font-family:system-ui, -apple-system, Segoe UI, Roboto, sans-serif;'><strong style='font-size:14px; color:#065F46;'>端侧轻量生理时序基模</strong><br/><span style='font-size:12px; color:#047857;'>• 1D U-Net / Patch 编码器<br/>• 毫秒级波形特征提取与去噪<br/>• 连续物理时间 RoPE 锚定</span></div>"]:::edgeSensor

    TOOLS["<div style='width:270px; text-align:left; line-height:1.7; font-family:system-ui, -apple-system, Segoe UI, Roboto, sans-serif;'><strong style='font-size:14px; color:#9A3412;'>CodeAct 与专业信号工具库</strong><br/><span style='font-size:12px; color:#C2410C;'>• pyPPG / SciPy 数字滤波与峰值检测<br/>• HRV 频域分析与 SQI 质量评估<br/>• 动态绘图与指南阈值核验</span></div>"]:::toolBox

    BRAIN <==>|"生理 Token 上传<br/>按需唤醒高频重采样"| EDGE
    BRAIN <==>|"执行 Python 脚本<br/>返回精确数值与图表"| TOOLS
"""

# 3. RisQ 多模态共生示意图 (docs/figures/risq/multimodal_coexistence.png)
mmd_risq = """flowchart LR
    classDef edgeBox fill:#F0FDF4,stroke:#10B981,stroke-width:2px,rx:10px,ry:10px;
    classDef centerBox fill:#EFF6FF,stroke:#3B82F6,stroke-width:2px,rx:10px,ry:10px;
    classDef staticBox fill:#F8FAFC,stroke:#64748B,stroke-width:2px,rx:10px,ry:10px;

    EDGE["<div style='width:250px; text-align:left; line-height:1.6;'><strong style='color:#065F46;'>动态穿戴时序感知</strong><br/>• 智能手表与体动仪传感器<br/>• 提取连续生理特征与昼夜节律</div>"]:::edgeBox
    STATIC["<div style='width:250px; text-align:left; line-height:1.6;'><strong style='color:#334155;'>静态组学与临床档案</strong><br/>• 全基因组变异与生化化验<br/>• 长期住院与就诊 ICD-10 记录</div>"]:::staticBox
    RISQ["<div style='width:250px; text-align:left; line-height:1.6;'><strong style='color:#1E3A8A;'>系统级多模态健康模型</strong><br/>• 统一跨模态潜表征空间<br/>• 自然语言 Prompt 零样本风险推演</div>"]:::centerBox

    EDGE ==>|"高频动态生理输入"| RISQ
    STATIC ==>|"先天遗传与临床基线"| RISQ
"""

# 4. 健康推理三层分工体系 (docs/figures/health_reasoning/hierarchical_ai_stack.png)
mmd_hierarchical = """flowchart TB
    classDef l3Brain fill:#EFF6FF,stroke:#3B82F6,stroke-width:2px,rx:10px,ry:10px;
    classDef l2Tools fill:#FFF7ED,stroke:#F97316,stroke-width:2px,rx:10px,ry:10px;
    classDef l1Edge fill:#F0FDF4,stroke:#10B981,stroke-width:2px,rx:10px,ry:10px;

    L3["<div style='width:560px; text-align:center; line-height:1.6;'><strong style='font-size:15px; color:#1E3A8A;'>顶层 · 认知与健康决策中枢 (Cognitive & Reasoning Layer)</strong><br/><span style='font-size:12px; color:#1D4ED8;'>• 健康推理 Agent (WearableQA / 临床指南循证库)<br/>• 跨模态健康画像、纵向病理因果推演与前瞻干预建议</span></div>"]:::l3Brain

    L2["<div style='width:560px; text-align:center; line-height:1.6;'><strong style='font-size:14px; color:#9A3412;'>中层 · 神经符号与工具调度执行层 (Neuro-Symbolic & CodeAct Layer)</strong><br/><span style='font-size:12px; color:#C2410C;'>• Python 科学计算运行时 (SciPy / NeuroKit2 / BioSPPy)<br/>• 动态高精度数值积分、假设检验与频域功率谱解算</span></div>"]:::l2Tools

    L1["<div style='width:560px; text-align:center; line-height:1.6;'><strong style='font-size:14px; color:#065F46;'>底层 · 微观高频物理感知与生理基座 (Edge Sensor Foundation Models)</strong><br/><span style='font-size:12px; color:#047857;'>• 专用时序基座: SensorFM / PaPaGei / UniCardio / SleepFM / GlucoFM<br/>• 毫秒级滤波去噪、运动伪影抑制与连续潜表征抽取</span></div>"]:::l1Edge

    L3 <==>|"意图解析与计算指令派发<br/>返回高可靠结构化计算结果"| L2
    L2 <==>|"提取生理指标特征<br/>按需调度波形计算"| L1
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
