# -*- coding: utf-8 -*-
"""
归一化如何驯服深层网络的数值分布 可视化
配套文章：《为什么大模型能堆到几十层还训得动？答案藏在一个最不起眼的动作里》

左面板：不做归一化 —— 同一份输入穿过一层层线性变换，数值分布逐层被放大，
        直方图越来越宽、σ 一路飙升（数值爆炸 -> 梯度爆炸）。
右面板：每层做 LayerNorm —— 同样的网络、同样的权重，但每层之后"减均值、除标准差"，
        直方图被反复"居中 + 收窄"，稳稳待在标准钟形（σ≈1，40 层也不飘）。

一句话：同样深度、同样权重，一个飘到失控，一个稳如水平线 —— 差别只在每层之间那次归一化。

运行：
    python layernorm_visualization.py
需要：torch, numpy, matplotlib（在 conda `torch` 环境里直接跑）
若想存 GIF，把下面的 SAVE_GIF 改成 True（需要 pillow）。
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 中文字体（macOS）——参照仓库既有脚本的口径
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False

SAVE_GIF = False
N_LAYERS = 40           # 网络深度（帧数 = N_LAYERS + 1）
DIM = 512               # 每层宽度
GAIN = 1.15             # 权重增益 > 1 -> 不归一化时逐层放大，制造可见的"爆炸"

torch.manual_seed(0)
np.random.seed(0)

# --- 同一份输入，走两条路：一条裸奔、一条每层归一化（用完全相同的权重）---
x0 = torch.randn(DIM)

frames_no = [(0, x0.numpy().copy(), x0.std().item())]      # 不归一化
frames_yes = [(0, x0.numpy().copy(), x0.std().item())]     # 每层 LayerNorm

x_no = x0.clone()
x_yes = x0.clone()
for layer in range(1, N_LAYERS + 1):
    W = torch.randn(DIM, DIM) * (GAIN / DIM ** 0.5)        # 两条路共用同一套权重
    x_no = x_no @ W                                        # 裸奔：只做线性变换
    x_yes = x_yes @ W
    x_yes = (x_yes - x_yes.mean()) / (x_yes.std() + 1e-5)  # 归一化：减均值、除标准差
    frames_no.append((layer, x_no.numpy().copy(), x_no.std().item()))
    frames_yes.append((layer, x_yes.numpy().copy(), x_yes.std().item()))

# --- 画图：左"不归一化" + 右"每层 LayerNorm" ---
fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5.2))
fig.suptitle("归一化如何驯服深层网络的数值分布", fontsize=15, fontweight="bold")

RED, BLUE = "#d62728", "#1f77b4"


def draw(idx):
    layerL, valsL, stdL = frames_no[idx]
    layerR, valsR, stdR = frames_yes[idx]

    # 左：不归一化 —— 横轴随 σ 动态放大，直观看到"越撑越宽"
    axL.clear()
    mL = max(4.0, 4.0 * stdL)
    axL.hist(valsL, bins=40, range=(-mL, mL), color=RED,
             alpha=0.85, edgecolor="white", linewidth=0.4)
    axL.set_xlim(-mL, mL)
    axL.set_title(f"不做归一化 · 第 {layerL} 层\nσ ≈ {stdL:.3g}（数值一路飙升）", fontsize=12)
    axL.set_xlabel("激活值"); axL.set_ylabel("频数")
    axL.grid(alpha=0.2)

    # 右：每层 LayerNorm —— 横轴固定，σ 稳稳≈1
    axR.clear()
    axR.hist(valsR, bins=40, range=(-4, 4), color=BLUE,
             alpha=0.85, edgecolor="white", linewidth=0.4)
    axR.set_xlim(-4, 4)
    axR.set_title(f"每层 LayerNorm · 第 {layerR} 层\nσ ≈ {stdR:.3g}（稳定在标准状态）", fontsize=12)
    axR.set_xlabel("激活值"); axR.set_ylabel("频数")
    axR.grid(alpha=0.2)


anim = FuncAnimation(fig, draw, frames=len(frames_no), interval=200,
                     repeat=True, repeat_delay=1800)
plt.tight_layout(rect=[0, 0, 1, 0.94])

if SAVE_GIF:
    out = os.path.join(os.path.dirname(__file__), "out")
    os.makedirs(out, exist_ok=True)
    anim.save(os.path.join(out, "layernorm.gif"), writer="pillow", fps=6)
    print("已保存 GIF 到", out)
else:
    plt.show()
