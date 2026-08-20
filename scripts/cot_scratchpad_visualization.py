# -*- coding: utf-8 -*-
"""
固定深度 vs 草稿纸：用输出长度购买计算深度 可视化
配套文章：《老师逼你"写出过程"，原来是大模型学会推理的全部秘密》

左面板：直接答 —— 一根固定高度的柱子。题目再难，可用的串行步数就那么多（层数写死）。
中面板：打草稿 —— 每写一句中间结果，就叠上一层新的前向传播，可用步数阶梯式上升。
右面板：草稿长度 vs 正确率 —— 在某个长度上，曲线会从"瞎猜"突然跳到"全对"。

一句话：Transformer 的思考深度写死在层数里，唯一能延长思考的方式就是多写几个字。

运行：
    python cot_scratchpad_visualization.py
需要：torch, numpy, matplotlib（在 conda `torch` 环境里直接跑）
若想存 GIF，把下面的 SAVE_GIF 改成 True（需要 pillow）。
"""
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 中文字体（macOS）——参照仓库既有脚本的口径
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False

SAVE_GIF = False
RED, BLUE, GREEN, GRAY = "#d62728", "#1f77b4", "#2ca02c", "#999999"

LAYERS = 12          # 网络深度：一次前向能走的串行步数
MAX_STEPS = 8        # 草稿最多写几句
SEQ_LEN = 8          # 奇偶校验的序列长度

torch.manual_seed(0)
np.random.seed(0)

SCRATCH = [
    "题目：9.11 和 9.9 哪个大？",
    "① 补齐位数：9.11 vs 9.90",
    "② 整数部分相同：9 = 9",
    "③ 比较小数第一位：1 vs 9",
    "④ 1 < 9",
    "⑤ 所以 9.90 更大",
    "⑥ 检查：9.90 - 9.11 = 0.79 > 0 ✓",
    "结论：9.9 更大",
]


# ------------------------------------------- 右面板：草稿长度 vs 正确率（预计算）
class Scratchpad(nn.Module):
    """每写一步 = 一次额外前向；写 k 步就只允许扫描前 k 位。"""

    def __init__(self, k):
        super().__init__()
        self.k = k
        self.step = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        s = torch.zeros(x.shape[0], 1)
        for i in range(self.k):                       # 草稿写多长，就能往前算多少步
            s = torch.sigmoid(self.step(torch.cat([x[:, i:i + 1], s], 1)))
        return s


def accuracy_at(k, n=3000, steps=1200):
    bits = torch.randint(0, 2, (n, SEQ_LEN)).float()
    label = (bits.sum(1) % 2).unsqueeze(1)
    net = Scratchpad(k)
    opt = torch.optim.Adam(net.parameters(), lr=5e-3)
    lossfn = nn.BCELoss()
    for _ in range(steps):
        loss = lossfn(net(bits).clamp(1e-6, 1 - 1e-6), label)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return ((net(bits) > 0.5).float() == label).float().mean().item()


print("正在跑「草稿长度 vs 正确率」实验（约半分钟）...")
KS = list(range(1, SEQ_LEN + 1))
ACC = []
for k in KS:
    a = accuracy_at(k)
    ACC.append(a)
    print(f"  草稿写 {k} 步 -> 准确率 {a:.1%}")

# ------------------------------------------------------------------- 画图
fig, (axL, axM, axR) = plt.subplots(1, 3, figsize=(15.5, 5.6))
fig.suptitle("深度不够，长度来凑 —— 思维链到底在买什么",
             fontsize=15, fontweight="bold")


def draw_direct(ax):
    """左：直接答，永远只有 LAYERS 步。"""
    ax.clear()
    ax.bar([0], [LAYERS], width=0.5, color=RED, alpha=0.85,
           edgecolor="white", linewidth=2)
    ax.text(0, LAYERS + 2, f"{LAYERS} 步", ha="center", fontsize=14,
            fontweight="bold", color=RED)
    ax.axhline(LAYERS, color=GRAY, ls="--", lw=1)
    ax.set_xlim(-0.8, 0.8); ax.set_ylim(0, LAYERS * (MAX_STEPS + 1) * 1.1)
    ax.set_xticks([]); ax.set_ylabel("可用的串行计算步数")
    ax.set_title("直接答\n题目再难，也只有这么高", fontsize=12, color=RED)
    ax.grid(axis="y", alpha=0.2)


def draw_scratch(ax, n):
    """中：每写一句，叠一层。"""
    ax.clear()
    for j in range(n + 1):
        ax.bar([0], [LAYERS], bottom=[j * LAYERS], width=0.5,
               color=BLUE, alpha=0.55 + 0.05 * (j % 3),
               edgecolor="white", linewidth=2)
        if j < len(SCRATCH):
            ax.text(0.35, j * LAYERS + LAYERS / 2, SCRATCH[j],
                    va="center", fontsize=9, color="#333333")
    total = LAYERS * (n + 1)
    ax.text(0, total + 2, f"{total} 步", ha="center", fontsize=14,
            fontweight="bold", color=BLUE)
    ax.set_xlim(-0.8, 2.6); ax.set_ylim(0, LAYERS * (MAX_STEPS + 1) * 1.1)
    ax.set_xticks([]); ax.set_ylabel("可用的串行计算步数")
    ax.set_title(f"打草稿 · 已写 {n} 句\n每写一句 = 多买一次前向传播",
                 fontsize=12, color=BLUE)
    ax.grid(axis="y", alpha=0.2)


def draw_curve(ax, upto):
    """右：草稿长度 vs 正确率，逐点显现。"""
    ax.clear()
    k = max(1, min(upto, len(KS)))
    ax.axhline(0.5, color=GRAY, ls="--", lw=1.2)
    ax.text(SEQ_LEN * 0.55, 0.52, "瞎猜水平 50%", fontsize=9, color=GRAY)
    ax.plot(KS[:k], ACC[:k], "o-", color=GREEN, lw=2.5, ms=8)
    ax.set_xlim(0.6, SEQ_LEN + 0.4); ax.set_ylim(0.35, 1.05)
    ax.set_xticks(KS)
    ax.set_xlabel("草稿写了几步"); ax.set_ylabel("准确率")
    ax.set_title("同一个模型，只是允许它多写几步", fontsize=12)
    ax.grid(alpha=0.25)


def update(frame):
    n = frame % (MAX_STEPS + 1)
    draw_direct(axL)
    draw_scratch(axM, n)
    draw_curve(axR, n + 1)


anim = FuncAnimation(fig, update, frames=MAX_STEPS + 1, interval=1100,
                     repeat=True, repeat_delay=1800)
plt.tight_layout(rect=[0, 0, 1, 0.93])

if SAVE_GIF:
    out = os.path.join(os.path.dirname(__file__), "out")
    os.makedirs(out, exist_ok=True)
    anim.save(os.path.join(out, "cot_scratchpad.gif"), writer="pillow", fps=1)
    print("已保存 GIF 到", out)
else:
    plt.show()
