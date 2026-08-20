# -*- coding: utf-8 -*-
"""
数位对齐 vs 分词切块 可视化
配套文章：《小学一年级学的"数位对齐"，原来是大模型至今算不明白数的根本原因》

左面板：竖式 —— 数字规规矩矩按数位对齐，进位一格一格往左传。
中面板：分词器 —— 同一个数字被切成大小不一的块，块的边界随上下文乱跳，
        "第几位"这个信息在切分那一刻就没了。
右面板：位数 vs 平均误差 —— 块表示那条线在位数变大后直接跳水，
        按位对齐那条线基本贴着地面。

一句话：人类算数靠位置，大模型看数字靠频率，两套地基对不上。

运行：
    python tokenizer_number_visualization.py
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
RED, BLUE, GRAY = "#d62728", "#1f77b4", "#999999"
PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

torch.manual_seed(0)
np.random.seed(0)

# ---------------------------------------------------------------- 玩具分词器
FREQUENT = {"20", "19", "11", "100", "000", "12", "10", "24", "37"}


def toy_tokenize(s: str):
    """极简 BPE：优先匹配语料里的高频块，匹配不上才单字符成块。"""
    tokens, i = [], 0
    while i < len(s):
        for L in (3, 2):
            if s[i:i + L] in FREQUENT:
                tokens.append(s[i:i + L]); i += L; break
        else:
            tokens.append(s[i]); i += 1
    return tokens


# 演示用的一串数字：同一批数字，看竖式怎么看、分词器怎么看
CASES = ["9.11", "9.9", "1234", "01234", "2019", "100000", "24371"]


# ------------------------------------------------- 右面板：位数 vs 误差（预计算）
def digits_of(x, n_digits):
    """把整数拆成逐位通道 —— 这就是竖式在做的事。"""
    return torch.stack([(x // 10 ** k) % 10 for k in range(n_digits)],
                       dim=1).float() / 9.0


def train_once(X, y, steps=700):
    net = nn.Sequential(nn.Linear(X.shape[1], 64), nn.ReLU(), nn.Linear(64, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(steps):
        loss = ((net(X) - y) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return (net(X) - y).abs().mean().item()


def sweep():
    """对 1~5 位数分别训练：块表示 vs 按位对齐，返回相对误差(%)。"""
    err_block, err_digit, xs = [], [], []
    for nd in range(1, 6):
        hi, N = 10 ** nd, 4000
        a = torch.randint(0, hi, (N,))
        b = torch.randint(0, hi, (N,))
        scale = float(2 * hi)
        y = (a + b).float().unsqueeze(1) / scale

        XA = torch.stack([a, b], dim=1).float() / hi          # 块：整个数一个值
        XB = torch.cat([digits_of(a, nd), digits_of(b, nd)], 1)  # 按位对齐

        err_block.append(train_once(XA, y) * 100)             # 已归一化 -> 直接当 %
        err_digit.append(train_once(XB, y) * 100)
        xs.append(nd)
        print(f"  {nd} 位数: 块表示 {err_block[-1]:.2f}%  |  按位对齐 {err_digit[-1]:.2f}%")
    return xs, err_block, err_digit


print("正在跑「位数 vs 误差」实验（约十几秒）...")
XS, ERR_BLOCK, ERR_DIGIT = sweep()

# ------------------------------------------------------------------- 画图
fig, (axL, axM, axR) = plt.subplots(1, 3, figsize=(15.5, 5.4))
fig.suptitle("竖式靠位置，分词靠频率 —— 大模型算不对数的根本原因",
             fontsize=15, fontweight="bold")


def draw_vertical(ax, s):
    """左：竖式对齐 —— 每一位落在固定的格子里。"""
    ax.clear()
    digits = [c for c in s if c.isdigit()]
    n = len(digits)
    for k, ch in enumerate(digits):
        col = n - 1 - k                      # 从右往左：个位、十位、百位...
        ax.add_patch(plt.Rectangle((col, 0.35), 0.9, 0.9, facecolor="white",
                                   edgecolor=BLUE, linewidth=2))
        ax.text(col + 0.45, 0.8, ch, ha="center", va="center", fontsize=20)
        ax.text(col + 0.45, 0.12, f"10^{n-1-col}", ha="center", va="center",
                fontsize=10, color=GRAY)
    ax.set_xlim(-0.4, max(n, 6) + 0.2)
    ax.set_ylim(-0.2, 2.0)
    ax.set_title(f"竖式眼中的 {s}\n每一位都有固定的权重（位置即意义）", fontsize=12)
    ax.axis("off")


def draw_tokens(ax, s):
    """中：分词切块 —— 块大小不一，且随上下文变化。"""
    ax.clear()
    toks = toy_tokenize(s)
    x = 0.0
    for j, t in enumerate(toks):
        w = 0.75 * len(t)
        ax.add_patch(plt.Rectangle((x, 0.35), w - 0.1, 0.9,
                                   facecolor=PALETTE[j % len(PALETTE)],
                                   alpha=0.75, edgecolor="white", linewidth=2))
        ax.text(x + (w - 0.1) / 2, 0.8, t, ha="center", va="center",
                fontsize=18, color="white", fontweight="bold")
        x += w
    ax.set_xlim(-0.3, max(x, 4.8) + 0.3)
    ax.set_ylim(-0.2, 2.0)
    ax.set_title(f"分词器眼中的 {s}\n切成 {len(toks)} 块 —— 「第几位」已经消失",
                 fontsize=12, color=RED)
    ax.axis("off")


def draw_curve(ax, upto):
    """右：位数 vs 误差，逐点显现。"""
    ax.clear()
    k = min(upto, len(XS))
    ax.plot(XS[:k], ERR_BLOCK[:k], "o-", color=RED, lw=2.5, ms=8,
            label="块表示（分词器的做法）")
    ax.plot(XS[:k], ERR_DIGIT[:k], "s-", color=BLUE, lw=2.5, ms=8,
            label="按位对齐（竖式的做法）")
    ax.set_xlim(0.7, 5.3)
    ax.set_ylim(0, max(ERR_BLOCK) * 1.25 + 0.5)
    ax.set_xticks(XS)
    ax.set_xlabel("加数的位数"); ax.set_ylabel("平均相对误差 (%)")
    ax.set_title("同样的网络，只换了「怎么看这个数」", fontsize=12)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=10)


def update(frame):
    s = CASES[frame % len(CASES)]
    draw_vertical(axL, s)
    draw_tokens(axM, s)
    draw_curve(axR, frame // 2 + 1)


anim = FuncAnimation(fig, update, frames=len(CASES) * 2, interval=1400,
                     repeat=True, repeat_delay=1500)
plt.tight_layout(rect=[0, 0, 1, 0.93])

if SAVE_GIF:
    out = os.path.join(os.path.dirname(__file__), "out")
    os.makedirs(out, exist_ok=True)
    anim.save(os.path.join(out, "tokenizer_number.gif"), writer="pillow", fps=1)
    print("已保存 GIF 到", out)
else:
    plt.show()
