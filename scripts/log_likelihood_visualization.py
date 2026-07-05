# -*- coding: utf-8 -*-
"""
对数似然 / 交叉熵 可视化
配套文章：《高中觉得最没用的 log，其实是大模型判断"自己错得有多离谱"的唯一标尺》

左面板：意外曲线 y = -ln(p)（那道 log 的陡墙）。一个小球代表"模型对正确词的把握"，
        训练时从左边（概率低、意外高、贴着墙）一路滑向右边（概率高、意外趋近 0）。
右面板：交叉熵（蓝，log 空间的平均意外）与困惑度（红，脱掉 log 的人话版）同步跳水。

小球横坐标 p = exp(-loss)（几何平均概率），正好落在 y=-ln(p) 曲线上。

运行：
    python log_likelihood_visualization.py
需要：torch, numpy, matplotlib（在 conda `torch` 环境里直接跑）
若想存 GIF，把下面的 SAVE_GIF 改成 True（需要 pillow）。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 中文字体（macOS）——参照仓库既有脚本的口径
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False

SAVE_GIF = False
STEPS = 300
SAMPLE_EVERY = 2       # 每几步记录一帧（帧数 = STEPS/SAMPLE_EVERY，约 150 帧）

torch.manual_seed(0)
np.random.seed(0)

# --- 玩具语料：一段不断重复的字符串，用当前字符预测下一个字符（bigram）---
text = "hello world " * 60
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text])
x, y = data[:-1], data[1:]

V = len(chars)
model = nn.Embedding(V, V)                              # 最简 bigram：查一行 logits
opt = torch.optim.Adam(model.parameters(), lr=0.1)

# --- 先把整段训练跑完，逐帧记录（loss / 困惑度 / 正确词把握）---
frames = []
for step in range(STEPS + 1):
    logits = model(x)
    loss = F.cross_entropy(logits, y)                  # 平均 -log(正确字符的概率)
    if step % SAMPLE_EVERY == 0:
        ce = loss.item()
        ppl = float(np.exp(ce))                        # 困惑度 = exp(交叉熵)
        p_correct = float(np.exp(-ce))                 # 几何平均概率，落在 -ln(p) 曲线上
        frames.append((step, ce, ppl, p_correct))
    opt.zero_grad()
    loss.backward()
    opt.step()

steps_hist = [f[0] for f in frames]
ce_hist = [f[1] for f in frames]
ppl_hist = [f[2] for f in frames]

# --- 左面板固定的意外曲线 y = -ln(p) ---
p_grid = np.linspace(0.008, 1.0, 500)
surprise_grid = -np.log(p_grid)
ce0 = ce_hist[0]                                        # 起始交叉熵，用来定纵轴范围

# --- 画图 ---
fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5))


def draw(idx):
    step, ce, ppl, p_correct = frames[idx]

    # 左：意外曲线 + 滑动的小球
    axL.clear()
    axL.plot(p_grid, surprise_grid, color="#1f77b4", lw=2.2,
             label="意外 = -ln(p)")
    axL.scatter([p_correct], [ce], s=140, color="#d62728",
                edgecolor="white", zorder=5)
    axL.annotate(f"p={p_correct:.2f}\n意外={ce:.2f}",
                 xy=(p_correct, ce),
                 xytext=(p_correct + 0.12, ce + 0.6),
                 fontsize=11, color="#d62728")
    axL.axhline(0, color="gray", ls=":", alpha=0.5)
    axL.set_xlim(0, 1.05)
    axL.set_ylim(-0.3, max(ce0 + 1.0, 5.0))
    axL.set_title("那道 log 的陡墙：越靠近 p=0，意外越冲向无穷")
    axL.set_xlabel("模型对正确词的把握 p")
    axL.set_ylabel("意外 = -ln(p)")
    axL.legend(loc="upper right")
    axL.grid(alpha=0.25)

    # 右：交叉熵 + 困惑度 一起跳水
    axR.clear()
    k = idx + 1
    axR.plot(steps_hist[:k], ce_hist[:k], color="#1f77b4", lw=2,
             label="交叉熵（log 空间的平均意外）")
    axR.plot(steps_hist[:k], ppl_hist[:k], color="#d62728", lw=2,
             label="困惑度 = exp(交叉熵)")
    axR.axhline(1, color="gray", ls=":", alpha=0.6)    # 困惑度理想下界：毫不犹豫
    axR.set_xlim(0, STEPS)
    axR.set_ylim(0, max(ppl_hist) * 1.1)
    axR.set_title(f"交叉熵 = {ce:.3f}    困惑度 = {ppl:.2f}")
    axR.set_xlabel("训练步数")
    axR.legend(loc="upper right")
    axR.grid(alpha=0.25)


anim = FuncAnimation(fig, draw, frames=len(frames), interval=200,
                     repeat=True, repeat_delay=1500)   # 放慢播放 + 循环重播
plt.tight_layout()

if SAVE_GIF:
    import os
    out = os.path.join(os.path.dirname(__file__), "out")
    os.makedirs(out, exist_ok=True)
    anim.save(os.path.join(out, "log_likelihood.gif"),
              writer="pillow", fps=15)
    print("已保存 GIF 到", out)
else:
    plt.show()
