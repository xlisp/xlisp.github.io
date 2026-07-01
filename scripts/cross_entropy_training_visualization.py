# -*- coding: utf-8 -*-
"""
交叉熵训练可视化
配套文章：《大学4年没讲明白的信息论，被一段 PyTorch 代码讲透了》

左面板：三类数据 + 模型决策边界，随训练一帧帧收紧
右面板：交叉熵（蓝，平均意外）与困惑度（红，在几类间纠结）同步下滑

运行：
    python cross_entropy_training_visualization.py
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
STEPS = 200

torch.manual_seed(0)
np.random.seed(0)

# --- 数据：三类高斯团 ---
centers = torch.tensor([[2.0, 2.0], [-2.0, 2.0], [0.0, -2.0]])
X = torch.cat([c + torch.randn(200, 2) * 0.6 for c in centers])
y = torch.cat([torch.full((200,), i, dtype=torch.long) for i in range(3)])

model = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 3))
opt = torch.optim.Adam(model.parameters(), lr=0.05)

# --- 决策边界用的网格 ---
pad = 1.0
x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
gx, gy = torch.meshgrid(torch.linspace(x_min, x_max, 220),
                        torch.linspace(y_min, y_max, 220), indexing="xy")
mesh = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)

# --- 先把整段训练跑完，逐帧记录（边界预测 + loss + 困惑度）---
frames = []
for step in range(STEPS + 1):
    logits = model(X)
    loss = F.cross_entropy(logits, y)     # 平均交叉熵 = 平均意外
    ppl = torch.exp(loss)                 # 困惑度 = 在几类之间纠结
    if step % 2 == 0:
        with torch.no_grad():
            pred = model(mesh).argmax(1).reshape(gx.shape).numpy()
        frames.append((step, pred, loss.item(), ppl.item()))
    opt.zero_grad()
    loss.backward()
    opt.step()

steps_hist = [f[0] for f in frames]
ce_hist = [f[2] for f in frames]
ppl_hist = [f[3] for f in frames]

# --- 画图 ---
fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5))
colors = np.array([[0.86, 0.20, 0.18],
                   [0.12, 0.47, 0.71],
                   [0.20, 0.63, 0.29]])
Xn, yn = X.numpy(), y.numpy()
GX, GY = gx.numpy(), gy.numpy()


def draw(idx):
    step, pred, ce, ppl = frames[idx]

    axL.clear()
    axL.contourf(GX, GY, pred, levels=[-0.5, 0.5, 1.5, 2.5],
                 colors=colors * 0.35 + 0.65, alpha=0.55)
    for c in range(3):
        axL.scatter(Xn[yn == c, 0], Xn[yn == c, 1], s=12,
                    color=colors[c], edgecolor="white", linewidth=0.3)
    axL.set_title(f"决策边界（step {step}）—— 从混沌到切开三坨点")
    axL.set_xlabel("特征 1")
    axL.set_ylabel("特征 2")

    axR.clear()
    k = idx + 1
    axR.plot(steps_hist[:k], ce_hist[:k], color="#1f77b4", lw=2,
             label="交叉熵（平均意外）")
    axR.plot(steps_hist[:k], ppl_hist[:k], color="#d62728", lw=2,
             label="困惑度（在几类间纠结）")
    axR.axhline(np.log(3), color="#1f77b4", ls="--", alpha=0.4)   # ln3 = 三类瞎猜
    axR.axhline(3, color="#d62728", ls="--", alpha=0.4)           # 困惑度瞎猜上界
    axR.axhline(1, color="gray", ls=":", alpha=0.6)               # 困惑度理想下界
    axR.set_xlim(0, STEPS)
    axR.set_ylim(0, 3.6)
    axR.set_title(f"交叉熵 = {ce:.3f}    困惑度 = {ppl:.2f}")
    axR.set_xlabel("训练步数")
    axR.legend(loc="upper right")
    axR.grid(alpha=0.25)


anim = FuncAnimation(fig, draw, frames=len(frames), interval=60, repeat=False)
plt.tight_layout()

if SAVE_GIF:
    import os
    out = os.path.join(os.path.dirname(__file__), "out")
    os.makedirs(out, exist_ok=True)
    anim.save(os.path.join(out, "cross_entropy_training.gif"),
              writer="pillow", fps=15)
    print("已保存 GIF 到", out)
else:
    plt.show()
