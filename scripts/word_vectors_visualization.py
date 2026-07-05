# -*- coding: utf-8 -*-
"""
词向量自我组织 可视化
配套文章：《高中觉得只是"带箭头的线段"的向量，其实是大模型把"意思"变成"方向"的地基》

左面板：6 个词向量的箭头（动物 3 个 / 水果 3 个）。训练时你会看到同类的箭头慢慢转到一起、
        异类的转向两边 —— "意思相近 -> 方向相近"。
右面板：两两余弦相似度热力图。从一片模糊，逐渐"同类发亮、异类变暗"。

目标：让同类词的余弦相似度趋近 +1、异类趋近 -1（用点积/余弦当损失，直接把语义摆进方向里）。

运行：
    python word_vectors_visualization.py
需要：torch, numpy, matplotlib（在 conda `torch` 环境里直接跑）
若想存 GIF，把下面的 SAVE_GIF 改成 True（需要 pillow）。
"""
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 中文字体（macOS）——参照仓库既有脚本的口径
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False

SAVE_GIF = False
STEPS = 300
SAMPLE_EVERY = 3        # 每几步记录一帧（帧数 = STEPS/SAMPLE_EVERY，约 100 帧）

torch.manual_seed(3)
np.random.seed(0)

# --- 6 个词，前 3 个动物、后 3 个水果 ---
words = ["猫", "狗", "虎", "苹果", "香蕉", "葡萄"]
cat_id = torch.tensor([0, 0, 0, 1, 1, 1])          # 0=动物, 1=水果
n = len(words)

# 目标余弦矩阵：同类 +1，异类 -1（对角线不参与）
same = (cat_id[:, None] == cat_id[None, :]).float()
target = same * 2 - 1                                # 同类->1, 异类->-1
mask = 1 - torch.eye(n)                              # 去掉对角线

emb = torch.randn(n, 2, requires_grad=True)          # 随机初始化的 2 维词向量
opt = torch.optim.Adam([emb], lr=0.05)

# --- 先把训练跑完，逐帧记录（词向量 + 余弦矩阵）---
frames = []
for step in range(STEPS + 1):
    normed = F.normalize(emb, dim=1)                 # 只看方向
    sim = normed @ normed.t()                        # 两两余弦相似度
    loss = (((sim - target) ** 2) * mask).sum() / mask.sum()
    if step % SAMPLE_EVERY == 0:
        frames.append((step, normed.detach().numpy().copy(),
                       sim.detach().numpy().copy(), loss.item()))
    opt.zero_grad()
    loss.backward()
    opt.step()

# --- 画图：左箭头 + 右热力图 ---
fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5.2))
colors = ["#d62728", "#d62728", "#d62728", "#1f77b4", "#1f77b4", "#1f77b4"]

# 右面板的热力图 + 颜色条只建一次（之后只更新数据，避免重复叠加）
im = axR.imshow(frames[0][2], vmin=-1, vmax=1, cmap="coolwarm")
fig.colorbar(im, ax=axR, fraction=0.046, pad=0.04)
axR.set_xticks(range(n)); axR.set_xticklabels(words)
axR.set_yticks(range(n)); axR.set_yticklabels(words)

theta = np.linspace(0, 2 * np.pi, 100)               # 左面板的参考单位圆


def draw(idx):
    step, vecs, sim, loss = frames[idx]

    axL.clear()
    axL.plot(np.cos(theta), np.sin(theta), color="gray", ls=":", alpha=0.4)
    for i in range(n):
        axL.quiver(0, 0, vecs[i, 0], vecs[i, 1], color=colors[i],
                   angles="xy", scale_units="xy", scale=1, width=0.012)
        axL.text(vecs[i, 0] * 1.15, vecs[i, 1] * 1.15, words[i],
                 color=colors[i], fontsize=13, ha="center", va="center")
    axL.set_xlim(-1.35, 1.35); axL.set_ylim(-1.35, 1.35)
    axL.set_aspect("equal")
    axL.set_title(f"词向量方向（step {step}）—— 红=动物  蓝=水果")
    axL.grid(alpha=0.2)

    im.set_data(sim)
    axR.set_title(f"两两余弦相似度（loss={loss:.3f}）—— 同类亮 异类暗")


anim = FuncAnimation(fig, draw, frames=len(frames), interval=120,
                     repeat=True, repeat_delay=1500)
plt.tight_layout()

if SAVE_GIF:
    import os
    out = os.path.join(os.path.dirname(__file__), "out")
    os.makedirs(out, exist_ok=True)
    anim.save(os.path.join(out, "word_vectors.gif"),
              writer="pillow", fps=15)
    print("已保存 GIF 到", out)
else:
    plt.show()
