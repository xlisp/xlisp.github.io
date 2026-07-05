# -*- coding: utf-8 -*-
"""
数学归纳法 / 自回归 可视化
配套文章：《高中最容易被忽略的数学归纳法，其实就是大模型"逐字生成"的骨架》

左面板：逐词生成的"骨牌链"。每落下一个新词（归纳步），都有一束箭头指回前面所有已生成的词
        —— 这就是"强归纳：看全部前文"。第 1 个词是奠基（绿色）。
右面板：归纳链断裂 = 曝光偏差。两条本该一样的序列（逻辑斯蒂映射），只在第 k 步被扰动一点点，
        之后沿着归纳步越拉越远 —— 一步错、后面全歪。

运行：
    python induction_autoregression_visualization.py
需要：numpy, matplotlib（不依赖 torch）
若想存 GIF，把下面的 SAVE_GIF 改成 True（需要 pillow）。
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 中文字体（macOS）——参照仓库既有脚本的口径
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False

SAVE_GIF = False

N = 10                 # 生成 N 步（共 N+1 个词）
R, X0 = 3.9, 0.40      # 逻辑斯蒂映射：对初值极敏感，用来放大"一步之差"
K_ERR, EPS = 3, 0.02

# --- 右面板：干净序列 vs 第 K 步出错的序列 ---
clean = [X0]
for _ in range(N):
    clean.append(R * clean[-1] * (1 - clean[-1]))
pert = clean[:K_ERR] + [clean[K_ERR] + EPS]        # 第 K 步被扰动
while len(pert) < len(clean):
    pert.append(R * pert[-1] * (1 - pert[-1]))
steps = np.arange(len(clean))

fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5.2))


def arc(ax, j, t, color):
    """从词 j 到词 t 画一条上凸的弧线（表示回看前文）。"""
    xs = np.linspace(j, t, 30)
    mid, span = (j + t) / 2, max(t - j, 1)
    ys = 0.28 * span / N * 4 * (1 - ((xs - mid) / (span / 2)) ** 2)
    ax.plot(xs, ys, color=color, lw=1.0, alpha=0.55, zorder=1)


def draw(t):
    # 左：逐词生成的骨牌链 + 回看全部前文的箭头
    axL.clear()
    for j in range(t):                             # 新词 t 回看前面每一个词
        arc(axL, j, t, "#d85a30")
    for i in range(t + 1):
        color = "#1d9e75" if i == 0 else ("#e24b4a" if i == t else "#378add")
        axL.scatter([i], [0], s=460, color=color, zorder=3,
                    edgecolor="white", linewidth=1.5)
        axL.text(i, 0, str(i), ha="center", va="center",
                 color="white", fontsize=11, zorder=4)
    axL.text(0, -0.42, "奠基", ha="center", color="#1d9e75", fontsize=11)
    if t > 0:
        axL.text(t, -0.42, f"归纳步{t}", ha="center", color="#e24b4a", fontsize=11)
    axL.set_xlim(-0.6, N + 0.6)
    axL.set_ylim(-0.7, 1.5)
    axL.axis("off")
    axL.set_title(f"逐词生成（第 {t} 步）—— 每个新词回看全部前文（强归纳）")

    # 右：归纳链断裂 = 曝光偏差
    axR.clear()
    k = t + 1
    axR.plot(steps[:k], clean[:k], "-o", color="#378add", ms=4, label="正常归纳链")
    axR.plot(steps[:k], pert[:k], "-o", color="#e24b4a", ms=4, label="第 3 步出错后")
    axR.axvline(K_ERR, color="gray", ls=":", alpha=0.6)
    axR.text(K_ERR + 0.1, 0.05, "一步错", color="gray", fontsize=9)
    axR.set_xlim(-0.3, N + 0.3)
    axR.set_ylim(-0.05, 1.05)
    diff = abs(clean[t] - pert[t])
    axR.set_title(f"曝光偏差：第 {t} 步偏差 = {diff:.3f}（起点只差 {EPS}）")
    axR.set_xlabel("归纳步")
    axR.set_ylabel("序列取值")
    axR.legend(loc="upper right", fontsize=9)
    axR.grid(alpha=0.25)


anim = FuncAnimation(fig, draw, frames=len(clean), interval=550,
                     repeat=True, repeat_delay=1800)
plt.tight_layout()

if SAVE_GIF:
    import os
    out = os.path.join(os.path.dirname(__file__), "out")
    os.makedirs(out, exist_ok=True)
    anim.save(os.path.join(out, "induction_autoregression.gif"),
              writer="pillow", fps=6)
    print("已保存 GIF 到", out)
else:
    plt.show()
