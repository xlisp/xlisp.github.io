# -*- coding: utf-8 -*-
"""
组合爆炸 可视化
配套文章：《高中背到头秃的排列组合，其实解释了大模型为什么"不可能靠背、只能靠学"》

左面板：一棵每层分 3 个叉的树（示意），逐层生长，节点数 = 3^depth，直观感受"指数膨胀"。
右面板：真实量级下句子空间 50000^L 的位数（对数尺），随句长 L 增长，
        一路冲破"训练集规模"和"全宇宙原子数"两条参考线。

运行：
    python combinatorial_explosion_visualization.py
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

BRANCH = 3           # 左侧示意树的分叉数
MAX_DEPTH = 5        # 左侧树最多画到第几层（3^5=243 个节点）
V = 50000            # 右侧真实词表
L_MAX = 30           # 右侧最长句长
FRAMES = 90

log10V = np.log10(V)
L_full = np.linspace(0, L_MAX, 200)
digits_full = L_full * log10V          # 50000^L 的位数 = L·log10(V)

UNIVERSE_DIGITS = 80                    # 全宇宙原子数 ~ 80 位
TRAIN_DIGITS = 13                       # 训练集 ~ 10^13 句 -> 13 位

fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5.2))


def draw(fi):
    f = fi / (FRAMES - 1)              # 0 -> 1
    depth = int(round(f * MAX_DEPTH))
    L_cur = f * L_MAX

    # 左：逐层生长的分叉树
    axL.clear()
    prev_xs = None
    for level in range(depth + 1):
        ncount = BRANCH ** level
        xs = (np.linspace(0, 1, ncount + 2)[1:-1] if ncount > 1
              else np.array([0.5]))
        y = -level
        if prev_xs is not None:        # 画父->子的连线
            for ci, x in enumerate(xs):
                px = prev_xs[ci // BRANCH]
                axL.plot([px, x], [-(level - 1), y],
                         color="#9ecae1", lw=0.6, zorder=1)
        axL.scatter(xs, [y] * ncount, s=18, color="#1f77b4", zorder=2)
        prev_xs = xs
    axL.set_xlim(-0.05, 1.05)
    axL.set_ylim(-MAX_DEPTH - 0.5, 0.5)
    axL.axis("off")
    axL.set_title(f"每层分 {BRANCH} 个叉（示意）：第 {depth} 层已有 {BRANCH**depth} 个节点")

    # 右：真实句子空间的位数，冲破两条参考线
    axR.clear()
    axR.plot(L_full, digits_full, color="#1f77b4", lw=2,
             label="50000^L 的位数")
    mask = L_full <= L_cur
    axR.fill_between(L_full[mask], 0, digits_full[mask],
                     color="#1f77b4", alpha=0.15)
    axR.axhline(UNIVERSE_DIGITS, color="#d62728", ls="--", alpha=0.8,
                label="全宇宙原子数 (~80 位)")
    axR.axhline(TRAIN_DIGITS, color="gray", ls=":", alpha=0.8,
                label="训练集 (~13 位)")
    d_cur = L_cur * log10V
    axR.scatter([L_cur], [d_cur], s=110, color="#d62728",
                edgecolor="white", zorder=5)
    axR.set_xlim(0, L_MAX)
    axR.set_ylim(0, digits_full.max() * 1.05)
    over = "  ← 已超过全宇宙原子数！" if d_cur > UNIVERSE_DIGITS else ""
    axR.set_title(f"句长 L={L_cur:.0f} 时，可能句子有约 {d_cur:.0f} 位数{over}")
    axR.set_xlabel("句子长度 L（词数）")
    axR.set_ylabel("可能句子数的位数（log10）")
    axR.legend(loc="upper left", fontsize=9)
    axR.grid(alpha=0.25)


anim = FuncAnimation(fig, draw, frames=FRAMES, interval=120,
                     repeat=True, repeat_delay=1500)
plt.tight_layout()

if SAVE_GIF:
    import os
    out = os.path.join(os.path.dirname(__file__), "out")
    os.makedirs(out, exist_ok=True)
    anim.save(os.path.join(out, "combinatorial_explosion.gif"),
              writer="pillow", fps=15)
    print("已保存 GIF 到", out)
else:
    plt.show()
