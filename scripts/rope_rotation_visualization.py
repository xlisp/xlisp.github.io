# -*- coding: utf-8 -*-
"""
RoPE 旋转位置编码可视化
配套文章：《高中背了三年的 sin 和 cos，原来是大模型记住"谁先谁后"的秘密》

左  ：同一个词向量，随位置 0→N 像钟表指针一样每次多转固定角度
中  ：多频率指针（快针分辨相邻、慢针记住远方），随位置同步旋转
右  ：query·key 点积只取决于相对距离——三条不同绝对位置的曲线完全重合

运行：
    python rope_rotation_visualization.py
需要：numpy, matplotlib
若想存 GIF，把下面的 SAVE_GIF 改成 True（需要 pillow）。
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False

SAVE_GIF = False
MAXPOS = 12
THETA = math.radians(35)          # 左图：每个位置转 35°


def rot(v, ang):                  # 把 2 维向量转 ang 弧度（第 2 节的旋转矩阵）
    c, s = math.cos(ang), math.sin(ang)
    return np.array([v[0] * c - v[1] * s, v[0] * s + v[1] * c])


# 中图：多频率指针。真实 RoPE 用 theta_i = 10000^(-2i/d)；
# 这里为了动画可读，用一组更温和、区分明显的转速做演示。
DEMO_FREQS = [1.0, 0.6, 0.32, 0.14]
HAND_COLORS = ["#d62728", "#1b9e77", "#7570b3", "#e6ab02"]
BASE = np.array([1.0, 0.0])


def unit_circle(ax):
    t = np.linspace(0, 2 * np.pi, 240)
    ax.plot(np.cos(t), np.sin(t), color="gray", lw=1, alpha=0.4)
    ax.axhline(0, color="gray", lw=0.5, alpha=0.3)
    ax.axvline(0, color="gray", lw=0.5, alpha=0.3)
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal")


fig = plt.figure(figsize=(15, 5))
axA = fig.add_subplot(1, 3, 1)
axB = fig.add_subplot(1, 3, 2)
axC = fig.add_subplot(1, 3, 3)

# --- 右图（静态，一次画好）：点积只依赖相对距离 ---
q = np.array([0.8, 0.4])
k = np.array([0.3, 0.9])
THETA_C = 0.5
rel = np.arange(-6, 7)
# 三条线数学上完全相等（点积只依赖相对距离），会严丝合缝地重叠。
# 用「粗→细、实→虚」套画，才能肉眼看出是三条叠在一起，而不是只有一条。
styles = [
    dict(color="#1f77b4", lw=7.0, ls="-",  alpha=0.35),          # 底层：粗、半透明
    dict(color="#d62728", lw=3.5, ls="-",  alpha=0.9),           # 中层
    dict(color="#2ca02c", lw=1.5, ls="--", marker="o", ms=4),    # 顶层：细虚线+圆点
]
for m, st in zip([0, 3, 6], styles):
    # score(m, m+r) = (R(mθ)q)·(R((m+r)θ)k) = qᵀ R(rθ) k —— 只剩相对量 r
    scores = [rot(q, m * THETA_C) @ rot(k, (m + r) * THETA_C) for r in rel]
    axC.plot(rel, scores, label=f"query 在绝对位置 {m}", **st)
axC.set_title("点积只取决于相对距离\n（三条线完全重合，粗蓝→红→细绿虚线层层叠）")
axC.set_xlabel("相对距离 (n − m)")
axC.set_ylabel("query · key 点积")
axC.legend(fontsize=9)
axC.grid(alpha=0.25)


def draw(pos):
    # 左图：同一个词随位置转动，留下轨迹
    axA.clear()
    unit_circle(axA)
    word = np.array([1.0, 0.0])
    for p in range(pos + 1):
        tip = rot(word, p * THETA)
        alpha = 0.2 + 0.8 * (p / max(pos, 1))
        axA.annotate("", xy=tip, xytext=(0, 0),
                     arrowprops=dict(arrowstyle="->", color="#1f77b4",
                                     alpha=alpha, lw=2))
        axA.text(tip[0] * 1.12, tip[1] * 1.12, str(p),
                 fontsize=9, ha="center", va="center")
    axA.set_title(f"同一个词，位置 0 → {pos}\n每往后一位就多转 {int(math.degrees(THETA))}°")

    # 中图：多频率指针
    axB.clear()
    unit_circle(axB)
    for f, col in zip(DEMO_FREQS, HAND_COLORS):
        tip = rot(BASE, pos * f * THETA)
        axB.annotate("", xy=tip, xytext=(0, 0),
                     arrowprops=dict(arrowstyle="->", color=col, lw=2.2))
        axB.text(tip[0] * 1.16, tip[1] * 1.16, f"转速 {f:.2g}",
                 fontsize=8, color=col, ha="center", va="center")
    axB.set_title(f"多频率指针（位置 {pos}）\n快针分辨相邻，慢针记住远方")


anim = FuncAnimation(fig, draw, frames=MAXPOS + 1, interval=450, repeat=True)
plt.tight_layout()

if SAVE_GIF:
    import os
    out = os.path.join(os.path.dirname(__file__), "out")
    os.makedirs(out, exist_ok=True)
    anim.save(os.path.join(out, "rope_rotation.gif"), writer="pillow", fps=4)
    print("已保存 GIF 到", out)
else:
    plt.show()
