# -*- coding: utf-8 -*-
"""
复数版 RoPE 可视化
配套文章：《高中觉得最没用的"虚数 i"，其实是大模型旋转位置编码最优雅的写法》

左面板：复平面上 d/2 根"单位复数指针"，随相对位置 r 增大各自旋转 r·freq。
        快指针（红）负责分辨相邻词，慢指针（蓝）负责记住远距离 —— 时针分针秒针。
右面板：注意力分数 = Re(conj(q)·k·e^(i·r·freq)) 随相对距离 r 的曲线。
        它只依赖相对距离 r，与绝对位置无关（RoPE 的核心性质）。

运行：
    python complex_rope_visualization.py
需要：numpy, matplotlib（不依赖 torch，纯 numpy 复数即可）
若想存 GIF，把下面的 SAVE_GIF 改成 True（需要 pillow）。
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 中文字体（macOS）——参照仓库既有脚本的口径
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False

SAVE_GIF = False

d, base = 8, 10000.0
half = d // 2
idx = np.arange(half)
freqs = 1.0 / (base ** (2 * idx / d))          # 每一对的转速（快 -> 慢）

# 固定的 query / key（每对一个复数）
rng = np.random.default_rng(0)
qc = rng.standard_normal(half) + 1j * rng.standard_normal(half)
kc = rng.standard_normal(half) + 1j * rng.standard_normal(half)
base_prod = np.conj(qc) * kc                   # 与绝对位置无关的部分

R_MAX = 40
offsets = np.linspace(0, R_MAX, 160)           # 相对距离从 0 平滑扫到 R_MAX


def score(r):
    return float(np.sum((base_prod * np.exp(1j * r * freqs)).real))


score_curve = [score(r) for r in offsets]

# 指针颜色：快红 -> 慢蓝
hand_colors = plt.cm.coolwarm(np.linspace(1, 0, half))
theta_circle = np.linspace(0, 2 * np.pi, 100)

fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5.2))


def draw(fi):
    r = offsets[fi]

    # 左：复平面上的多频率旋转指针
    axL.clear()
    axL.plot(np.cos(theta_circle), np.sin(theta_circle),
             color="gray", ls=":", alpha=0.4)
    for j in range(half):
        ang = r * freqs[j]
        axL.quiver(0, 0, np.cos(ang), np.sin(ang), color=hand_colors[j],
                   angles="xy", scale_units="xy", scale=1, width=0.012,
                   label=f"第{j+1}对 (转速{freqs[j]:.3f})")
    axL.set_xlim(-1.3, 1.3)
    axL.set_ylim(-1.3, 1.3)
    axL.set_aspect("equal")
    axL.set_title(f"复平面指针（相对位置 r = {r:.1f}）—— 红快 蓝慢")
    axL.legend(loc="upper right", fontsize=8)
    axL.grid(alpha=0.2)

    # 右：注意力分数 vs 相对距离
    axR.clear()
    axR.plot(offsets, score_curve, color="#1f77b4", lw=2)
    axR.scatter([r], [score(r)], s=120, color="#d62728",
                edgecolor="white", zorder=5)
    axR.axhline(0, color="gray", ls=":", alpha=0.5)
    axR.set_xlim(0, R_MAX)
    axR.set_title("注意力分数只依赖相对距离 r（绝对位置平移它不变）")
    axR.set_xlabel("相对距离 r = n − m")
    axR.set_ylabel("注意力分数")
    axR.grid(alpha=0.25)


anim = FuncAnimation(fig, draw, frames=len(offsets), interval=90,
                     repeat=True, repeat_delay=1500)
plt.tight_layout()

if SAVE_GIF:
    import os
    out = os.path.join(os.path.dirname(__file__), "out")
    os.makedirs(out, exist_ok=True)
    anim.save(os.path.join(out, "complex_rope.gif"),
              writer="pillow", fps=15)
    print("已保存 GIF 到", out)
else:
    plt.show()
