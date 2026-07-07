# -*- coding: utf-8 -*-
"""
复数版 RoPE 可视化（2×2 四面板，与文章逐节对应）
配套文章：《高中觉得最没用的"虚数 i"，其实是大模型旋转位置编码最优雅的写法》

左上（第 -1 节）：虚数的出生 —— Bombelli 1572 解 x³=15x+4：
        从实数轴出发，借道 2±11i、2±i，虚部对消回到实数解 4。"以虚补实"的桥。
右上（第 1 节）：乘 i = 逆时针转 90°。指针连续旋转，i²=−1 就是"转两次 90° = 掉头"。
左下（疑惑点二/三）：apply_rope 之后，d/2 根"单位复数指针"随相对位置 r 各自旋转 r·freq。
        快指针（红）负责分辨相邻词，慢指针（蓝）负责记住远距离 —— 时针分针秒针。
右下（第 3 节 & 动手实验）：注意力分数 = Re(conj(qₘ)·kₙ) 随相对距离 r 的曲线。
        把 query/key 的绝对位置整体平移到 m=0 / 10 / 20，三条曲线完全重合 ——
        分数只依赖相对距离 r，与绝对位置无关（RoPE 的核心性质）。

运行：
    python scripts/complex_rope_visualization.py
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

R_MAX = 40
offsets = np.linspace(0, R_MAX, 160)           # 相对距离从 0 平滑扫到 R_MAX


def score_abs(m, r):
    """query 放在绝对位置 m、key 放在 m+r 时的注意力分数（文章"动手"一节的实验）"""
    qm = qc * np.exp(1j * m * freqs)
    kn = kc * np.exp(1j * (m + r) * freqs)
    return float(np.sum((np.conj(qm) * kn).real))


# 三个不同的绝对位置起点 —— 数学上分数只含 e^(i·r·freq)，三条曲线应当完全重合
SHIFTS = [0, 10, 20]
curves = {m: [score_abs(m, r) for r in offsets] for m in SHIFTS}

# 指针颜色：快红 -> 慢蓝
hand_colors = plt.cm.coolwarm(np.linspace(1, 0, half))
theta_circle = np.linspace(0, 2 * np.pi, 100)

fig, ((axB, axI), (axL, axR)) = plt.subplots(2, 2, figsize=(13, 10))


def draw_bombelli(ax):
    """左上：静态图 —— 虚数的出生（Bombelli 1572），以虚补实的桥"""
    ax.axhline(0, color="black", lw=1.5)                       # 实数轴：现实世界
    ax.text(5.9, -2.6, "实数轴（现实）", fontsize=9, ha="right", color="black")

    pts = {"2+11i": 2 + 11j, "2−11i": 2 - 11j, "2+i": 2 + 1j, "2−i": 2 - 1j}
    for label, z in pts.items():
        ax.scatter(z.real, z.imag, s=60, color="#9467bd", zorder=5)
        ax.annotate(label, (z.real, z.imag), textcoords="offset points",
                    xytext=(8, 4), fontsize=10, color="#9467bd")
    ax.scatter(4, 0, s=110, color="#d62728", zorder=6)
    ax.annotate("x = 4（实数解）", (4, 0), textcoords="offset points",
                xytext=(6, 14), fontsize=10, color="#d62728")
    ax.scatter(0, 0, s=60, color="#2ca02c", zorder=6)
    ax.annotate("问题 x³=15x+4", (0, 0), textcoords="offset points",
                xytext=(-6, 10), fontsize=10, color="#2ca02c")

    arrow = dict(arrowstyle="->", color="gray", lw=1.4,
                 connectionstyle="arc3,rad=-0.25")
    ax.annotate("", xy=(2, 11), xytext=(0.1, 0.3), arrowprops=arrow)     # 公式冒出 √-121
    ax.annotate("", xy=(2, -11), xytext=(0.1, -0.3), arrowprops=arrow)
    ax.annotate("", xy=(2, 1.2), xytext=(2, 10.6), arrowprops=dict(
        arrowstyle="->", color="#9467bd", lw=1.4))                       # 开立方
    ax.annotate("", xy=(2, -1.2), xytext=(2, -10.6), arrowprops=dict(
        arrowstyle="->", color="#9467bd", lw=1.4))
    ax.text(2.25, 5.6, "开立方", fontsize=9, color="#9467bd")
    ax.annotate("", xy=(3.8, 0.15), xytext=(2.15, 1), arrowprops=dict(
        arrowstyle="->", color="#d62728", lw=1.4))                       # 相加，虚部对消
    ax.annotate("", xy=(3.8, -0.15), xytext=(2.15, -1), arrowprops=dict(
        arrowstyle="->", color="#d62728", lw=1.4))
    ax.text(3.0, 1.7, "相加：虚部对消", fontsize=9, color="#d62728")
    ax.text(0.15, -6.5, "卡尔达诺公式中间\n冒出 √(−121)", fontsize=9, color="gray")

    ax.set_xlim(-1.2, 6.2)
    ax.set_ylim(-13, 13)
    ax.set_title("第 -1 节｜虚数的出生：实数进，借道虚数，实数出（Bombelli 1572）")
    ax.grid(alpha=0.2)


def draw_mul_i(ax, r):
    """右上：乘 i = 转 90°。指针随动画连续旋转，走到 90° 的整数倍就是乘了几次 i"""
    ax.plot(np.cos(theta_circle), np.sin(theta_circle),
            color="gray", ls=":", alpha=0.4)
    for z, label in [(1, "1"), (1j, "i"), (-1, "i² = −1"), (-1j, "i³ = −i")]:
        ax.scatter(np.real(z), np.imag(z), s=50, color="#7f7f7f", zorder=4)
        ax.annotate(label, (np.real(z), np.imag(z)), textcoords="offset points",
                    xytext=(8, 6), fontsize=11)
    ang = (r / R_MAX) * 2 * np.pi                  # 动画扫满一整圈
    n_i = ang / (np.pi / 2)                        # 已经乘了几次 i
    arc = np.linspace(0, ang, 60)
    ax.plot(0.35 * np.cos(arc), 0.35 * np.sin(arc), color="#ff7f0e", lw=2)
    ax.quiver(0, 0, np.cos(ang), np.sin(ang), color="#ff7f0e",
              angles="xy", scale_units="xy", scale=1, width=0.014)
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal")
    ax.set_title(f"第 1 节｜乘 i = 逆时针转 90°（当前已乘 i 共 {n_i:.2f} 次）")
    ax.grid(alpha=0.2)


def draw_hands(ax, r):
    """左下：apply_rope 之后，d/2 根指针各自按自己的转速旋转"""
    ax.plot(np.cos(theta_circle), np.sin(theta_circle),
            color="gray", ls=":", alpha=0.4)
    for j in range(half):
        ang = r * freqs[j]
        ax.quiver(0, 0, np.cos(ang), np.sin(ang), color=hand_colors[j],
                  angles="xy", scale_units="xy", scale=1, width=0.012,
                  label=f"第{j+1}对 (转速{freqs[j]:.3f})")
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.set_title(f"疑惑点三｜apply_rope：d/2 根指针各自旋转（r = {r:.1f}）—— 红快 蓝慢")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.2)


def draw_scores(ax, r):
    """右下：三个绝对位置起点的分数曲线完全重合 —— 只认相对距离"""
    styles = [dict(lw=5, alpha=0.30, color="#1f77b4"),
              dict(lw=2, ls="--", color="#2ca02c"),
              dict(lw=1.4, ls=":", color="#d62728")]
    for m, st in zip(SHIFTS, styles):
        ax.plot(offsets, curves[m], label=f"query@{m}, key@{m}+r", **st)
    ax.scatter([r], [score_abs(0, r)], s=120, color="#d62728",
               edgecolor="white", zorder=5)
    ax.axhline(0, color="gray", ls=":", alpha=0.5)
    ax.set_xlim(0, R_MAX)
    ax.set_title("第 3 节｜绝对位置平移 0/10/20，三条曲线完全重合 —— 只认相对距离")
    ax.set_xlabel("相对距离 r = n − m")
    ax.set_ylabel("注意力分数")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.25)


def draw(fi):
    r = offsets[fi]
    for ax in (axB, axI, axL, axR):
        ax.clear()
    draw_bombelli(axB)
    draw_mul_i(axI, r)
    draw_hands(axL, r)
    draw_scores(axR, r)


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
