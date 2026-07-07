# -*- coding: utf-8 -*-
"""
复数版 RoPE 三维动画：交流电相量图的画法
配套文章：《高中觉得最没用的"虚数 i"，其实是大模型旋转位置编码最优雅的写法》

画法借自电气工程的经典相量图：复平面竖起来（y=实部, z=虚部），
"位置 pos"作为水平轴向前延伸（交流电里这根轴是时间，RoPE 里是第几个词）。

左图（第 2.5 节 & 疑惑点二）：三根不同转速的"相量"（快/中/慢，即 d/2 对维度中的三对）
        随位置前进一边旋转一边拖出三根螺旋；投影到底面，就是一族相交的正弦波——
        快波分辨相邻词，慢波记住远距离。这就是 freqs_cis 的全部内容。

右图（第 3 节 & 动手实验）：query（橙）、key（绿）两根螺旋，动画分两个阶段——
        阶段 1｜整体平移：距离 r=3 锁死，位置对 (m, m+3) 沿轴滑动，
                两支箭头之间的红色夹角扇形像焊死了一样纹丝不动 → 分数不变；
        阶段 2｜改变距离：m 固定，r 从 0 拉大到 8，扇形张开、分数跟着变。
        —— 模型看得见的只有这个夹角：位置平移隐形，相对距离显形。

运行：
    python scripts/complex_rope_visualization.py
需要：numpy, matplotlib（不依赖 torch）；SAVE_GIF=True 可存 GIF（需要 pillow）。
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 中文字体（macOS）——参照仓库既有脚本的口径
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False

SAVE_GIF = False

N_POS = 24                                     # 位置轴长度
FRAMES = 160
FLOOR, WALL = -1.45, 1.45                      # 投影墙的位置

# ── 左图：三根转速不同的相量（对应 d/2 对维度里的快/中/慢三对）──
FREQS = [1.0, 0.35, 0.10]
F_COLORS = ["#d62728", "#ff7f0e", "#1f77b4"]
F_LABELS = ["快（分辨相邻词）", "中", "慢（记住远距离）"]

# ── 右图：query / key 各一根螺旋（只画一对维度，转速取中档）──
THETA = 0.35
PHI_Q, PHI_K = 0.6, 1.7                        # 两个词本身不同 -> 初始幅角不同
C_Q, C_K, C_FAN = "#ff7f0e", "#2ca02c", "#d62728"

fig = plt.figure(figsize=(14.5, 7))
axL = fig.add_subplot(1, 2, 1, projection="3d")
axR = fig.add_subplot(1, 2, 2, projection="3d")


def setup_axis(ax, title):
    ax.set_xlim(0, N_POS)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)
    ax.set_xlabel("位置 pos（第几个词）")
    ax.set_ylabel("实部")
    ax.set_zlabel("虚部")
    ax.set_title(title, fontsize=11)
    ax.set_box_aspect((2.7, 1, 1))               # 位置轴拉长，螺旋才像螺旋而不是叠圈
    ax.view_init(elev=20, azim=-58)


def draw_cross_section(ax, x, alpha=0.5):
    """在位置 x 处画一个竖起来的复平面截面（单位圆 + 两根轴）"""
    t = np.linspace(0, 2 * np.pi, 80)
    ax.plot(np.full_like(t, x), np.cos(t), np.sin(t),
            color="gray", ls=":", lw=0.9, alpha=alpha)
    ax.plot([x, x], [-1.2, 1.2], [0, 0], color="gray", lw=0.5, alpha=alpha * 0.7)
    ax.plot([x, x], [0, 0], [-1.2, 1.2], color="gray", lw=0.5, alpha=alpha * 0.7)


def draw_left(front):
    """左图：三根相量转到位置 front，拖出螺旋 + 底面正弦投影"""
    axL.clear()
    setup_axis(axL, "疑惑点二｜每对维度 = 一根旋转相量（交流电同款画法）\n"
                    "螺旋在底面的投影 = 一族相交的正弦波（快 / 中 / 慢）")
    draw_cross_section(axL, 0, alpha=0.35)
    draw_cross_section(axL, front, alpha=0.8)          # 跟着前沿走的复平面
    axL.plot([0, N_POS], [0, 0], [0, 0], color="black", lw=0.8, alpha=0.6)

    ps = np.linspace(0, front, max(int(front * 16), 2))
    for f, c, lab in zip(FREQS, F_COLORS, F_LABELS):
        ang = ps * f
        axL.plot(ps, np.cos(ang), np.sin(ang), color=c, lw=1.8, label=lab)  # 螺旋
        axL.plot(ps, np.cos(ang), np.full_like(ps, FLOOR),                  # 底面投影
                 color=c, lw=1.2, ls="--", alpha=0.65)
        a = front * f                                                        # 前沿相量
        axL.quiver(front, 0, 0, 0, np.cos(a), np.sin(a),
                   color=c, lw=2.2, arrow_length_ratio=0.14)
        axL.scatter([front], [np.cos(a)], [FLOOR], color=c, s=18)
    axL.text(front, 0, 1.62, f"pos = {front:.1f}", fontsize=10, ha="center")
    axL.legend(loc="upper left", fontsize=8)


def draw_fan(ax, x, a_from, a_to, color, radius=0.6):
    """位置 x 的复平面截面里，两方向之间的夹角扇形"""
    t = np.linspace(a_from, a_to, 40)
    ax.plot(np.full_like(t, x), radius * np.cos(t), radius * np.sin(t),
            color=color, lw=3)
    for a in (a_from, a_to):
        ax.plot([x, x], [0, radius * np.cos(a)], [0, radius * np.sin(a)],
                color=color, lw=1.2, alpha=0.85)


def draw_right(fi):
    """右图两阶段动画：阶段1 平移不变；阶段2 距离改变"""
    axR.clear()
    setup_axis(axR, "第 3 节｜注意力分数 = q、k 相量夹角的余弦\n"
                    "阶段1：整体平移，夹角焊死不动；阶段2：拉开距离，夹角才变")

    half = FRAMES // 2
    if fi < half:                                       # 阶段 1：r=3，m 滑动
        phase, m, r = 1, 18.0 * fi / (half - 1), 3.0
    else:                                               # 阶段 2：m 固定，r 拉大
        phase, m = 2, 10.0
        r = 5.0 * (fi - half) / (half - 1)
    n = m + r

    ps = np.linspace(0, N_POS, 300)
    axR.plot(ps, np.cos(PHI_Q + ps * THETA), np.sin(PHI_Q + ps * THETA),
             color=C_Q, lw=1.4, alpha=0.4, label="query 螺旋")
    axR.plot(ps, np.cos(PHI_K + ps * THETA), np.sin(PHI_K + ps * THETA),
             color=C_K, lw=1.4, alpha=0.4, label="key 螺旋")
    axR.plot([0, N_POS], [0, 0], [0, 0], color="black", lw=0.8, alpha=0.6)

    aq, ak = PHI_Q + m * THETA, PHI_K + n * THETA
    delta = (PHI_K - PHI_Q) + r * THETA                 # 夹角只含 r，不含 m
    score = np.cos(delta)

    # q、k 各自位置上的实心箭头 + 连到比较截面的虚线
    axR.quiver(m, 0, 0, 0, np.cos(aq), np.sin(aq), color=C_Q, lw=2.4,
               arrow_length_ratio=0.14)
    axR.quiver(n, 0, 0, 0, np.cos(ak), np.sin(ak), color=C_K, lw=2.4,
               arrow_length_ratio=0.14)
    draw_cross_section(axR, n, alpha=0.8)
    axR.plot([m, n], [np.cos(aq)] * 2, [np.sin(aq)] * 2,
             color=C_Q, ls=":", lw=1.2, alpha=0.8)      # 把 q 的方向平移过来比夹角
    axR.quiver(n, 0, 0, 0, np.cos(aq), np.sin(aq), color=C_Q, lw=1.4,
               arrow_length_ratio=0.14, alpha=0.5)
    draw_fan(axR, n, aq, ak, C_FAN)

    msg = ("阶段1｜整体平移：q@%.1f, k@%.1f，距离 r=3 锁死\n"
           "夹角 %.1f° 纹丝不动 → 分数恒为 %.3f" if phase == 1 else
           "阶段2｜改变距离：m=%.0f 固定，k 滑到 %.1f，r 在拉大\n"
           "夹角 %.1f° 在变 → 分数 %.3f 跟着变")
    axR.text2D(0.03, 0.02, msg % (m, n, np.degrees(delta), score),
               transform=axR.transAxes, fontsize=10.5,
               color=C_FAN if phase == 2 else "black")
    axR.legend(loc="upper left", fontsize=8)


def draw(fi):
    front = 2.0 + (N_POS - 2.0) * min(fi / (FRAMES * 0.55), 1.0)  # 左图先长满后保持
    draw_left(front)
    draw_right(fi)


anim = FuncAnimation(fig, draw, frames=FRAMES, interval=80,
                     repeat=True, repeat_delay=1000)
plt.tight_layout()
fig.subplots_adjust(top=0.88)

if SAVE_GIF:
    import os
    out = os.path.join(os.path.dirname(__file__), "out")
    os.makedirs(out, exist_ok=True)
    anim.save(os.path.join(out, "complex_rope_3d.gif"),
              writer="pillow", fps=15)
    print("已保存 GIF 到", out)
else:
    plt.show()
