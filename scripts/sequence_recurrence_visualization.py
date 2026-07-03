# -*- coding: utf-8 -*-
"""
数列递推可视化
配套文章：《高中背了三年的数列，原来是 GPT 一个字一个字蹦出来的秘密》

左：一条数列一项一项被"推"出来 —— 每个新项由前一项经规则 f 算出（自回归生成的骨架）
右：等比数列 rⁿ 的三种命运 —— r<1 塌向 0（梯度消失）、r=1 守恒、r>1 冲天（梯度爆炸）
两栏用同一个步数 n 同步：递推推得越多步，等比数列的命运就越极端。

运行：
    python sequence_recurrence_visualization.py
需要：numpy, matplotlib
若想存 GIF，把下面的 SAVE_GIF 改成 True（需要 pillow）。
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False

SAVE_GIF = False
N = 11            # 数列展开到第 N 项（右图同步画到 rⁿ 的第 N 项）
WINDOW = 6        # 左图镜头里同时显示的节点数（超出就右移）

# --- 左图：一条递推数列。用一个有界的非线性规则 f，值不会爆也不会趴平 ---
def f(a):
    return 3.2 * a * (1 - a)          # 一条示例递推规则 aₙ₊₁ = f(aₙ)

seq = [0.2]
for _ in range(N - 1):
    seq.append(f(seq[-1]))            # 反复套用 f，一项一项往下推

# --- 右图：等比数列 rⁿ 的三种命运 ---
R_VALUES = [0.7, 1.0, 1.3]
R_COLORS = ["#1f77b4", "#7f7f7f", "#d62728"]
R_LABELS = ["r=0.7　梯度消失", "r=1.0　守恒", "r=1.3　梯度爆炸"]
ns = np.arange(1, N + 1)

fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5))


def draw(step):
    n = step + 1                     # 当前展开到第 n 项

    # ---------- 左图：数列逐项递推 ----------
    axL.clear()
    axL.set_ylim(-1.2, 1.2)
    axL.axis("off")
    lo = max(0, n - WINDOW)          # 镜头右移：只显示最近 WINDOW 个节点
    hi = n
    axL.set_xlim(lo - 0.6, lo + WINDOW + 0.4)

    for i in range(lo, hi):
        x = i
        # 节点
        axL.add_patch(plt.Circle((x, 0), 0.30, color="#1f77b4",
                                  alpha=0.85, zorder=3))
        axL.text(x, 0, f"a{i+1}", color="white", ha="center", va="center",
                 fontsize=10, fontweight="bold", zorder=4)
        axL.text(x, -0.55, f"{seq[i]:.3f}", ha="center", va="center",
                 fontsize=9, color="#333")
        # 指向下一项的 f 箭头
        if i + 1 < hi:
            axL.annotate("", xy=(x + 0.68, 0.0), xytext=(x + 0.32, 0.0),
                         arrowprops=dict(arrowstyle="-|>", color="#e6842e",
                                         lw=2, connectionstyle="arc3,rad=-0.35"))
            axL.text(x + 0.5, 0.42, "f", color="#e6842e", ha="center",
                     fontsize=11, fontstyle="italic")
    axL.set_title(f"递推数列展开（已推到第 {n} 项）\n"
                  f"每个新项 = 规则 f 作用在前一项上　（f(a)=3.2·a·(1−a)）",
                  fontsize=11)

    # ---------- 右图：等比数列 rⁿ 的命运 ----------
    axR.clear()
    for r, col, lab in zip(R_VALUES, R_COLORS, R_LABELS):
        y = r ** ns
        axR.plot(ns[:n], y[:n], "-", color=col, lw=2, label=lab, alpha=0.4)
        axR.plot(n, r ** n, "o", color=col, ms=9, zorder=5)   # 当前项高亮
    axR.axhline(1, color="#7f7f7f", ls=":", alpha=0.5)
    axR.set_yscale("log")
    axR.set_xlim(1, N)
    axR.set_ylim(5e-3, 1e2)
    axR.set_xlabel("递推步数 n")
    axR.set_ylabel("rⁿ（对数轴）")
    axR.set_title(f"等比数列 rⁿ 的三种命运（第 {n} 步）\n"
                  f"多传一步，就多乘一个 r —— 消失 / 守恒 / 爆炸", fontsize=11)
    axR.legend(loc="center left", fontsize=9)
    axR.grid(alpha=0.25, which="both")


anim = FuncAnimation(fig, draw, frames=N, interval=650, repeat=True)
plt.tight_layout()

if SAVE_GIF:
    import os
    out = os.path.join(os.path.dirname(__file__), "out")
    os.makedirs(out, exist_ok=True)
    anim.save(os.path.join(out, "sequence_recurrence.gif"), writer="pillow", fps=2)
    print("已保存 GIF 到", out)
else:
    plt.show()
