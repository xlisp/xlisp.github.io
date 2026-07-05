# -*- coding: utf-8 -*-
"""
温度极限 可视化
配套文章：《高中觉得最玄乎的"极限"，其实是大模型"从犹豫到笃定"的那个旋钮》

左面板：固定 logits 下 softmax 分布的条形图。温度从高滑到低时，
        分布从"平坦的均匀分布"逐渐收紧成"一根独苗"（one-hot）。
右面板：分布的熵（不确定性）随温度的曲线（对数横轴）。
        T->∞ 熵逼近最大值 ln(K)（最犹豫）；T->0 熵逼近 0（最笃定）。两端各是一个极限。

运行：
    python temperature_limit_visualization.py
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

logits = np.array([3.0, 2.0, 1.0, 0.5])
K = len(logits)
labels = [f"词{i+1}" for i in range(K)]


def softmax(l):
    e = np.exp(l - l.max())
    return e / e.sum()


def entropy(p):
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


T_hi, T_lo = 30.0, 0.05
T_frames = np.geomspace(T_hi, T_lo, 110)          # 温度从高滑到低
T_grid = np.geomspace(T_lo, T_hi, 200)            # 右图熵曲线用的温度网格
H_grid = [entropy(softmax(logits / T)) for T in T_grid]
H_MAX = np.log(K)                                 # 均匀分布的熵上界

bar_colors = ["#1f77b4", "#4a90d9", "#7fb0e6", "#a9c9ef"]

fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5.2))


def draw(fi):
    T = T_frames[fi]
    p = softmax(logits / T)

    # 左：softmax 分布条形图
    axL.clear()
    axL.bar(labels, p, color=bar_colors)
    for i, v in enumerate(p):
        axL.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)
    axL.set_ylim(0, 1.05)
    axL.axhline(1 / K, color="gray", ls=":", alpha=0.6)   # 均匀分布参考线
    state = ("趋近 one-hot（T→0 的极限：绝对笃定）" if T < 0.5
             else "趋近均匀（T→∞ 的极限：完全犹豫）" if T > 5
             else "中间过渡")
    axL.set_title(f"softmax 分布  (T={T:.2f})\n{state}")
    axL.set_ylabel("概率")

    # 右：熵随温度的曲线（对数横轴）
    axR.clear()
    axR.semilogx(T_grid, H_grid, color="#1f77b4", lw=2)
    axR.scatter([T], [entropy(p)], s=120, color="#d62728",
                edgecolor="white", zorder=5)
    axR.axhline(H_MAX, color="#d62728", ls="--", alpha=0.6,
                label=f"熵上界 ln(K)={H_MAX:.2f}（均匀）")
    axR.axhline(0, color="gray", ls=":", alpha=0.6, label="熵下界 0（one-hot）")
    axR.set_xlim(T_lo, T_hi)
    axR.set_ylim(-0.1, H_MAX + 0.2)
    axR.set_title(f"分布的熵（不确定性）= {entropy(p):.3f}")
    axR.set_xlabel("温度 T（对数轴）")
    axR.set_ylabel("熵")
    axR.legend(loc="center right", fontsize=9)
    axR.grid(alpha=0.25, which="both")


anim = FuncAnimation(fig, draw, frames=len(T_frames), interval=90,
                     repeat=True, repeat_delay=1500)
plt.tight_layout()

if SAVE_GIF:
    import os
    out = os.path.join(os.path.dirname(__file__), "out")
    os.makedirs(out, exist_ok=True)
    anim.save(os.path.join(out, "temperature_limit.gif"),
              writer="pillow", fps=15)
    print("已保存 GIF 到", out)
else:
    plt.show()
