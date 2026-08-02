# -*- coding: utf-8 -*-
"""
FIM（Fill-in-the-Middle）"剪切—搬运"可视化
配套文章：《英语考试的完形填空，就是大模型学会写代码的全部方法》

左  ：原始顺序 P→M→S。光标处那一格只能向左看，后文被灰掉（看不见）
中  ：FIM 重排 <P> P <S> S <M> M。同一道视野，后文已被搬到左边，合法可见
右  ：动画——middle 逐 token 生成，绿色可见窗口始终只向左延伸，却始终含着后文

运行：
    python cloze_fim_visualization.py
需要：numpy, matplotlib
若想存 GIF，把下面的 SAVE_GIF 改成 True（需要 pillow）。
"""
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False

SAVE_GIF = False

# ---------- 数据：一段被挖空的代码 ----------
TOKENS = ["def", "add", "(a,b)", ":", "result", "=", "a+b", "return", "result"]
LO, HI = 4, 7                                   # 挖掉 tokens[4:7] = result = a+b
PREFIX, MIDDLE, SUFFIX = TOKENS[:LO], TOKENS[LO:HI], TOKENS[HI:]

# PSM 重排：<P> prefix <S> suffix <M> middle —— suffix 被挪到了 middle 前面
FIM = ["<P>"] + PREFIX + ["<S>"] + SUFFIX + ["<M>"] + MIDDLE
I_S = 1 + len(PREFIX)                           # <S> 标记的下标
I_M = I_S + 1 + len(SUFFIX)                     # <M> 标记的下标

C_PREFIX, C_SUFFIX, C_MIDDLE, C_MARK = "#cfe3f7", "#ffd8a8", "#f8ccd4", "#e6e6e6"
COL_ORIG = [C_PREFIX] * LO + [C_MIDDLE] * len(MIDDLE) + [C_SUFFIX] * len(SUFFIX)
COL_FIM = ([C_MARK] + [C_PREFIX] * len(PREFIX)
           + [C_MARK] + [C_SUFFIX] * len(SUFFIX)
           + [C_MARK] + [C_MIDDLE] * len(MIDDLE))

XMAX = max(len(TOKENS), len(FIM))
BAND_H = 0.62


def setup(ax, title):
    ax.clear()
    ax.set_xlim(-0.4, XMAX + 0.4)
    ax.set_ylim(-1.15, 1.35)
    ax.axis("off")
    ax.set_title(title, fontsize=11)


def draw_band(ax, toks, colors, y=0.0, alpha=None):
    """把 token 序列画成一条彩色带子"""
    for i, (t, c) in enumerate(zip(toks, colors)):
        a = 1.0 if alpha is None else alpha[i]
        ax.add_patch(Rectangle((i + 0.04, y), 0.92, BAND_H, facecolor=c,
                               edgecolor="#555", lw=1.0, alpha=a, zorder=2))
        ax.text(i + 0.5, y + BAND_H / 2, t, ha="center", va="center",
                fontsize=7.5, alpha=a, zorder=3)


def draw_vision(ax, x_from, y, color, label):
    """从 x_from 这一格向左射出的『因果视野』箭头"""
    ax.annotate("", xy=(0.05, y), xytext=(x_from, y),
                arrowprops=dict(arrowstyle="->", color=color, lw=2.4))
    ax.text(x_from / 2, y - 0.20, label, ha="center", va="top",
            fontsize=8.5, color=color)


fig = plt.figure(figsize=(16, 4.6))
axA, axB, axC = (fig.add_subplot(1, 3, i) for i in (1, 2, 3))

# ---------- 左图（静态）：原始顺序，后文看不见 ----------
setup(axA, "① 原始顺序 P → M → S\n光标只能往左看，后文完全在视野之外")
draw_band(axA, TOKENS, COL_ORIG,
          alpha=[1.0] * HI + [0.22] * len(SUFFIX))
draw_vision(axA, LO + 0.04, -0.30, "#1f77b4", "因果掩码：只能往左看")
axA.text(LO + len(MIDDLE) / 2, BAND_H + 0.16, "▮ 光标在这里，要补的是这段",
         ha="center", fontsize=8.5, color="#c2255c")
axA.text(HI + len(SUFFIX) / 2, -0.72, "❌ 后文在右边，看不见",
         ha="center", fontsize=9, color="#888")

# ---------- 中图（静态）：FIM 重排，后文合法可见 ----------
setup(axB, "② FIM 重排 <P> P <S> S <M> M\n后文被搬到左边，同一道视野就够到了")
draw_band(axB, FIM, COL_FIM)
draw_vision(axB, I_M + 1.04, -0.30, "#1f77b4", "规矩一条没破：依然只往左看")
axB.add_patch(Rectangle((I_S + 1.0, -0.10), len(SUFFIX), BAND_H + 0.20,
                        fill=False, edgecolor="#2f9e44", lw=2.0, ls="--", zorder=4))
axB.text(I_S + 1 + len(SUFFIX) / 2, BAND_H + 0.16, "✅ 后文已搬到左边",
         ha="center", fontsize=8.5, color="#2f9e44")
axB.text(I_M + 1 + len(MIDDLE) / 2, BAND_H + 0.16, "现在开始作答",
         ha="center", fontsize=8.5, color="#c2255c")

# ---------- 右图（动画）：middle 逐 token 生成 ----------
FRAMES = len(MIDDLE) + 3          # 末尾多留几帧停顿


def draw_step(frame):
    k = min(frame, len(MIDDLE))                       # 已经生成了几个 middle token
    setup(axC, "③ 逐 token 生成 middle\n可见窗口始终只向左，却始终含着后文")

    shown = list(FIM[: I_M + 1 + k]) + ["?"] * (len(MIDDLE) - k)
    draw_band(axC, shown, COL_FIM,
              alpha=[1.0] * (I_M + 1 + k) + [0.25] * (len(MIDDLE) - k))

    # 绿色可见窗口：0 .. 当前位置
    axC.add_patch(Rectangle((0.0, -0.14), I_M + 1 + k, BAND_H + 0.28,
                            facecolor="#2f9e44", alpha=0.12, zorder=1))
    if k < len(MIDDLE):                               # 高亮正在生成的那一格
        axC.add_patch(Rectangle((I_M + 1 + k + 0.04, 0.0), 0.92, BAND_H,
                                fill=False, edgecolor="#c2255c", lw=2.4, zorder=5))
        axC.text(I_M + 1 + k + 0.5, BAND_H + 0.16, "正在写",
                 ha="center", fontsize=8.5, color="#c2255c")
        draw_vision(axC, I_M + 1 + k + 0.04, -0.34, "#2f9e44",
                    f"第 {k + 1} 个 middle token 的可见范围（含后文）")
    else:
        axC.text(XMAX / 2, -0.62, "✅ 补全完成 —— 全程没有偷看过任何『右边』",
                 ha="center", fontsize=9.5, color="#2f9e44")


anim = FuncAnimation(fig, draw_step, frames=FRAMES, interval=900, repeat=True)
plt.tight_layout()

if SAVE_GIF:
    out = os.path.join(os.path.dirname(__file__), "out")
    os.makedirs(out, exist_ok=True)
    anim.save(os.path.join(out, "cloze_fim.gif"), writer="pillow", fps=1.2)
    print("已保存 GIF 到", out)
else:
    plt.show()
