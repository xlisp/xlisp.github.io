# -*- coding: utf-8 -*-
"""
训狗 与 训大模型(GRPO)：同一套"试错 + 奖励"机制 可视化
配套文章：《训一条狗，和训出会"思考"的 DeepSeek，用的是同一套办法》

左面板：训狗 —— 同一个动作（比如"握手"）自己试 8 遍。
右面板：训大模型(GRPO) —— 同一道数学题自己答 8 遍。

两边跑的是一模一样的五步循环，逐帧演示：
    ① Rollout：同一件事自己试 8 遍（狗试动作 / 模型答同一道题）
    ② 打分：做对给 1、做错给 0（给不给零食 / 对不对标准答案）
    ③ 基线：算出这一组的平均分（虚线 = baseline）
    ④ 优势：优势 = 自己的分 − 平均分（高于线为正，低于线为负）
    ⑤ 更新：高于平均的被强化↑、低于平均的被压制↓ —— 这就是"组相对(Group Relative)"

一句话：把"优势 = 回报 − 平均回报"这行式子画出来，你会发现训狗和训 DeepSeek
        是同一台发动机——差别只在奖励是"零食"还是"答案对不对"。

运行：
    python rl_dog_to_reasoning_visualization.py
需要：numpy, matplotlib（在 conda `torch` 环境里直接跑）
若想存 GIF，把下面的 SAVE_GIF 改成 True（需要 pillow）。
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 中文字体（macOS）——参照仓库既有脚本的口径
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False

SAVE_GIF = False
N = 8                                   # 一组尝试次数（group size）

# 两边各一组"对/错"（1=做对给奖励，0=做错不给）——不同命中数，同一套机制
DOG = np.array([1, 0, 1, 0, 0, 1, 0, 0])   # 狗：8 次里 3 次抬对了爪
LLM = np.array([0, 1, 0, 0, 1, 0, 0, 0])   # 模型：8 次里 2 次答对了题

GREEN, RED, GRAY, BASE = "#2ca02c", "#d62728", "#b0b0b0", "#444444"

# --- 帧编排：五个阶段依次推进，最后停留几帧再循环 ---
R_END, S_END, B_END, A_END, U_END, HOLD = 8, 16, 21, 29, 41, 49   # 累计帧边界
TOTAL = HOLD


def get_state(f):
    """把帧号翻译成'该显示到哪一步、进度多少'。"""
    n_reveal = min(N, f + 1) if f < R_END else N                       # ① 逐个亮出尝试
    n_score = 0 if f < R_END else (min(N, f - R_END + 1) if f < S_END else N)  # ② 逐个打分
    show_base = f >= S_END                                             # ③ 平均线
    adv = 0.0 if f < B_END else (min(1.0, (f - B_END + 1) / (A_END - B_END)) if f < A_END else 1.0)   # ④ 优势
    upd = 0.0 if f < A_END else (min(1.0, (f - A_END + 1) / (U_END - A_END)) if f < U_END else 1.0)   # ⑤ 强化/压制
    if f < R_END:
        cap = "①  Rollout：同一件事，自己试 8 遍（狗试同一个动作 / 模型答同一道题）"
    elif f < S_END:
        cap = "②  打分：做对给 1、做错给 0（给不给零食 · 对不对标准答案）"
    elif f < B_END:
        cap = "③  基线：算出这一组的平均分（虚线）—— 不需要额外的裁判网络"
    elif f < A_END:
        cap = "④  优势 = 自己的分 − 平均分：高于虚线为正，低于虚线为负"
    else:
        cap = "⑤  更新：高于平均的被强化 ↑，低于平均的被压制 ↓  —— 这就是 GRPO"
    return n_reveal, n_score, show_base, adv, upd, cap


def draw_panel(ax, rewards, title, item_ok, item_no, n_reveal, n_score, show_base, adv, upd):
    ax.clear()
    mean = rewards.mean()
    xs = np.arange(N)

    for i in range(N):
        if i < n_score:                                   # 已打分：绿(对) / 红(错)
            h = float(rewards[i])
            color = GREEN if h == 1 else RED
            ax.bar(i, max(h, 0.035), width=0.62, color=color, alpha=0.9,
                   edgecolor="white", linewidth=0.6, zorder=3)
            tag = f"对 +1" if h == 1 else "错 0"
            ax.text(i, max(h, 0.035) + 0.06, tag, ha="center", va="bottom",
                    fontsize=9, color=color, zorder=5)
        elif i < n_reveal:                                # 已试出、还没打分：灰占位
            ax.bar(i, 0.5, width=0.62, color=GRAY, alpha=0.45,
                   edgecolor="white", linewidth=0.6, zorder=2)
            ax.text(i, 0.25, "试", ha="center", va="center", fontsize=10,
                    color="white", zorder=4)

    # ③ 平均线（基线）
    if show_base:
        ax.axhline(mean, ls="--", lw=1.6, color=BASE, zorder=1)
        ax.text(N - 0.4, mean + 0.03, f"平均 {mean:.2f}", ha="right", va="bottom",
                fontsize=9.5, color=BASE, style="italic")

    # ④ 优势：从平均线到自己的分的那段“缝隙”，绿=正、红=负
    if adv > 0 and show_base:
        for i in range(N):
            h = float(rewards[i])
            c = GREEN if h > mean else RED
            ax.plot([i, i], [mean, h], color=c, lw=7, alpha=0.30 * adv,
                    solid_capstyle="round", zorder=2)

    # ⑤ 更新：正优势 → ↑强化(多给零食)，负优势 → ↓压制
    if upd > 0 and show_base:
        for i in range(N):
            h = float(rewards[i])
            if h > mean:
                ax.plot(i, h + 0.20, marker="^", ms=11, color=GREEN,
                        alpha=upd, zorder=6)
                ax.text(i, h + 0.30, "强化", ha="center", va="bottom",
                        fontsize=8.5, color=GREEN, alpha=upd)
            else:
                ax.plot(i, 0.12, marker="v", ms=11, color=RED, alpha=upd, zorder=6)
                ax.text(i, -0.10, "压制", ha="center", va="top",
                        fontsize=8.5, color=RED, alpha=upd)

    ax.set_xlim(-0.7, N - 0.3)
    ax.set_ylim(-0.42, 1.62)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"第{i+1}次" for i in xs], fontsize=8)
    ax.set_yticks([0, mean, 1])
    ax.set_yticklabels(["0（不给）", "平均", "1（给）"], fontsize=8)
    ax.set_title(title, fontsize=12.5, fontweight="bold", pad=8)
    ax.grid(axis="y", alpha=0.15)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.4, 5.6))
fig.suptitle("同一套办法：训一条狗  vs  训一个会思考的大模型(GRPO)",
             fontsize=15, fontweight="bold", y=0.98)
cap_text = fig.text(0.5, 0.03, "", ha="center", va="bottom", fontsize=12,
                    color="#222", fontweight="bold")


def draw(f):
    n_reveal, n_score, show_base, adv, upd, cap = get_state(f)
    draw_panel(axL, DOG, "训狗 · 同一个动作试 8 遍", "抬对爪", "没抬",
               n_reveal, n_score, show_base, adv, upd)
    draw_panel(axR, LLM, "训大模型(GRPO) · 同一道题答 8 遍", "答对", "答错",
               n_reveal, n_score, show_base, adv, upd)
    cap_text.set_text(cap)


anim = FuncAnimation(fig, draw, frames=TOTAL, interval=380,
                     repeat=True, repeat_delay=2200)
plt.tight_layout(rect=[0, 0.07, 1, 0.95])

if SAVE_GIF:
    out = os.path.join(os.path.dirname(__file__), "out")
    os.makedirs(out, exist_ok=True)
    anim.save(os.path.join(out, "rl_dog_to_reasoning.gif"), writer="pillow", fps=4)
    print("已保存 GIF 到", out)
else:
    plt.show()
