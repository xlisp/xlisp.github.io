# -*- coding: utf-8 -*-
"""
抄答案 vs 抄过程：知识蒸馏 可视化
配套文章：《老师最痛恨的"抄作业"，原来是小模型追上大模型的唯一办法》

左面板：硬标签 —— 一根孤零零的柱子，其余全是零。熵为 0，不含任何结构信息。
中面板：软标签随温度 T 升高逐渐"打开" —— 次优选项一个个浮出水面（暗知识显影）。
右面板：训练数据量 vs 考试准确率 —— 在数据很少的左半边，
        「抄过程」那条线把「抄答案」甩开一大截。

一句话：硬标签是句号，软标签是推导；温度是让推导重新可见的放大镜。

运行：
    python distillation_visualization.py
需要：torch, numpy, matplotlib（在 conda `torch` 环境里直接跑）
若想存 GIF，把下面的 SAVE_GIF 改成 True（需要 pillow）。
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 中文字体（macOS）——参照仓库既有脚本的口径
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False

SAVE_GIF = False
RED, BLUE, GRAY = "#d62728", "#1f77b4", "#999999"

CLASSES = ["猫", "狗", "狐狸", "老虎", "汽车"]
TEACHER_LOGITS = torch.tensor([8.0, 3.2, 2.4, 1.5, -3.0])   # 老师的原始输出
TEMPS = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]

D, C = 20, len(CLASSES)
torch.manual_seed(0)
np.random.seed(0)


def entropy(p):
    p = p.clamp_min(1e-9)
    return -(p * p.log()).sum().item()


# ------------------------------------------- 右面板：数据量 vs 准确率（预计算）
W = torch.randn(D, C)


def make(n):
    X = torch.randn(n, D)
    return X, (X @ W).argmax(1)


print("正在训练老师模型...")
Xt, yt = make(6000)
teacher = nn.Sequential(nn.Linear(D, 128), nn.ReLU(), nn.Linear(128, C))
_opt = torch.optim.Adam(teacher.parameters(), lr=1e-2)
for _ in range(1200):
    _loss = F.cross_entropy(teacher(Xt), yt)
    _opt.zero_grad(); _loss.backward(); _opt.step()

Xe, ye = make(3000)
with torch.no_grad():
    TEACHER_ACC = (teacher(Xe).argmax(1) == ye).float().mean().item()
print(f"  老师考试准确率 {TEACHER_ACC:.1%}")


def student(Xs, ys, t_logits, mode, T=4.0, steps=1200):
    net = nn.Sequential(nn.Linear(D, 32), nn.ReLU(), nn.Linear(32, C))
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(steps):
        out = net(Xs)
        if mode == "hard":                                   # 只抄答案
            loss = F.cross_entropy(out, ys)
        else:                                                # 抄整个分布
            loss = F.kl_div(F.log_softmax(out / T, 1),
                            F.softmax(t_logits / T, 1),
                            reduction="batchmean") * T * T
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return (net(Xe).argmax(1) == ye).float().mean().item()


print("正在跑「数据量 vs 准确率」实验（约半分钟）...")
SIZES = [50, 100, 200, 400, 800, 1600]
ACC_HARD, ACC_SOFT = [], []
for n in SIZES:
    Xs, ys = make(n)
    with torch.no_grad():
        tl = teacher(Xs)
    ACC_HARD.append(student(Xs, ys, tl, "hard"))
    ACC_SOFT.append(student(Xs, ys, tl, "soft"))
    print(f"  {n:>4} 条: 抄答案 {ACC_HARD[-1]:.1%}  |  抄过程 {ACC_SOFT[-1]:.1%}")

# ------------------------------------------------------------------- 画图
fig, (axL, axM, axR) = plt.subplots(1, 3, figsize=(15.5, 5.4))
fig.suptitle("抄答案 vs 抄过程 —— 小模型追上大模型的唯一办法",
             fontsize=15, fontweight="bold")


def draw_hard(ax):
    ax.clear()
    p = torch.zeros(C); p[0] = 1.0
    ax.bar(CLASSES, p.numpy(), color=RED, alpha=0.85,
           edgecolor="white", linewidth=2)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("概率")
    ax.set_title(f"硬标签（标准答案）\n熵 = {entropy(p):.3f} —— 零信息，只有一个句号",
                 fontsize=12, color=RED)
    ax.grid(axis="y", alpha=0.2)


def draw_soft(ax, T):
    ax.clear()
    p = F.softmax(TEACHER_LOGITS / T, dim=0)
    ax.bar(CLASSES, p.numpy(), color=BLUE, alpha=0.85,
           edgecolor="white", linewidth=2)
    for i, v in enumerate(p):
        if v > 0.004:
            ax.text(i, float(v) + 0.02, f"{float(v):.3f}",
                    ha="center", fontsize=9, color="#333333")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("概率")
    ax.set_title(f"软标签（解题过程）· T = {T:g}\n熵 = {entropy(p):.3f} —— 次优选项浮出水面",
                 fontsize=12, color=BLUE)
    ax.grid(axis="y", alpha=0.2)


def draw_curve(ax, upto):
    ax.clear()
    k = max(1, min(upto, len(SIZES)))
    ax.axhline(TEACHER_ACC, color=GRAY, ls="--", lw=1.2)
    ax.text(SIZES[0], TEACHER_ACC + 0.012, "老师水平", fontsize=9, color=GRAY)
    ax.plot(SIZES[:k], ACC_HARD[:k], "o-", color=RED, lw=2.5, ms=8,
            label="只抄答案（硬标签）")
    ax.plot(SIZES[:k], ACC_SOFT[:k], "s-", color=BLUE, lw=2.5, ms=8,
            label="抄解题过程（软标签）")
    ax.set_xscale("log")
    ax.set_xlim(SIZES[0] * 0.8, SIZES[-1] * 1.25)
    ax.set_ylim(min(ACC_HARD) - 0.05, 1.02)
    ax.set_xticks(SIZES); ax.set_xticklabels([str(s) for s in SIZES])
    ax.set_xlabel("学生拿到的训练数据条数"); ax.set_ylabel("考试准确率")
    ax.set_title("数据越少，抄过程的优势越大", fontsize=12)
    ax.grid(alpha=0.25); ax.legend(loc="lower right", fontsize=10)


def update(frame):
    draw_hard(axL)
    draw_soft(axM, TEMPS[frame % len(TEMPS)])
    draw_curve(axR, frame + 1)


anim = FuncAnimation(fig, update, frames=max(len(TEMPS), len(SIZES)),
                     interval=1200, repeat=True, repeat_delay=1800)
plt.tight_layout(rect=[0, 0, 1, 0.93])

if SAVE_GIF:
    out = os.path.join(os.path.dirname(__file__), "out")
    os.makedirs(out, exist_ok=True)
    anim.save(os.path.join(out, "distillation.gif"), writer="pillow", fps=1)
    print("已保存 GIF 到", out)
else:
    plt.show()
