# -*- coding: utf-8 -*-
"""
核磁共振 → sin/cos 序列 → 多模态表征 可视化
配套文章：《医院那台核磁共振，和大模型的位置编码、多模态，是同一件事》

图一（静态，六格）：第一幕 + 第二幕
    左上  ：三个质子（3/6/9 Hz）的"合唱"——线圈实际收到的一团混沌波形
    中上  ：一次 FFT，三根指针的频率与强度原样掉出来
    右上  ：一维"人体" → k 空间 → 逆傅里叶还原，误差 ~1e-6
    左下  ：同一根旋转指针的两个名字——MRI 的 e^{ikx} 与 RoPE 的 e^{imθ}
    中下  ：相干度 / 点积都只依赖相对距离，两条曲线严丝合缝地重合
    右下  ：只采低频 = 只有轮廓；病灶边缘和血管细线全藏在高频里

图二（动画）：第三幕——多角度采样
    左  ：频域被一条条"投影切片"逐渐填满（1 个角度只占 0.8%）
    中  ：对应的重建图像，从一片条纹糊影慢慢长成真实世界
    右  ：还原误差随视角数单调下降，永不饱和 —— 多模态的数学理由

运行：
    python mri_to_multimodal_visualization.py
需要：numpy, matplotlib
若想存 GIF / PNG，把下面的 SAVE 改成 True（GIF 需要 pillow）。
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False

SAVE = False
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


# ---------------------------------------------------------------- 素材

def phantom(n=128):
    """一张 n×n 的"断层图"：器官（大圆）+ 病灶（小圆）+ 血管（细线）。"""
    c, s = n // 2, n / 128.0                      # s: 相对 128×128 的缩放，保证和文章数字一致
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    img = (((xx - c) ** 2 + (yy - c) ** 2) < (40 * s) ** 2) * 0.5              # 器官
    img = img + ((((xx - 80 * s) ** 2 + (yy - 52 * s) ** 2) < (10 * s) ** 2) * 0.5)  # 病灶
    img[int(60 * s):int(68 * s), int(24 * s):int(104 * s)] += 0.4             # 血管
    return img


def radial_mask(n, n_angles):
    """n_angles 个视角，每个在频域里划一条过原点的线（投影切片定理）。"""
    c = n // 2
    m = np.zeros((n, n), dtype=bool)
    t = np.arange(-c, c, dtype=float)
    for a in np.pi * np.arange(n_angles) / n_angles:
        iy = np.clip(np.round(t * np.sin(a)).astype(int) + c, 0, n - 1)
        ix = np.clip(np.round(t * np.cos(a)).astype(int) + c, 0, n - 1)
        m[iy, ix] = True
    return m


def recon_from(K, mask):
    return np.real(np.fft.ifft2(np.fft.ifftshift(K * mask)))


# ================================================================ 图一

fig1, ax = plt.subplots(2, 3, figsize=(16, 9))

# --- 左上：三个质子的合唱（第 1 节） ---
fs = 256
t = np.arange(fs) / fs
freqs, amps = [3.0, 6.0, 9.0], [1.0, 0.6, 0.3]
signal = sum(a * np.cos(2 * np.pi * f * t) for f, a in zip(freqs, amps))
for f, a, col in zip(freqs, amps, ["#1f77b4", "#2ca02c", "#d62728"]):
    ax[0, 0].plot(t, a * np.cos(2 * np.pi * f * t), color=col, lw=1,
                  alpha=0.45, label=f"{int(f)} Hz 质子群")
ax[0, 0].plot(t, signal, color="black", lw=2, label="线圈实际收到的（求和）")
ax[0, 0].set_title("① 线圈收到的不是照片，是一团合唱\n每个质子按各自的拉莫尔频率旋转")
ax[0, 0].set_xlabel("时间 (s)")
ax[0, 0].legend(fontsize=8, loc="upper right")
ax[0, 0].grid(alpha=0.25)

# --- 中上：一次 FFT 把合唱拆回独唱 ---
spec = np.abs(np.fft.rfft(signal)) / (len(t) / 2)
fr = np.fft.rfftfreq(len(t), 1 / fs)
ax[0, 1].stem(fr[:20], spec[:20], basefmt=" ")
for f, a in zip(freqs, amps):
    ax[0, 1].annotate(f"{int(f)} Hz\n强度 {a}", xy=(f, a), xytext=(f + 0.8, a + 0.06),
                      fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))
ax[0, 1].set_title("② 傅里叶：频率与强度一个不差地被找回\n（特斯拉的 369，说的就是频率可分离）")
ax[0, 1].set_xlabel("频率 (Hz)")
ax[0, 1].set_ylim(0, 1.25)
ax[0, 1].grid(alpha=0.25)

# --- 右上：一维人体 → k 空间 → 还原（疑惑点一） ---
N1 = 64
x = np.arange(N1, dtype=float)
body = np.zeros(N1)
body[20:30], body[40:45] = 1.0, 0.6
kspace = np.array([np.sum(body * np.exp(-2j * np.pi * k * x / N1)) for k in range(N1)])
recon1 = np.real(np.fft.ifft(kspace))
ax[0, 2].plot(body, color="black", lw=4, alpha=0.35, label="真实的身体")
ax[0, 2].plot(recon1, color="#d62728", lw=1.4, ls="--", label="从 k 空间还原")
ax[0, 2].set_title(f"③ 位置→相位→k空间→逆变换\n还原误差 {np.abs(recon1 - body).max():.1e}（=完美还原）")
ax[0, 2].set_xlabel("位置 x")
ax[0, 2].legend(fontsize=9, loc="upper left")
ax[0, 2].grid(alpha=0.25)
inset = ax[0, 2].inset_axes([0.56, 0.58, 0.42, 0.35])
inset.plot(np.abs(np.fft.fftshift(kspace)), color="#1f77b4", lw=1)
inset.set_title("机器真正存下的 k 空间", fontsize=7)
inset.tick_params(labelsize=6)

# --- 左下：同一根指针的两个名字（第 2 节） ---
th = np.linspace(0, 2 * np.pi, 240)
ax[1, 0].plot(np.cos(th), np.sin(th), color="gray", lw=1, alpha=0.4)
for p in range(6):
    a_mri = p * 1.0            # MRI: e^{ikx}，k=1
    ax[1, 0].annotate("", xy=(np.cos(a_mri), np.sin(a_mri)), xytext=(0, 0),
                      arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=6, alpha=0.3))
    a_rope = p * 1.0           # RoPE: e^{imθ}，θ=1
    ax[1, 0].annotate("", xy=(np.cos(a_rope), np.sin(a_rope)), xytext=(0, 0),
                      arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.6))
    ax[1, 0].text(np.cos(a_rope) * 1.15, np.sin(a_rope) * 1.15, str(p),
                  fontsize=9, ha="center", va="center")
ax[1, 0].plot([], [], color="#1f77b4", lw=6, alpha=0.3, label=r"MRI 相位编码 $e^{ikx}$（位置 x）")
ax[1, 0].plot([], [], color="#d62728", lw=1.6, label=r"RoPE 位置编码 $e^{im\theta}$（位置 m）")
ax[1, 0].set_xlim(-1.4, 1.4)
ax[1, 0].set_ylim(-1.4, 1.4)
ax[1, 0].set_aspect("equal")
ax[1, 0].set_xticks([]); ax[1, 0].set_yticks([])
ax[1, 0].set_title("④ 同一个算子换了个变量名\n蓝箭头（质子）与红箭头（词向量）完全重合")
ax[1, 0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.02))

# --- 中下：都只依赖相对距离 ---
rel = np.arange(-8, 9)
coh = [np.real(np.exp(-1j * 0.0) * np.exp(1j * 1.0 * r)) for r in rel]      # MRI 相干度
dot = [np.real(np.exp(-1j * 3.0) * np.exp(1j * (3.0 + r))) for r in rel]    # RoPE 点积（绝对位置 3）
ax[1, 1].plot(rel, coh, color="#1f77b4", lw=7, alpha=0.35, label="MRI 相干度（质子在位置 0）")
ax[1, 1].plot(rel, dot, color="#d62728", lw=2, ls="--", marker="o", ms=4,
              label="RoPE 点积（query 在位置 3）")
ax[1, 1].set_title("⑤ 绝对位置随便挪，曲线纹丝不动\n两边都只依赖「相对距离」")
ax[1, 1].set_xlabel("相对距离")
ax[1, 1].legend(fontsize=8)
ax[1, 1].grid(alpha=0.25)

# --- 右下：低频给轮廓，高频给细节（疑惑点二） ---
N = 128
img = phantom(N)
K = np.fft.fftshift(np.fft.fft2(img))
c = N // 2
yy, xx = np.mgrid[0:N, 0:N].astype(float)
r = np.sqrt((xx - c) ** 2 + (yy - c) ** 2)
strip, labels = [], []
for radius in [2, 8, 32]:
    kept = K * (r <= radius)
    strip.append(recon_from(K, r <= radius))
    e = np.sum(np.abs(kept) ** 2) / np.sum(np.abs(K) ** 2)
    labels.append(f"半径{radius}\n能量{e:.1%}")
strip.append(img)
labels.append("完整\n100%")
ax[1, 2].imshow(np.hstack(strip), cmap="gray")
ax[1, 2].set_xticks([(i + 0.5) * N for i in range(4)])
ax[1, 2].set_xticklabels(labels, fontsize=8)
ax[1, 2].set_yticks([])
ax[1, 2].set_title("⑥ 低频占 83% 能量却只给出轮廓\n病灶边缘、血管细线全在那几个 % 的高频里")

fig1.suptitle("第一幕 · 核磁共振把世界写进震动     |     第二幕 · 同一套 sin/cos 就是大模型的位置编码",
              fontsize=13)
fig1.tight_layout(rect=[0, 0, 1, 0.96])


# ================================================================ 图二（动画）

ANGLES = [1, 2, 3, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128]

fig2, bx = plt.subplots(1, 3, figsize=(15, 5))
covers, errs = [], []
for n in ANGLES:
    m = radial_mask(N, n)
    covers.append(m.mean())
    errs.append(np.abs(recon_from(K, m) - img).mean() / np.abs(img).mean())


def draw(i):
    n = ANGLES[i]
    m = radial_mask(N, n)
    rec = recon_from(K, m)

    bx[0].clear()
    bx[0].imshow(m, cmap="magma")
    bx[0].set_xticks([]); bx[0].set_yticks([])
    bx[0].set_title(f"频域采样：{n} 个视角\n每个视角只贡献一条过原点的线 —— 覆盖 {covers[i]:.1%}")

    bx[1].clear()
    bx[1].imshow(rec, cmap="gray", vmin=img.min(), vmax=img.max())
    bx[1].set_xticks([]); bx[1].set_yticks([])
    bx[1].set_title(f"从这些视角重建出的「世界」\n还原误差 {errs[i]:.1%}")

    bx[2].clear()
    bx[2].plot(ANGLES, errs, color="#cccccc", lw=1.5, zorder=1)
    bx[2].plot(ANGLES[:i + 1], errs[:i + 1], color="#d62728", lw=2.5,
               marker="o", ms=4, zorder=2)
    bx[2].scatter([n], [errs[i]], s=90, color="#d62728", zorder=3)
    bx[2].set_xscale("log", base=2)
    bx[2].set_xlabel("视角数（= 模态数）")
    bx[2].set_ylabel("还原误差")
    bx[2].set_ylim(0, 1.0)
    bx[2].grid(alpha=0.25)
    bx[2].set_title("单调下降，永不饱和\n这就是「多模态」的数学理由，不是产品功能")


anim = FuncAnimation(fig2, draw, frames=len(ANGLES), interval=700, repeat=True)
fig2.suptitle("第三幕 · 一个角度只能拿到世界的 0.8%：投影切片定理 → 柏拉图表征假说", fontsize=13)
fig2.tight_layout(rect=[0, 0.05, 1, 0.94])

if SAVE:
    os.makedirs(OUT, exist_ok=True)
    fig1.savefig(os.path.join(OUT, "mri_rope_static.png"), dpi=130)
    anim.save(os.path.join(OUT, "mri_multimodal_angles.gif"), writer="pillow", fps=2)
    print("已保存到", OUT)
else:
    plt.show()
