# -*- coding: utf-8 -*-
"""
梯度磁场 → 旋转相位 → k 空间 可视化
配套文章：《把磁场歪一点点，位置就变成了角度》

图一（静态，六格）：梯度磁场这一步到底干了什么
    ①  主磁场是一条水平线，加上梯度就变成一条斜线：B(x) = B0 + G·x
    ②  磁场斜了，转速就跟着斜：拉莫尔频率沿位置线性铺开（左耳右耳差 64 kHz）
    ③  指针风车：一排质子随时间扇开成螺旋 —— 位置被写进了角度
    ④  相位地图 φ(x,t)：斜条纹，斜率就是梯度强度
    ⑤  一个 k 点是怎么测出来的：身体 × 条纹尺子 → 复平面上首尾相接的箭头链
    ⑥  全部 k 点扫完 → 逆傅里叶 → 身体原样回来（误差 ~1e-15）

图二（静态，三格）：梯度回波 —— 相位散开，再原样收回来
    左  ：+G 阶段，指针越扇越开，总信号从 1.00 掉到 0
    中  ：-G 阶段，同一批指针倒着转回来，在回波时刻重新对齐
    右  ：信号幅度曲线，一个漂亮的对称回波
        （"只依赖相对量"—— 这条性质在大模型那边叫相对位置编码）

图三（动画）：k 空间一点点被填满
    左  ：当前这一步梯度在身体上画出的二维条纹尺子
    中  ：已经采到的 k 空间（从中心低频往外扩）
    右  ：用现有 k 点重建出的图像 —— 先出轮廓，细节最后才长出来

图四（静态，四格）：澄清一个常见误解 —— 这不是啁啾（chirp）
    ①  一个位置 = 一个固定频率，全程不变（不是频率一路递减的波形）
    ②  线圈真正收到的波形：几万条固定正弦之和，中间回波、两边碎、左右对称
    ③  对照组：真正的啁啾长什么样，以及它需要什么条件
    ④ 「近的高频、远的低频」的正确含义：一整套频率并存（k空间中心↔边缘 / RoPE）

运行：
    python gradient_field_phase_visualization.py
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

GAMMA = 42.58e6          # 氢质子旋磁比 Hz/T
B0, G = 3.0, 0.010       # 主磁场 3 T，梯度 10 mT/m


def phantom(n=64):
    """一张 n×n 的"断层图"：器官（大圆）+ 病灶（小圆）+ 血管（细线）。"""
    c, s = n // 2, n / 128.0
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    img = (((xx - c) ** 2 + (yy - c) ** 2) < (40 * s) ** 2) * 0.5
    img = img + ((((xx - 80 * s) ** 2 + (yy - 52 * s) ** 2) < (10 * s) ** 2) * 0.5)
    img[int(60 * s):int(68 * s), int(24 * s):int(104 * s)] += 0.4
    return img


# ================================================================ 图一

fig1, ax = plt.subplots(2, 3, figsize=(16, 9))

# --- ① 梯度磁场本身：一条被掰斜的线 ---
xm = np.linspace(-0.15, 0.15, 200)                 # 左耳到右耳，30 厘米
ax[0, 0].axhline(B0, color="#888888", lw=2, ls="--", label="只有主磁场 $B_0$：处处一样")
ax[0, 0].plot(xm * 100, B0 + G * xm, color="#d62728", lw=2.5,
              label=r"加上梯度：$B(x)=B_0+G\cdot x$")
ax[0, 0].fill_between(xm * 100, B0, B0 + G * xm, color="#d62728", alpha=0.12)
ax[0, 0].text(-13, B0 - 0.0012, "左边弱一点 → 转得慢", fontsize=9, color="#a01f1f")
ax[0, 0].text(1, B0 + 0.0009, "右边强一点 → 转得快", fontsize=9, color="#a01f1f")
ax[0, 0].set_title("① 整台机器最天才的一步：把磁场掰斜\n"
                   "梯度只有主磁场的万分之五（3 T vs ±1.5 mT）")
ax[0, 0].set_xlabel("位置 x (cm)")
ax[0, 0].set_ylabel("磁场强度 (T)")
ax[0, 0].legend(fontsize=8, loc="lower right")
ax[0, 0].grid(alpha=0.25)

# --- ② 磁场斜了，转速就跟着斜 ---
df = GAMMA * G * xm / 1e3                          # 相对中心的频率差，kHz
ax[0, 1].plot(xm * 100, df, color="#1f77b4", lw=2.5)
ax[0, 1].axhline(0, color="#888888", lw=1, ls=":")
for xp in [-0.15, -0.075, 0.075, 0.15]:
    v = GAMMA * G * xp / 1e3
    ax[0, 1].scatter([xp * 100], [v], s=45, color="#1f77b4", zorder=3)
    ax[0, 1].annotate(f"{v:+.0f} kHz", xy=(xp * 100, v), xytext=(xp * 100 + 1, v - 9),
                      fontsize=8)
ax[0, 1].margins(0.12)
ax[0, 1].set_title("② 转速 = 位置的一次函数\n"
                   "身体每个位置，从此有了独一无二的转速")
ax[0, 1].set_xlabel("位置 x (cm)")
ax[0, 1].set_ylabel("比中心快多少 (kHz)")
ax[0, 1].grid(alpha=0.25)

# --- ③ 指针风车：一排质子随时间扇开 ---
xs = np.linspace(-0.15, 0.15, 13)
times = np.array([0, 1, 2, 3]) * 1e-6              # 微秒
cols = ["#cccccc", "#8ab4f8", "#4a7fd6", "#d62728"]
for row, (t, col) in enumerate(zip(times, cols)):
    ph = 2 * np.pi * GAMMA * G * xs * t
    ax[0, 2].quiver(xs * 100, np.full_like(xs, -row), np.cos(ph), np.sin(ph),
                    color=col, angles="xy", scale=9, width=0.007, pivot="mid")
    ax[0, 2].text(-22.5, -row, f"t={t*1e6:.0f}μs", fontsize=9, color=col, va="center")
ax[0, 2].set_xlim(-24, 18)
ax[0, 2].set_ylim(-3.7, 0.7)
ax[0, 2].set_yticks([])
ax[0, 2].set_xlabel("位置 x (cm)")
ax[0, 2].set_title("③ 梯度一开，指针就扇开成螺旋\n"
                   "位置 → 角度：空间信息被写进了相位")

# --- ④ 相位地图 φ(x, t)：斜条纹 ---
tt = np.linspace(0, 16e-6, 200)
XX, TT = np.meshgrid(np.linspace(-0.15, 0.15, 200), tt)
PH = np.angle(np.exp(1j * 2 * np.pi * GAMMA * G * XX * TT))
im = ax[1, 0].imshow(PH, cmap="twilight", aspect="auto", origin="lower",
                     extent=[-15, 15, 0, 16])
ax[1, 0].set_xlabel("位置 x (cm)")
ax[1, 0].set_ylabel("时间 t (μs)")
ax[1, 0].set_title("④ 相位地图 $\\varphi=2\\pi\\gamma G x t$\n"
                   "条纹越密 = 梯度开得越久 = k 越大")
plt.colorbar(im, ax=ax[1, 0], fraction=0.046, label="相位")

# --- ⑤ 一个 k 点怎么测出来：复平面上的箭头链 ---
N1 = 64
xi = np.arange(N1, dtype=float)
body = np.zeros(N1)
body[20:30], body[40:45] = 1.0, 0.6
for k, col in [(0, "#cccccc"), (1, "#1f77b4"), (4, "#d62728")]:
    contrib = body * np.exp(-2j * np.pi * k * xi / N1)
    path = np.concatenate([[0], np.cumsum(contrib)])       # 首尾相接的矢量链
    ax[1, 1].plot(path.real, path.imag, color=col, lw=1.6, alpha=0.9,
                  label=f"k={k}：终点 = {path[-1]:.1f}")
    ax[1, 1].scatter([path[-1].real], [path[-1].imag], s=60, color=col, zorder=3)
ax[1, 1].scatter([0], [0], s=30, color="black", zorder=4)
ax[1, 1].set_aspect("equal")
ax[1, 1].set_title("⑤ 线圈只收到一个数：所有指针的矢量和\n"
                   "一条箭头链走完 = k 空间上的一个复数")
ax[1, 1].set_xlabel("实部"); ax[1, 1].set_ylabel("虚部")
ax[1, 1].legend(fontsize=8)
ax[1, 1].grid(alpha=0.25)

# --- ⑥ 扫完全部 k → 逆傅里叶 → 身体回来 ---
kspace = np.array([np.sum(body * np.exp(-2j * np.pi * k * xi / N1)) for k in range(N1)])
recon = np.real(np.fft.ifft(kspace))
ax[1, 2].plot(body, color="black", lw=5, alpha=0.3, label="真实的身体")
ax[1, 2].plot(recon, color="#d62728", lw=1.4, ls="--", label="从 k 空间还原")
ax[1, 2].set_title(f"⑥ 逆傅里叶：世界原样回来\n还原误差 {np.abs(recon - body).max():.1e}（浮点误差级别）")
ax[1, 2].set_xlabel("位置 x")
ax[1, 2].legend(fontsize=9, loc="upper left")
ax[1, 2].grid(alpha=0.25)
inset = ax[1, 2].inset_axes([0.58, 0.55, 0.40, 0.38])
inset.plot(np.abs(np.fft.fftshift(kspace)), color="#1f77b4", lw=1)
inset.set_title("机器真正存下的 k 空间", fontsize=7)
inset.tick_params(labelsize=6)

fig1.suptitle("梯度磁场：把「你在哪」翻译成「你转了多少度」", fontsize=14)
fig1.tight_layout(rect=[0, 0, 1, 0.95])


# ================================================================ 图二：梯度回波

fig2, bx = plt.subplots(1, 3, figsize=(16, 5))

xe = np.linspace(-1, 1, 13)
STEP = 0.25
ks = np.concatenate([STEP * np.arange(5), STEP * np.arange(3, -1, -1)])   # 先散相，再倒着转回来
sig = [np.abs(np.mean(np.exp(1j * k * xe * np.pi))) for k in ks]

for row, s in enumerate(range(1, 5)):               # 左：+G 散相
    ph = ks[s] * xe * np.pi
    bx[0].quiver(xe, np.full_like(xe, -row), np.cos(ph), np.sin(ph),
                 color="#1f77b4", angles="xy", scale=11, width=0.008, pivot="mid")
    bx[0].text(-1.85, -row, f"+G 第{s}步\n信号 {sig[s]:.2f}", fontsize=8, va="center")
bx[0].set_xlim(-2.0, 1.2); bx[0].set_ylim(-3.7, 0.7); bx[0].set_yticks([])
bx[0].set_title("+G 打开：指针越扇越开\n总信号一路衰减到 0（散相）")

for row, s in enumerate(range(5, 9)):               # 中：-G 重聚
    ph = ks[s] * xe * np.pi
    bx[1].quiver(xe, np.full_like(xe, -row), np.cos(ph), np.sin(ph),
                 color="#d62728", angles="xy", scale=11, width=0.008, pivot="mid")
    bx[1].text(-1.85, -row, f"-G 第{s-4}步\n信号 {sig[s]:.2f}", fontsize=8, va="center")
bx[1].set_xlim(-2.0, 1.2); bx[1].set_ylim(-3.7, 0.7); bx[1].set_yticks([])
bx[1].set_title("-G 反转：同一批指针倒着转回来\n最后一步全部重新对齐（回波）")

bx[2].plot(range(len(sig)), sig, color="#2ca02c", lw=2.5, marker="o")
bx[2].axvline(4, color="#888888", ls="--", lw=1)
bx[2].text(4.1, 0.5, "梯度反号", fontsize=9)
bx[2].set_xlabel("时间步"); bx[2].set_ylabel("线圈收到的信号幅度")
bx[2].set_title("散开的相位可以原样收回来\n因为信号只取决于「转了多少」的相对量")
bx[2].grid(alpha=0.25)

fig2.suptitle("梯度回波：相位不是丢了，是被藏进了角度里", fontsize=14)
fig2.tight_layout(rect=[0, 0, 1, 0.93])


# ================================================================ 图三：k 空间一点点被填满

N = 64
img = phantom(N)
K = np.fft.fftshift(np.fft.fft2(img))
c = N // 2
yy, xx = np.mgrid[0:N, 0:N].astype(float)
rr = np.sqrt((xx - c) ** 2 + (yy - c) ** 2)
RADII = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]

fig3, cx = plt.subplots(1, 3, figsize=(15, 5.4))


def draw(i):
    r = RADII[i]
    mask = rr <= r
    rec = np.real(np.fft.ifft2(np.fft.ifftshift(K * mask)))
    err = np.abs(rec - img).mean() / np.abs(img).mean()

    cx[0].clear()
    cx[0].imshow(np.cos(2 * np.pi * (r * xx + (r // 2) * yy) / N), cmap="gray")
    cx[0].set_xticks([]); cx[0].set_yticks([])
    cx[0].set_title(f"这一步梯度画出的条纹尺子\n梯度越强，条纹越密（k = {r}）")

    cx[1].clear()
    cx[1].imshow(np.log1p(np.abs(K * mask)), cmap="magma")
    cx[1].set_xticks([]); cx[1].set_yticks([])
    cx[1].set_title(f"已经采到的 k 空间\n从中心低频往外扩，覆盖 {mask.mean():.1%}")

    cx[2].clear()
    cx[2].imshow(rec, cmap="gray", vmin=img.min(), vmax=img.max())
    cx[2].set_xticks([]); cx[2].set_yticks([])
    cx[2].set_title(f"此刻重建出的身体\n还原误差 {err:.1%}（先有轮廓，细节最后才长出来）")


anim = FuncAnimation(fig3, draw, frames=len(RADII), interval=800, repeat=True)
fig3.suptitle("每换一次梯度，就是拿一把更密的条纹尺子去量一次身体", fontsize=13)
fig3.tight_layout(rect=[0, 0, 1, 0.93])

# ================================================================ 图四：不是啁啾（常见误解）

fig4, dx = plt.subplots(2, 2, figsize=(15, 8.5))

# --- ① 一个位置 = 一个固定频率 ---
tc = np.linspace(0, 1, 1200)
for fq, col, lab in [(3, "#1f77b4", "左边的质子（慢）"),
                     (6, "#2ca02c", "中间的质子"),
                     (9, "#d62728", "右边的质子（快）")]:
    dx[0, 0].plot(tc, np.cos(2 * np.pi * fq * tc), color=col, lw=1.4,
                  label=f"{lab}   f={fq}，全程不变")
dx[0, 0].legend(fontsize=9); dx[0, 0].grid(alpha=0.25); dx[0, 0].set_xlabel("时间 t")
dx[0, 0].set_title("① 梯度磁场下：一个位置 = 一个【固定】频率\n"
                   "频率只和位置有关、和时间无关（所以不是啁啾）")

# --- ② 线圈真正收到的波形：全部求和 ---
xb = np.linspace(-1, 1, 400)
rho = (np.abs(xb) < 0.55) * 1.0 + (np.abs(xb - 0.75) < 0.12) * 0.6      # 一维身体
tb = np.linspace(-1, 1, 1500)
wave = np.array([np.mean(rho * np.exp(2j * np.pi * 6 * xb * tv)) for tv in tb]).real
dx[0, 1].plot(tb, wave, color="#d62728", lw=1.2)
dx[0, 1].axvline(0, color="#888888", ls="--", lw=1)
dx[0, 1].annotate("回波中心\n（所有指针恰好对齐）", xy=(0, wave.max()),
                  xytext=(0.15, wave.max() * 0.8), fontsize=9,
                  arrowprops=dict(arrowstyle="->", lw=0.8))
dx[0, 1].grid(alpha=0.25); dx[0, 1].set_xlabel("时间 t（读出梯度打开期间）")
dx[0, 1].set_title("② 线圈真正收到的波形：把①里几万条正弦全加起来\n"
                   "中间最高、两边碎、左右对称 —— 是干涉，不是频率在滑动")

# --- ③ 啁啾长什么样（对照组） ---
fi = 12 - 9 * tc                                        # 瞬时频率随时间递减
dx[1, 0].plot(tc, (0.4 + 0.9 * tc) * np.cos(2 * np.pi * np.cumsum(fi) * (tc[1] - tc[0])),
              color="#ff7f0e", lw=1.4)
dx[1, 0].grid(alpha=0.25); dx[1, 0].set_xlabel("时间 t")
dx[1, 0].set_title("③ 对照：啁啾 chirp\n"
                   "频率【沿时间轴一路递减】—— 要让梯度自己在变才会出现")

# --- ④ 「近高频、远低频」真正的意思：一整套频率并存 ---
d = 8
thetas = 10000.0 ** (-np.arange(d) / d)
mm = np.arange(0, 64)
for i in range(d):
    dx[1, 1].plot(mm, np.cos(mm * thetas[i]) + i * 2.4, lw=1.3, color=plt.cm.viridis(i / d))
    dx[1, 1].text(65, i * 2.4, f"θ={thetas[i]:.3f}", fontsize=8, va="center")
dx[1, 1].set_yticks([]); dx[1, 1].set_xlim(0, 78)
dx[1, 1].set_xlabel("位置 m（第几个词）/ 空间位置 x")
dx[1, 1].set_title("④「近的高频、远的低频」说的是这个：一整套频率同时存在\n"
                   "下面转得快=分辨相邻，上面转得慢=管住远距离（k空间中心↔边缘 / RoPE）")

fig4.suptitle("澄清：梯度给的是「一个位置一个固定频率」，不是频率递减的啁啾", fontsize=14)
fig4.tight_layout(rect=[0, 0, 1, 0.94])

if SAVE:
    os.makedirs(OUT, exist_ok=True)
    fig1.savefig(os.path.join(OUT, "gradient_field_phase.png"), dpi=130)
    fig2.savefig(os.path.join(OUT, "gradient_echo.png"), dpi=130)
    fig4.savefig(os.path.join(OUT, "not_a_chirp.png"), dpi=130)
    anim.save(os.path.join(OUT, "kspace_filling.gif"), writer="pillow", fps=1.5)
    print("已保存到", OUT)
else:
    plt.show()
