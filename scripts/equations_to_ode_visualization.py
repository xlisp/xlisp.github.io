# -*- coding: utf-8 -*-
"""
一般方程 → 微分方程 可视化
配套文章：docs/equations_to_differential_equations.md

图 1  根 → 流：同一个 f(y)，代数只看过零点，微分方程看整条箭头
图 2  牛顿法本身就是一个动力系统：连续牛顿流 vs 离散牛顿法的分形吸引域
图 3  判别式 → 分岔：根的个数随参数变化 = 平衡点的诞生与湮灭
图 4  Ax=b → x'=Ax：六种线性相图，由特征值决定
图 5  迹-行列式平面：二维版的"判别式"
图 6  特征方程的根轨迹：阻尼从 0 到 2，根在复平面上走，运动随之变形
图 7  守恒量：单摆的每条轨迹都躺在一条代数曲线 H(θ,ω)=E 上
图 8  代数没有的怪事：有限时间爆破、解不唯一
图 9  离散化的代价：欧拉法稳定域 |1+hλ|<1（= 梯度下降的 lr < 2/λ）
图 10 代数到头了：洛伦兹吸引子与蝴蝶效应
动画  单摆相空间里一团初值的流动（刘维尔：面积守恒）

运行：
    python scripts/equations_to_ode_visualization.py
输出到 images/diffeq/
需要：numpy, matplotlib, scipy, pillow
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.integrate import solve_ivp

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "images", "diffeq")
os.makedirs(OUT, exist_ok=True)

BLUE, RED, GREEN, ORANGE, PURPLE, GRAY = "#2563eb", "#dc2626", "#16a34a", "#ea580c", "#7c3aed", "#6b7280"


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", os.path.relpath(path, ROOT))


# ---------------------------------------------------------------------------
# 图 1：根 → 流（Allee 效应：y' = r y (y/A - 1)(1 - y/K)）
# ---------------------------------------------------------------------------
def fig1_roots_to_flow():
    r, A, K = 1.2, 0.35, 1.0
    f = lambda y: r * y * (y / A - 1) * (1 - y / K)
    roots = [0.0, A, K]
    stable = [True, False, True]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True,
                                   gridspec_kw={"width_ratios": [1, 2.2]})
    ys = np.linspace(-0.15, 1.25, 400)
    ax0.plot(f(ys), ys, color="black", lw=2)
    ax0.axvline(0, color=GRAY, lw=1)
    ax0.fill_betweenx(ys, 0, f(ys), where=f(ys) > 0, color=GREEN, alpha=0.15)
    ax0.fill_betweenx(ys, 0, f(ys), where=f(ys) < 0, color=RED, alpha=0.15)
    for y0, s in zip(roots, stable):
        ax0.plot(0, y0, "o", ms=11, mfc=BLUE if s else "white", mec=BLUE, mew=2, zorder=5)
    # 相线箭头
    for yc in [-0.1, 0.17, 0.62, 0.85, 1.15]:
        d = np.sign(f(yc))
        ax0.annotate("", xy=(-0.02, yc + 0.07 * d), xytext=(-0.02, yc - 0.07 * d),
                     arrowprops=dict(arrowstyle="-|>", color=GREEN if d > 0 else RED, lw=2.5))
    ax0.set_xlabel("f(y)")
    ax0.set_ylabel("y")
    ax0.set_title("代数视角：解 f(y)=0\n只得到 3 个点（●稳定 ○不稳定）", fontsize=11)
    ax0.text(0.02, 0.5, "f>0 ↑", color=GREEN, transform=ax0.transAxes, fontsize=10)
    ax0.text(0.02, 0.1, "f<0 ↓", color=RED, transform=ax0.transAxes, fontsize=10)

    t = np.linspace(0, 12, 400)
    for y0 in np.concatenate([np.linspace(-0.12, 1.22, 23)]):
        sol = solve_ivp(lambda t, y: f(y), (0, 12), [y0], t_eval=t, rtol=1e-8)
        yv = sol.y[0]
        c = BLUE if (yv[-1] > 0.5 or y0 > A) else PURPLE
        ax1.plot(sol.t, yv, color=c, lw=1.2, alpha=0.8)
    for y0, s in zip(roots, stable):
        ax1.axhline(y0, color=BLUE if s else ORANGE, ls="-" if s else "--", lw=2)
    ax1.text(12.2, K, "K=1 承载量（稳定）", va="center", fontsize=10, color=BLUE)
    ax1.text(12.2, A, "A=0.35 阈值（不稳定）", va="center", fontsize=10, color=ORANGE)
    ax1.text(12.2, 0, "0 灭绝（稳定）", va="center", fontsize=10, color=BLUE)
    ax1.set_xlim(0, 12)
    ax1.set_ylim(-0.15, 1.25)
    ax1.set_xlabel("t")
    ax1.set_title("微分方程视角：y' = f(y)\n同一个 f，给出的是一整族轨迹——根变成了轨迹的终点和分水岭", fontsize=11)
    fig.suptitle("图 1  根 → 流：一般方程只用了 f 的零点，微分方程用了 f 的每一个值", fontsize=13, y=1.02)
    save(fig, "fig1_roots_to_flow.png")


# ---------------------------------------------------------------------------
# 图 2：牛顿法 = 动力系统（z^3 = 1）
# ---------------------------------------------------------------------------
def fig2_newton():
    roots = np.exp(2j * np.pi * np.arange(3) / 3)
    cols = np.array([[0.15, 0.39, 0.92], [0.86, 0.15, 0.15], [0.09, 0.64, 0.29]])

    n = 700
    x = np.linspace(-1.6, 1.6, n)
    X, Y = np.meshgrid(x, x)
    Z = X + 1j * Y

    def classify(Zf):
        d = np.abs(Zf[..., None] - roots[None, None, :])
        return np.argmin(d, axis=-1)

    # 离散牛顿
    Zd = Z.copy()
    iters = np.zeros(Z.shape)
    with np.errstate(all="ignore"):
        for k in range(40):
            Zd = Zd - (Zd ** 3 - 1) / (3 * Zd ** 2)
            conv = np.min(np.abs(Zd[..., None] - roots), axis=-1) < 1e-6
            iters += ~conv
    Zd = np.nan_to_num(Zd)
    img_d = cols[classify(Zd)] * (0.45 + 0.55 * (1 - iters / 40))[..., None]

    # 连续牛顿流 dz/dt = -f/f' 有解析解：f(z(t)) = f(z0)·e^{-t}，
    # 即 f 的值沿直线缩向 0。只有当这条线段撞上临界值 f(0) = -1 时才会"走岔"，
    # 也就是 z0³ 为负实数 —— 所以吸引域的边界恰好是 arg z = ±π/3, π 三条射线，
    # 每个根的吸引域就是以它为中心、张角 120° 的扇形。
    sector = np.round(np.angle(Z) / (2 * np.pi / 3)).astype(int) % 3
    Zc = roots[sector]
    img_c = cols[classify(Zc)] * 0.9

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4))
    for ax, img, title in [
        (axes[0], img_c, "连续牛顿流  dz/dt = −f(z)/f'(z)\n吸引域的边界是三条干净的射线"),
        (axes[1], img_d, "离散牛顿法  z ← z − f(z)/f'(z)（步长 = 1）\n边界炸成了分形"),
    ]:
        ax.imshow(img, extent=[-1.6, 1.6, -1.6, 1.6], origin="lower")
        ax.plot(roots.real, roots.imag, "wo", ms=9, mec="black")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Re z")
        ax.set_ylabel("Im z")
    # 在左图上叠加流线
    xs = np.linspace(-1.6, 1.6, 30)
    XS, YS = np.meshgrid(xs, xs)
    ZS = XS + 1j * YS
    with np.errstate(all="ignore"):
        V = -(ZS ** 3 - 1) / (3 * ZS ** 2)
    V = V / np.maximum(1e-9, np.abs(V))
    axes[0].streamplot(XS, YS, V.real, V.imag, color="white", density=1.0, linewidth=0.6, arrowsize=0.8)
    axes[0].set_xlim(-1.6, 1.6)
    axes[0].set_ylim(-1.6, 1.6)
    fig.suptitle("图 2  解一般方程 z³ = 1，本身就是在跑一个动力系统：三个根 = 三个吸引子", fontsize=13)
    save(fig, "fig2_newton_as_dynamics.png")


# ---------------------------------------------------------------------------
# 图 3：判别式 → 分岔
# ---------------------------------------------------------------------------
def fig3_bifurcation():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    y = np.linspace(-2, 2, 300)
    for rr, c in [(-1, BLUE), (0, PURPLE), (1, RED)]:
        ax.plot(y, y ** 2 + rr, color=c, lw=2, label=f"r = {rr:+d}：{'两个根' if rr < 0 else ('一个重根' if rr == 0 else '无实根')}")
        if rr <= 0:
            for s in [-1, 1]:
                ax.plot(s * np.sqrt(-rr), 0, "o", color=c, ms=8)
    ax.axhline(0, color="black", lw=1)
    ax.set_ylim(-1.5, 3)
    ax.set_title("一般方程  y² + r = 0\n判别式 Δ = −4r 决定根的个数", fontsize=11)
    ax.legend(fontsize=9, loc="upper center")
    ax.set_xlabel("y")

    def quiver_field(ax, g, rr_range, yy_range):
        R, Yq = np.meshgrid(np.linspace(*rr_range, 17), np.linspace(*yy_range, 15))
        V = np.sign(g(R, Yq))
        ax.quiver(R, Yq, np.zeros_like(V), V, color=np.where(V.ravel() > 0, GREEN, RED),
                  alpha=0.35, scale=28, width=0.004, pivot="middle")

    ax = axes[1]
    quiver_field(ax, lambda r, y: r + y ** 2, (-2, 1), (-2, 2))
    rr = np.linspace(-2, 0, 200)
    ax.plot(rr, -np.sqrt(-rr), color=BLUE, lw=3, label="稳定平衡点")
    ax.plot(rr, np.sqrt(-rr), color=ORANGE, lw=3, ls="--", label="不稳定平衡点")
    ax.plot(0, 0, "ko", ms=8)
    ax.annotate("Δ = 0：两个平衡点\n相撞后一起消失", xy=(0, 0), xytext=(0.05, -1.4),
                arrowprops=dict(arrowstyle="->"), fontsize=10)
    ax.set_xlim(-2, 1)
    ax.set_title("鞍结分岔  y' = r + y²\n同一个判别式，决定平衡点的生死", fontsize=11)
    ax.set_xlabel("参数 r")
    ax.set_ylabel("平衡点 y*")
    ax.legend(fontsize=9, loc="upper left")

    ax = axes[2]
    quiver_field(ax, lambda r, y: r * y - y ** 3, (-1, 1), (-1.2, 1.2))
    rl, rp = np.linspace(-1, 0, 100), np.linspace(0, 1, 100)
    ax.plot(rl, 0 * rl, color=BLUE, lw=3, label="稳定")
    ax.plot(rp, 0 * rp, color=ORANGE, lw=3, ls="--", label="不稳定")
    ax.plot(rp, np.sqrt(rp), color=BLUE, lw=3)
    ax.plot(rp, -np.sqrt(rp), color=BLUE, lw=3)
    ax.set_title("叉式分岔  y' = r·y − y³\n对称破缺：一个解分裂成两个", fontsize=11)
    ax.set_xlabel("参数 r")
    ax.legend(fontsize=9, loc="upper left")
    fig.suptitle("图 3  判别式 → 分岔：一般方程里\"根的个数变了\"，在微分方程里就是\"系统的命运变了\"", fontsize=13, y=1.02)
    save(fig, "fig3_discriminant_to_bifurcation.png")


# ---------------------------------------------------------------------------
# 图 4 & 5：线性相图 + 迹-行列式平面
# ---------------------------------------------------------------------------
CASES = [
    ("稳定结点", np.array([[-2.0, 0.5], [0.3, -1.0]])),
    ("鞍点", np.array([[1.0, 1.0], [2.0, -1.0]])),
    ("稳定螺旋", np.array([[-0.4, -2.0], [2.0, -0.4]])),
    ("中心", np.array([[0.0, -1.5], [1.5, 0.0]])),
    ("不稳定螺旋", np.array([[0.3, -2.0], [2.0, 0.3]])),
    ("不稳定结点", np.array([[1.5, 0.3], [0.2, 0.8]])),
]
CASE_COLORS = [BLUE, RED, GREEN, PURPLE, ORANGE, "#0891b2"]


def fig4_phase_portraits():
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    g = np.linspace(-2, 2, 25)
    X, Y = np.meshgrid(g, g)
    for ax, (name, A), c in zip(axes.flat, CASES, CASE_COLORS):
        U = A[0, 0] * X + A[0, 1] * Y
        V = A[1, 0] * X + A[1, 1] * Y
        ax.streamplot(X, Y, U, V, color=GRAY, density=1.1, linewidth=0.7, arrowsize=0.9)
        for th in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            z0 = 1.8 * np.array([np.cos(th), np.sin(th)])
            ev = np.linalg.eigvals(A).real
            span = (0, -6) if np.all(ev > 0) else ((0, 3) if np.any(ev > 0) else (0, 6))
            sol = solve_ivp(lambda t, z: A @ z, span, z0, max_step=0.02,
                            events=lambda t, z: np.max(np.abs(z)) - 4)
            ax.plot(sol.y[0], sol.y[1], color=c, lw=1.6)
        lam, vec = np.linalg.eig(A)
        if np.all(np.isreal(lam)):
            for k in range(2):
                v = vec[:, k].real
                ax.plot([-3 * v[0], 3 * v[0]], [-3 * v[1], 3 * v[1]], "k--", lw=1)
        ax.plot(0, 0, "ko", ms=6)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect("equal")
        lam_s = ",  ".join(f"{l.real:+.2f}{l.imag:+.2f}i" if abs(l.imag) > 1e-9 else f"{l.real:+.2f}" for l in lam)
        ax.set_title(f"{name}\nA = {A.tolist()}\nλ = {lam_s}", fontsize=10)
    fig.suptitle("图 4  Ax = b 只有一个解 x = A⁻¹b；x' = Ax 的全部行为由 A 的特征值决定（虚线 = 实特征向量）", fontsize=13)
    fig.tight_layout()
    save(fig, "fig4_linear_phase_portraits.png")


def fig5_trace_det():
    fig, ax = plt.subplots(figsize=(10, 7))
    tau = np.linspace(-4, 4, 400)
    ax.fill_between(tau, tau ** 2 / 4, 6, where=tau < 0, color=GREEN, alpha=0.12)
    ax.fill_between(tau, tau ** 2 / 4, 6, where=tau > 0, color=ORANGE, alpha=0.12)
    ax.fill_between(tau, -3, 0, color=RED, alpha=0.08)
    ax.plot(tau, tau ** 2 / 4, color="black", lw=2.5, label="τ² − 4Δ = 0（二维的\"判别式 = 0\"）")
    ax.axhline(0, color="black", lw=1)
    ax.plot([0, 0], [0, 6], color=PURPLE, lw=2.5, label="τ = 0, Δ > 0：中心")
    ax.text(-3.6, 5.2, "稳定螺旋\n（复根，实部 < 0）", fontsize=11, color=GREEN)
    ax.text(2.0, 5.2, "不稳定螺旋\n（复根，实部 > 0）", fontsize=11, color=ORANGE)
    ax.text(-3.9, 0.6, "稳定结点\n（两负实根）", fontsize=11)
    ax.text(2.9, 0.3, "不稳定结点\n（两正实根）", fontsize=11)
    ax.text(-0.9, -2, "鞍点（Δ < 0：一正一负）", fontsize=12, color=RED)
    for (name, A), c in zip(CASES, CASE_COLORS):
        t, d = np.trace(A), np.linalg.det(A)
        ax.plot(t, d, "o", color=c, ms=12, mec="black")
        ax.annotate(name, (t, d), xytext=(6, 6), textcoords="offset points", fontsize=10, color=c)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-3.6, 6)
    ax.set_xlabel("τ = trace(A) = λ₁ + λ₂")
    ax.set_ylabel("Δ = det(A) = λ₁·λ₂")
    ax.set_title("图 5  迹-行列式平面：特征方程 λ² − τλ + Δ = 0 就是一个一元二次方程\n"
                 "韦达定理 + 判别式，把平面切成了所有线性系统的\"命运地图\"", fontsize=12)
    ax.legend(loc="lower right", fontsize=10)
    save(fig, "fig5_trace_determinant.png")


# ---------------------------------------------------------------------------
# 图 6：阻尼振子的根轨迹
# ---------------------------------------------------------------------------
def fig6_root_locus():
    w = 2.0
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1, 1.5]})
    zs = np.linspace(0, 2.2, 400)
    locus = np.array([np.roots([1, 2 * z * w, w * w]) for z in zs])
    ax0.plot(locus[:, 0].real, locus[:, 0].imag, color=GRAY, lw=1)
    ax0.plot(locus[:, 1].real, locus[:, 1].imag, color=GRAY, lw=1)
    picks = [0.0, 0.15, 0.4, 1.0, 2.0]
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(picks)))
    t = np.linspace(0, 8, 600)
    for z, c in zip(picks, cmap):
        r = np.roots([1, 2 * z * w, w * w])
        ax0.plot(r.real, r.imag, "o", color=c, ms=11, mec="black", label=f"ζ = {z}")
        sol = solve_ivp(lambda t, s: [s[1], -2 * z * w * s[1] - w * w * s[0]], (0, 8), [1, 0], t_eval=t)
        ax1.plot(t, sol.y[0], color=c, lw=2, label=f"ζ = {z}")
    ax0.axvline(0, color="black", lw=1)
    ax0.axhline(0, color="black", lw=1)
    ax0.set_aspect("equal")
    ax0.set_xlim(-8.2, 1)
    ax0.set_xlabel("Re λ（衰减速度）")
    ax0.set_ylabel("Im λ（振动频率）")
    ax0.set_title("特征方程 λ² + 2ζωλ + ω² = 0 的两个根\nζ 增大：根沿圆弧汇合（Δ=0），再沿实轴分开", fontsize=11)
    ax0.legend(fontsize=9, loc="lower left")
    ax1.axhline(0, color="black", lw=1)
    ax1.set_xlabel("t")
    ax1.set_ylabel("y(t)")
    ax1.set_title("对应的运动  y'' + 2ζω y' + ω² y = 0,  y(0)=1\n虚部 → 振荡，实部 → 衰减；ζ=1 回零最快且不过冲", fontsize=11)
    ax1.legend(fontsize=9)
    fig.suptitle("图 6  一元二次方程的根，逐点决定了一个物理系统的运动", fontsize=13, y=1.02)
    save(fig, "fig6_root_locus.png")


# ---------------------------------------------------------------------------
# 图 7：单摆 —— 守恒量把微分方程压回代数曲线
# ---------------------------------------------------------------------------
def pendulum(t, s):
    return [s[1], -np.sin(s[0])]


def fig7_pendulum():
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15, 5.8), gridspec_kw={"width_ratios": [1.6, 1]})
    th = np.linspace(-2 * np.pi, 2 * np.pi, 400)
    om = np.linspace(-3, 3, 300)
    TH, OM = np.meshgrid(th, om)
    H = OM ** 2 / 2 - np.cos(TH)
    ax0.contourf(TH, OM, H, levels=30, cmap="Blues", alpha=0.35)
    ax0.contour(TH, OM, H, levels=np.linspace(-0.9, 3, 14), colors=GRAY, linewidths=0.8, linestyles="solid")
    ax0.contour(TH, OM, H, levels=[1.0], colors=RED, linewidths=2.5)
    g = np.linspace(-2 * np.pi, 2 * np.pi, 30)
    GT, GO = np.meshgrid(g, np.linspace(-3, 3, 20))
    ax0.streamplot(GT, GO, GO, -np.sin(GT), color="black", density=0.8, linewidth=0.5, arrowsize=0.8)
    for k in [-2, -1, 0, 1, 2]:
        ax0.plot(k * np.pi, 0, "o", ms=8, mfc=BLUE if k % 2 == 0 else "white", mec=BLUE, mew=2)
    ax0.set_xlim(-2 * np.pi, 2 * np.pi)
    ax0.set_ylim(-3, 3)
    ax0.set_xlabel("角度 θ")
    ax0.set_ylabel("角速度 ω")
    ax0.set_title("θ'' = −sin θ 的相图：每条轨迹都是代数曲线  ω²/2 − cos θ = E\n"
                  "红线 E = 1 是分界线：里面来回摆，外面一直转圈", fontsize=11)

    t = np.linspace(0, 20, 800)
    for E, c in [(-0.8, BLUE), (0.5, GREEN), (0.99, ORANGE), (1.3, RED)]:
        w0 = np.sqrt(2 * (E + 1))
        sol = solve_ivp(pendulum, (0, 20), [0, w0], t_eval=t, rtol=1e-9)
        ax1.plot(t, sol.y[0], color=c, lw=2, label=f"E = {E}")
    ax1.set_xlabel("t")
    ax1.set_ylabel("θ(t)")
    ax1.set_title("同样四条轨迹的 θ(t)\n小 E 近似正弦，E→1 周期趋于无穷，E>1 单调翻转", fontsize=11)
    ax1.legend(fontsize=9)
    fig.suptitle("图 7  守恒量：能量方程把\"求一条函数\"降回\"画一条代数曲线\"", fontsize=13, y=1.02)
    save(fig, "fig7_pendulum_energy.png")


# ---------------------------------------------------------------------------
# 图 8：有限时间爆破 & 解不唯一
# ---------------------------------------------------------------------------
def fig8_blowup():
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 5.2))
    for y0, c in [(0.5, BLUE), (1.0, GREEN), (2.0, RED)]:
        tb = 1 / y0
        t = np.linspace(0, tb * 0.985, 300)
        ax0.plot(t, y0 / (1 - y0 * t), color=c, lw=2, label=f"y' = y²,  y(0) = {y0}  → t = {tb:g} 时爆破")
        ax0.axvline(tb, color=c, ls=":", lw=1.5)
        t2 = np.linspace(0, 2.5, 200)
        ax0.plot(t2, y0 * np.exp(t2), color=c, lw=1.2, ls="--", alpha=0.6)
    ax0.set_ylim(0, 30)
    ax0.set_xlim(0, 2.5)
    ax0.set_xlabel("t")
    ax0.set_title("有限时间爆破：y' = y² 的解 y = y₀/(1 − y₀t)\n虚线是 y' = y（指数），实线在有限时刻冲到无穷", fontsize=11)
    ax0.legend(fontsize=9, loc="center right")

    t = np.linspace(0, 4, 400)
    for c0, c in zip([0, 0.5, 1.0, 1.5, 2.0], plt.cm.plasma(np.linspace(0.1, 0.85, 5))):
        y = np.where(t < c0, 0, (t - c0) ** 2 / 4)
        ax1.plot(t, y, color=c, lw=2, label=f"在 t = {c0} 起飞")
    ax1.plot(t, 0 * t, color="black", lw=3, label="y ≡ 0 也是解")
    ax1.set_xlabel("t")
    ax1.set_title("解不唯一：y' = √|y|,  y(0) = 0\n同一个初值，无穷多条解（√ 在 0 处斜率无穷，违反利普希茨条件）", fontsize=11)
    ax1.legend(fontsize=9)
    fig.suptitle("图 8  一般方程里没有的现象：解会\"跑掉\"，也会\"分叉\"", fontsize=13, y=1.02)
    save(fig, "fig8_blowup_nonunique.png")


# ---------------------------------------------------------------------------
# 图 9：欧拉法稳定域 = 学习率上限
# ---------------------------------------------------------------------------
def fig9_stability():
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 5.8), gridspec_kw={"width_ratios": [1, 1.3]})
    x = np.linspace(-4, 2, 500)
    X, Y = np.meshgrid(x, np.linspace(-3.2, 3.2, 500))
    Z = X + 1j * Y
    euler = np.abs(1 + Z)
    rk4 = np.abs(1 + Z + Z ** 2 / 2 + Z ** 3 / 6 + Z ** 4 / 24)
    ax0.contourf(X, Y, (rk4 < 1).astype(float), levels=[0.5, 1.5], colors=[GREEN], alpha=0.18)
    ax0.contour(X, Y, rk4, levels=[1], colors=GREEN, linewidths=2)
    ax0.contourf(X, Y, (euler < 1).astype(float), levels=[0.5, 1.5], colors=[BLUE], alpha=0.3)
    ax0.contour(X, Y, euler, levels=[1], colors=BLUE, linewidths=2.5)
    ax0.axhline(0, color="black", lw=1)
    ax0.axvline(0, color="black", lw=1)
    for hl, c in [(-0.5, BLUE), (-1.5, ORANGE), (-2.1, RED)]:
        ax0.plot(hl, 0, "o", color=c, ms=11, mec="black")
    ax0.text(-1.85, 0.35, "欧拉 |1+hλ|<1", color=BLUE, fontsize=10)
    ax0.text(-3.4, 2.4, "RK4 稳定域", color=GREEN, fontsize=11)
    ax0.set_aspect("equal")
    ax0.set_xlabel("Re(hλ)")
    ax0.set_ylabel("Im(hλ)")
    ax0.set_title("数值格式的稳定域：\n一个关于 hλ 的代数不等式", fontsize=11)

    lam = -10.0
    tt = np.linspace(0, 1, 300)
    ax1.plot(tt, np.exp(lam * tt), color="black", lw=3, label="精确解 e^{−10t}")
    for h, c in [(0.05, BLUE), (0.15, ORANGE), (0.21, RED)]:
        n = int(1 / h) + 1
        ys = (1 + h * lam) ** np.arange(n)
        ax1.plot(np.arange(n) * h, ys, "o-", color=c, lw=1.8, ms=5,
                 label=f"h = {h}：hλ = {h*lam:.1f}，1+hλ = {1+h*lam:+.1f}")
    ax1.set_ylim(-3, 3)
    ax1.axhline(0, color="black", lw=1)
    ax1.set_xlabel("t")
    ax1.set_title("y' = −10y 用欧拉法：h > 2/10 就发散\n（换个名字：二次损失上的梯度下降，lr > 2/λ 就发散）", fontsize=11)
    ax1.legend(fontsize=9, loc="lower left")
    fig.suptitle("图 9  把微分方程离散成一般方程是有代价的：步长必须满足一个代数不等式", fontsize=13, y=1.02)
    save(fig, "fig9_euler_stability.png")


# ---------------------------------------------------------------------------
# 图 10：洛伦兹
# ---------------------------------------------------------------------------
def lorenz(t, s, sigma=10, rho=28, beta=8 / 3):
    x, y, z = s
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]


def fig10_lorenz():
    t = np.linspace(0, 40, 12000)
    a = solve_ivp(lorenz, (0, 40), [1, 1, 1], t_eval=t, rtol=1e-10, atol=1e-12)
    b = solve_ivp(lorenz, (0, 40), [1 + 1e-8, 1, 1], t_eval=t, rtol=1e-10, atol=1e-12)
    fig = plt.figure(figsize=(15, 5.8))
    ax0 = fig.add_subplot(1, 2, 1, projection="3d")
    ax0.plot(*a.y, color=BLUE, lw=0.4)
    eq = np.sqrt(8 / 3 * 27)
    ax0.scatter([eq, -eq, 0], [eq, -eq, 0], [27, 27, 0], color=RED, s=40)
    ax0.set_title("洛伦兹吸引子：三个平衡点（红）都不稳定\n轨迹永远在两个\"翅膀\"间游荡，永不重复", fontsize=11)
    ax0.set_xlabel("x"); ax0.set_ylabel("y"); ax0.set_zlabel("z")
    ax1 = fig.add_subplot(1, 2, 2)
    ax1.plot(t, a.y[0], color=BLUE, lw=1, label="x(0) = 1")
    ax1.plot(t, b.y[0], color=RED, lw=1, alpha=0.8, label="x(0) = 1 + 10⁻⁸")
    ax1.set_xlabel("t")
    ax1.set_ylabel("x(t)")
    ax1.set_title("蝴蝶效应：初值差 10⁻⁸，t ≈ 33 后完全分道扬镳", fontsize=11)
    ax1.legend(fontsize=9, loc="lower left")
    fig.suptitle("图 10  代数的尽头：平衡点 = 解一个方程组，但它们全都不稳定时，长期行为不再由任何\"根\"决定", fontsize=13, y=1.04)
    save(fig, "fig10_lorenz.png")


# ---------------------------------------------------------------------------
# 动画：单摆相空间里一团初值的流动
# ---------------------------------------------------------------------------
def anim_pendulum_flow():
    rng = np.random.default_rng(0)
    n = 1500
    blobs = [
        (np.array([0.8, 0.0]), 0.25, BLUE),
        (np.array([-0.3, 1.95]), 0.18, RED),
    ]
    pts, colors = [], []
    for c, r, col in blobs:
        a = rng.uniform(0, 2 * np.pi, n)
        rr = r * np.sqrt(rng.uniform(0, 1, n))
        pts.append(c + np.c_[rr * np.cos(a), rr * np.sin(a)])
        colors += [col] * n
    P = np.vstack(pts)

    def step(P, h):
        # 辛欧拉：保持相空间面积（刘维尔定理）
        w = P[:, 1] - h * np.sin(P[:, 0])
        th = P[:, 0] + h * w
        return np.c_[th, w]

    fig, ax = plt.subplots(figsize=(8, 5))
    th = np.linspace(-np.pi, 3 * np.pi, 300)
    om = np.linspace(-3, 3, 200)
    TH, OM = np.meshgrid(th, om)
    H = OM ** 2 / 2 - np.cos(TH)
    ax.contour(TH, OM, H, levels=np.linspace(-0.9, 3, 12), colors="#d1d5db", linewidths=0.8, linestyles="solid")
    ax.contour(TH, OM, H, levels=[1.0], colors=GRAY, linewidths=1.5)
    sc = ax.scatter(P[:, 0], P[:, 1], s=2, c=colors)
    ax.set_xlim(-np.pi, 3 * np.pi)
    ax.set_ylim(-3, 3)
    ax.set_xlabel("θ")
    ax.set_ylabel("ω")
    title = ax.set_title("")
    state = {"P": P}

    def update(k):
        for _ in range(4):
            state["P"] = step(state["P"], 0.02)
        Q = state["P"].copy()
        sc.set_offsets(Q)
        title.set_text(f"单摆相空间中两团初值的流动  t = {k * 0.08:.1f}\n"
                       "沿能量曲线滑动、被剪切拉长，但面积始终不变")
        return sc, title

    ani = FuncAnimation(fig, update, frames=90, interval=60, blit=False)
    path = os.path.join(OUT, "anim_pendulum_flow.gif")
    ani.save(path, writer=PillowWriter(fps=15), dpi=80)
    plt.close(fig)
    print("saved", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    fig1_roots_to_flow()
    fig2_newton()
    fig3_bifurcation()
    fig4_phase_portraits()
    fig5_trace_det()
    fig6_root_locus()
    fig7_pendulum()
    fig8_blowup()
    fig9_stability()
    fig10_lorenz()
    anim_pendulum_flow()
