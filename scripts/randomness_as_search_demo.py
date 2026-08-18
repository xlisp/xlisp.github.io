#!/usr/bin/env python3
"""随机性作为搜索工具：四个可复现的小实验（只依赖 numpy）。

A. 维数灾难：固定 1e6 次函数求值，网格求积 vs 蒙特卡洛，误差随维度的走向。
B. 能量地形：贪心下降（零温）vs 模拟退火（有温度），命中全局最小的比例。
C. 温度扫描：固定温度从 0.01 到 5，命中率不是单调的——随机性有最优剂量。
D. 随机过程代替穷举：Gibbs 采样估的 <E>、<s_i s_j> 与穷举 2^N 的精确值对比。

    python scripts/randomness_as_search_demo.py
"""

from __future__ import annotations

import math
import numpy as np


# ---------------------------------------------------------------- A ----------
def experiment_a(budget: int = 1_000_000, seed: int = 0) -> None:
    """I(d) = ∫_{[0,1]^d} exp(-|x|^2/2) dx，精确值 = (∫_0^1 e^{-t^2/2} dt)^d。

    被积函数可分离，所以中点网格法的结果等于一维中点和的 d 次方——
    不必真的把 n^d 个点存下来，数值与老老实实铺网格完全一致。
    """
    one_dim_exact = math.sqrt(math.pi / 2) * math.erf(1 / math.sqrt(2))
    rng = np.random.default_rng(seed)

    print("A. 固定 1e6 次函数求值：网格 vs 蒙特卡洛")
    print(f"   {'d':>3} {'每维格点 n':>10} {'网格实际点数':>12} "
          f"{'网格相对误差':>12} {'MC 相对误差':>12} {'MC 标准差':>10}")
    for d in (1, 2, 3, 5, 10, 20):
        exact = one_dim_exact ** d

        n = max(int(budget ** (1.0 / d)), 1)
        xs = (np.arange(n) + 0.5) / n                      # 中点
        grid = (np.exp(-xs**2 / 2).mean()) ** d
        grid_err = abs(grid - exact) / exact

        errs = []
        for _ in range(5):
            x = rng.random((budget, d))
            mc = np.exp(-(x**2).sum(axis=1) / 2).mean()
            errs.append(abs(mc - exact) / exact)
        errs = np.array(errs)

        print(f"   {d:>3} {n:>10} {n**d:>12} "
              f"{grid_err:>12.2e} {errs.mean():>12.2e} {errs.std():>10.2e}")
    print()


# ------------------------------------------------------------- B, C, D -------
def make_landscape(n_spins: int = 12, seed: int = 7):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 1, (n_spins, n_spins))
    W = (W + W.T) / 2
    np.fill_diagonal(W, 0.0)
    return W


def all_states(n: int) -> np.ndarray:
    idx = np.arange(2 ** n)
    bits = ((idx[:, None] >> np.arange(n)[None, :]) & 1).astype(np.float64)
    return 2 * bits - 1


def energy(S: np.ndarray, W: np.ndarray) -> np.ndarray:
    return -0.5 * np.einsum("ki,ij,kj->k", S, W, S)


def greedy_settle(s, W, rng, max_sweeps: int = 200):
    """Hopfield：s_i <- sign(h_i)，确定性，只能下山。"""
    n = s.size
    for _ in range(max_sweeps):
        changed = False
        for i in rng.permutation(n):
            new = 1.0 if W[i] @ s > 0 else -1.0
            if new != s[i]:
                s[i] = new
                changed = True
        if not changed:
            break
    return s


def anneal_settle(s, W, rng, T_hi=2.0, T_lo=0.05, sweeps=60):
    """玻尔兹曼机：p(s_i=+1) = sigmoid(2 h_i / T)，允许上山。"""
    n = s.size
    for k in range(sweeps):
        T = T_hi * (T_lo / T_hi) ** (k / (sweeps - 1))
        for i in rng.permutation(n):
            p_up = 1.0 / (1.0 + math.exp(-2.0 * (W[i] @ s) / T))
            s[i] = 1.0 if rng.random() < p_up else -1.0
    return s


def fixed_T_settle(s, W, rng, T, sweeps=60):
    n = s.size
    for _ in range(sweeps):
        for i in rng.permutation(n):
            z = max(min(2.0 * (W[i] @ s) / T, 60.0), -60.0)
            p_up = 1.0 / (1.0 + math.exp(-z))
            s[i] = 1.0 if rng.random() < p_up else -1.0
    return s


def experiment_b(W, E_min, trials: int = 200, seed: int = 1) -> None:
    rng = np.random.default_rng(seed)
    n = W.shape[0]
    rows = []
    for name, fn in (("贪心下降（T=0）", greedy_settle), ("模拟退火（2.0→0.05）", anneal_settle)):
        finals = []
        for _ in range(trials):
            s0 = rng.choice([-1.0, 1.0], size=n)
            s = fn(s0.copy(), W, rng)
            finals.append(energy(s[None, :], W)[0])
        finals = np.array(finals)
        hit = np.mean(np.isclose(finals, E_min, atol=1e-9))
        rows.append((name, hit, finals.mean(), finals.min()))

    print(f"B. 同一地形（{n} 自旋，全局最小 E = {E_min:.4f}），{trials} 个随机初态")
    print(f"   {'策略':<22} {'命中全局最小':>12} {'平均终态能量':>14} {'最好终态':>10}")
    for name, hit, mean, best in rows:
        print(f"   {name:<22} {hit:>11.1%} {mean:>14.4f} {best:>10.4f}")
    print()


def experiment_c(W, E_min, trials: int = 200, seed: int = 2) -> None:
    rng = np.random.default_rng(seed)
    n = W.shape[0]
    print("C. 固定温度扫描：随机性有最优剂量")
    print(f"   {'T':>6} {'命中全局最小':>12} {'平均终态能量':>14}")
    for T in (0.01, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0):
        finals = []
        for _ in range(trials):
            s0 = rng.choice([-1.0, 1.0], size=n)
            s = fixed_T_settle(s0.copy(), W, rng, T)
            finals.append(energy(s[None, :], W)[0])
        finals = np.array(finals)
        hit = np.mean(np.isclose(finals, E_min, atol=1e-9))
        print(f"   {T:>6.2f} {hit:>11.1%} {finals.mean():>14.4f}")
    print()


def experiment_d(W, S, E_all, T: float = 1.0, sweeps: int = 20000, seed: int = 3) -> None:
    """穷举 2^N 的精确期望 vs 一条 Gibbs 链的时间平均（遍历定理的数值版）。"""
    logits = -E_all / T
    p = np.exp(logits - logits.max())
    p /= p.sum()
    E_exact = float(p @ E_all)
    C_exact = np.einsum("k,ki,kj->ij", p, S, S)

    rng = np.random.default_rng(seed)
    n = W.shape[0]
    s = rng.choice([-1.0, 1.0], size=n)
    burn = sweeps // 10
    E_acc, C_acc, kept = 0.0, np.zeros((n, n)), 0
    for k in range(sweeps):
        for i in rng.permutation(n):
            z = max(min(2.0 * (W[i] @ s) / T, 60.0), -60.0)
            s[i] = 1.0 if rng.random() < 1.0 / (1.0 + math.exp(-z)) else -1.0
        if k >= burn:
            E_acc += energy(s[None, :], W)[0]
            C_acc += np.outer(s, s)
            kept += 1
    E_gibbs = E_acc / kept
    C_gibbs = C_acc / kept

    print(f"D. T={T} 下的期望：穷举 {2**n} 个状态 vs 一条跑了 {sweeps} sweep 的 Gibbs 链")
    print(f"   <E>      精确 {E_exact:>9.4f}   采样 {E_gibbs:>9.4f}   "
          f"绝对误差 {abs(E_gibbs - E_exact):.4f}")
    off = ~np.eye(n, dtype=bool)
    err = np.abs(C_gibbs - C_exact)[off]
    print(f"   <s_i s_j>  最大绝对误差 {err.max():.4f}   平均绝对误差 {err.mean():.4f}")
    print(f"   代价对比：穷举 {2**n} 次能量计算 vs 采样 {sweeps * n} 次单点更新")
    print()


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)
    experiment_a()

    W = make_landscape()
    S = all_states(W.shape[0])
    E_all = energy(S, W)
    E_min = float(E_all.min())

    experiment_b(W, E_min)
    experiment_c(W, E_min)
    experiment_d(W, S, E_all)


if __name__ == "__main__":
    main()
