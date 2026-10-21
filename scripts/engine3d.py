# -*- coding: utf-8 -*-
"""
极简软件 3D 渲染器（numpy + PIL），给 rocket_transformer_video.py 画 3D 发动机用。

网格 = 一堆四边形；画家算法排序；金属感光照（主光 + 青色轮廓光 + 火焰底光 + 环境反射）；
2 倍超采样抗锯齿；支持剖切（cutaway）的内表面。
"""
import math
import numpy as np
from PIL import Image, ImageDraw


def _norm(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class Mesh:
    def __init__(self, Q=None, col=None, spec=None, emis=None, inner=None):
        self.Q = np.zeros((0, 4, 3)) if Q is None else Q        # 四边形顶点
        self.col = np.zeros((0, 3)) if col is None else col     # 外表面颜色
        self.spec = np.zeros(0) if spec is None else spec       # 金属度/高光
        self.emis = np.zeros((0, 3)) if emis is None else emis  # 自发光
        self.inner = np.full((len(self.Q), 3), -1.0) if inner is None else inner  # 背面颜色，-1 = 剔除

    def __add__(self, o):
        return Mesh(np.concatenate([self.Q, o.Q]), np.concatenate([self.col, o.col]),
                    np.concatenate([self.spec, o.spec]), np.concatenate([self.emis, o.emis]),
                    np.concatenate([self.inner, o.inner]))

    def transform(self, R=np.eye(3), t=(0, 0, 0)):
        return Mesh(self.Q @ np.asarray(R).T + np.asarray(t, float), self.col, self.spec, self.emis, self.inner)


def _make(quads, outward, col, spec, emis, inner):
    Q = np.asarray(quads, float)
    n = np.cross(Q[:, 2] - Q[:, 0], Q[:, 3] - Q[:, 1])
    flip = (n * outward).sum(-1) < 0
    Q[flip] = Q[flip][:, ::-1]
    N = len(Q)
    e = np.zeros((N, 3)) if emis is None else (np.asarray(emis, float) if np.ndim(emis) == 2 else np.tile(emis, (N, 1)))
    inn = np.full((N, 3), -1.0) if inner is None else np.tile(np.asarray(inner, float), (N, 1))
    return Mesh(Q, np.tile(np.asarray(col, float), (N, 1)), np.full(N, spec), e, inn)


def lathe(profile, nseg=48, col=(180, 180, 190), spec=0.6, a0=0.0, a1=2 * math.pi,
          emis=None, emis_fn=None, inner=None, flip=False, center=(0, 0, 0)):
    """绕 y 轴旋转体；profile = [(r, y), ...] 沿外表面从上往下"""
    prof = np.asarray(profile, float)
    full = abs(a1 - a0 - 2 * math.pi) < 1e-6
    th = np.linspace(a0, a1, nseg + 1)
    cx, cy, cz = center
    quads, outs, em = [], [], []
    for i in range(len(prof) - 1):
        (r0, y0), (r1, y1) = prof[i], prof[i + 1]
        dr, dy = r1 - r0, y1 - y0
        nr, ny = -dy, dr
        if flip:
            nr, ny = -nr, -ny
        for j in range(nseg):
            t0, t1 = th[j], th[j + 1]
            p = [(cx + r0 * math.cos(t0), cy + y0, cz + r0 * math.sin(t0)),
                 (cx + r1 * math.cos(t0), cy + y1, cz + r1 * math.sin(t0)),
                 (cx + r1 * math.cos(t1), cy + y1, cz + r1 * math.sin(t1)),
                 (cx + r0 * math.cos(t1), cy + y0, cz + r0 * math.sin(t1))]
            tm = (t0 + t1) / 2
            quads.append(p)
            outs.append((nr * math.cos(tm), ny, nr * math.sin(tm)))
            if emis_fn is not None:
                em.append(emis_fn((y0 + y1) / 2))
    del full
    e = np.asarray(em, float) if emis_fn is not None else emis
    return _make(quads, np.asarray(outs), col, spec, e, inner)


def section(profile_out, profile_in, angle, col=(220, 120, 70), spec=0.3, emis=None):
    """剖切面：外轮廓与内轮廓之间、在某个角度上的平面带"""
    c, s = math.cos(angle), math.sin(angle)
    quads = []
    for i in range(len(profile_out) - 1):
        (ro0, y0), (ro1, y1) = profile_out[i], profile_out[i + 1]
        (ri0, _), (ri1, _) = profile_in[i], profile_in[i + 1]
        quads.append([(ro0 * c, y0, ro0 * s), (ro1 * c, y1, ro1 * s), (ri1 * c, y1, ri1 * s), (ri0 * c, y0, ri0 * s)])
    n = np.tile((-s, 0, c), (len(quads), 1))
    m = _make(quads, n, col, spec, emis, None)
    m.inner = m.col.copy()
    return m


def _smooth(pts, k=6):
    pts = np.asarray(pts, float)
    if len(pts) < 3:
        return pts
    P = np.vstack([pts[0], pts, pts[-1]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for u in np.linspace(0, 1, k, endpoint=False):
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3))
    out.append(P[-2])
    return np.asarray(out)


def tube_path(pts, k=6):
    return _smooth(pts, k)


def tube(pts, radius, nseg=12, col=(170, 175, 185), spec=0.7, emis=None, k=6):
    path = _smooth(pts, k)
    rings = []
    for i in range(len(path)):
        T = path[min(i + 1, len(path) - 1)] - path[max(i - 1, 0)]
        T = T / (np.linalg.norm(T) + 1e-9)
        up = np.array([0, 1, 0]) if abs(T[1]) < 0.9 else np.array([1, 0, 0])
        N = np.cross(T, up)
        N /= np.linalg.norm(N)
        B = np.cross(T, N)
        rings.append([path[i] + radius * (math.cos(a) * N + math.sin(a) * B)
                      for a in np.linspace(0, 2 * math.pi, nseg, endpoint=False)])
    quads, outs = [], []
    for i in range(len(rings) - 1):
        for j in range(nseg):
            j2 = (j + 1) % nseg
            q = [rings[i][j], rings[i + 1][j], rings[i + 1][j2], rings[i][j2]]
            quads.append(q)
            outs.append(np.mean(q, 0) - (path[i] + path[i + 1]) / 2)
    return _make(quads, np.asarray(outs), col, spec, emis, None)


def box(center, size, R=np.eye(3), col=(190, 190, 200), spec=0.6, emis=None):
    sx, sy, sz = np.asarray(size) / 2
    c = np.array([[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)]) @ np.asarray(R).T + center
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    quads = [[c[i] for i in f] for f in faces]
    outs = [np.mean(q, 0) - center for q in quads]
    return _make(quads, np.asarray(outs), col, spec, emis, None)


# ---------------------------------------------------------------------------
# 相机与渲染
# ---------------------------------------------------------------------------
class Camera:
    def __init__(self, yaw=0.0, pitch=0.25, dist=8.0, target=(0, 0, 0), fov=1100, cx=960, cy=540):
        self.R = rot_x(pitch) @ rot_y(yaw)
        self.dist, self.target, self.f, self.cx, self.cy = dist, np.asarray(target, float), fov, cx, cy

    def to_cam(self, P):
        C = (np.asarray(P, float) - self.target) @ self.R.T
        C[..., 2] += self.dist
        return C

    def project(self, P):
        """世界坐标 → (屏幕 x, 屏幕 y, 深度, 透视缩放)"""
        C = self.to_cam(P)
        z = np.maximum(C[..., 2], 0.05)
        return self.cx + self.f * C[..., 0] / z, self.cy - self.f * C[..., 1] / z, z, self.f / z

    def azimuth(self):
        """相机在世界里所处的方位角（xz 平面），剖切时用"""
        p = self.R.T @ np.array([0, 0, -self.dist])
        return math.atan2(p[2], p[0])


L_KEY = _norm(np.array([-0.55, 0.75, -0.6]))
L_RIM = _norm(np.array([0.85, 0.25, 0.6]))
L_FIRE = _norm(np.array([0.0, -1.0, -0.25]))


def render(mesh, cam, W=1920, H=1080, ss=2, fire=0.0):
    """返回 RGBA 图（已抗锯齿到 W×H）"""
    img = Image.new("RGBA", (W * ss, H * ss), (0, 0, 0, 0))
    if len(mesh.Q) == 0:
        return img.reduce(ss)
    C = cam.to_cam(mesh.Q)
    zmin = C[..., 2].min(1)
    keep = zmin > 0.1
    n = _norm(np.cross(C[:, 2] - C[:, 0], C[:, 3] - C[:, 1]))
    ctr = C.mean(1)
    V = _norm(-ctr)
    facing = (n * V).sum(-1) > 0
    col = mesh.col.copy()
    back = ~facing
    has_inner = mesh.inner[:, 0] >= 0
    keep &= facing | has_inner
    n[back] = -n[back]
    col[back & has_inner] = mesh.inner[back & has_inner]
    nd = lambda L: np.maximum((n * L).sum(-1), 0)
    diff = nd(L_KEY)
    rim = nd(L_RIM)
    firel = nd(L_FIRE) * fire
    Hh = _norm(L_KEY + V)
    spec = np.maximum((n * Hh).sum(-1), 0) ** 40
    refl = V - 2 * (n * V).sum(-1, keepdims=True) * n
    env = np.clip(0.5 - 0.5 * refl[:, 1], 0, 1)  # 反射向上 = 亮天空
    fres = (1 - np.abs((n * V).sum(-1))) ** 3
    m = mesh.spec[:, None]
    shade = (col * (0.16 + 0.85 * diff[:, None] * (1 - 0.5 * m))
             + m * (env[:, None] * np.array([70, 95, 135]) + spec[:, None] * 255 * 1.1)
             + rim[:, None] * np.array([0, 150, 200]) * (0.25 + 0.6 * fres[:, None])
             + firel[:, None] * np.array([255, 120, 30]) * 0.7
             + mesh.emis)
    shade = np.clip(shade, 0, 255).astype(np.uint8)
    xs = cam.cx + cam.f * C[..., 0] / np.maximum(C[..., 2], 0.05)
    ys = cam.cy - cam.f * C[..., 1] / np.maximum(C[..., 2], 0.05)
    xs, ys = xs * ss, ys * ss
    order = np.argsort(-ctr[:, 2])
    d = ImageDraw.Draw(img)
    for i in order:
        if not keep[i]:
            continue
        c = tuple(shade[i]) + (255,)
        d.polygon([(xs[i, k], ys[i, k]) for k in range(4)], fill=c, outline=c)
    return img.reduce(ss)


# ---------------------------------------------------------------------------
# 零件：发动机各部分
# ---------------------------------------------------------------------------
COPPER = (200, 115, 70)
STEEL = (165, 172, 185)
DARK = (70, 78, 92)
NOZ = (95, 100, 112)


def chamber_profile(n_bell=18):
    """燃烧室 + 喷管外轮廓（r, y），从喷注器顶部到喷管出口"""
    p = [(0.0, 2.45), (0.3, 2.42), (0.5, 2.32), (0.6, 2.18), (0.6, 2.0)]
    for y in np.linspace(1.9, 1.15, 5):
        p.append((0.6, y))
    for k in range(1, 8):
        u = k / 7
        p.append((0.26 + 0.34 * (1 + math.cos(math.pi * u)) / 2, 1.15 - 0.4 * u))
    for k in range(1, n_bell + 1):
        u = k / n_bell
        p.append((0.26 + 1.0 * u ** 0.62, 0.75 - 2.35 * u))
    return p


def throat_y():
    return 0.75


def exit_y():
    return -1.6


def exit_r():
    return 1.26


def radius_at(y):
    p = chamber_profile()
    for (r0, y0), (r1, y1) in zip(p, p[1:]):
        if y1 <= y <= y0:
            u = (y0 - y) / (y0 - y1 + 1e-9)
            return r0 + (r1 - r0) * u
    return p[-1][0]


def engine_body(nseg=48, cut=None, heat=0.0):
    """燃烧室 + 喷管。cut = (a0, a1) 保留的角度范围（剖切）；heat = 内壁发光程度"""
    p = chamber_profile()
    a0, a1 = cut if cut else (0.0, 2 * math.pi)
    # 铜合金燃烧室部分 + 深色喷管裙，用 y 分段着色
    top = [q for q in p if q[1] >= 1.0]
    bell = [q for q in p if q[1] <= 1.0 + 1e-9]
    if top[-1] != bell[0]:
        bell = [top[-1]] + bell

    def hot(y):
        k = math.exp(-((y - throat_y()) / 0.55) ** 2)
        return (255 * 0.55 * k * heat, 90 * 0.55 * k * heat, 20 * 0.55 * k * heat)

    m = lathe(top, nseg, COPPER, 0.75, a0, a1, emis_fn=hot)
    m = m + lathe(bell, nseg, NOZ, 0.85, a0, a1, emis_fn=hot)
    # 出口唇边
    m = m + lathe([(exit_r(), exit_y() + 0.001), (exit_r() + 0.04, exit_y())], nseg, STEEL, 0.8, a0, a1)
    if cut:
        pin = [(max(0.0, r - 0.07), y) for r, y in p]

        def inner_glow(y):
            if y > 1.0:
                k = 1.0
            else:
                k = max(0.0, 1 - (1.0 - y) / 2.6)
            return (255 * k * heat, 120 * k * heat, 40 * k * heat)

        m = m + lathe(pin, nseg, (60, 40, 35), 0.2, a0, a1, emis_fn=inner_glow, flip=True)
        sec = [q for q in p if q[0] > 0.05]
        sin_ = [(max(0.0, r - 0.07), y) for r, y in sec]
        for ang in (a0, a1):
            m = m + section(sec, sin_, ang, (235, 140, 90), 0.3)
    # 喷注器顶盖 + 顶部法兰
    m = m + lathe([(0.0, 2.62), (0.35, 2.6), (0.45, 2.5), (0.45, 2.42)], nseg, STEEL, 0.8)
    return m


def turbopump(x, z, col=STEEL, spin=0.0, y0=1.35):
    """竖直放置的涡轮泵：下面涡轮壳，中间轴承段，上面泵蜗壳"""
    c = (x, 0, z)
    m = lathe([(0.0, y0 + 1.25), (0.2, y0 + 1.22), (0.26, y0 + 1.1), (0.26, y0 + 0.9)], 24, col, 0.8, center=c)
    m = m + lathe([(0.26, y0 + 0.9), (0.38, y0 + 0.85), (0.38, y0 + 0.62), (0.18, y0 + 0.55)], 24, col, 0.8, center=c)
    m = m + lathe([(0.18, y0 + 0.55), (0.18, y0 + 0.32)], 24, DARK, 0.6, center=c)
    m = m + lathe([(0.18, y0 + 0.32), (0.34, y0 + 0.27), (0.34, y0 + 0.05), (0.0, y0)], 24, (150, 120, 105), 0.7, center=c)
    # 外壳加强筋，随转子微转动以示转动（视觉提示）
    for k in range(6):
        a = spin + k * math.pi / 3
        m = m + box((x + 0.39 * math.cos(a), y0 + 0.74, z + 0.39 * math.sin(a)), (0.05, 0.2, 0.05), rot_y(-a), (120, 200, 230), 0.4)
    return m


def full_engine(spin=0.0, cut=None, heat=0.0, gg=None):
    """完整发动机：燃烧室/喷管 + 两台涡轮泵 + 管路。gg = 'open' / 'closed' / None"""
    m = engine_body(48, cut, heat)
    m = m + turbopump(1.15, 0.0, (175, 185, 205), spin)
    m = m + turbopump(-1.15, 0.0, (185, 175, 165), -spin)
    # 推进剂进口（从贮箱下来）
    m = m + tube([(1.15, 3.4, 0), (1.15, 2.6, 0)], 0.1, 12, (200, 140, 90), 0.6)
    m = m + tube([(-1.15, 3.4, 0), (-1.15, 2.6, 0)], 0.1, 12, (110, 160, 230), 0.6)
    # 泵出口 → 喷注器
    m = m + tube(FUEL_DISCHARGE, 0.085, 12, (200, 140, 90), 0.7)
    m = m + tube(LOX_DISCHARGE, 0.085, 12, (110, 160, 230), 0.7)
    if gg == "open":
        m = m + tube(GG_EXHAUST, 0.11, 12, (95, 90, 90), 0.5)
    elif gg == "closed":
        m = m + tube(HOT_GAS, 0.13, 12, (210, 120, 80), 0.6)
    return m


FUEL_IN = [(1.15, 3.6, 0), (1.15, 2.6, 0)]
LOX_IN = [(-1.15, 3.6, 0), (-1.15, 2.6, 0)]
FUEL_DISCHARGE = [(1.5, 2.05, 0.0), (1.55, 2.5, 0.25), (0.9, 2.75, 0.3), (0.35, 2.68, 0.1), (0.15, 2.62, 0.0)]
LOX_DISCHARGE = [(-1.5, 2.05, 0.0), (-1.55, 2.5, -0.25), (-0.9, 2.75, -0.3), (-0.35, 2.68, -0.1), (-0.15, 2.62, 0.0)]
GG_EXHAUST = [(1.15, 1.35, 0.0), (1.25, 0.9, 0.15), (1.5, 0.0, 0.3), (1.75, -1.0, 0.35), (1.85, -1.5, 0.35)]
HOT_GAS = [(1.15, 1.35, 0.0), (1.05, 1.15, 0.0), (0.75, 1.2, 0.0), (0.55, 1.6, 0.0)]


def turbopump_closeup(spin):
    """横放的涡轮泵特写：轴 + 涡轮盘与叶片 + 泵叶轮"""
    Rh = rot_z(-math.pi / 2)  # y 轴 → x 轴
    m = lathe([(0.09, 1.9), (0.09, -1.9)], 16, STEEL, 0.9).transform(Rh)
    # 涡轮盘（右，x>0）
    m = m + lathe([(0.0, 0.12), (0.62, 0.12), (0.66, 0.0), (0.62, -0.12), (0.0, -0.12)], 40, (150, 95, 60), 0.6).transform(Rh, (1.1, 0, 0))
    for k in range(26):
        a = spin + k * 2 * math.pi / 26
        cpos = np.array([1.1, -0.82 * math.sin(a), 0.82 * math.cos(a)])
        m = m + box(cpos, (0.2, 0.08, 0.36), rot_x(a) @ rot_z(0.45), (230, 180, 120), 0.9)
    # 泵叶轮（左，x<0）：锥形轮毂 + 弯叶片
    m = m + lathe([(0.12, 0.5), (0.3, 0.2), (0.75, -0.05), (0.75, -0.12), (0.0, -0.14)], 40, (150, 190, 230), 0.8).transform(rot_z(math.pi / 2), (-1.1, 0, 0))
    for k in range(8):
        a = -spin * 1.0 + k * 2 * math.pi / 8
        for s in range(3):
            rr = 0.3 + s * 0.17
            aa = a + s * 0.22
            cpos = np.array([-1.1 + 0.18 - s * 0.08, -rr * math.sin(aa), rr * math.cos(aa)])
            m = m + box(cpos, (0.2 - 0.04 * s, 0.05, 0.2), rot_x(aa) @ rot_z(0.3), (170, 210, 245), 0.9)
    return m
