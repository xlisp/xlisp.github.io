# -*- coding: utf-8 -*-
"""
横屏视频：从火箭发动机的动力循环，看懂 Transformer 的计算图
配套文章：posts/rocket-engine-to-transformer.html

运行：
    python scripts/rocket_transformer_video.py            # 渲染完整视频
    python scripts/rocket_transformer_video.py preview    # 每个场景导出一张静帧
输出到 scripts/out/rocket_transformer/
需要：numpy, pillow, ffmpeg
"""
import os
import sys
import math
import wave
import subprocess
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops

import engine3d as e3

W, H, FPS = 1920, 1080, 30
SR = 44100
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "scripts", "out", "rocket_transformer")
FONT = "/system/fonts/NotoSansCJK-Regular.ttc"
SANS_I, MONO_I = 2, 7

CYAN = (0, 229, 255)
ORANGE = (255, 140, 20)
YELLOW = (255, 216, 90)
MAGENTA = (255, 60, 172)
RED = (255, 72, 72)
GREEN = (70, 255, 150)
WHITE = (235, 242, 255)
GRAY = (120, 135, 160)
PURPLE = (165, 115, 255)
BLUE = (70, 140, 255)
PIPE = (30, 70, 110)

# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
_fonts = {}


def font(size, mono=False):
    k = (size, mono)
    if k not in _fonts:
        _fonts[k] = ImageFont.truetype(FONT, size, index=MONO_I if mono else SANS_I)
    return _fonts[k]


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


def win(t, t0, t1=1e9, fi=0.4, fo=0.4):
    return clamp((t - t0) / fi) * clamp((t1 - t) / fo)


def frac(x):
    return x - math.floor(x)


def rnd(i, k=0):
    return frac(math.sin(i * 12.9898 + k * 78.233) * 43758.5453)


def sc(c, a):
    a = clamp(a)
    return tuple(int(v * a) for v in c)


def mix(c1, c2, u):
    u = clamp(u)
    return tuple(int(a + (b - a) * u) for a, b in zip(c1, c2))


def pl_at(pts, u):
    """折线上按弧长比例 u 取点"""
    segs = [math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    total = sum(segs)
    d = clamp(u) * total
    for i, s in enumerate(segs):
        if d <= s or i == len(segs) - 1:
            r = d / s if s else 0
            (x0, y0), (x1, y1) = pts[i], pts[i + 1]
            return x0 + (x1 - x0) * r, y0 + (y1 - y0) * r
        d -= s
    return pts[-1]


def pl_part(pts, u):
    """折线的前 u 部分"""
    if u >= 1:
        return list(pts)
    segs = [math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    d = clamp(u) * sum(segs)
    out = [pts[0]]
    for i, s in enumerate(segs):
        if d >= s:
            out.append(pts[i + 1])
            d -= s
        else:
            r = d / s if s else 0
            (x0, y0), (x1, y1) = pts[i], pts[i + 1]
            out.append((x0 + (x1 - x0) * r, y0 + (y1 - y0) * r))
            break
    return out


# ---------------------------------------------------------------------------
# 帧：清晰层 + 半分辨率辉光层，最后加法合成（霓虹风）
# ---------------------------------------------------------------------------
class Fr:
    def __init__(self):
        self.S = Image.new("RGB", (W, H))
        self.G = Image.new("RGB", (W // 2, H // 2))
        self.ds = ImageDraw.Draw(self.S)
        self.dg = ImageDraw.Draw(self.G)
        self.M = None  # 3D 网格遮罩（挡住背景）

    def line(self, pts, c, w=4, a=1, g=0.8):
        if a <= 0.01 or len(pts) < 2:
            return
        self.ds.line(pts, fill=sc(c, a), width=w, joint="curve")
        if g > 0:
            self.dg.line([(x / 2, y / 2) for x, y in pts], fill=sc(c, a * g),
                         width=max(2, w // 2 + 2), joint="curve")

    def dashed(self, p0, p1, c, w=3, a=1, dash=14, g=0.5):
        n = max(1, int(math.dist(p0, p1) / dash))
        for i in range(0, n, 2):
            self.line([pl_at([p0, p1], i / n), pl_at([p0, p1], (i + 1) / n)], c, w, a, g)

    def dot(self, x, y, r, c, a=1, g=1.0):
        if a <= 0.01:
            return
        self.ds.ellipse([x - r, y - r, x + r, y + r], fill=sc(c, a))
        if g > 0:
            rg = r / 2 + 1.5
            self.dg.ellipse([x / 2 - rg, y / 2 - rg, x / 2 + rg, y / 2 + rg], fill=sc(c, a * g))

    def ring(self, x, y, r, c, w=3, a=1, g=0.8):
        if a <= 0.01:
            return
        self.ds.ellipse([x - r, y - r, x + r, y + r], outline=sc(c, a), width=w)
        if g > 0:
            self.dg.ellipse([x / 2 - r / 2, y / 2 - r / 2, x / 2 + r / 2, y / 2 + r / 2],
                            outline=sc(c, a * g), width=max(2, w // 2 + 1))

    def box(self, b, c, a=1, w=3, r=14, fill=None, fa=0.18, g=0.6):
        if a <= 0.01:
            return
        x0, y0, x1, y1 = b
        if fill is not None:
            self.ds.rounded_rectangle(b, r, fill=sc(fill, a * fa))
        self.ds.rounded_rectangle(b, r, outline=sc(c, a), width=w)
        if g > 0:
            self.dg.rounded_rectangle([x0 / 2, y0 / 2, x1 / 2, y1 / 2], r / 2,
                                      outline=sc(c, a * g), width=max(2, w // 2 + 1))

    def fillrect(self, b, c, a=1, g=0.6):
        if a <= 0.01 or b[3] <= b[1]:
            return
        self.ds.rectangle(b, fill=sc(c, a))
        if g > 0:
            self.dg.rectangle([v / 2 for v in b], fill=sc(c, a * g))

    def poly(self, pts, fill=None, outline=None, a=1, w=3, g=0.6, fa=1.0):
        if a <= 0.01:
            return
        if fill is not None:
            self.ds.polygon(pts, fill=sc(fill, a * fa))
        if outline is not None:
            self.line(list(pts) + [pts[0]], outline, w, a, g)

    def text(self, xy, s, size, c=WHITE, a=1, g=0.45, anchor="mm", mono=False, bold=False):
        if a <= 0.01 or not s:
            return
        sw = max(1, size // 30) if bold else 0
        self.ds.text(xy, s, font=font(size, mono), fill=sc(c, a), anchor=anchor,
                     stroke_width=sw, stroke_fill=sc(c, a))
        g *= 0.45
        if g > 0:
            self.dg.text((xy[0] / 2, xy[1] / 2), s, font=font(max(8, size // 2), mono),
                         fill=sc(c, a * g), anchor=anchor)

    def rich(self, xy, s, size, a=1, base=WHITE, acc=CYAN, align="c", n=None, bold=False, mono=False):
        """【】内的文字用强调色；n = 打字机效果显示的字数"""
        segs, cur, hot = [], "", False
        for ch in s:
            if ch in "【】":
                if cur:
                    segs.append((cur, hot))
                cur, hot = "", ch == "【"
            else:
                cur += ch
        if cur:
            segs.append((cur, hot))
        f = font(size, mono)
        total = sum(f.getlength(t) for t, _ in segs)
        x = xy[0] - total / 2 if align == "c" else xy[0]
        left = 10 ** 9 if n is None else int(n)
        for t, hot in segs:
            show = t[:max(0, left)]
            left -= len(t)
            if show:
                self.text((x, xy[1]), show, size, acc if hot else base, a,
                          g=0.75 if hot else 0.12, anchor="lm", mono=mono, bold=bold or hot)
            x += f.getlength(t)


# ---------------------------------------------------------------------------
# 背景：星云 + 网格 + 暗角（预计算），星星每帧漂移
# ---------------------------------------------------------------------------
BG = None
STARS = None


def init_bg():
    global BG, STARS
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    bg = np.zeros((H, W, 3), np.float32)
    bg += np.array([3, 5, 12], np.float32)
    bg += (yy / H)[..., None] * np.array([4, 8, 18], np.float32)
    for x0, y0, s, c in [(300, 250, 380, (40, 10, 60)), (1600, 850, 460, (0, 35, 55)),
                         (1500, 150, 300, (35, 12, 45)), (500, 950, 350, (0, 22, 40)),
                         (960, 540, 600, (8, 10, 26))]:
        g = np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * s * s))
        bg += g[..., None] * np.array(c, np.float32)
    grid = ((xx % 80) < 1.2) | ((yy % 80) < 1.2)
    bg += grid[..., None] * np.array([6, 10, 16], np.float32)
    vig = 1 - 0.55 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) ** 1.2 / 2 ** 1.2
    bg *= vig[..., None]
    BG = bg
    rng = np.random.default_rng(7)
    n = 320
    STARS = (rng.uniform(0, W, n), rng.uniform(0, H, n), rng.uniform(4, 30, n),
             rng.uniform(0, 6.28, n), rng.uniform(0.3, 1, n))


def compose(fr, fade, T):
    g1 = fr.G.filter(ImageFilter.GaussianBlur(5)).resize((W, H), Image.BILINEAR)
    g2 = fr.G.resize((W // 4, H // 4), Image.BILINEAR).filter(ImageFilter.GaussianBlur(9)).resize((W, H), Image.BILINEAR)
    fg = np.asarray(fr.S, np.float32) + np.asarray(g1, np.float32) * 1.3 + np.asarray(g2, np.float32) * 1.7
    bg = BG if fr.M is None else BG * (1 - fr.M[..., None] * 0.95 * fade)
    out = bg + fg * fade
    xs, ys, sp, ph, br = STARS
    x = ((xs - T * sp) % W).astype(int)
    y = ys.astype(int)
    b = (br * (0.55 + 0.45 * np.sin(T * 2.2 + ph)) * 200)[:, None]
    out[y, x] += b
    out[np.minimum(y + 1, H - 1), x] += b * 0.5
    out[y, np.minimum(x + 1, W - 1)] += b * 0.5
    return np.clip(out, 0, 255).astype(np.uint8)


def glitch(arr, strength, seed):
    """场景切换时的错位 + RGB 分离"""
    if strength <= 0.02:
        return arr
    arr = arr.copy()
    rng = np.random.default_rng(seed)
    for _ in range(int(3 + 7 * strength)):
        y0 = int(rng.uniform(0, H - 40))
        h = int(rng.uniform(6, 60))
        dx = int(rng.uniform(-80, 80) * strength)
        arr[y0:y0 + h] = np.roll(arr[y0:y0 + h], dx, axis=1)
    s = int(10 * strength)
    if s:
        arr[..., 0] = np.roll(arr[..., 0], s, axis=1)
        arr[..., 2] = np.roll(arr[..., 2], -s, axis=1)
    return arr


# ---------------------------------------------------------------------------
# 复用部件：发动机、Transformer block、标题、字幕、粒子流
# ---------------------------------------------------------------------------
def engine(f, cx, ty, s, t, a=1, flame=1.0, col=(150, 200, 255)):
    def r_of(y):
        if y < 120:
            return 62
        if y < 190:
            u = (y - 120) / 70
            return 22 + 40 * (1 + math.cos(math.pi * u)) / 2
        return 22 + 128 * ((y - 190) / 230) ** 0.55

    ys = list(range(0, 421, 10))
    right = [(cx + r_of(y) * s, ty + y * s) for y in ys]
    left = [(cx - r_of(y) * s, ty + y * s) for y in reversed(ys)]
    shape = right + left
    f.poly(shape, fill=(22, 34, 58), a=a, fa=1)
    f.line(right, col, max(2, int(4 * s)), a, 0.7)
    f.line(left, col, max(2, int(4 * s)), a, 0.7)
    # 冷却通道纹理
    for k in range(1, 9):
        y = 190 + k * 26
        f.line([(cx - r_of(y) * s + 4, ty + y * s), (cx + r_of(y) * s - 4, ty + y * s)], col, 1, a * 0.35, 0)
    # 喷注器盖板 + 涡轮泵
    f.fillrect([cx - 80 * s, ty - 18 * s, cx + 80 * s, ty], col, a * 0.8, 0.5)
    for sgn, c in ((-1, BLUE), (1, ORANGE)):
        bx = cx + sgn * 150 * s
        f.box([bx - 34 * s, ty + 10 * s, bx + 34 * s, ty + 110 * s], c, a, max(2, int(3 * s)), int(10 * s), fill=c, fa=0.15)
        f.line([(bx, ty + 10 * s), (bx, ty - 40 * s), (cx + sgn * 40 * s, ty - 40 * s), (cx + sgn * 40 * s, ty - 18 * s)], c, max(2, int(4 * s)), a, 0.6)
        f.line([(bx - sgn * 34 * s, ty + 70 * s), (cx + sgn * 62 * s, ty + 70 * s)], c, max(2, int(3 * s)), a * 0.8, 0.5)
    if flame <= 0:
        return
    ex = ty + 420 * s
    L = 560 * s * flame
    for k in range(6):
        yk = ex + (40 + 80 * k) * s * flame
        if yk > ex + L * 0.85:
            break
        fl = 0.75 + 0.25 * math.sin(t * 40 + k)
        rw = (55 - 6 * k) * s
        f.ds.ellipse([cx - rw, yk - 22 * s, cx + rw, yk + 22 * s], fill=sc((255, 190, 225), a * 0.2 * fl))
        f.dg.ellipse([(cx - rw) / 2, (yk - 22 * s) / 2, (cx + rw) / 2, (yk + 22 * s) / 2], fill=sc((255, 110, 190), a * 0.35 * fl))
    for i in range(160):
        r1, r2, r3 = rnd(i, 1), rnd(i, 2), rnd(i, 3)
        u = frac(r1 + t * (1.2 + 0.8 * r2))
        y = ex + u * L
        w = (140 * (1 - 0.55 * u) + 20) * s
        x = cx + (r3 - 0.5) * 2 * w * (0.6 + 0.4 * math.sin(t * 9 + i))
        c = mix((255, 235, 160), ORANGE, u * 2.2) if u < 0.6 else mix(ORANGE, (200, 30, 20), (u - 0.6) * 2.5)
        f.dot(x, y, (13 * (1 - u) + 3) * s, c, a * (1 - u) ** 1.3 * 0.7, 0.9)


def tblock(f, cx, ty, s, t, a=1, hl=-1):
    rows = [("LayerNorm", GRAY, 60), ("Multi-Head Attention", CYAN, 90), ("⊕", WHITE, 50),
            ("LayerNorm", GRAY, 60), ("FFN  (MLP)", ORANGE, 90), ("⊕", WHITE, 50)]
    y = ty
    w = 300 * s
    centers = []
    for i, (name, c, h) in enumerate(rows):
        h *= s
        aa = a * (1.0 if hl < 0 or hl == i else 0.45)
        if name == "⊕":
            f.ring(cx, y + h / 2, 22 * s, c, 3, aa)
            f.text((cx, y + h / 2 - 2), "+", int(36 * s), c, aa, bold=True)
        else:
            f.box([cx - w / 2, y, cx + w / 2, y + h], c, aa, 3, int(12 * s), fill=c, fa=0.12)
            f.text((cx, y + h / 2), name, int(30 * s), c, aa)
        centers.append(y + h / 2)
        y += h + 26 * s
    for i in range(len(rows) - 1):
        f.line([(cx, centers[i] + 28 * s), (cx, centers[i + 1] - 28 * s)], WHITE, 2, a * 0.6, 0.2)
    # 残差旁路
    for top, bot in ((ty - 18 * s, centers[2]), (centers[2] + 25 * s, centers[5])):
        xr = cx + w / 2 + 50 * s
        pts = [(cx, top), (xr, top), (xr, bot), (cx + 24 * s, bot)]
        f.line(pts, GREEN, 3, a * 0.9, 0.8)
        u = frac(t * 0.5)
        px, py = pl_at(pts, u)
        f.dot(px, py, 6 * s, GREEN, a)
    return y


def header(f, t, num, title, c=CYAN):
    a = ease(t / 0.5)
    x = 70 - (1 - a) * 60
    f.fillrect([x, 52, x + 8, 100], c, a, 1)
    f.text((x + 28, 76), num, 44, c, a, 0.8, anchor="lm", bold=True)
    f.text((x + 100, 76), title, 40, WHITE, a, 0.25, anchor="lm")


def captions(f, t, caps):
    for t0, t1, s in caps:
        a = win(t, t0, t1, 0.3, 0.3)
        if a <= 0:
            continue
        y = 990 + (1 - ease((t - t0) / 0.35)) * 24
        f.rich((W / 2, y), s, 46, a, acc=YELLOW)
        f.line([(W / 2 - 60 * a, 1036), (W / 2 + 60 * a, 1036)], YELLOW, 3, a * 0.8)


def flow(f, pts, t, n, c, speed=0.15, r=5, a=1, skip=None, fade_end=False):
    for i in range(n):
        u = frac(i / n + t * speed)
        x, y = pl_at(pts, u)
        if skip and skip[0] < x < skip[2] and skip[1] < y < skip[3]:
            continue
        aa = a * ((1 - u) if fade_end else 1)
        f.dot(x, y, r, c, aa, 1.0)


def pipe(f, pts, c=PIPE, a=1, w=16, core=None):
    f.line(pts, c, w, a, 0.2)
    if core:
        f.line(pts, core, 2, a * 0.5, 0.3)


def chevrons(f, x0, x1, y, t, c, a=1, n=7, speed=260, gain=None):
    """反向梯度：向左移动的 <<< ；gain(x) 返回该位置的强度"""
    span = x1 - x0
    for i in range(n):
        x = x1 - ((i * span / n + t * speed) % span)
        k = gain(x) if gain else 1
        aa = a * k
        f.line([(x + 14, y - 16), (x, y), (x + 14, y + 16)], c, 5, aa, 1.0)
        f.line([(x + 30, y - 16), (x + 16, y), (x + 30, y + 16)], c, 5, aa * 0.6, 1.0)


# ---------------------------------------------------------------------------
# 场景
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 3D：网格绘制、火焰、粒子、标注
# ---------------------------------------------------------------------------
_paths = {}


def path3(pts):
    k = tuple(map(tuple, pts))
    if k not in _paths:
        P = e3.tube_path(pts)
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        _paths[k] = (P, np.concatenate([[0], np.cumsum(seg)]))
    return _paths[k]


def at3(pts, u):
    P, cum = path3(pts)
    d = clamp(u) * cum[-1]
    i = min(int(np.searchsorted(cum, d, side="right") - 1), len(P) - 2)
    r = (d - cum[i]) / max(cum[i + 1] - cum[i], 1e-9)
    return P[i] + (P[i + 1] - P[i]) * r


def draw3d(f, mesh, cam, fire=0.0):
    im = e3.render(mesh, cam, W, H, 2, fire)
    f.S.paste(im, (0, 0), im)
    a = np.asarray(im, np.float32)
    m = a[..., 3] / 255
    f.M = m if f.M is None else np.maximum(f.M, m)
    hl = np.clip(a[..., :3] - 170, 0, 255) * 1.2 * m[..., None]
    g = Image.fromarray(hl.astype(np.uint8)).resize((W // 2, H // 2), Image.BILINEAR)
    f.G = ImageChops.add(f.G, g)
    f.dg = ImageDraw.Draw(f.G)


def dots3(f, cam, P, cols, rad, alphas, glow=1.0, streak=None):
    """P: (N,3)；streak: (N,3) 上一时刻位置，画运动拖尾"""
    if len(P) == 0:
        return
    x, y, z, s = cam.project(np.asarray(P, float))
    order = np.argsort(-z)
    if streak is not None:
        x0, y0, _, _ = cam.project(np.asarray(streak, float))
    for i in order:
        r = max(1.5, rad[i] * s[i] / 1000)
        if streak is not None:
            f.line([(x0[i], y0[i]), (x[i], y[i])], cols[i], max(2, int(r * 1.4)), alphas[i] * 0.7, glow)
        f.dot(x[i], y[i], r, cols[i], alphas[i], glow)


def flame3d(f, cam, t, a=1.0, L=4.2, n=220, y0=None, scale=1.0):
    ey = e3.exit_y() if y0 is None else y0
    er = e3.exit_r() * scale
    P, C, R, A = [], [], [], []
    for i in range(n):
        u = frac(rnd(i, 31) + t * (1.1 + 0.7 * rnd(i, 32)))
        ang = rnd(i, 33) * 6.283 + t
        rr = er * (0.9 - 0.45 * u) * math.sqrt(rnd(i, 34))
        P.append((rr * math.cos(ang), ey - u * L * scale, rr * math.sin(ang)))
        C.append(mix((255, 240, 190), ORANGE, u * 2.2) if u < 0.5 else mix(ORANGE, (190, 30, 30), (u - 0.5) * 2))
        R.append(40 * (1 - u) + 12)
        A.append(a * min(1, u * 8) * (1 - u) ** 1.2 * 0.6)
    # 火焰内核：一条渐细的发光锥
    for k in range(8):
        u0, u1 = k / 8, (k + 1) / 8
        rr0, rr1 = er * (0.6 - 0.45 * u0), er * (0.6 - 0.45 * u1)
        q = [(-rr0, ey - u0 * L * 0.8 * scale, 0), (rr0, ey - u0 * L * 0.8 * scale, 0),
             (rr1, ey - u1 * L * 0.8 * scale, 0), (-rr1, ey - u1 * L * 0.8 * scale, 0)]
        Rm = cam.R.T  # 让锥面始终正对相机
        qq = np.asarray(q) @ Rm.T * np.array([1, 0, 1]) + np.asarray(q) * np.array([0, 1, 0])
        x, y, _, _ = cam.project(qq)
        cc = mix((255, 230, 170), (255, 90, 30), u0)
        f.ds.polygon(list(zip(x, y)), fill=sc(cc, a * 0.1 * (1 - u0)))
        f.dg.polygon([(px / 2, py / 2) for px, py in zip(x, y)], fill=sc(cc, a * 0.35 * (1 - u0)))
    dots3(f, cam, P, C, R, A, 0.9)
    for k in range(5):  # 马赫环
        yk = ey - (0.45 + 0.75 * k) * scale
        rk = er * (0.55 - 0.08 * k)
        ring = [(rk * math.cos(q), yk, rk * math.sin(q)) for q in np.linspace(0, 6.283, 24)]
        x, y, _, _ = cam.project(np.asarray(ring))
        fl = (0.7 + 0.3 * math.sin(t * 37 + k)) * a * (1 - k / 5)
        pts = list(zip(x, y))
        f.ds.polygon(pts, fill=sc((255, 200, 235), 0.28 * fl))
        f.dg.polygon([(px / 2, py / 2) for px, py in pts], fill=sc((255, 110, 200), 0.55 * fl))


def flow3(f, cam, pts, t, n, c, speed=0.25, rad=22, a=1.0, glow=1.0):
    P = [at3(pts, frac(i / n + t * speed)) for i in range(n)]
    dots3(f, cam, P, [c] * n, [rad] * n, [a] * n, glow)


def label3(f, cam, p, txt, sub, tx, ty, c, a, align="r"):
    """左/右侧文字 + 指向 3D 点的引线"""
    if a <= 0.01:
        return
    x, y, _, _ = cam.project(np.asarray([p], float))
    px, py = float(x[0]), float(y[0])
    anchor = "rm" if align == "r" else "lm"
    ex = tx + 16 if align == "r" else tx - 16
    f.line([(ex, ty), (ex + (px - ex) * 0.25, ty), (px, py)], c, 2, a * 0.8, 0.5)
    f.dot(px, py, 6, c, a, 1.2)
    f.ring(px, py, 13 + 3 * math.sin(f_time[0] * 5), c, 2, a * 0.7)
    f.text((tx, ty - 6), txt, 40, c, a, 0.7, anchor=anchor[0] + "s", bold=True)
    if sub:
        f.text((tx, ty + 34), sub, 27, WHITE, a * 0.85, 0.1, anchor=anchor[0] + "s")


f_time = [0.0]


def cut_range(cam, open_ang=2.2):
    az = cam.azimuth()
    return az + open_ang / 2, az + 2 * math.pi - open_ang / 2


# ---------------------------------------------------------------------------
# A 段：发动机原理（3D）
# ---------------------------------------------------------------------------
def s_a_overview(f, t, d):
    f_time[0] = t
    header(f, t, "A1", "一台液体火箭发动机长什么样", ORANGE)
    cam = e3.Camera(0.35 + t * 0.22, 0.12, 9.5, (0, 0.75, 0), 1120, 1230, 420)
    draw3d(f, e3.full_engine(t * 9, heat=1.0), cam, fire=1.0)
    flame3d(f, cam, t)
    fa = ease((t - 4.5) / 0.6)
    flow3(f, cam, e3.FUEL_IN + e3.FUEL_DISCHARGE, t, 26, ORANGE, 0.3, 20, fa)
    flow3(f, cam, e3.LOX_IN + e3.LOX_DISCHARGE, t, 26, CYAN, 0.3, 20, fa)
    parts = [((1.15, 2.1, 0), "涡轮泵", "把推进剂加压到几百 bar", CYAN),
             ((0, 2.55, 0), "喷注器", "燃料和氧化剂在这里混合", PURPLE),
             ((0.6, 1.6, 0), "燃烧室", "3500 K 高温高压燃烧", ORANGE),
             ((0.26, 0.75, 0), "喉部", "最窄处：气流刚好到音速", YELLOW),
             ((0.95, -0.9, 0), "喷管", "热能 → 动能，高速喷出", GREEN)]
    for i, (p, s1, s2, c) in enumerate(parts):
        label3(f, cam, p, s1, s2, 560, 230 + i * 150, c, ease((t - 0.5 - i * 0.6) / 0.5))
    captions(f, t, [(0.3, 4.5, "液体火箭发动机 = 【涡轮泵】+【燃烧室】+【喷管】"),
                    (4.5, d, "贮箱 → 泵加压 → 喷注混合 → 燃烧 → 喷管加速喷出")])


def s_a_thrust(f, t, d):
    f_time[0] = t
    header(f, t, "A2", "推力从哪来", YELLOW)
    cam = e3.Camera(0.5 + 0.12 * math.sin(t * 0.5), 0.18, 7.2, (0, 0.45, 0), 1250, 600, 470)
    draw3d(f, e3.engine_body(48, cut_range(cam), 0.9), cam, fire=1.0)
    # 燃气：燃烧室里慢，过喉部后越来越快（拖尾越长）
    n = 140
    P, P0, C, R, A = [], [], [], [], []
    for i in range(n):
        u = frac(rnd(i, 41) + t * 0.32)
        u0 = max(0, u - 0.025)
        ang, rr = rnd(i, 42) * 6.283, math.sqrt(rnd(i, 43)) * 0.85
        for uu, lst in ((u, P), (u0, P0)):
            y = 2.3 - 3.9 * uu ** 2.2
            r = e3.radius_at(y) * rr
            lst.append((r * math.cos(ang), y, r * math.sin(ang)))
        C.append(mix((255, 245, 200), ORANGE, u * 1.3) if u < 0.75 else mix(ORANGE, RED, (u - 0.75) * 4))
        R.append(16)
        A.append(0.85)
    dots3(f, cam, P, C, R, A, 1.0, streak=P0)
    flame3d(f, cam, t, 0.8, 2.6, 90)
    stations = [((0.45, 1.7, 0), "燃烧室", "3500 K · 300 bar · 流速很慢", ORANGE, 0.6),
                ((0.26, 0.75, 0), "喉部", "马赫 1：流量由这里决定", YELLOW, 2.0),
                ((1.1, -1.45, 0), "出口", "≈ 3.3 km/s 超音速", CYAN, 3.4)]
    for i, (p, s1, s2, c, t0) in enumerate(stations):
        label3(f, cam, p, s1, s2, 1120, 240 + i * 150, c, ease((t - t0) / 0.5), align="l")
    # 沿轴向：压力 ↓ 速度 ↑
    ag = ease((t - 4.2) / 0.6)
    X0, X1, Y0, Y1 = 1120, 1780, 650, 840
    f.line([(X0, Y1), (X1, Y1)], GRAY, 2, ag, 0)
    f.text((X1, Y1 + 26), "沿喷管轴向 →", 24, GRAY, ag, 0, anchor="rm")
    xt = X0 + (X1 - X0) * 0.38
    f.dashed((xt, Y0), (xt, Y1), YELLOW, 2, ag)
    f.text((xt, Y0 - 18), "M=1", 26, YELLOW, ag, 0.5)
    up = clamp((t - 4.4) / 2.0)
    pc = [(X0 + (X1 - X0) * q, Y0 + 20 + (Y1 - Y0 - 30) * (1 / (1 + math.exp(-(q - 0.38) * 9)))) for q in np.linspace(0, 1, 50)]
    vc = [(X0 + (X1 - X0) * q, Y1 - 10 - (Y1 - Y0 - 30) * (1 / (1 + math.exp(-(q - 0.38) * 7)))) for q in np.linspace(0, 1, 50)]
    f.line(pl_part(pc, up), RED, 4, ag)
    f.line(pl_part(vc, up), CYAN, 4, ag)
    f.text((X0 + 10, Y0 + 2), "压力", 26, RED, ag * up, 0.4, anchor="lm")
    f.text((X0 + 10, Y1 - 40), "速度", 26, CYAN, ag * up, 0.4, anchor="lm")
    ae = ease((t - 6.8) / 0.5)
    f.box([1100, 880, 1800, 945], YELLOW, ae, 3, 16, fill=YELLOW, fa=0.08)
    f.text((1450, 912), "F = (dm/dt)·ve + (pe − pa)·Ae", 36, YELLOW, ae, 0.6, mono=True)
    captions(f, t, [(0.3, 3.5, "燃烧把化学能变成【高温高压】的燃气"),
                    (3.5, 6.8, "收敛段加速 → 喉部【音速】→ 扩张段继续加速到【超音速】"),
                    (6.8, d, "每秒喷出几百公斤 3 km/s 的燃气，反作用力就是【推力】")])


def s_a_pump(f, t, d):
    f_time[0] = t
    header(f, t, "A3", "涡轮泵：发动机的心脏", CYAN)
    spin = t * 5
    cam = e3.Camera(0.22 + 0.05 * t, 0.38, 5.4, (0, 0, 0), 1150, 640, 520)
    draw3d(f, e3.turbopump_closeup(spin), cam, fire=0.0)
    for xc, c in ((1.1, ORANGE), (-1.1, CYAN)):  # 全息外壳
        for rr in (0.98, 1.06):
            ring = [(xc, rr * math.sin(q), rr * math.cos(q)) for q in np.linspace(0, 6.283, 48)]
            x, y, _, _ = cam.project(np.asarray(ring))
            f.line(list(zip(x, y)), c, 2, 0.5, 0.6)
    # 燃气吹动涡轮叶片；推进剂被叶轮甩出去
    P, C, A = [], [], []
    for i in range(70):
        u = frac(rnd(i, 51) + t * 0.6)
        q = spin * 0.3 + rnd(i, 52) * 6.283 + u * 2.5
        P.append((1.75 - u * 1.3, -0.82 * math.sin(q), 0.82 * math.cos(q)))
        C.append(mix(YELLOW, RED, u))
        A.append(0.9 * (1 - u * 0.6))
    for i in range(70):
        u = frac(rnd(i, 53) + t * 0.5)
        q = -spin * 0.4 + rnd(i, 54) * 6.283 + max(0, u - 0.45) * 5
        if u < 0.45:
            x, r = -2.1 + u / 0.45 * 0.9, 0.12 + 0.05 * rnd(i, 55)
        else:
            x, r = -1.1, 0.15 + (u - 0.45) / 0.55 * 0.9
        P.append((x, -r * math.sin(q), r * math.cos(q)))
        C.append(mix(CYAN, WHITE, (u - 0.45) * 1.5))
        A.append(0.9)
    dots3(f, cam, P, C, [16] * len(P), A, 1.0)
    label3(f, cam, (1.1, 0.0, 0.95), "涡轮", "燃气吹动叶片 → 带动轴", 1240, 250, ORANGE, ease((t - 0.5) / 0.5), "l")
    label3(f, cam, (-1.1, 0.0, 0.8), "泵叶轮", "把推进剂甩出去、加压", 1240, 410, CYAN, ease((t - 1.2) / 0.5), "l")
    label3(f, cam, (0.0, 0.0, 0.0), "同一根轴", "转速 7 万转/分", 1240, 570, WHITE, ease((t - 1.9) / 0.5), "l")
    # 自举环
    ar = ease((t - 5.6) / 0.6)
    cx, cy, R = 1500, 800, 110
    nodes = ["泵加压", "烧出燃气", "推涡轮", "带动泵"]
    for k, s in enumerate(nodes):
        q = -math.pi / 2 + k * math.pi / 2
        f.text((cx + R * 1.45 * math.cos(q), cy + R * 0.9 * math.sin(q)), s, 28, YELLOW, ar, 0.5)
    arc = [(cx + R * math.cos(q), cy + R * 0.6 * math.sin(q)) for q in np.linspace(0, 6.0, 40)]
    f.line(arc, YELLOW, 3, ar * 0.7, 0.6)
    qd = t * 2.5
    f.dot(cx + R * math.cos(qd), cy + R * 0.6 * math.sin(qd), 8, YELLOW, ar, 1.4)
    captions(f, t, [(0.3, 3.0, "要把推进剂顶进 300 bar 的燃烧室，得靠【涡轮泵】"),
                    (3.0, 5.6, "RS-25 的燃料泵：出口 480 bar，几万千瓦，【比火车头还大】"),
                    (5.6, d, "可功率只能来自推进剂自己：这是一个【自举】难题")])


def s_a_cycles(f, t, d):
    f_time[0] = t
    closed = t >= 5.5
    header(f, t, "A4", "涡轮的废气往哪去？", MAGENTA)
    cam = e3.Camera(0.95 + 0.1 * t, 0.12, 9.5, (0.3, 0.75, 0), 1120, 1300, 420)
    draw3d(f, e3.full_engine(t * 9, heat=1.0, gg="closed" if closed else "open"), cam, fire=1.0)
    flame3d(f, cam, t)
    flow3(f, cam, e3.FUEL_IN + e3.FUEL_DISCHARGE, t, 22, ORANGE, 0.3, 20)
    flow3(f, cam, e3.LOX_IN + e3.LOX_DISCHARGE, t, 22, CYAN, 0.3, 20)
    if not closed:
        flow3(f, cam, e3.GG_EXHAUST, t, 14, (200, 200, 200), 0.35, 22)
        P, C, R, A = [], [], [], []
        for i in range(40):
            u = frac(rnd(i, 61) + t * 0.35)
            P.append((1.85 + u * 1.4 + (rnd(i, 62) - 0.5) * u, -1.5 - u * 1.2 + (rnd(i, 63) - 0.5) * u, 0.35 + (rnd(i, 64) - 0.5) * u))
            C.append((90, 90, 100))
            R.append(30 + 120 * u)
            A.append((1 - u) * 0.5)
        dots3(f, cam, P, C, R, A, 0.3)
    else:
        flow3(f, cam, e3.HOT_GAS, t, 14, RED, 0.5, 24)
    a1 = win(t, 0.4, 5.5, 0.5, 0.3)
    a2 = ease((t - 5.7) / 0.5)
    for a, title, c, rows in ((a1, "燃气发生器循环（开式）", ORANGE,
                               ["抽 2–5% 推进剂烧一小炉，推涡轮", "推完涡轮的废气【直接排到舷外】",
                                "比冲损失 2–3%：F-1、Merlin", "≈ 【detach()】"]),
                              (a2, "分级燃烧循环（闭式）", GREEN,
                               ["预燃室故意不烧完（~900 K）", "涡轮废气【接回主燃烧室】再烧",
                                "一分推进剂都不丢：RS-25、猛禽", "≈ 【残差连接 x + f(x)】"])):
        if a <= 0:
            continue
        f.text((110, 230), title, 52, c, a, 0.8, anchor="lm", bold=True)
        for i, r in enumerate(rows):
            aa = a * ease((t - (0.8 if c == ORANGE else 6.0) - i * 0.5) / 0.4)
            f.dot(126, 340 + i * 105, 7, c, aa)
            f.rich((150, 340 + i * 105), r, 38, aa, acc=c, align="l")
    captions(f, t, [(0.3, 5.5, "开式：推完涡轮的废气【排到舷外】 —— 土星五号侧面那股黑烟"),
                    (5.5, d, "闭式：废气【接回主燃烧室】再烧一次 —— 一点都不浪费")])


def s_a_regen(f, t, d):
    f_time[0] = t
    header(f, t, "A5", "再生冷却：废热用两次", BLUE)
    cam = e3.Camera(-0.4 + 0.1 * t, 0.2, 7.0, (0, 0.45, 0), 1250, 640, 470)
    a0, a1 = cut_range(cam, 2.4)
    draw3d(f, e3.engine_body(48, (a0, a1), 1.0), cam, fire=1.0)
    # 冷却剂在壁内由下往上流（剖面 + 透视）
    P, C, R, A = [], [], [], []
    for side, ang in enumerate((a0, a1)):
        for i in range(36):
            u = frac(rnd(i + side * 50, 71) + t * 0.22)
            y = -1.55 + u * 3.6
            r = e3.radius_at(y) - 0.035
            P.append((r * math.cos(ang), y, r * math.sin(ang)))
            C.append(mix(BLUE, (255, 230, 200), u))
            R.append(14)
            A.append(1.0)
    dots3(f, cam, P, C, R, A, 1.0)
    for i in range(70):  # 内部燃气
        u = frac(rnd(i, 72) + t * 0.32)
        y = 2.3 - 3.9 * u ** 2.2
        ang, rr = rnd(i, 73) * 6.283, math.sqrt(rnd(i, 74)) * 0.8
        r = e3.radius_at(y) * rr
        f_p = [(r * math.cos(ang), y, r * math.sin(ang))]
        dots3(f, cam, f_p, [mix((255, 240, 190), ORANGE, u)], [14], [0.6], 0.8)
    flame3d(f, cam, t, 0.7, 2.4, 80)
    label3(f, cam, (0.0, 1.6, 0.0), "燃气 3500 K", "比钢的熔点还高一倍", 1180, 260, ORANGE, ease((t - 0.4) / 0.5), "l")
    p_wall = (e3.radius_at(0.0) * math.cos(a1), 0.0, e3.radius_at(0.0) * math.sin(a1))
    label3(f, cam, p_wall, "燃料在壁内逆流而上", "带走热量，壁面保持 < 800 K", 1180, 430, BLUE, ease((t - 1.4) / 0.5), "l")
    label3(f, cam, (0.0, 2.55, 0.0), "预热后进喷注器", "废热没有浪费，回到燃烧室", 1180, 600, GREEN, ease((t - 2.6) / 0.5), "l")
    ak = ease((t - 4.5) / 0.5)
    f.box([1160, 720, 1800, 840], CYAN, ak, 3, 18, fill=CYAN, fa=0.08)
    f.rich((1480, 780), "≈ 【KV Cache】：付过的成本用两次", 38, ak, acc=CYAN)
    captions(f, t, [(0.3, 4.0, "燃气 3500 K，壁面扛不住：让【燃料先流过壁面】带走热量"),
                    (4.0, d, "废热被回收、燃料被预热 —— 已经付过的成本，【用第二次】")])


def s_cover(f, t, d):
    f_time[0] = t
    cam = e3.Camera(0.5 + 0.12 * t, 0.12, 9.0, (0, 0.6, 0), 1050, 1430, 430)
    draw3d(f, e3.full_engine(t * 9, heat=1.0), cam, fire=1.0)
    flame3d(f, cam, t)
    a = ease(t / 0.6)
    f.box([140, 175, 520, 240], MAGENTA, a, 3, 30, fill=MAGENTA, fa=0.15)
    f.text((330, 207), "硬核类比 · AI 原理", 32, MAGENTA, a, 0.6)
    x1 = 140 - (1 - ease((t - 0.2) / 0.6)) * 200
    f.text((x1, 400), "火箭发动机", 140, ORANGE, ease((t - 0.2) / 0.6), 0.9, anchor="ls", bold=True)
    x2 = 140 - (1 - ease((t - 0.5) / 0.6)) * 200
    f.text((x2, 560), "= Transformer ?", 118, CYAN, ease((t - 0.5) / 0.6), 0.9, anchor="ls", bold=True)
    f.rich((140, 655), "SpaceX【猛禽】和【GPT】，用的是同一张图", 48, ease((t - 0.9) / 0.6), align="l", acc=YELLOW)
    chips = [("残差连接", GREEN), ("detach()", ORANGE), ("因果掩码", CYAN), ("warmup", PURPLE)]
    x = 140
    for i, (s, c) in enumerate(chips):
        aa = ease((t - 1.2 - i * 0.15) / 0.4)
        wdt = font(34).getlength(s) + 48
        f.box([x, 730, x + wdt, 790], c, aa, 3, 30, fill=c, fa=0.12)
        f.text((x + wdt / 2, 760), s, 34, c, aa, 0.6)
        x += wdt + 22
    f.text((140, 880), "看懂一台发动机，就看懂了残差、注意力和训练稳定性", 34, GRAY, ease((t - 1.8) / 0.6), 0.1, anchor="lm")


def s_same(f, t, d):
    header(f, t, "00", "同一道题，两张图")
    a1 = ease((t - 0.2) / 0.6)
    f.text((480, 170), "火箭发动机", 46, ORANGE, a1, 0.7, bold=True)
    cam = e3.Camera(0.4 + 0.2 * t, 0.12, 9.5, (0, 0.2, 0), 820 * (0.6 + 0.4 * a1), 480, 520)
    draw3d(f, e3.full_engine(t * 9, heat=1.0), cam, fire=1.0)
    flame3d(f, cam, t, a1, 3.0, 120)
    a2 = ease((t - 0.5) / 0.6)
    f.text((1440, 170), "Transformer Block", 46, CYAN, a2, 0.7, bold=True)
    tblock(f, 1400, 250, 0.95, t, a2)
    for i, (s, c) in enumerate((("分流", BLUE), ("做功", ORANGE), ("汇流", GREEN))):
        aa = ease((t - 1.0 - i * 0.5) / 0.4)
        y = 380 + i * 170
        f.box([860, y - 50, 1060, y + 50], c, aa, 3, 50, fill=c, fa=0.15)
        f.text((960, y), s, 50, c, aa, 0.8, bold=True)
        if i < 2:
            f.line([(960, y + 58), (960, y + 112)], WHITE, 3, aa * 0.6, 0.3)
            f.line([(948, y + 100), (960, y + 112), (972, y + 100)], WHITE, 3, aa * 0.6, 0.3)
    captions(f, t, [(0.3, 3.0, "一台【闭式循环火箭发动机】和一个【Transformer block】"),
                    (3.0, d, "解的是同一道题：【分流 → 做功 → 原样汇回主流】")])


MAIN = [(200, 560), (1720, 560)]
BR = [(560, 560), (660, 400), (740, 330), (1180, 330), (1260, 400), (1360, 560)]
BOX = (760, 270, 1160, 390)


def s_cycle(f, t, d):
    header(f, t, "01", "cycle ≠ loop", CYAN)
    a = ease((t - 0.2) / 0.6)
    pipe(f, MAIN, a=a)
    pipe(f, BR, a=a)
    flow(f, MAIN, t, 26, CYAN, 0.07, 6, a)
    flow(f, BR, t, 14, ORANGE, 0.1, 6, a, skip=BOX)
    m = clamp((t - 4.0) / 0.8)
    f.box(BOX, mix(ORANGE, CYAN, m), a, 3, 18, fill=(40, 30, 10), fa=1)
    pulse = 0.5 + 0.5 * math.sin(t * 6)
    f.dot(960, 330, 30 + 6 * pulse, ORANGE, a * 0.25, 1.5)
    f.text((960, 330), "预燃室 → 涡轮", 44, ORANGE, a * (1 - m), 0.6)
    f.text((960, 330), "f(x) = attn / mlp", 44, CYAN, a * m, 0.6, mono=True)
    f.ring(1360, 560, 30, WHITE, 4, a)
    f.text((1360, 557), "+", 44, WHITE, a, bold=True)
    for (x, y, s1, s2) in ((200, 620, "推进剂", "x"), (1720, 620, "主燃烧室", "x + f(x)"), (960, 620, "主流：直通", "恒等映射 x")):
        f.text((x, y), s1, 34, GRAY, a * (1 - m), 0.2)
        f.text((x, y), s2, 34, CYAN, a * m, 0.4, mono=True)
    ac = ease((t - 6.2) / 0.5)
    f.box([610, 720, 1310, 830], GREEN, ac, 3, 18, fill=GREEN, fa=0.08)
    f.text((960, 775), "x = x + f(x)", 60, GREEN, ac, 0.8, mono=True)
    da = win(t, 3.2, 6.4, 0.4, 0.4)
    f.box([1440, 200, 1800, 300], GREEN, da, 3, 18)
    f.text((1620, 250), "DAG ✓   loop ×", 40, GREEN, da, 0.6)
    captions(f, t, [(0.3, 3.2, "误会：发动机的“循环”(cycle) ≠ 计算图里的【loop】"),
                    (3.2, 6.2, "稳态下它是一张【DAG】：分流 → 做功 → 汇流"),
                    (6.2, d, "这正是【残差块】的形状")])


def s_open(f, t, d):
    header(f, t, "02", "开式循环 = detach()", ORANGE)
    a = ease((t - 0.2) / 0.6)
    br = [(560, 560), (660, 400), (740, 330), (1180, 330), (1300, 240), (1420, 150), (1540, 50)]
    pipe(f, MAIN, a=a)
    pipe(f, br, a=a)
    flow(f, MAIN, t, 26, CYAN, 0.07, 6, a)
    flow(f, br, t, 16, ORANGE, 0.1, 6, a, skip=BOX)
    for i in range(30):  # 舷外黑烟
        u = frac(rnd(i, 4) + t * 0.25)
        x = 1420 + u * 260 + (rnd(i, 5) - 0.5) * 120 * u
        y = 150 - u * 200 + (rnd(i, 6) - 0.5) * 100 * u
        f.dot(x, y, 8 + 30 * u, (110, 110, 120), a * (1 - u) * 0.35, 0.6)
    f.box(BOX, ORANGE, a, 3, 18, fill=(40, 30, 10), fa=1)
    f.text((960, 330), "燃气发生器 → 涡轮", 42, ORANGE, a, 0.6)
    ax = ease((t - 1.2) / 0.4)
    f.dashed((1260, 400), (1360, 560), GRAY, 3, ax * 0.6)
    f.line([(1290, 450), (1340, 500)], RED, 6, ax)
    f.line([(1340, 450), (1290, 500)], RED, 6, ax)
    f.text((1660, 120), "比冲 −2~3%", 56, RED, ease((t - 1.5) / 0.5), 0.8, bold=True)
    f.text((1660, 185), "废气排到舷外", 30, GRAY, ease((t - 1.5) / 0.5), 0.1)
    ab = ease((t - 3.0) / 0.5)
    chevrons(f, 600, 1720, 560, t, MAGENTA, ab, n=6, speed=200)
    f.text((1780, 560), "∂L", 34, MAGENTA, ab, 0.6, anchor="lm")
    f.ring(610, 470, 34, RED, 4, ab)
    f.text((610, 468), "断", 36, RED, ab, 0.8, bold=True)
    f.text((380, 470), "反向压力链断在这里 →", 30, RED, ab, 0.3)
    ac = ease((t - 4.0) / 0.5)
    f.box([200, 700, 900, 800], GREEN, ac, 3, 18, fill=GREEN, fa=0.08)
    f.text((550, 750), "h = branch(x.detach())", 42, GREEN, ac, 0.7, mono=True)
    items = [("DQN 目标网络", CYAN), ("VQ-VAE 直通估计", PURPLE), ("EMA teacher", YELLOW), ("梯度裁剪", ORANGE)]
    for i, (s, c) in enumerate(items):
        aa = ease((t - 5.0 - i * 0.3) / 0.4)
        x = 1000 + (i % 2) * 380
        y = 720 + (i // 2) * 80
        f.box([x, y - 32, x + 350, y + 32], c, aa, 2, 30, fill=c, fa=0.1)
        f.text((x + 175, y), s, 32, c, aa, 0.5)
    captions(f, t, [(0.3, 3.0, "燃气发生器循环：涡轮废气直接【排到舷外】"),
                    (3.0, 6.0, "反向的压力链断在这里 —— 这就是【detach()】"),
                    (6.0, d, "拿效率换稳定：【没把握驯服闭环时，先 detach 它】")])


def s_preln(f, t, d):
    header(f, t, "03", "闭式循环 = 残差连接：主管路上什么都别串", GREEN)
    xs = [640, 860, 1080, 1300, 1520]
    for li, (y, name, form, post) in enumerate(((360, "Post-LN", "x = LN(x + f(x))", True),
                                                 (720, "Pre-LN", "x = x + f(LN(x))", False))):
        a = ease((t - 0.3 - li * 0.6) / 0.6)
        c = RED if post else GREEN
        f.text((150, y - 22), name, 52, c, a, 0.8, anchor="lm", bold=True)
        f.text((150, y + 36), form, 28, GRAY, a, 0.1, anchor="lm", mono=True)
        pipe(f, [(520, y), (1760, y)], a=a, w=14)
        for x in xs:
            arc = [(x, y), (x + 30, y - 80), (x + 130, y - 80), (x + 160, y)]
            f.line(arc, PIPE, 6, a, 0.2)
            f.box([x + 40, y - 112, x + 120, y - 50], ORANGE, a, 2, 10, fill=ORANGE, fa=0.12)
            f.text((x + 80, y - 81), "f", 32, ORANGE, a, 0.5, mono=True)
            if post:
                f.box([x + 172, y - 24, x + 214, y + 24], RED, a, 3, 6, fill=RED, fa=0.35)
                f.text((x + 193, y), "LN", 20, WHITE, a, 0.3)
            else:
                f.box([x - 2, y - 72, x + 40, y - 40], GREEN, a, 2, 6, fill=GREEN, fa=0.3)
                f.text((x + 19, y - 56), "LN", 18, WHITE, a, 0.3)
        ab = ease((t - 2.5) / 0.6) * a

        def gain(x, post=post):
            if not post:
                return 1
            k = sum(1 for v in xs if v + 193 > x)
            return 0.55 ** k

        chevrons(f, 520, 1760, y, t, MAGENTA, ab, n=8, speed=230, gain=gain)
        # 到达第 1 层的梯度
        g = (0.55 ** 5 if post else 1) * ease((t - 3.5) / 1.0)
        hbar = 200 * g
        f.fillrect([440, y + 60 - hbar, 480, y + 60], c, a, 0.8)
        f.box([436, y - 144, 484, y + 64], GRAY, a * 0.6, 2, 6, g=0)
    ab = ease((t - 3.5) / 0.6)
    f.text((460, 460), "↑ 到达第 1 层的梯度", 26, GRAY, ab, 0.1, anchor="lm")
    f.text((1660, 200), "越深越衰减 ×", 34, RED, ease((t - 4.5) / 0.5), 0.6)
    f.text((1660, 560), "一路无阻 ✓", 34, GREEN, ease((t - 6.8) / 0.5), 0.6)
    captions(f, t, [(0.3, 3.3, "闭式循环：涡轮排气汇回主室 = 【残差连接】"),
                    (3.3, 6.6, "Post-LN 把阀门串在主管路上：越深，【压力传不到顶】"),
                    (6.6, d, "Pre-LN 只装在支路：一条【无阻的高压直通道】")])


def s_fwdbwd(f, t, d):
    header(f, t, "04", "前向是质量流，反向是压力链", MAGENTA)
    names = ["泵", "预燃室", "涡轮", "喷注器", "主燃烧室", "喷管"]
    xs = [300 + i * 264 for i in range(6)]
    a = ease((t - 0.2) / 0.6)
    pipe(f, [(220, 460), (1700, 460)], a=a, w=90, c=(10, 25, 40))
    for i, (x, n) in enumerate(zip(xs, names)):
        aa = ease((t - 0.2 - i * 0.12) / 0.4)
        f.box([x - 95, 395, x + 95, 525], WHITE, aa * 0.8, 2, 16, fill=(30, 50, 80), fa=1, g=0.3)
        f.text((x, 445), n, 38, WHITE, aa, 0.2)
        f.text((x, 495), f"layer {i + 1}", 24, CYAN, aa, 0.3, mono=True)
    af = ease((t - 0.8) / 0.5)
    flow(f, [(220, 410), (1700, 410)], t, 30, CYAN, 0.08, 5, af)
    f.text((960, 330), "前向：质量流 / 激活  →→→", 40, CYAN, af, 0.7)
    ab = ease((t - 1.6) / 0.5)
    chevrons(f, 220, 1700, 560, t, MAGENTA, ab, n=9, speed=240)
    f.text((960, 620), "←←←  反向：压力需求 / 梯度", 40, MAGENTA, ab, 0.7)
    for i, (s, c, sz) in enumerate((("x[l+1] = x[l] + f(x[l])", CYAN, 40),
                                    ("∂L/∂x[l] = ∂L/∂x[l+1] · (I + ∂f/∂x[l])", MAGENTA, 40),
                                    ("P泵出口 = Pc + ΔP喷注 + ΔP涡轮 + ΔP冷却 + …", GRAY, 32))):
        aa = ease((t - 3.0 - i * 0.6) / 0.5)
        f.text((960, 720 + i * 70), s, sz, c, aa, 0.5, mono=True)
    captions(f, t, [(0.3, 4.0, "前向传【流量】，反向传【需求】，共用同一张图"),
                    (4.0, d, "管网叫 Hardy Cross，电路叫基尔霍夫，网络里叫【反向传播】")])


def s_staged(f, t, d):
    header(f, t, "05", "为什么必须“分级”", ORANGE)
    base, top = 860, 260
    ky = lambda K: base - K / 3600 * (base - top)
    lim = ky(1100)
    bars = [("一次烧完", 3500, RED, 0.3), ("预燃室·富燃", 900, GREEN, 3.0), ("主燃烧室", 3500, ORANGE, 4.5)]
    for i, (name, K, c, t0) in enumerate(bars):
        x = 300 + i * 230
        u = ease((t - t0) / 1.2)
        f.box([x - 55, top - 10, x + 55, base + 6], GRAY, 0.5, 2, 12, g=0)
        f.fillrect([x - 48, ky(K * u), x + 48, base], c, 0.6 * clamp((t - t0) * 3), 0.25)
        f.text((x, ky(K * u) - 30), f"{int(K * u)} K", 34, c, clamp((t - t0) * 3), 0.6, bold=True)
        f.text((x, base + 44), name, 30, WHITE, clamp((t - t0) * 3), 0.2)
    f.dashed((220, lim), (820, lim), YELLOW, 3, ease(t / 0.6))
    f.text((830, lim), "← 涡轮叶片极限 ~1100K", 28, YELLOW, ease(t / 0.6), 0.5, anchor="lm")
    boom = win(t, 1.4, 3.0, 0.2, 0.4)
    f.text((300, 210), "涡轮气化！", 38, RED, boom * (0.6 + 0.4 * math.sin(t * 30)), 1.0, bold=True)
    f.text((760, 210), "无转动件，扛得住", 28, ORANGE, ease((t - 5.5) / 0.4), 0.4)
    # 右侧：残差流逐层小步累积
    ar = ease((t - 5.8) / 0.5)
    f.text((1440, 300), "残差流：每层只写一点点", 38, CYAN, ar, 0.6)
    n = 12
    shown = int(clamp((t - 6.0) / 2.2) * n)
    x = 1130
    for i in range(n):
        if i < shown:
            f.line([(x, 400), (x + 46, 400)], CYAN, 10, ar, 0.9)
            f.dot(x + 46, 400, 7, WHITE, ar)
        x += 50
    f.text((1440, 450), "‖Δx‖ ≪ ‖x‖", 34, WHITE, ar, 0.3, mono=True)
    ac = ease((t - 6.6) / 0.5)
    f.box([1090, 540, 1790, 640], GREEN, ac, 3, 18, fill=GREEN, fa=0.08)
    f.text((1440, 590), "std = 0.02 / √(2·n_layer)", 40, GREEN, ac, 0.7, mono=True)
    f.text((1440, 770), "ReZero · LayerScale · DeepNorm = 调预燃室混合比", 28, GRAY, ac, 0.1)
    captions(f, t, [(0.3, 3.0, "主燃烧室 3500K，涡轮叶片只扛得住 ~1000K"),
                    (3.0, 6.0, "预燃室【故意不烧完】：富燃降温，能量留到主室"),
                    (6.0, d, "= 每层残差【只做一点点更新】")])


def s_injector(f, t, d):
    header(f, t, "06", "喷注器 = Attention", CYAN)
    cx, cy, R = 700, 520, 300
    a = ease((t - 0.2) / 0.6)
    f.ring(cx, cy, R, (150, 200, 255), 4, a)
    f.ring(cx, cy, R - 14, (150, 200, 255), 1, a * 0.4, 0)
    pts = []
    for i in range(16):
        ang = rnd(i, 7) * 6.283 if i else 0
        rr = (0.25 + 0.65 * rnd(i, 8)) * R if i else 0
        pts.append((cx + rr * math.cos(ang), cy + rr * math.sin(ang), math.atan2(math.sin(ang), math.cos(ang)), rr))
    pb = ease((t - 6.0) / 0.8)
    rot = (1 - pb) * 1.2
    sector = lambda p: int(((p[2] - rot * 0 + math.pi) / (2 * math.pi)) * 4) % 4
    heads = [CYAN, MAGENTA, YELLOW, GREEN]
    ma = win(t, 0.5, 1e9)
    for i in range(16):
        for j in range(i + 1, 16):
            same = sector(pts[i]) == sector(pts[j])
            w = 0.5 + 0.5 * math.sin(t * 2.5 + i * 1.7 + j * 0.9)
            aa = ma * a * w * 0.55 * (1 if same else (1 - pb))
            c = heads[sector(pts[i])] if (pb > 0 and same) else CYAN
            f.line([pts[i][:2], pts[j][:2]], c, 2, aa, 0.6)
    fb = ease((t - 3.0) / 0.6)
    for i, p in enumerate(pts):
        c = mix(WHITE, heads[sector(p)], pb)
        f.dot(p[0], p[1], 10, c, a)
        if fb > 0:
            for k in range(5):
                u = frac(rnd(i * 5 + k, 9) + t * 1.5)
                f.dot(p[0] + (rnd(i * 5 + k, 10) - 0.5) * 14, p[1] - 14 - u * 50, 7 * (1 - u) + 2,
                      mix(YELLOW, RED, u), fb * (1 - u) * 0.8, 1.2)
    for k in range(4):
        ang = k * math.pi / 2 - math.pi + rot
        f.line([(cx, cy), (cx + R * math.cos(ang), cy + R * math.sin(ang))], PURPLE, 6, pb, 1.0)
    rows = [("喷注器 → Attention", "唯一让不同位置相互作用的地方", CYAN, 0.6),
            ("燃烧室 → FFN", "每个位置独立释放能量 · 参数占 2/3", ORANGE, 3.2),
            ("隔板 → 多头注意力", "F-1 炸弹测试：45ms 内恢复稳定", PURPLE, 6.2)]
    for i, (s1, s2, c, t0) in enumerate(rows):
        aa = ease((t - t0) / 0.5)
        y = 320 + i * 200
        x = 1150 + (1 - aa) * 60
        f.fillrect([x, y - 40, x + 8, y + 60], c, aa, 1)
        f.text((x + 30, y), s1, 48, c, aa, 0.7, anchor="lm", bold=True)
        f.text((x + 30, y + 52), s2, 30, GRAY, aa, 0.1, anchor="lm")
    captions(f, t, [(0.3, 3.0, "喷注器负责【混合】—— 这就是 Attention"),
                    (3.0, 6.0, "燃烧室负责【反应】—— 这就是逐位置的 FFN"),
                    (6.0, d, "F-1 靠【隔板】打碎燃烧共振 —— 多头也是在切腔体")])


def s_throat(f, t, d):
    header(f, t, "07", "喉部 = 因果掩码", YELLOW)
    cy = 470
    x0, xt, x1 = 180, 620, 1240

    def r_of(x):
        if x < 400:
            return 170
        if x < xt:
            u = (x - 400) / (xt - 400)
            return 60 + 110 * (1 + math.cos(math.pi * u)) / 2
        return 60 + 170 * ((x - xt) / (x1 - xt)) ** 0.7

    a = ease((t - 0.2) / 0.6)
    xs = list(range(x0, x1 + 1, 10))
    top = [(x, cy - r_of(x)) for x in xs]
    bot = [(x, cy + r_of(x)) for x in xs]
    f.line(top, (150, 200, 255), 5, a)
    f.line(bot, (150, 200, 255), 5, a)
    for i in range(90):
        u = frac(rnd(i, 11) + t * 0.35)
        x = x0 + (x1 - x0) * u ** 1.8
        y = cy + (rnd(i, 12) - 0.5) * 1.7 * r_of(x)
        sup = x > xt
        c = mix(BLUE, YELLOW, (x - xt) / 300) if sup else BLUE
        f.dot(x, y, 4 + 3 * sup, c, a * 0.9, 1.0)
    f.dashed((xt, cy - 120), (xt, cy + 120), YELLOW, 3, a)
    f.text((xt, cy - 150), "M = 1", 38, YELLOW, a, 0.8, bold=True)
    f.text((380, 720), "M < 1  扰动可以逆行 ⇄", 32, BLUE, a, 0.4)
    f.text((960, 720), "M > 1  扰动无法逆行 →", 32, ORANGE, a, 0.4)
    # 扰动波：亚音速区向两边扩散；超音速区被冲向下游
    for k in range(3):
        tt = (t - 1.5 - k * 0.7) % 2.1
        if t < 1.5:
            break
        r = tt * 90
        f.ring(330, cy, r, BLUE, 2, (1 - tt / 2.1) * a * 0.9, 0.6)
        f.ring(1000 + tt * 160, cy, r * 0.8, ORANGE, 2, (1 - tt / 2.1) * a * 0.9, 0.6)
    f.text((1000, cy - 235), "点一根火柴：燃烧室完全不知道", 26, GRAY, ease((t - 2) / 0.5), 0.1)
    # 因果掩码矩阵
    am = ease((t - 4.0) / 0.6)
    N, cs = 8, 54
    mx, my = 1370, 220
    f.text((mx + N * cs / 2, my - 40), "causal mask", 34, CYAN, am, 0.6, mono=True)
    for i in range(N):
        for j in range(N):
            b = [mx + j * cs + 3, my + i * cs + 3, mx + (j + 1) * cs - 3, my + (i + 1) * cs - 3]
            if j <= i:
                u = clamp((t - 4.3 - (i + j) * 0.08) / 0.3)
                c = YELLOW if i == j else CYAN
                f.fillrect(b, c, am * u * (0.85 if i == j else 0.45), 0.7)
            else:
                f.box(b, RED, am * 0.35, 1, 4, g=0)
                f.text(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2), "-∞", 16, RED, am * 0.5, 0)
    f.text((mx + N * cs / 2, my + N * cs + 40), "位置 t 只看 ≤ t", 34, WHITE, am, 0.3)
    f.text((mx + N * cs / 2, my + N * cs + 90), "对角线 = 喉部", 28, YELLOW, am, 0.5)
    captions(f, t, [(0.3, 3.3, "喉部之后是超音速：【下游扰动传不回上游】"),
                    (3.3, 6.6, "因果掩码是同一个数学对象：【依赖域】"),
                    (6.6, d, "所以 K、V 永远不会被未来改写 —— 【KV Cache 才合法】")])


def s_raptor(f, t, d):
    header(f, t, "08", "全流量分级燃烧：猛禽", ORANGE)
    a = ease((t - 0.2) / 0.6)
    top = [(160, 330), (450, 330), (640, 330), (700, 330), (860, 330), (920, 460), (980, 500)]
    bot = [(160, 690), (450, 690), (640, 690), (700, 690), (860, 690), (920, 560), (980, 520)]
    pipe(f, top, a=a, w=14)
    pipe(f, bot, a=a, w=14)
    flow(f, top, t, 20, BLUE, 0.12, 6, a)
    flow(f, bot, t, 20, ORANGE, 0.12, 6, a)
    f.text((160, 280), "LOX", 36, BLUE, a, 0.6, anchor="lm", bold=True)
    f.text((160, 750), "CH4", 36, ORANGE, a, 0.6, anchor="lm", bold=True)
    for (b, s, c) in (((440, 280, 600, 380), "富氧预燃室", BLUE), ((660, 280, 860, 380), "氧涡轮泵", BLUE),
                      ((440, 640, 600, 740), "富燃预燃室", ORANGE), ((660, 640, 860, 740), "燃涡轮泵", ORANGE)):
        f.box(b, c, a, 3, 14, fill=(15, 25, 45), fa=1)
        f.text(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2), s, 28, c, a, 0.4)
    f.ring(1000, 510, 80, WHITE, 4, a)
    f.text((1000, 495), "主燃烧室", 26, WHITE, a, 0.3)
    f.text((1000, 530), "气-气", 24, YELLOW, a, 0.5)
    for i in range(50):
        u = frac(rnd(i, 13) + t * 1.4)
        x = 1080 + u * 160
        y = 510 + (rnd(i, 14) - 0.5) * (40 + 120 * u)
        f.dot(x, y, 10 * (1 - u) + 2, mix(WHITE, ORANGE, u * 1.5), a * (1 - u), 1.2)
    rows = [("100% 推进剂过涡轮", "没有 dead branch，没有 detach"),
            ("富氧 / 富燃 两路分离", "关注点分离：QKV、多头、专家"),
            ("流量最大 → 涡轮最凉", "更宽 → 每个单元负荷更低"),
            ("气-气喷注", "soft attention 胜过硬路由")]
    for i, (s1, s2) in enumerate(rows):
        aa = ease((t - 1.0 - i * 1.2) / 0.5)
        y = 250 + i * 160
        x = 1300 + (1 - aa) * 60
        f.text((x, y), f"0{i + 1}", 40, ORANGE, aa, 0.8, anchor="lm", bold=True)
        f.text((x + 70, y), s1, 38, WHITE, aa, 0.2, anchor="lm")
        f.text((x + 70, y + 50), "→ " + s2, 30, CYAN, aa, 0.4, anchor="lm")
    captions(f, t, [(0.3, 4.0, "把前面所有原则推到极限，你会得到【猛禽】"),
                    (4.0, d, "RD-270 当年没飞起来：不是想不到，是【控制系统没追上】")])


def s_control(f, t, d):
    header(f, t, "09", "闭环要再付一次稳定性的钱", PURPLE)
    a = ease((t - 0.2) / 0.6)
    # 左：开环启动时序 / warmup
    X0, Y0, X1, Y1 = 170, 280, 880, 800
    f.box([X0 - 30, Y0 - 90, X1 + 30, Y1 + 40], GRAY, a * 0.5, 2, 18, g=0)
    f.text(((X0 + X1) / 2, Y0 - 45), "启动瞬态 = warmup", 40, ORANGE, a, 0.6, bold=True)
    f.line([(X0, Y1), (X1, Y1)], GRAY, 2, a, 0)
    f.line([(X0, Y0), (X0, Y1)], GRAY, 2, a, 0)
    u = clamp((t - 0.6) / 3.5)
    steps = []
    for k in range(8):
        x = X0 + k * (X1 - X0) / 8
        yv = Y1 - (Y1 - Y0) * min(1, (k + 1) / 5) * 0.85
        steps += [(x, yv), (x + (X1 - X0) / 8, yv)]
    f.line(pl_part(steps, u), ORANGE, 5, a)
    lr = []
    for k in range(60):
        p = k / 59
        v = p / 0.3 if p < 0.3 else 0.5 * (1 + math.cos(math.pi * (p - 0.3) / 0.7))
        lr.append((X0 + p * (X1 - X0), Y1 - (Y1 - Y0) * 0.7 * v))
    f.line(pl_part(lr, u), CYAN, 4, a)
    f.text((X0 + 20, Y0 + 20), "■ 开环阀门时序表 (RS-25)", 26, ORANGE, a * u, 0.3, anchor="lm")
    f.text((X0 + 20, Y0 + 60), "■ lr warmup + cosine", 26, CYAN, a * u, 0.3, anchor="lm", mono=True)
    # 右：POGO
    Xa, Xb = 1020, 1780
    f.box([Xa - 30, Y0 - 90, Xb + 30, Y1 + 40], GRAY, a * 0.5, 2, 18, g=0)
    f.text(((Xa + Xb) / 2, Y0 - 45), "POGO 振荡 = loss spike", 40, RED, a, 0.6, bold=True)
    ur = clamp((t - 2.5) / 2.5)
    red = [(Xa + p / 99 * (Xb - Xa), 420 + 90 * math.exp(2.4 * (p / 99 - 1)) * math.sin(p * 0.55)) for p in range(100)]
    f.line(pl_part(red, ur), RED, 4, a)
    f.text((Xa, 300), "正反馈：振幅指数增长", 28, RED, a * ur, 0.3, anchor="lm")
    ug = clamp((t - 5.5) / 2.5)
    grn = [(Xa + p / 99 * (Xb - Xa), 680 + 90 * math.exp(-p / 30) * math.sin(p * 0.55)) for p in range(100)]
    f.line(pl_part(grn, ug), GREEN, 4, a)
    f.text((Xa, 565), "加蓄压器 → 改相位 → 衰减", 28, GREEN, a * ug, 0.3, anchor="lm")
    f.text(((Xa + Xb) / 2, 790), "动量 · EMA · 梯度裁剪 · 降 lr", 30, GREEN, a * ug, 0.5)
    captions(f, t, [(0.3, 3.0, "瞬态里闭环来不及：只能按【开环时序表】走 = warmup"),
                    (3.0, 6.0, "POGO：结构和推进耦合，振幅【指数增长】"),
                    (6.0, d, "解法不是更硬，而是插个柔性节点：【动量 / EMA / 裁剪】")])


def s_rocket_eq(f, t, d):
    header(f, t, "10", "为什么值得这么复杂", YELLOW)
    a = ease((t - 0.2) / 0.6)
    f.text((960, 240), "Δv = Isp · g0 · ln(m0 / mf)", 66, YELLOW, a, 0.8, bold=True)
    f.text((960, 315), "比冲在指数的位置上", 32, GRAY, a, 0.1)
    u = ease((t - 1.0) / 2.0)
    f.text((520, 430), "比冲 Isp", 40, WHITE, a, 0.2)
    f.text((520, 530), f"{340 + 10 * u:.0f} s", 96, CYAN, a, 0.8, bold=True, mono=True)
    f.text((520, 640), f"+{2.9 * u:.1f}%", 60, CYAN, a, 0.6)
    f.text((1400, 430), "有效载荷", 40, WHITE, a, 0.2)
    f.text((1400, 530), f"{1.46 + 0.50 * u:.2f} t", 96, ORANGE, a, 0.8, bold=True, mono=True)
    pulse = 1 + 0.06 * math.sin(t * 6) * (u > 0.99)
    f.text((1400, 660), f"+{34 * u:.0f}%", int(110 * pulse), ORANGE, a, 1.0, bold=True)
    arr = ease((t - 1.5) / 0.6)
    f.line([(780, 530), (1140, 530)], WHITE, 4, arr, 0.6)
    f.line([(1115, 510), (1140, 530), (1115, 550)], WHITE, 4, arr, 0.6)
    f.text((960, 495), "指数放大", 30, WHITE, arr, 0.3)
    al = ease((t - 4.0) / 0.6)
    f.box([440, 770, 1480, 860], CYAN, al, 3, 18, fill=CYAN, fa=0.06)
    f.text((960, 815), "Scaling law：L(N,D) ≈ E + A/N^α + B/D^β", 40, CYAN, al, 0.7, mono=True)
    captions(f, t, [(0.3, 4.0, "比冲只多【2.9%】，载荷多出【34%】"),
                    (4.0, d, "收益在【指数】上，成本在多项式上 —— 所以值得手写 CUDA")])


def s_five(f, t, d):
    header(f, t, "11", "如果只记五句话", CYAN)
    rows = ["cycle 不是 loop：废气汇回主室 = 【x + f(x)】",
            "主管路上什么都别串：【Pre-LN】，部件全挂支路",
            "分级 = 材料扛不住一次转化：【每层只做小步更新】",
            "闭环要付稳定性的钱：【warmup / 动量 / 多头 / 裁剪】",
            "值不值得复杂，看收益在不在【指数】上"]
    cols = [GREEN, CYAN, ORANGE, PURPLE, YELLOW]
    for i, (s, c) in enumerate(zip(rows, cols)):
        t0 = 0.4 + i * 1.8
        aa = ease((t - t0) / 0.5)
        y = 230 + i * 150
        x = 200 + (1 - aa) * 80
        f.ring(x, y, 40, c, 4, aa)
        f.text((x, y - 2), str(i + 1), 44, c, aa, 0.8, bold=True)
        n = (t - t0) * 30
        f.rich((x + 80, y), s, 46, aa, acc=c, align="l", n=n)


def s_end(f, t, d):
    a = ease(t / 0.6)
    ty = 700 - ease(t / d) * 900
    cam = e3.Camera(0.6 + 0.3 * t, 0.12, 12.0, (0, 4.0 - ease(t / d) * 9, 0), 900, 1560, 540)
    draw3d(f, e3.full_engine(t * 9, heat=1.0), cam, fire=1.0)
    flame3d(f, cam, t, a, 5.0)
    s1 = ease((t - 0.2) / 0.6)
    f.text((760, 330), "关注", 60, WHITE, s1, 0.3)
    k = 1 + 0.02 * math.sin(t * 4)
    f.text((760, 460), "桥帮主AI", int(150 * k), CYAN, ease((t - 0.4) / 0.6), 1.0, bold=True)
    f.rich((760, 610), "每天一起【学点AI】", 64, ease((t - 0.9) / 0.5), acc=YELLOW, n=(t - 0.9) * 14)
    ab = ease((t - 2.0) / 0.5)
    p = 0.5 + 0.5 * math.sin(t * 5)
    f.box([610, 710, 910, 800], ORANGE, ab, 4, 45, fill=ORANGE, fa=0.25 + 0.2 * p, g=0.6 + 0.6 * p)
    f.text((760, 754), "＋ 关注", 46, WHITE, ab, 0.5, bold=True)
    for i in range(40):  # 星火
        u = frac(rnd(i, 20) + t * 0.3)
        ang = rnd(i, 21) * 6.283
        r = 120 + u * 500
        f.dot(760 + r * math.cos(ang), 470 + r * math.sin(ang) * 0.6, 3, mix(CYAN, MAGENTA, rnd(i, 22)),
              s1 * (1 - u) * 0.8, 1.0)


SCENES = [(s_cover, 5), (s_same, 6), (s_a_overview, 9), (s_a_thrust, 10), (s_a_pump, 9),
          (s_a_cycles, 11), (s_a_regen, 8), (s_cycle, 9), (s_open, 9), (s_preln, 10), (s_fwdbwd, 8),
          (s_staged, 9), (s_injector, 9), (s_throat, 10), (s_raptor, 8), (s_control, 9),
          (s_rocket_eq, 8), (s_five, 11), (s_end, 6)]
STARTS = np.cumsum([0] + [d for _, d in SCENES])
TOTAL = float(STARTS[-1])
NFRAMES = int(TOTAL * FPS)


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------
def hud(img, T):
    d = ImageDraw.Draw(img)
    p = T / TOTAL
    d.rectangle([0, 0, int(W * p), 5], fill=CYAN)
    d.text((W - 60, 76), "桥帮主AI", font=font(30), fill=(150, 165, 190), anchor="rm")


def render(i, T=None, with_glitch=True):
    if BG is None:
        init_bg()
    T = i / FPS if T is None else T
    k = int(np.searchsorted(STARTS, T, side="right") - 1)
    k = min(k, len(SCENES) - 1)
    fn, d = SCENES[k]
    t = T - STARTS[k]
    fr = Fr()
    fn(fr, t, d)
    fade = min(ease(t / 0.35) if k else 1.0, ease((d - t) / 0.35) if k < len(SCENES) - 1 else 1.0)
    arr = compose(fr, fade, T)
    if with_glitch and k:
        arr = glitch(arr, clamp(1 - t / 0.3), i)
    img = Image.fromarray(arr)
    hud(img, T)
    return img


def render_bytes(i):
    return render(i).tobytes()


# ---------------------------------------------------------------------------
# 配乐：合成器 pad + 贝斯 + 鼓 + 琶音 + 转场 whoosh
# ---------------------------------------------------------------------------
def music(path):
    n = int(TOTAL * SR)
    t = np.arange(n) / SR
    out = np.zeros(n, np.float64)
    rng = np.random.default_rng(1)
    beat = 60 / 100
    chords = [[57, 60, 64, 69], [53, 57, 60, 65], [48, 55, 60, 64], [55, 59, 62, 67]]
    hz = lambda m: 440 * 2 ** ((m - 69) / 12)
    bar = 4 * beat
    for b in range(int(TOTAL / bar) + 1):
        ch = chords[b % 4]
        s0 = int(b * bar * SR)
        s1 = min(n, int((b + 1) * bar * SR + 0.6 * SR))
        if s0 >= n:
            break
        tt = t[s0:s1] - b * bar
        env = np.minimum(1, tt / 0.4) * np.minimum(1, np.maximum(0, (bar + 0.6 - tt) / 0.6))
        seg = np.zeros_like(tt)
        for m in ch:
            for det in (-0.12, 0.12):
                fr = hz(m - 12) * 2 ** (det / 12)
                for h in range(1, 6):
                    seg += np.sin(2 * np.pi * fr * h * tt + h) / h ** 1.6
        out[s0:s1] += 0.012 * seg * env
        # 贝斯 + 琶音
        for q in range(8):
            ts = b * bar + q * beat / 2
            s = int(ts * SR)
            e = min(n, s + int(0.5 * SR))
            if s >= n:
                break
            tq = t[s:e] - ts
            if q % 2 == 0:
                out[s:e] += 0.16 * np.sin(2 * np.pi * hz(ch[0] - 24) * tq) * np.exp(-tq * 5)
            if 5 < ts < TOTAL - 6:
                m = ch[[0, 1, 2, 3, 2, 1, 3, 2][q]] + 12
                out[s:e] += 0.035 * (np.sin(2 * np.pi * hz(m) * tq) + 0.3 * np.sin(4 * np.pi * hz(m) * tq)) * np.exp(-tq * 9)
    for k in range(int(TOTAL / beat)):
        ts = k * beat
        if ts < 5 or ts > TOTAL - 6:
            continue
        s = int(ts * SR)
        e = min(n, s + int(0.4 * SR))
        tk = t[s:e] - ts
        out[s:e] += 0.35 * np.sin(2 * np.pi * (45 * tk + 105 / 30 * (1 - np.exp(-30 * tk)))) * np.exp(-tk * 9)
        sh = s + int(beat / 2 * SR)
        eh = min(n, sh + int(0.08 * SR))
        if sh < n:
            nz = np.diff(rng.standard_normal(eh - sh + 1))
            out[sh:eh] += 0.025 * nz * np.exp(-np.arange(eh - sh) / SR * 50)
    for st in STARTS[1:-1]:  # whoosh
        s = int((st - 0.5) * SR)
        e = min(n, int((st + 0.4) * SR))
        nz = np.convolve(rng.standard_normal(e - s), np.ones(12) / 12, "same")
        tw = np.arange(e - s) / SR
        env = np.where(tw < 0.5, (tw / 0.5) ** 2, np.exp(-(tw - 0.5) * 10))
        out[s:e] += 0.5 * nz * env
    for st in (0.0, STARTS[-2]):  # 冲击
        s = int(st * SR)
        e = min(n, s + int(2.0 * SR))
        tb = t[s:e] - st
        out[s:e] += 0.6 * np.sin(2 * np.pi * (40 * tb + 80 / 8 * (1 - np.exp(-8 * tb)))) * np.exp(-tb * 2.5)
    out = np.tanh(out * 1.3)
    out *= np.minimum(1, (TOTAL - t) / 1.5)
    out = out / np.max(np.abs(out)) * 0.85
    pcm = (out * 32767).astype(np.int16)
    stereo = np.stack([pcm, pcm], 1)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(stereo.tobytes())


def main():
    os.makedirs(OUT, exist_ok=True)
    if len(sys.argv) > 1 and sys.argv[1] == "preview":
        for k, (fn, d) in enumerate(SCENES):
            T = STARTS[k] + d * 0.8
            render(0, T, False).save(os.path.join(OUT, f"preview_{k:02d}.jpg"), quality=85)
            print("preview", k, fn.__name__)
        return
    render(0, 3.5, False).save(os.path.join(OUT, "cover.png"))
    print("cover saved")
    wav = os.path.join(OUT, "music.wav")
    music(wav)
    print("music saved")
    mp4 = os.path.join(OUT, "rocket_engine_to_transformer.mp4")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-i", wav, "-c:v", "libx264", "-preset", "fast", "-crf", "20",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", mp4]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    with Pool(max(1, os.cpu_count() - 1)) as pool:
        for i, b in enumerate(pool.imap(render_bytes, range(NFRAMES), chunksize=4)):
            p.stdin.write(b)
            if i % 150 == 0:
                print(f"frame {i}/{NFRAMES}", flush=True)
    p.stdin.close()
    p.wait()
    print("saved", mp4)


if __name__ == "__main__":
    main()
