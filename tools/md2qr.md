
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
md2qr.py —— 把一个 Markdown 文件切分成多张二维码图片。

每张二维码里装的是**明文的 md 片段**（不是压缩/base64），
所以用微信、系统相机等任意扫码工具扫出来就是可读的原文。

用法:
    python scripts/md2qr.py docs/cloze_to_fim.md
    python scripts/md2qr.py docs/xxx.md -b 500 -e M --sheet
    python scripts/md2qr.py docs/xxx.md -o scripts/out/mycode --no-header

依赖:
    pip install qrcode[pil]
"""

import argparse
import os
import sys

import qrcode
from qrcode.constants import (
    ERROR_CORRECT_L,
    ERROR_CORRECT_M,
    ERROR_CORRECT_Q,
    ERROR_CORRECT_H,
)
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------- 配置

EC_MAP = {
    "L": ERROR_CORRECT_L,
    "M": ERROR_CORRECT_M,
    "Q": ERROR_CORRECT_Q,
    "H": ERROR_CORRECT_H,
}

# version 40 在 byte 模式下的理论容量（字节），用于给 --bytes 封顶
EC_CAPACITY = {"L": 2953, "M": 2331, "Q": 1663, "H": 1273}

DEFAULT_OUT_DIR = "scripts/out"

FONT_CANDIDATES = [
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/PingFang.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def load_font(size):
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


# ---------------------------------------------------------------- 切分

def blen(s):
    return len(s.encode("utf-8"))


def split_long_line(line, budget):
    """按字符切一行超长的文本，保证每段 utf-8 字节数不超 budget（不会切坏汉字）"""
    pieces, cur, cur_b = [], [], 0
    for ch in line:
        cb = blen(ch)
        if cur_b + cb > budget and cur:
            pieces.append("".join(cur))
            cur, cur_b = [], 0
        cur.append(ch)
        cur_b += cb
    if cur:
        pieces.append("".join(cur))
    return pieces


def split_text(text, budget):
    """优先按行切，尽量让每张二维码扫出来是完整的段落"""
    chunks, cur = [], ""
    for line in text.splitlines(keepends=True):
        if blen(line) > budget:
            if cur:
                chunks.append(cur)
                cur = ""
            pieces = split_long_line(line, budget)
            chunks.extend(pieces[:-1])
            cur = pieces[-1]
            continue
        if blen(cur) + blen(line) > budget:
            chunks.append(cur)
            cur = line
        else:
            cur += line
    if cur:
        chunks.append(cur)
    return chunks


def build_payloads(text, max_bytes, with_header, title):
    """切分并加上 [i/n] 序号头。因为序号头本身占字节，这里迭代到稳定为止。"""
    if not with_header:
        return split_text(text, max_bytes)

    reserve = 16
    for _ in range(6):
        chunks = split_text(text, max_bytes - reserve)
        n = len(chunks)
        header_len = max(
            blen("[%d/%d] %s\n" % (i + 1, n, title)) for i in range(n)
        )
        if header_len <= reserve:
            return [
                "[%d/%d] %s\n%s" % (i + 1, n, title, c)
                for i, c in enumerate(chunks)
            ]
        reserve = header_len + 4
    raise RuntimeError("切分未收敛，请调小 --bytes")


# ---------------------------------------------------------------- 出图

def make_qr(data, ec, box_size, border):
    qr = qrcode.QRCode(
        version=None,
        error_correction=EC_MAP[ec],
        box_size=box_size,
        border=border,
    )
    qr.add_data(data)
    qr.make(fit=True)          # 数据超限会抛 DataOverflowError
    return qr.make_image(fill_color="black", back_color="white").convert("RGB"), qr.version


def add_caption(img, caption, font):
    pad = 14
    bbox = font.getbbox(caption)
    th = bbox[3] - bbox[1]
    canvas = Image.new("RGB", (img.width, img.height + th + pad * 2), "white")
    canvas.paste(img, (0, 0))
    d = ImageDraw.Draw(canvas)
    tw = bbox[2] - bbox[0]
    d.text(((img.width - tw) // 2, img.height + pad - bbox[1]), caption,
           fill="black", font=font)
    return canvas


def make_sheet(images, cols=3, gap=20):
    cols = min(cols, len(images))
    rows = (len(images) + cols - 1) // cols
    w = max(im.width for im in images)
    h = max(im.height for im in images)
    sheet = Image.new("RGB", (cols * w + gap * (cols + 1),
                              rows * h + gap * (rows + 1)), "white")
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        sheet.paste(im, (gap + c * (w + gap), gap + r * (h + gap)))
    return sheet


# ---------------------------------------------------------------- 主流程

def main():
    ap = argparse.ArgumentParser(
        description="把 Markdown 文件切分成多张可直接扫码阅读的二维码图片")
    ap.add_argument("md_path", help="Markdown 文件路径")
    ap.add_argument("-o", "--outdir", default=None,
                    help="输出目录（默认 %s/<文件名>_qr）" % DEFAULT_OUT_DIR)
    ap.add_argument("-b", "--bytes", type=int, default=700,
                    help="每张二维码最多装多少 utf-8 字节，默认 700（越小越好扫）")
    ap.add_argument("-e", "--ec", choices=list(EC_MAP), default="M",
                    help="纠错等级 L/M/Q/H，默认 M")
    ap.add_argument("--box-size", type=int, default=8, help="每个码点的像素，默认 8")
    ap.add_argument("--border", type=int, default=4, help="静默区宽度，默认 4")
    ap.add_argument("--no-header", action="store_true",
                    help="不在片段前加 [i/n] 序号头")
    ap.add_argument("--no-caption", action="store_true", help="图片下方不写序号文字")
    ap.add_argument("--sheet", action="store_true", help="额外生成一张拼版总图")
    ap.add_argument("--cols", type=int, default=3, help="拼版每行几张，默认 3")
    args = ap.parse_args()

    if not os.path.isfile(args.md_path):
        sys.exit("找不到文件: %s" % args.md_path)

    with open(args.md_path, "r", encoding="utf-8") as f:
        text = f.read()
    if not text.strip():
        sys.exit("文件内容为空")

    stem = os.path.splitext(os.path.basename(args.md_path))[0]
    outdir = args.outdir or os.path.join(DEFAULT_OUT_DIR, stem + "_qr")
    os.makedirs(outdir, exist_ok=True)

    cap = EC_CAPACITY[args.ec]
    max_bytes = min(args.bytes, cap - 24)
    if args.bytes > max_bytes:
        print("提示: --bytes 超过纠错等级 %s 的容量上限，已自动降到 %d"
              % (args.ec, max_bytes))

    # 自适应：真装不下就缩小片段重来
    for _ in range(8):
        payloads = build_payloads(text, max_bytes, not args.no_header, stem)
        try:
            results = [make_qr(p, args.ec, args.box_size, args.border)
                       for p in payloads]
            break
        except qrcode.exceptions.DataOverflowError:
            max_bytes = int(max_bytes * 0.8)
            print("单张超限，片段上限下调到 %d 字节后重试" % max_bytes)
    else:
        sys.exit("无法生成，请手动指定更小的 --bytes")

    font = load_font(26)
    total = len(results)
    width = max(2, len(str(total)))
    images, paths = [], []

    for i, (img, ver) in enumerate(results, 1):
        if not args.no_caption:
            img = add_caption(img, "%d / %d  ·  %s" % (i, total, stem), font)
        name = "part_%s.png" % str(i).zfill(width)
        path = os.path.join(outdir, name)
        img.save(path)
        images.append(img)
        paths.append(path)
        print("  %s  version=%-2d  %d 字节" % (name, ver, blen(payloads[i - 1])))

    if args.sheet:
        sheet_path = os.path.join(outdir, "sheet.png")
        make_sheet(images, cols=args.cols).save(sheet_path)
        print("  sheet.png  拼版总图")

    with open(os.path.join(outdir, "manifest.txt"), "w", encoding="utf-8") as f:
        f.write("source: %s\ntotal: %d\nbytes_per_qr: %d\nec: %s\n\n"
                % (args.md_path, total, max_bytes, args.ec))
        for i, p in enumerate(payloads, 1):
            f.write("--- part %d (%d bytes) ---\n%s\n" % (i, blen(p), p))

    print("\n共 %d 张，原文 %d 字节 → %s" % (total, blen(text), outdir))


if __name__ == "__main__":
    main()
    
