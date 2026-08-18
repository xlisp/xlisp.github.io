#!/usr/bin/env python3
"""把一个 Markdown 文件切成若干块，每块生成一个 SVG 二维码。

扫任意一张码，得到的就是原文对应的那一段 Markdown 原文（纯文本，不是链接、
不做压缩编码），按编号把各段首尾相接就还原成完整文件。

    python scripts/md_to_qrcodes.py posts/xxx.md
    python scripts/md_to_qrcodes.py posts/xxx.md --bytes 500 --ecc h --outdir /tmp/qr

默认每段前面加一行 `<!-- name i/n -->` 的注释头，用来标记顺序；它是合法的
Markdown 注释，渲染时不显示。不想要就加 --no-header。

依赖：segno（纯 Python，pip install segno）。
"""

from __future__ import annotations

import argparse
import html
import os
import sys

try:
    import segno
except ImportError:  # pragma: no cover
    sys.exit("缺少依赖：pip install segno")


# 版本 40、字节模式下各纠错级别的最大字节数（含头部），用来给 --bytes 封顶
MAX_BYTES = {"l": 2953, "m": 2331, "q": 1663, "h": 1273}

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def split_text(text: str, budget: int) -> list[str]:
    """按 UTF-8 字节数切分，优先在行边界断开，超长的行再按字符切。

    budget 是每段正文的字节上限（不含注释头）。返回的各段直接拼接 == text。
    """
    chunks: list[str] = []
    buf: list[str] = []
    used = 0

    def flush() -> None:
        nonlocal used
        if buf:
            chunks.append("".join(buf))
            buf.clear()
            used = 0

    for line in text.splitlines(keepends=True):
        size = len(line.encode("utf-8"))
        if size > budget:                      # 单行就超了：拆成字符级碎片
            flush()
            piece: list[str] = []
            piece_size = 0
            for ch in line:
                ch_size = len(ch.encode("utf-8"))
                if piece_size + ch_size > budget:
                    chunks.append("".join(piece))
                    piece, piece_size = [], 0
                piece.append(ch)
                piece_size += ch_size
            if piece:                          # 尾巴留给后面的行继续填
                buf.append("".join(piece))
                used = piece_size
            continue
        if used + size > budget:
            flush()
        buf.append(line)
        used += size

    flush()
    return chunks or [""]


def write_index(path: str, name: str, files: list[str], payloads: list[str]) -> None:
    """一张便于打印/翻页扫码的联系表。"""
    cards = []
    for i, (svg, payload) in enumerate(zip(files, payloads), 1):
        head = payload.strip().splitlines()[0] if payload.strip() else ""
        cards.append(
            f'<figure><img src="{html.escape(os.path.basename(svg))}" alt="part {i}">'
            f"<figcaption>{i} / {len(files)}"
            f'<br><code>{html.escape(head[:40])}</code></figcaption></figure>'
        )
    doc = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(name)} · {len(files)} 张二维码</title>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 2rem; }}
 .grid {{ display: flex; flex-wrap: wrap; gap: 1.5rem; }}
 figure {{ margin: 0; width: 260px; text-align: center; }}
 img {{ width: 100%; border: 1px solid #ddd; }}
 figcaption {{ font-size: .85rem; color: #555; margin-top: .4rem;
               word-break: break-all; }}
 @media print {{ figure {{ page-break-inside: avoid; }} }}
</style></head>
<body>
<h1>{html.escape(name)}</h1>
<p>按编号顺序扫码，把各段文本首尾相接即可还原原文。</p>
<div class="grid">
{chr(10).join(cards)}
</div>
</body></html>
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)


def main() -> None:
    ap = argparse.ArgumentParser(description="Markdown 文件 → 多张 SVG 二维码")
    ap.add_argument("mdfile", help="要编码的 .md 文件路径")
    ap.add_argument("-o", "--outdir", default=None,
                    help="输出目录，默认 scripts/out/qr/<文件名>/")
    ap.add_argument("-b", "--bytes", type=int, default=600, dest="budget",
                    help="每张码的正文字节上限，默认 600（越小越容易扫）")
    ap.add_argument("-e", "--ecc", default="m", choices=list(MAX_BYTES),
                    help="纠错级别 l/m/q/h，默认 m")
    ap.add_argument("-s", "--scale", type=int, default=8, help="SVG 模块边长，默认 8")
    ap.add_argument("--border", type=int, default=4, help="静区宽度，默认 4 个模块")
    ap.add_argument("--no-header", action="store_true", help="不加 <!-- i/n --> 注释头")
    ap.add_argument("--no-index", action="store_true", help="不生成 index.html")
    args = ap.parse_args()

    with open(args.mdfile, encoding="utf-8") as f:
        text = f.read()
    name = os.path.splitext(os.path.basename(args.mdfile))[0]

    outdir = args.outdir or os.path.join(OUT, "qr", name)
    os.makedirs(outdir, exist_ok=True)

    # 注释头本身也占容量，先按最坏情况（三位数编号）留出余量
    overhead = 0 if args.no_header else len(f"<!-- {name} 999/999 -->\n".encode("utf-8"))
    budget = min(args.budget, MAX_BYTES[args.ecc] - overhead)
    if budget <= 0:
        sys.exit(f"--ecc {args.ecc} 的单码容量放不下 {overhead} 字节的注释头，换 -e l 或 --no-header")

    chunks = split_text(text, budget)
    assert "".join(chunks) == text, "切分后无法还原，请报 bug"

    files, payloads = [], []
    width = len(str(len(chunks)))
    for i, chunk in enumerate(chunks, 1):
        payload = chunk if args.no_header else f"<!-- {name} {i}/{len(chunks)} -->\n{chunk}"
        qr = segno.make(payload, error=args.ecc, mode="byte", encoding="utf-8")
        path = os.path.join(outdir, f"{name}-{i:0{width}d}.svg")
        qr.save(path, scale=args.scale, border=args.border)
        files.append(path)
        payloads.append(payload)
        print(f"  {os.path.basename(path)}  版本 {qr.version}-{qr.error.upper()}  "
              f"{len(payload.encode('utf-8'))} 字节")

    if not args.no_index:
        index = os.path.join(outdir, "index.html")
        write_index(index, name, files, payloads)
        print(f"\n索引页：{index}")

    print(f"共 {len(chunks)} 张，{len(text.encode('utf-8'))} 字节 → {outdir}")


if __name__ == "__main__":
    main()
