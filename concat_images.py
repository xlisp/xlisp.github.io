#!/usr/bin/env python3
"""
Vertically concatenate multiple images into one.

Usage:
    python concat_images.py a.jpg b.jpg c.png -o out.jpg
    python concat_images.py *.png                       # default -> concat.jpg
    python concat_images.py a.jpg b.jpg --width 1080    # resize to common width

Images are stacked top-to-bottom in the order given. By default each image
keeps its original width and the canvas width equals the widest input;
narrower images are centered on a white background. Pass --width to scale
every image to a common width first.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    sys.stderr.write(
        "Missing dependency: Pillow\n"
        "Install with:  pip install Pillow\n"
    )
    sys.exit(1)


def concat_vertical(
    paths: list[Path],
    out_path: Path,
    target_width: int | None = None,
    background: tuple[int, int, int] = (255, 255, 255),
) -> None:
    if not paths:
        raise ValueError("no input images")

    images = [Image.open(p).convert("RGB") for p in paths]

    if target_width is not None:
        scaled = []
        for im in images:
            w, h = im.size
            new_h = round(h * target_width / w)
            scaled.append(im.resize((target_width, new_h), Image.LANCZOS))
        images = scaled
        canvas_width = target_width
    else:
        canvas_width = max(im.width for im in images)

    canvas_height = sum(im.height for im in images)
    canvas = Image.new("RGB", (canvas_width, canvas_height), background)

    y = 0
    for im in images:
        x = (canvas_width - im.width) // 2
        canvas.paste(im, (x, y))
        y += im.height

    canvas.save(out_path)
    print(f"wrote {out_path}  ({canvas_width}x{canvas_height}, {len(images)} images)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Vertically concatenate images.")
    ap.add_argument("inputs", nargs="+", help="input image paths (in order)")
    ap.add_argument("-o", "--output", default="concat.jpg", help="output file (default: concat.jpg)")
    ap.add_argument("--width", type=int, default=None, help="resize every image to this width first")
    ap.add_argument("--bg", default="255,255,255", help="background RGB for padding (default 255,255,255)")
    args = ap.parse_args()

    paths = [Path(p) for p in args.inputs]
    missing = [p for p in paths if not p.exists()]
    if missing:
        sys.stderr.write("missing files:\n  " + "\n  ".join(str(p) for p in missing) + "\n")
        sys.exit(1)

    bg = tuple(int(x) for x in args.bg.split(","))
    if len(bg) != 3:
        sys.stderr.write("--bg must be 'R,G,B'\n")
        sys.exit(1)

    concat_vertical(paths, Path(args.output), args.width, bg)


if __name__ == "__main__":
    main()
