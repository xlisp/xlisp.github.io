#!/usr/bin/env python3
"""
Vertically concatenate multiple images into one.

Usage:
    python concat_images.py                             # latest 2 screenshots -> Screenshots dir
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
from datetime import datetime
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    sys.stderr.write(
        "Missing dependency: Pillow\n"
        "Install with:  pip install Pillow\n"
    )
    sys.exit(1)


SCREENSHOT_DIR = Path("/storage/emulated/0/DCIM/Screenshots")
SCREENSHOT_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def latest_screenshots(directory: Path, n: int = 2) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"screenshot directory not found: {directory}")
    candidates = [
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in SCREENSHOT_EXTS
    ]
    if len(candidates) < n:
        raise ValueError(f"need at least {n} images in {directory}, found {len(candidates)}")
    candidates.sort(key=lambda p: p.stat().st_mtime)
    return candidates[-n:]


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
    ap.add_argument("inputs", nargs="*", help="input image paths (in order); if omitted, latest 2 screenshots are used")
    ap.add_argument("-o", "--output", default=None, help="output file (default: concat.jpg, or Screenshot_<ts>_concat.jpg in screenshots mode)")
    ap.add_argument("--width", type=int, default=None, help="resize every image to this width first")
    ap.add_argument("--bg", default="255,255,255", help="background RGB for padding (default 255,255,255)")
    ap.add_argument("--dir", default=str(SCREENSHOT_DIR), help=f"screenshot directory (default: {SCREENSHOT_DIR})")
    args = ap.parse_args()

    if args.inputs:
        paths = [Path(p) for p in args.inputs]
        missing = [p for p in paths if not p.exists()]
        if missing:
            sys.stderr.write("missing files:\n  " + "\n  ".join(str(p) for p in missing) + "\n")
            sys.exit(1)
        out_path = Path(args.output) if args.output else Path("concat.jpg")
    else:
        directory = Path(args.dir)
        paths = latest_screenshots(directory, 2)
        print("using latest screenshots:")
        for p in paths:
            print(f"  {p.name}")
        if args.output:
            out_path = Path(args.output)
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = directory / f"Screenshot_{ts}_concat.jpg"

    bg = tuple(int(x) for x in args.bg.split(","))
    if len(bg) != 3:
        sys.stderr.write("--bg must be 'R,G,B'\n")
        sys.exit(1)

    concat_vertical(paths, out_path, args.width, bg)


if __name__ == "__main__":
    main()
