#!/usr/bin/env python3
"""
Static site generator for xlisp.github.io.

- Reads markdown files from `docs/`
- Renders each as `posts/<slug>.html` using `templates/post.html`
- Regenerates `index.html` from `templates/index.html` with the post list

Usage:
    pip install markdown pygments
    python build.py
"""

from __future__ import annotations

import datetime as dt
import re
import shutil
import sys
from pathlib import Path

try:
    import markdown
except ImportError:
    sys.stderr.write(
        "Missing dependency: markdown\n"
        "Install with:  pip install markdown pygments\n"
    )
    sys.exit(1)


ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"
POSTS = ROOT / "posts"
TEMPLATES = ROOT / "templates"
ASSETS = ROOT / "assets"

SITE_TITLE = "Steve Chan — xlisp"
AUTHOR = "Steve Chan"


def slugify(name: str) -> str:
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9一-鿿]+", "-", s)
    return s.strip("-") or "post"


def parse_front_matter(text: str) -> tuple[dict, str]:
    """Parse very small YAML-ish front matter delimited by --- lines."""
    meta: dict = {}
    if not text.startswith("---"):
        return meta, text
    end = text.find("\n---", 3)
    if end == -1:
        return meta, text
    block = text[3:end].strip()
    body = text[end + 4 :].lstrip("\n")
    for line in block.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip().strip('"').strip("'")
    return meta, body


def render_markdown(body: str) -> tuple[str, str | None]:
    """Return (html, first_h1_title)."""
    md = markdown.Markdown(
        extensions=[
            "extra",
            "fenced_code",
            "codehilite",
            "tables",
            "toc",
            "sane_lists",
        ],
        extension_configs={
            "codehilite": {"guess_lang": False, "css_class": "highlight"},
        },
    )
    html = md.convert(body)
    title = None
    m = re.search(r"<h1[^>]*>(.*?)</h1>", html, re.S)
    if m:
        title = re.sub(r"<[^>]+>", "", m.group(1)).strip()
    return html, title


def read_template(name: str) -> str:
    return (TEMPLATES / name).read_text(encoding="utf-8")


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"  wrote {path.relative_to(ROOT)}")


def build() -> None:
    if not DOCS.exists():
        DOCS.mkdir(parents=True)
    if POSTS.exists():
        for p in POSTS.glob("*.html"):
            p.unlink()
    POSTS.mkdir(exist_ok=True)

    post_tpl = read_template("post.html")
    index_tpl = read_template("index.html")

    posts: list[dict] = []
    for md_path in sorted(DOCS.glob("*.md")):
        raw = md_path.read_text(encoding="utf-8")
        meta, body = parse_front_matter(raw)
        html, h1_title = render_markdown(body)

        title = meta.get("title") or h1_title or md_path.stem
        date = meta.get("date") or dt.date.fromtimestamp(
            md_path.stat().st_mtime
        ).isoformat()
        slug = meta.get("slug") or slugify(md_path.stem)
        summary = meta.get("summary", "")

        out_path = POSTS / f"{slug}.html"
        page = (
            post_tpl.replace("{{title}}", title)
            .replace("{{date}}", date)
            .replace("{{author}}", AUTHOR)
            .replace("{{site_title}}", SITE_TITLE)
            .replace("{{content}}", html)
        )
        write(out_path, page)

        posts.append(
            {
                "title": title,
                "date": date,
                "slug": slug,
                "summary": summary,
                "href": f"posts/{slug}.html",
            }
        )

    posts.sort(key=lambda p: p["date"], reverse=True)

    if posts:
        items = "\n".join(
            f'        <li><span class="post-date">{p["date"]}</span> '
            f'<a href="{p["href"]}">{p["title"]}</a>'
            + (f' <span class="post-summary">— {p["summary"]}</span>' if p["summary"] else "")
            + "</li>"
            for p in posts
        )
        post_list = f'      <ul class="post-list">\n{items}\n      </ul>'
    else:
        post_list = (
            '      <p class="empty">No posts yet — drop a markdown file into '
            "<code>docs/</code> and run <code>python build.py</code>.</p>"
        )

    index_html = index_tpl.replace("{{site_title}}", SITE_TITLE).replace(
        "{{post_list}}", post_list
    ).replace("{{year}}", str(dt.date.today().year))
    write(ROOT / "index.html", index_html)

    if ASSETS.exists():
        for css in ASSETS.glob("*.css"):
            shutil.copy2(css, ROOT / css.name)

    print(f"\nBuilt {len(posts)} post(s).")


if __name__ == "__main__":
    print("Building xlisp.github.io ...")
    build()
