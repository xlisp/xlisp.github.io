#!/usr/bin/env python3
"""
Static site generator for xlisp.github.io.

- Reads markdown files from `docs/`
- Renders each as `posts/<slug>.html` using `templates/post.html`
- Regenerates `index.html` from `templates/index.html` with the post list

Usage:
    pip install markdown pygments
    python build.py                    # full rebuild (renders every docs/*.md)
    python build.py path/to/foo.md     # incremental: render only foo.md, then refresh index
    python build.py a.md b.md ...      # multiple files OK
"""

from __future__ import annotations

import datetime as dt
import html
import json
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
SITE_DESCRIPTION = (
    "Steve Chan — Clojure / Emacs / Python Lisp hacker. Notes on machine "
    "learning, deep learning, reinforcement learning, and large language models."
)


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
    """Return (html, first_h1_title). The first <h1> is stripped from html
    because the post template already renders the title in its header."""
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
    out = md.convert(body)
    title = None
    m = re.search(r"<h1[^>]*>(.*?)</h1>\s*", out, re.S)
    if m:
        title = re.sub(r"<[^>]+>", "", m.group(1)).strip()
        out = out[: m.start()] + out[m.end() :]
    return out, title


def first_paragraph_text(body: str, limit: int = 160) -> str:
    """Extract a plain-text snippet from the body for use as a description."""
    for raw in body.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(("#", ">", "-", "*", "`", "|", "!")):
            continue
        line = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", line)
        line = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", line)
        line = re.sub(r"[*_`]+", "", line).strip()
        if line:
            return line if len(line) <= limit else line[: limit - 1].rstrip() + "…"
    return ""


def read_template(name: str) -> str:
    return (TEMPLATES / name).read_text(encoding="utf-8")


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"  wrote {path.relative_to(ROOT)}")


def post_meta(md_path: Path, render_html: bool) -> tuple[dict, str | None]:
    """Read a markdown file, return (meta-dict, rendered-html-or-None)."""
    raw = md_path.read_text(encoding="utf-8")
    meta, body = parse_front_matter(raw)
    rendered = None
    h1_title = None
    if render_html:
        rendered, h1_title = render_markdown(body)
    else:
        m = re.search(r"^#\s+(.+)$", body, re.M)
        if m:
            h1_title = m.group(1).strip()

    title = meta.get("title") or h1_title or md_path.stem
    date = meta.get("date") or dt.date.fromtimestamp(md_path.stat().st_mtime).isoformat()
    slug = meta.get("slug") or slugify(md_path.stem)
    summary = meta.get("summary") or first_paragraph_text(body)

    return (
        {
            "title": title,
            "date": date,
            "slug": slug,
            "summary": summary,
            "href": f"posts/{slug}.html",
        },
        rendered,
    )


def render_post(info: dict, body_html: str, post_tpl: str) -> None:
    description = info["summary"] or SITE_DESCRIPTION
    page = (
        post_tpl.replace("{{title}}", html.escape(info["title"]))
        .replace("{{title_json}}", json.dumps(info["title"]))
        .replace("{{date}}", info["date"])
        .replace("{{author}}", AUTHOR)
        .replace("{{site_title}}", SITE_TITLE)
        .replace("{{description}}", html.escape(description, quote=True))
        .replace("{{content}}", body_html)
    )
    write(POSTS / f"{info['slug']}.html", page)


def parse_existing_index() -> list[dict]:
    """Recover post info from the previously rendered index.html, if present."""
    f = ROOT / "index.html"
    if not f.exists():
        return []
    text = f.read_text(encoding="utf-8")
    items: list[dict] = []
    for li in re.finditer(
        r'<li><span class="post-date">([^<]+)</span>\s*'
        r'<a href="([^"]+)">(.*?)</a>'
        r'(?:\s*<span class="post-summary">—\s*(.*?)</span>)?\s*</li>',
        text,
        re.S,
    ):
        date, href, title, summary = li.groups()
        slug_m = re.match(r"posts/(.+)\.html$", href)
        items.append(
            {
                "title": html.unescape(title.strip()),
                "date": date.strip(),
                "slug": slug_m.group(1) if slug_m else href,
                "summary": html.unescape((summary or "").strip()),
                "href": href,
            }
        )
    return items


def render_index(posts: list[dict], index_tpl: str) -> None:
    posts = sorted(posts, key=lambda p: p["date"], reverse=True)
    if posts:
        items = "\n".join(
            f'        <li><span class="post-date">{p["date"]}</span> '
            f'<a href="{p["href"]}">{html.escape(p["title"])}</a></li>'
            for p in posts
        )
        post_list = f'      <ul class="post-list">\n{items}\n      </ul>'
    else:
        post_list = (
            '      <p class="empty">No posts yet — drop a markdown file into '
            "<code>docs/</code> and run <code>python build.py</code>.</p>"
        )

    index_html = (
        index_tpl.replace("{{site_title}}", SITE_TITLE)
        .replace("{{description}}", html.escape(SITE_DESCRIPTION, quote=True))
        .replace("{{post_list}}", post_list)
        .replace("{{year}}", str(dt.date.today().year))
    )
    write(ROOT / "index.html", index_html)


def copy_assets() -> None:
    if ASSETS.exists():
        for css in ASSETS.glob("*.css"):
            shutil.copy2(css, ROOT / css.name)


def resolve_targets(args: list[str]) -> list[Path]:
    """Return md paths to render. Files outside docs/ are copied into docs/."""
    out: list[Path] = []
    for a in args:
        p = Path(a).expanduser().resolve()
        if not p.exists():
            sys.stderr.write(f"file not found: {a}\n")
            sys.exit(1)
        if p.suffix.lower() != ".md":
            sys.stderr.write(f"not a markdown file: {a}\n")
            sys.exit(1)
        DOCS.mkdir(parents=True, exist_ok=True)
        try:
            p.relative_to(DOCS)
            out.append(p)
        except ValueError:
            dst = DOCS / p.name
            shutil.copy2(p, dst)
            print(f"  copied {p} -> {dst.relative_to(ROOT)}")
            out.append(dst)
    return out


def build(targets: list[Path] | None = None) -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    POSTS.mkdir(exist_ok=True)

    post_tpl = read_template("post.html")
    index_tpl = read_template("index.html")

    if targets is None:
        target_set: set[Path] = set(DOCS.glob("*.md"))
    else:
        target_set = set(targets)

    by_slug: dict[str, dict] = {
        p["slug"]: p
        for p in parse_existing_index()
        if (POSTS / f"{p['slug']}.html").exists()
    }

    rendered_count = 0
    for md_path in sorted(DOCS.glob("*.md")):
        should_render = md_path in target_set
        info, body_html = post_meta(md_path, render_html=should_render)
        if should_render and body_html is not None:
            render_post(info, body_html, post_tpl)
            rendered_count += 1
        by_slug[info["slug"]] = info

    posts = list(by_slug.values())
    render_index(posts, index_tpl)
    copy_assets()

    if targets is None:
        print(f"\nFull build: rendered {rendered_count} post(s).")
    else:
        names = ", ".join(p.name for p in targets)
        print(f"\nIncremental build: rendered {rendered_count} post(s) ({names}); index refreshed.")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args:
        targets = resolve_targets(args)
        print(f"Building xlisp.github.io (incremental: {len(targets)} file) ...")
        build(targets=targets)
    else:
        print("Building xlisp.github.io (full) ...")
        build()
