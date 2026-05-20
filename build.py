#!/usr/bin/env python3
"""
Static site generator for xlisp.github.io.

- Reads markdown files from `docs/`
- Renders each as `posts/<slug>.html` using `templates/post.html`
- Regenerates `index.html` from `templates/index.html` with the post list

Usage:
    pip install markdown pygments
    python build.py                          # full rebuild (renders every docs/*.md)
    python build.py path/to/foo.md           # incremental: render only foo.md, then refresh index
    python build.py a.md b.md ...            # multiple files OK
    python build.py --sources DIR [DIR ...]  # copy every *.md from each DIR into docs/
                                             # (mtime preserved -> post date), then full rebuild
    python build.py --prune                  # drop index entries whose posts/<slug>.html
                                             # is gone, and delete the matching docs/*.md
    python build.py --remove posts/foo.html  # delete the html + matching docs/*.md,
                                             # update index, and remember it in
                                             # .build-ignore so --sources won't restore it
"""

from __future__ import annotations

import argparse
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
IGNORE_FILE = ROOT / ".build-ignore"

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


def load_ignore() -> set[str]:
    """Markdown basenames to skip during --sources. One name per line."""
    if not IGNORE_FILE.exists():
        return set()
    out: set[str] = set()
    for line in IGNORE_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.add(line)
    return out


def add_to_ignore(names: set[str]) -> None:
    """Persist newly-ignored md basenames to .build-ignore."""
    if not names:
        return
    merged = sorted(load_ignore() | names)
    header = (
        "# Markdown basenames to skip when syncing from --sources.\n"
        "# Managed by build.py (--remove / --prune). One filename per line.\n"
    )
    IGNORE_FILE.write_text(header + "\n".join(merged) + "\n", encoding="utf-8")


def sync_from_sources(sources: list[str]) -> int:
    """Copy every *.md from each source directory into docs/.

    `shutil.copy2` preserves the source mtime, so the post date in the
    generated index.html reflects the original file's modification time.
    Names listed in .build-ignore are skipped.
    """
    DOCS.mkdir(parents=True, exist_ok=True)
    ignore = load_ignore()
    if ignore:
        print(f"  ignore list ({len(ignore)}): {', '.join(sorted(ignore))}")
    n = 0
    skipped = 0
    for s in sources:
        src = Path(s).expanduser().resolve()
        if not src.is_dir():
            sys.stderr.write(f"source not a directory: {s}\n")
            sys.exit(1)
        mds = sorted(src.glob("*.md"))
        if not mds:
            print(f"  (no *.md in {src})")
            continue
        print(f"  from {src}:")
        for md in mds:
            if md.name in ignore:
                print(f"    {md.name}  [skipped]")
                skipped += 1
                continue
            dst = DOCS / md.name
            shutil.copy2(md, dst)
            mtime = dt.date.fromtimestamp(dst.stat().st_mtime).isoformat()
            print(f"    {md.name}  [{mtime}]")
            n += 1
    if skipped:
        print(f"  (skipped {skipped} ignored file(s))")
    return n


def _slug_from_arg(arg: str) -> str:
    """Accept `posts/foo.html`, `foo.html`, or `foo` and return the slug."""
    s = arg.strip()
    if s.startswith("posts/"):
        s = s[len("posts/") :]
    if s.endswith(".html"):
        s = s[: -len(".html")]
    return s


def remove(paths: list[str]) -> None:
    """Delete posts/<slug>.html + matching docs/*.md, drop the index entry,
    and add the md basename(s) to .build-ignore so --sources won't restore them."""
    POSTS.mkdir(exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)

    slugs = [_slug_from_arg(p) for p in paths]
    ignored_mds: set[str] = set()

    for slug in slugs:
        html_path = POSTS / f"{slug}.html"
        if html_path.exists():
            html_path.unlink()
            print(f"  deleted {html_path.relative_to(ROOT)}")
        for md in DOCS.glob("*.md"):
            if slugify(md.stem) == slug:
                ignored_mds.add(md.name)
                md.unlink()
                print(f"  deleted {md.relative_to(ROOT)}")

    if ignored_mds:
        add_to_ignore(ignored_mds)
        print(f"  remembered in .build-ignore: {', '.join(sorted(ignored_mds))}")

    existing_slugs = {p.stem for p in POSTS.glob("*.html")}
    entries = [e for e in parse_existing_index() if e["slug"] in existing_slugs]
    render_index(entries, read_template("index.html"))
    print(f"\nRemoved {len(slugs)} post(s): {', '.join(slugs)}")


def prune() -> None:
    """Sync index.html with what actually lives in posts/.

    For every index entry whose `posts/<slug>.html` no longer exists, the
    entry is dropped from index.html and the matching `docs/*.md` (any file
    whose slugified stem equals that slug) is deleted so the next build
    won't regenerate it. If the source still exists in an upstream project
    folder, you'll need to delete it there too — otherwise `--sources` will
    bring it back on the next sync.
    """
    POSTS.mkdir(exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)

    existing_slugs = {p.stem for p in POSTS.glob("*.html")}
    entries = parse_existing_index()

    kept: list[dict] = []
    removed: list[str] = []
    for e in entries:
        if e["slug"] in existing_slugs:
            kept.append(e)
        else:
            removed.append(e["slug"])

    if not removed:
        print("Nothing to prune — every index entry still has its post html.")
        return

    removed_set = set(removed)
    ignored_mds: set[str] = set()
    for md in sorted(DOCS.glob("*.md")):
        if slugify(md.stem) in removed_set:
            ignored_mds.add(md.name)
            md.unlink()
            print(f"  deleted {md.relative_to(ROOT)}")

    if ignored_mds:
        add_to_ignore(ignored_mds)
        print(f"  remembered in .build-ignore: {', '.join(sorted(ignored_mds))}")

    render_index(kept, read_template("index.html"))
    print(f"\nPruned {len(removed)} entry(ies): {', '.join(removed)}")


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
    parser = argparse.ArgumentParser(
        prog="build.py",
        description="Static site generator for xlisp.github.io.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        metavar="DIR",
        help="Project docs directories to sync into docs/ before building "
        "(preserves mtime so post dates come from the source files).",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Drop index entries whose posts/<slug>.html no longer exists "
        "and delete the matching docs/*.md source.",
    )
    parser.add_argument(
        "--remove",
        nargs="+",
        metavar="POST",
        help="Remove one or more posts (accepts posts/foo.html, foo.html, or foo). "
        "Deletes the html + matching docs/*.md, updates index.html, and remembers "
        "the md basename in .build-ignore so --sources won't restore it.",
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Optional markdown files for an incremental rebuild "
        "(ignored when --sources / --prune / --remove is given).",
    )
    args = parser.parse_args()

    if args.remove:
        remove(args.remove)
    elif args.prune:
        prune()
    elif args.sources:
        print(f"Syncing markdown from {len(args.sources)} source(s) ...")
        n = sync_from_sources(args.sources)
        print(f"Synced {n} markdown file(s).\n")
        print("Building xlisp.github.io (full) ...")
        build()
    elif args.files:
        targets = resolve_targets(args.files)
        print(f"Building xlisp.github.io (incremental: {len(targets)} file) ...")
        build(targets=targets)
    else:
        print("Building xlisp.github.io (full) ...")
        build()
