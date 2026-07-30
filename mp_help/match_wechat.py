#!/usr/bin/env python3
"""
Match scraped WeChat articles to site posts and refresh ../wechat.json.

`wechat.json` maps a post slug to its WeChat article url; build.py reads it and
renders a 微信 button next to that post in index.html.

The WeChat titles are rewritten versions of the site titles, so matching is
fuzzy: title similarity first, then a character-bigram overlap between the
WeChat digest and the post's markdown body.

Existing entries in wechat.json are never overwritten -- hand-fixed mappings
stay put. Only confident new matches are added; anything uncertain is printed
for you to add by hand.

Usage:
    python mp_help/match_wechat.py            # show what would change
    python mp_help/match_wechat.py --write    # actually update wechat.json
"""

from __future__ import annotations

import argparse
import difflib
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRAPE = ROOT / "mp_help" / "mp_articles.json"
MAPPING = ROOT / "wechat.json"
INDEX = ROOT / "index.html"
DOCS = ROOT / "docs"

TITLE_STRONG = 0.62   # title similarity above this -> accept
BODY_STRONG = 0.50    # digest/body overlap above this -> accept
PUNCT = re.compile(r'[\s"“”‘’\'`,，。：:；;！!？?、（）()《》<>\-—…·\.]+')


def norm(s: str) -> str:
    return PUNCT.sub("", html.unescape(s)).lower()


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9一-鿿]+", "-", name.lower().strip())
    return s.strip("-") or "post"


def bigrams(s: str) -> set[str]:
    s = re.sub(r"\s+", "", s)
    return {s[i : i + 2] for i in range(len(s) - 1)}


def load_posts() -> list[dict]:
    text = INDEX.read_text(encoding="utf-8")
    docs = {slugify(md.stem): md.read_text(encoding="utf-8") for md in DOCS.glob("*.md")}
    posts = []
    for date, href, title in re.findall(
        r'<li><span class="post-date">([^<]+)</span>\s*<a href="([^"]+)">(.*?)</a>',
        text,
        re.S,
    ):
        m = re.match(r"posts/(.+)\.html$", href)
        slug = m.group(1) if m else href
        title = html.unescape(re.sub(r"<[^>]+>", "", title)).strip()
        posts.append(
            {
                "slug": slug,
                "date": date,
                "title": title,
                "n": norm(title),
                "g": bigrams(norm(title)) | bigrams(docs.get(slug, "")[:3000]),
            }
        )
    return posts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="write wechat.json")
    args = ap.parse_args()

    articles = json.loads(SCRAPE.read_text(encoding="utf-8"))
    posts = load_posts()
    mapping: dict[str, str] = {}
    if MAPPING.exists():
        mapping = json.loads(MAPPING.read_text(encoding="utf-8"))

    taken = set(mapping)          # slugs already mapped by hand / earlier runs
    mapped_urls = set(mapping.values())
    added: list[tuple[str, str, str]] = []
    unsure: list[tuple[float, dict, list]] = []

    for art in articles:
        if art["url"] in mapped_urls:
            continue
        n = norm(art["title"])
        q = bigrams(n) | bigrams(re.sub(r"\s+", "", art.get("digest", "")))
        scored = []
        for p in posts:
            t = difflib.SequenceMatcher(None, n, p["n"]).ratio()
            b = len(q & p["g"]) / max(1, len(q))
            scored.append((max(t, b if t > 0.15 else 0.0), t, b, p))
        scored.sort(key=lambda x: -x[0])
        best = scored[0]
        if best[3]["slug"] in taken:            # post already has an article
            continue
        if best[1] >= TITLE_STRONG or best[2] >= BODY_STRONG:
            slug = best[3]["slug"]
            mapping[slug] = art["url"]
            taken.add(slug)
            added.append((slug, art["title"], best[3]["title"]))
        else:
            unsure.append((best[0], art, scored[:3]))

    print(f"{len(mapping)} slug(s) mapped, {len(added)} new")
    for slug, wx_title, site_title in added:
        print(f"  + {slug}\n      wx:   {wx_title}\n      site: {site_title}")

    if unsure:
        print(f"\n{len(unsure)} WeChat article(s) with no confident match "
              f"-- add by hand to wechat.json if one of these is right:")
        for _, art, scored in unsure:
            print(f"\n  wx: {art['title']}   {art['url']}")
            for s, t, b, p in scored:
                print(f"      title={t:.2f} body={b:.2f}  {p['slug']}  {p['title'][:40]}")

    unmapped = [p["slug"] for p in posts if p["slug"] not in mapping]
    print(f"\n{len(unmapped)} post(s) without a WeChat link: {', '.join(unmapped)}")

    if args.write:
        MAPPING.write_text(
            json.dumps(dict(sorted(mapping.items())), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote {MAPPING.relative_to(ROOT)}")
    else:
        print("\n(dry run -- pass --write to update wechat.json)")


if __name__ == "__main__":
    main()
