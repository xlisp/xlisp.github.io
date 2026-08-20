# -*- coding: utf-8 -*-
"""
隐形文章清单：把 wechat.json（已发布全量）和 mp_exports/（有数据的）对齐。

背景：wechat.json 记录了公众号真正发出去的每一篇（slug -> 链接），
而 tendency_*.xls 导出只包含"在统计窗口内产生过阅读"的文章。
两边一减，就是**发了但从没在数据里出现过**的那批——高度怀疑是次条位，
或者早于导出覆盖范围的老文。

用法：
    python find_invisible_articles.py                       # 用默认路径
    python find_invisible_articles.py --repo ~/PyPro/xlisp.github.io \\
                                      --exports ~/Downloads/mp_exports

需要：pandas, xlrd（在 conda `torch` 环境里跑；xlrd 只用于读 .xls）
输出：终端报告 + CSV（默认写到 exports 目录下的 converted/invisible_articles.csv）

两个实现要点：
1. 标题从 docs/*.md 的一级标题（# ...）取，slug 与文件名互转规则：
       docs/information_theory_through_pytorch.md  <->  information-theory-through-pytorch
   发布日期从 posts/<slug>.html 的首次 git 提交取（docs/*.md 基本没纳入 git 跟踪）。
2. 标题匹配用**全局贪心指派**而不是逐条取最优。原因：「大学4年没讲明白的信息论/
   微积分/线性代数」这类系列标题彼此相似度高达 0.89，逐条匹配会张冠李戴。
   全局指派保证 1.0 的完美匹配先被认领，剩下的才轮到模糊匹配。
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
from difflib import SequenceMatcher

try:
    import pandas as pd
except ImportError:
    sys.exit("需要 pandas / xlrd，请在 conda `torch` 环境里运行：conda activate torch")

DEFAULT_REPO = os.path.expanduser("~/PyPro/xlisp.github.io")
DEFAULT_EXPORTS = os.path.expanduser("~/Downloads/mp_exports")
# 「大学4年没讲明白的X」这类系列标题彼此相似度高达 0.889，阈值必须设在它之上，
# 否则一篇缺席的文章会去冒领它兄弟篇的数据。
MATCH_THRESHOLD = 0.93
REVIEW_BAND = (0.93, 1.0)       # 非完全一致的匹配，单独列出来让人肉核对
NEAR_MISS = 0.85                # 未匹配但相似度高于此 -> 提示"标题可能发布时改过"


# ──────────────────────────────────────────────────────────── 工具
def norm(s: str) -> str:
    """归一化标题：去掉所有标点/空白/引号，只留中日韩文字与字母数字。"""
    s = str(s)
    for ch in "“”\"'‘’":
        s = s.replace(ch, "")
    return re.sub(r"[^\w\u4e00-\u9fff]", "", s).lower()


def similar(a: str, b: str) -> float:
    return SequenceMatcher(None, norm(a), norm(b)).ratio()


def h1_of(md_path: str):
    """取 markdown 的一级标题。"""
    try:
        with open(md_path, encoding="utf-8") as f:
            for line in f:
                if line.startswith("# "):
                    return line[2:].strip()
    except OSError:
        pass
    return None


def git_added_date(repo: str, relpath: str):
    """该文件首次进入 git 的日期，作为发布日的近似。"""
    try:
        out = subprocess.run(
            ["git", "log", "--diff-filter=A", "--format=%ad", "--date=short", "--", relpath],
            cwd=repo, capture_output=True, text=True, timeout=15)
        lines = [x for x in out.stdout.strip().splitlines() if x]
        return lines[-1] if lines else None
    except Exception:
        return None


# ──────────────────────────────────────────────────── 读已发布清单
def load_published(repo: str):
    """wechat.json -> [{slug, url, title, md, git_date}]，标题取自对应 md 的 H1。"""
    with open(os.path.join(repo, "wechat.json"), encoding="utf-8") as f:
        mapping = json.load(f)

    docs = os.path.join(repo, "docs")
    by_slug = {}
    for fn in os.listdir(docs):
        if fn.endswith(".md"):
            by_slug[fn[:-3].replace("_", "-").lower()] = fn

    rows, missing_md = [], []
    for slug, url in sorted(mapping.items()):
        fn = by_slug.get(slug.lower())
        if not fn:
            missing_md.append(slug)
            rows.append({"slug": slug, "url": url, "title": None,
                         "md": None, "git_date": None})
            continue
        # docs/*.md 基本没纳入 git 跟踪，日期从对应的 posts/*.html 取
        rows.append({"slug": slug, "url": url,
                     "title": h1_of(os.path.join(docs, fn)),
                     "md": os.path.join("docs", fn),
                     "git_date": git_added_date(repo, os.path.join("posts", slug + ".html"))})
    return rows, missing_md


# ──────────────────────────────────────────────────── 读导出数据
def load_exports(exports_dir: str):
    """扫描所有 tendency_*.xls，聚合每篇文章的最高窗口阅读与发表日期。"""
    files = sorted(fn for fn in os.listdir(exports_dir)
                   if fn.startswith("tendency_") and fn.lower().endswith((".xls", ".xlsx")))
    if not files:
        sys.exit(f"在 {exports_dir} 里没找到 tendency_*.xls")

    seen, coverage, skipped = {}, [], []
    for fn in files:
        path = os.path.join(exports_dir, fn)
        try:
            engine = "xlrd" if path.lower().endswith(".xls") else "openpyxl"
            raw = pd.read_excel(path, header=None, engine=engine)
        except Exception as e:
            skipped.append(f"{fn}: {type(e).__name__}")
            continue

        # Block A 列[1-3]：日期/渠道/阅读人数 —— 用来确定这份导出的窗口范围
        a = raw.iloc[2:, [1, 2, 3]].copy()
        a.columns = ["date", "channel", "readers"]
        a = a[a["date"].astype(str).str.match(r"\d{4}-\d{2}-\d{2}")]
        if len(a):
            coverage.append((str(a["date"].min()), str(a["date"].max())))

        # Block C 列[11-15]：渠道/发表日期/标题/阅读人数/占比
        c = raw.iloc[2:, [11, 12, 13, 14, 15]].copy()
        c.columns = ["channel", "pubdate", "title", "readers", "ratio"]
        c = c.dropna(subset=["title"])
        c["readers"] = pd.to_numeric(c["readers"], errors="coerce").fillna(0).astype(int)
        c["pubdate"] = c["pubdate"].astype(str).str.replace(r"\.0$", "", regex=True)

        for title, grp in c.groupby("title"):
            total = int(grp["readers"].sum())          # 各渠道求和 = 该窗口内总阅读
            rec = seen.setdefault(str(title), {"title": str(title),
                                               "pubdate": grp["pubdate"].iloc[0],
                                               "max_window_reads": 0, "n_files": 0})
            rec["max_window_reads"] = max(rec["max_window_reads"], total)
            rec["n_files"] += 1

    cov_min = min(x[0] for x in coverage) if coverage else "?"
    cov_max = max(x[1] for x in coverage) if coverage else "?"
    return list(seen.values()), (cov_min, cov_max), len(files), skipped


# ──────────────────────────────────────────────────────────── 对齐
def align(published, exported):
    """全局贪心指派：把所有 (已发布, 导出) 配对按相似度降序，高分先认领。

    这样《大学4年…微积分》会先以 1.0 认领它自己那条，
    不会被相似度 0.89 的《大学4年…信息论》抢走。
    """
    taken_p, taken_e, matched = set(), set(), []

    # 第一遍：归一化后完全相等的，直接锁定。杜绝系列标题互相冒领。
    exact = {}
    for j, e in enumerate(exported):
        exact.setdefault(norm(e["title"]), j)
    for i, p in enumerate(published):
        if not p["title"]:
            continue
        j = exact.get(norm(p["title"]))
        if j is not None and j not in taken_e:
            taken_p.add(i); taken_e.add(j)
            e = exported[j]
            matched.append({**p, "export_title": e["title"], "pubdate": e["pubdate"],
                            "reads": e["max_window_reads"], "score": 1.0})

    # 第二遍：剩下的走高阈值模糊匹配，仍按相似度降序全局指派
    pairs = []
    for i, p in enumerate(published):
        if i in taken_p or not p["title"]:
            continue
        for j, e in enumerate(exported):
            if j in taken_e:
                continue
            s = similar(p["title"], e["title"])
            if s >= MATCH_THRESHOLD:
                pairs.append((s, i, j))
    pairs.sort(reverse=True)

    for s, i, j in pairs:
        if i in taken_p or j in taken_e:
            continue
        taken_p.add(i); taken_e.add(j)
        p, e = published[i], exported[j]
        matched.append({**p, "export_title": e["title"], "pubdate": e["pubdate"],
                        "reads": e["max_window_reads"], "score": round(s, 3)})

    invisible = []
    for i, p in enumerate(published):
        if i in taken_p:
            continue
        if not p["title"]:
            invisible.append({**p, "reason": "docs 里找不到对应 md，无法取标题"})
        else:
            best = max((similar(p["title"], e["title"]) for e in exported), default=0.0)
            note = "导出中无匹配"
            if best >= NEAR_MISS:
                note += "，但有近似标题 —— 发布时可能改过标题，请核对"
            invisible.append({**p, "reason": f"{note}（最高相似度 {best:.2f}）"})

    orphan = [e for j, e in enumerate(exported) if j not in taken_e]
    return matched, invisible, orphan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--exports", default=DEFAULT_EXPORTS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    published, missing_md = load_published(args.repo)
    exported, cov, n_files, skipped = load_exports(args.exports)
    matched, invisible, orphan = align(published, exported)

    W = 78
    print("=" * W)
    print("隐形文章清单")
    print("=" * W)
    print(f"wechat.json 已发布      : {len(published)} 篇")
    print(f"导出文件                : {n_files} 份，覆盖 {cov[0]} ~ {cov[1]}")
    print(f"导出中出现过的文章      : {len(exported)} 篇")
    print(f"  ├─ 与已发布对上       : {len(matched)} 篇")
    print(f"  └─ 对不上（孤儿）     : {len(orphan)} 篇")
    print(f"发布了但导出里查无此文  : {len(invisible)} 篇   <<< 隐形")
    if skipped:
        print(f"跳过的文件: {'; '.join(skipped[:3])}{' ...' if len(skipped) > 3 else ''}")
    if missing_md:
        print(f"docs 里缺 md 的 slug: {', '.join(missing_md)}")

    # 隐形文章按"是否在导出覆盖范围内"分两类 —— 这是关键的诊断分叉
    inside, before = [], []
    for it in invisible:
        d = it.get("git_date")
        (before if (d and d < cov[0]) else inside).append(it)

    print()
    print("-" * W)
    print(f"【A 类】发布日在覆盖范围内，却完全没数据 —— {len(inside)} 篇")
    print("        高度怀疑：次条位 / 未进入推荐池")
    print("-" * W)
    for it in sorted(inside, key=lambda x: x.get("git_date") or ""):
        print(f"  {it.get('git_date') or '  ????-??-??'}  {it['title'] or it['slug']}")
        print(f"              {it['url']}")

    print()
    print("-" * W)
    print(f"【B 类】发布早于导出覆盖（{cov[0]}）—— {len(before)} 篇，属正常盲区")
    print("-" * W)
    for it in sorted(before, key=lambda x: x.get("git_date") or ""):
        print(f"  {it.get('git_date')}  {it['title'] or it['slug']}")

    if orphan:
        print()
        print("-" * W)
        print(f"【C 类】导出里有、wechat.json 里没有 —— {len(orphan)} 篇")
        print("        可能是 wechat.json 漏登记，或标题发布时改过")
        print("-" * W)
        for e in sorted(orphan, key=lambda x: -x["max_window_reads"]):
            print(f"  {e['max_window_reads']:>6}  {e['pubdate']}  {e['title']}")

    # 需要人肉核对的边缘匹配
    fuzzy = [m for m in matched if REVIEW_BAND[0] <= m["score"] < REVIEW_BAND[1]]
    if fuzzy:
        print()
        print(f"⚠️  以下 {len(fuzzy)} 条是模糊匹配（非完全一致），建议核对：")
        for m in fuzzy:
            print(f"  {m['score']}  md  《{m['title']}》")
            print(f"         导出《{m['export_title']}》")

    print()
    print("-" * W)
    print("有数据的文章 · 按阅读排序")
    print("-" * W)
    for m in sorted(matched, key=lambda x: -x["reads"]):
        print(f"  {m['reads']:>7}  {m['pubdate']}  {m['title'][:44]}")

    # ── 写 CSV
    out = args.out or os.path.join(args.exports, "converted", "invisible_articles.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["分类", "git首次提交日", "发表日期(导出)", "窗口最高阅读",
                    "标题", "slug", "链接", "备注"])
        for it in inside:
            w.writerow(["A_覆盖内隐形", it.get("git_date") or "", "", "",
                        it["title"] or "", it["slug"], it["url"], it["reason"]])
        for it in before:
            w.writerow(["B_早于覆盖", it.get("git_date") or "", "", "",
                        it["title"] or "", it["slug"], it["url"], it["reason"]])
        for e in orphan:
            w.writerow(["C_导出孤儿", "", e["pubdate"], e["max_window_reads"],
                        e["title"], "", "", "wechat.json 无此条"])
        for m in sorted(matched, key=lambda x: -x["reads"]):
            w.writerow(["D_正常", m.get("git_date") or "", m["pubdate"], m["reads"],
                        m["title"], m["slug"], m["url"], f"匹配度 {m['score']}"])
    print()
    print(f"CSV 已写入: {out}")


if __name__ == "__main__":
    main()
