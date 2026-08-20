# -*- coding: utf-8 -*-
"""
隐形文章清单 v2：把 wechat.json（已发布全量）和 mp_exports/（有数据的）对齐。

v2 相对 v1 的三处修正（v1 的输出有系统性错误，不要再用）：

1. 【阅读数翻倍】Block C 里每篇文章有一行 `传播渠道=全部` 的汇总行，外加若干
   分渠道明细行。v1 把它们一起 sum，等于把每篇算了两遍。v2 只取 `全部` 行。
   同时表头在第 2 行（`传播渠道/发表日期/内容标题/...`），v1 的 iloc[2:] 把表头
   当数据读了进去，产生一条标题为"内容标题"的幽灵记录。

2. 【日期不可信】docs/*.md 未纳入 git，v1 退而取 posts/<slug>.html 的首次提交日，
   但仓库是批量导入的，一大批文件的首次提交日都挤在同一天（伪影）。v2 改为：
   优先用导出文件里的真实 pubdate；取不到才回落到 git，并自动识别"批量导入日"
   并标注不可信。

3. 【阈值过高错判】v1 阈值 0.93，而作者的实际习惯是发布时大改标题（相似度常
   掉到 0.6~0.8），导致同一篇文章在"隐形"和"孤儿"里各出现一次。v2 分三档匹配，
   并新增最有价值的输出：**改标题对照表（仓库标题 → 发布标题 → 阅读量）**。

口径提醒：本脚本报的是**单份导出窗口内的最高阅读**，不是生命周期累计。
导出是 30 天滚动窗口，老文会被严重低估（sin/cos 窗口值 7,631，真实累计 20,921）。
要真实累计请用 wechat_track_milestone。这里的数字只用于判断"有没有数据"和相对量级。

用法：
    python find_invisible_articles.py
    python find_invisible_articles.py --repo ~/PyPro/xlisp.github.io \\
                                      --exports ~/Downloads/mp_exports
需要：pandas, xlrd（conda `torch` 环境）
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
from collections import Counter
from difflib import SequenceMatcher

try:
    import pandas as pd
except ImportError:
    sys.exit("需要 pandas / xlrd，请在 conda `torch` 环境里运行：conda activate torch")

DEFAULT_REPO = os.path.expanduser("~/PyPro/xlisp.github.io")
DEFAULT_EXPORTS = os.path.expanduser("~/Downloads/mp_exports")

# 三档匹配。作者发布时常大改标题，所以下限必须放低；
# 但「大学4年没讲明白的X」这类系列标题彼此相似度就有 0.889，
# 所以 0.889 以下的匹配一律标记为"待核对"，绝不静默采信。
STRONG = 0.80          # 0.80 以上视为可信匹配
WEAK = 0.55            # 0.55~0.80 视为疑似，需人肉核对
SERIES_TRAP = 0.89     # 系列标题的天然相似度，用于提示误配风险

TOTAL_CHANNEL = "全部"   # Block C 里的汇总行标记（只收录 Top-N）
REC_CHANNEL = "推荐"     # 算法推荐渠道 —— 是否进过推荐池是最强的诊断信号


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
    try:
        with open(md_path, encoding="utf-8") as f:
            for line in f:
                if line.startswith("# "):
                    return line[2:].strip()
    except OSError:
        pass
    return None


def git_added_date(repo: str, relpath: str):
    try:
        out = subprocess.run(
            ["git", "log", "--diff-filter=A", "--format=%ad", "--date=short", "--", relpath],
            cwd=repo, capture_output=True, text=True, timeout=15)
        lines = [x for x in out.stdout.strip().splitlines() if x]
        return lines[-1] if lines else None
    except Exception:
        return None


def fmt_pubdate(d):
    """20260701 -> 2026-07-01"""
    d = str(d)
    m = re.fullmatch(r"(\d{4})(\d{2})(\d{2})", d)
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else d


def describe_change(repo_title: str, pub_title: str) -> str:
    """粗略描述标题改动方向，帮助肉眼快速扫。"""
    a, b = norm(repo_title), norm(pub_title)
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    delta = len(pub_title) - len(repo_title)
    tag = "前缀保留" if i >= 6 else "整句重写"
    return f"{tag}·长度{delta:+d}"


# ──────────────────────────────────────────────────── 读已发布清单
def load_published(repo: str):
    with open(os.path.join(repo, "wechat.json"), encoding="utf-8") as f:
        mapping = json.load(f)

    docs = os.path.join(repo, "docs")
    by_slug = {fn[:-3].replace("_", "-").lower(): fn
               for fn in os.listdir(docs) if fn.endswith(".md")}

    rows, missing_md = [], []
    for slug, url in sorted(mapping.items()):
        fn = by_slug.get(slug.lower())
        if not fn:
            missing_md.append(slug)
            rows.append({"slug": slug, "url": url, "title": None, "git_date": None})
            continue
        rows.append({"slug": slug, "url": url,
                     "title": h1_of(os.path.join(docs, fn)),
                     "git_date": git_added_date(repo, os.path.join("posts", slug + ".html"))})

    # 识别"批量导入日"：同一天首次提交的文件数异常多 -> 该日期不可信
    cnt = Counter(r["git_date"] for r in rows if r["git_date"])
    bulk = {d for d, n in cnt.items() if n >= 5}
    for r in rows:
        r["git_unreliable"] = r["git_date"] in bulk
    return rows, missing_md, sorted(bulk)


# ──────────────────────────────────────────────────── 读导出数据
def load_exports(exports_dir: str):
    """扫描 tendency_*.xls，只取 Block C 里 传播渠道=='全部' 的汇总行。"""
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

        # Block A 列[1-3]：日期/渠道/阅读人数 —— 确定这份导出的窗口范围
        a = raw.iloc[3:, [1, 2, 3]].copy()
        a.columns = ["date", "channel", "readers"]
        a = a[a["date"].astype(str).str.match(r"\d{4}-\d{2}-\d{2}")]
        if len(a):
            coverage.append((str(a["date"].min()), str(a["date"].max())))

        # Block C 列[11-15]：渠道/发表日期/标题/阅读人数/占比
        # 结构：每篇可能有一行 渠道=='全部' 的汇总，外加若干分渠道明细行。
        # 关键：'全部' 只收录 Top-N，长尾文章**只在分渠道明细里出现**
        #（本仓库实测：某份导出 18 篇有全部行，另有 25 篇只在明细里）。
        # 所以：优先用 '全部'，取不到就用分渠道求和兜底（两者实测差 <1.5%，
        # 因为同一人跨渠道会被去重）。
        c = raw.iloc[:, [11, 12, 13, 14, 15]].copy()
        c.columns = ["channel", "pubdate", "title", "readers", "ratio"]
        c = c[c["pubdate"].astype(str).str.match(r"\d{8}")].copy()   # 同时挡掉表头行
        c["readers"] = pd.to_numeric(c["readers"], errors="coerce").fillna(0).astype(int)
        c["title"] = c["title"].astype(str)
        c["channel"] = c["channel"].astype(str).str.strip()

        totals = c[c["channel"] == TOTAL_CHANNEL].set_index("title")["readers"].to_dict()
        detail = c[c["channel"] != TOTAL_CHANNEL]
        by_title_sum = detail.groupby("title")["readers"].sum().to_dict()
        rec_reads = (detail[detail["channel"] == REC_CHANNEL]
                     .groupby("title")["readers"].sum().to_dict())
        pubdates = c.drop_duplicates("title").set_index("title")["pubdate"].to_dict()

        for t in set(list(totals) + list(by_title_sum)):
            exact_total = totals.get(t)
            val = exact_total if exact_total is not None else by_title_sum.get(t, 0)
            rec = seen.setdefault(t, {"title": t, "pubdate": pubdates.get(t, ""),
                                      "max_window_reads": 0, "n_files": 0,
                                      "max_rec_reads": 0, "ever_recommended": False,
                                      "approx": True})
            rec["max_window_reads"] = max(rec["max_window_reads"], int(val))
            rec["max_rec_reads"] = max(rec["max_rec_reads"], int(rec_reads.get(t, 0)))
            if rec_reads.get(t, 0) > 0:
                rec["ever_recommended"] = True
            if exact_total is not None:
                rec["approx"] = False          # 至少有一份导出给了权威的"全部"值
            rec["n_files"] += 1

    cov = (min(x[0] for x in coverage), max(x[1] for x in coverage)) if coverage else ("?", "?")
    return list(seen.values()), cov, len(files), skipped


# ──────────────────────────────────────────────────────────── 对齐
def align(published, exported):
    """三档全局指派：精确 -> 强匹配 -> 弱匹配（标记待核对）。"""
    taken_p, taken_e, matched = set(), set(), []

    def claim(i, j, score, tier):
        taken_p.add(i); taken_e.add(j)
        p, e = published[i], exported[j]
        matched.append({**p, "export_title": e["title"],
                        "pubdate": e["pubdate"], "reads": e["max_window_reads"],
                        "score": round(score, 3), "tier": tier,
                        "changed": norm(p["title"]) != norm(e["title"]),
                        "rec": e.get("max_rec_reads", 0),
                        "ever_rec": e.get("ever_recommended", False),
                        "approx": e.get("approx", True)})

    # 第一遍：归一化后完全相等
    exact = {}
    for j, e in enumerate(exported):
        exact.setdefault(norm(e["title"]), j)
    for i, p in enumerate(published):
        if not p["title"]:
            continue
        j = exact.get(norm(p["title"]))
        if j is not None and j not in taken_e:
            claim(i, j, 1.0, "精确")

    # 第二、三遍：按相似度降序全局指派，高分先认领
    pairs = []
    for i, p in enumerate(published):
        if i in taken_p or not p["title"]:
            continue
        for j, e in enumerate(exported):
            if j in taken_e:
                continue
            s = similar(p["title"], e["title"])
            if s >= WEAK:
                pairs.append((s, i, j))
    pairs.sort(reverse=True)
    for s, i, j in pairs:
        if i in taken_p or j in taken_e:
            continue
        claim(i, j, s, "强" if s >= STRONG else "弱·待核对")

    invisible = []
    for i, p in enumerate(published):
        if i in taken_p:
            continue
        best = max((similar(p["title"], e["title"]) for e in exported), default=0.0)
        invisible.append({**p, "reason": ("docs 里找不到 md" if not p["title"]
                                          else f"最高相似度仅 {best:.2f}")})
    orphan = [e for j, e in enumerate(exported) if j not in taken_e]
    return matched, invisible, orphan


# ──────────────────────────────────────────────────────────── 输出
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--exports", default=DEFAULT_EXPORTS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    published, missing_md, bulk_days = load_published(args.repo)
    exported, cov, n_files, skipped = load_exports(args.exports)
    matched, invisible, orphan = align(published, exported)

    W = 80
    print("=" * W)
    print("隐形文章清单 v2")
    print("=" * W)
    print(f"wechat.json 已发布    : {len(published)} 篇")
    print(f"导出文件              : {n_files} 份，覆盖 {cov[0]} ~ {cov[1]}")
    print(f"导出中出现过的文章    : {len(exported)} 篇（只计 传播渠道='全部' 的汇总行）")
    print(f"  ├─ 对上             : {len(matched)} 篇"
          f"（其中标题改过 {sum(1 for m in matched if m['changed'])} 篇）")
    print(f"  └─ 孤儿             : {len(orphan)} 篇")
    print(f"真·查无此文           : {len(invisible)} 篇")
    if skipped:
        print(f"跳过文件: {'; '.join(skipped[:3])}{' ...' if len(skipped) > 3 else ''}")
    if missing_md:
        print(f"docs 里缺 md 的 slug: {', '.join(missing_md)}")
    if bulk_days:
        print(f"⚠️  git 批量导入日（该日期不可作发布日）: {', '.join(bulk_days)}")
    print("⚠️  阅读数为「单份导出窗口内最高值」，非生命周期累计，老文被低估。")

    # ── 核心产出：改标题对照表
    changed = sorted([m for m in matched if m["changed"]], key=lambda x: -x["reads"])
    print()
    print("=" * W)
    print(f"【改标题对照表】{len(changed)} 篇 —— 仓库标题 vs 实际发布标题")
    print("=" * W)
    for m in changed:
        flag = " ⚠️需核对" if m["tier"].startswith("弱") else ""
        print(f"\n  {m['reads']:>6} 阅读 · {fmt_pubdate(m['pubdate'])} "
              f"· 相似度 {m['score']} · {describe_change(m['title'], m['export_title'])}{flag}")
        print(f"    仓库: {m['title']}")
        print(f"    发布: {m['export_title']}")

    # ── 真隐形
    print()
    print("=" * W)
    print(f"【真·查无此文】{len(invisible)} 篇")
    print("=" * W)
    for it in sorted(invisible, key=lambda x: x.get("git_date") or ""):
        d = it.get("git_date") or "????-??-??"
        if it.get("git_unreliable"):
            d += "(批量导入日,不可信)"
        print(f"  {d}  {it['title'] or it['slug']}")
        print(f"      {it['reason']}  |  {it['url']}")

    # ── 孤儿
    if orphan:
        print()
        print("=" * W)
        print(f"【孤儿】导出里有、wechat.json 无 —— {len(orphan)} 篇")
        print("=" * W)
        for e in sorted(orphan, key=lambda x: -x["max_window_reads"]):
            print(f"  {e['max_window_reads']:>6}  {fmt_pubdate(e['pubdate'])}  {e['title']}")

    weak = [m for m in matched if m["tier"].startswith("弱")]
    if weak:
        print()
        print(f"⚠️  {len(weak)} 条弱匹配（相似度 < {STRONG}），"
              f"注意系列标题天然相似度可达 {SERIES_TRAP}，务必逐条核对：")
        for m in sorted(weak, key=lambda x: x["score"]):
            print(f"  {m['score']}  仓库《{m['title'][:34]}》")
            print(f"         发布《{m['export_title'][:34]}》")

    # ── 推荐池诊断：进没进过算法推荐，是最强的单一信号
    got = sorted([m for m in matched if m["ever_rec"]], key=lambda x: -x["reads"])
    nope = sorted([m for m in matched if not m["ever_rec"]], key=lambda x: -x["reads"])
    print()
    print("=" * W)
    print(f"【推荐池诊断】进过推荐 {len(got)} 篇 / 从未进过 {len(nope)} 篇")
    print("=" * W)
    if got:
        med = sorted(x["reads"] for x in got)[len(got) // 2]
        print(f"\n  ✅ 进过推荐池（中位数 {med:,}）：")
        for m in got:
            share = m["rec"] / m["reads"] if m["reads"] else 0
            print(f"    {m['reads']:>6}  推荐占 {share:>5.0%}  {fmt_pubdate(m['pubdate'])}"
                  f"  {m['export_title'][:38]}")
    if nope:
        med = sorted(x["reads"] for x in nope)[len(nope) // 2]
        print(f"\n  ❌ 从未进过推荐池（中位数 {med:,}）—— 天花板由社交转发决定：")
        for m in nope:
            print(f"    {m['reads']:>6}  {fmt_pubdate(m['pubdate'])}  {m['export_title'][:44]}")

    print()
    print("=" * W)
    print("有数据的文章 · 按窗口阅读排序")
    print("=" * W)
    for m in sorted(matched, key=lambda x: -x["reads"]):
        mark = "*" if m["changed"] else " "
        approx = "~" if m["approx"] else " "
        rec = "推" if m["ever_rec"] else "  "
        print(f"  {m['reads']:>6}{approx}{mark}{rec} {fmt_pubdate(m['pubdate'])}"
              f"  {m['export_title'][:44]}")
    print("  (* = 发布时改过标题 | ~ = 阅读为分渠道求和的近似值 | 推 = 进过推荐池)")

    # ── CSV
    out = args.out or os.path.join(args.exports, "converted", "invisible_articles.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["分类", "发表日期", "窗口最高阅读", "阅读是否近似", "进过推荐池",
                    "推荐渠道阅读", "仓库标题", "发布标题", "标题是否改过",
                    "匹配档位", "相似度", "改动描述", "slug", "链接", "备注"])
        for m in sorted(matched, key=lambda x: -x["reads"]):
            w.writerow(["已匹配", fmt_pubdate(m["pubdate"]), m["reads"],
                        "近似" if m["approx"] else "精确",
                        "是" if m["ever_rec"] else "否", m["rec"],
                        m["title"], m["export_title"], "是" if m["changed"] else "否",
                        m["tier"], m["score"],
                        describe_change(m["title"], m["export_title"]) if m["changed"] else "",
                        m["slug"], m["url"], ""])
        for it in invisible:
            note = it["reason"] + ("；git日期为批量导入日,不可信" if it.get("git_unreliable") else "")
            w.writerow(["真隐形", it.get("git_date") or "", "", "", "", "",
                        it["title"] or "", "", "", "", "", "",
                        it["slug"], it["url"], note])
        for e in orphan:
            w.writerow(["孤儿", fmt_pubdate(e["pubdate"]), e["max_window_reads"],
                        "近似" if e.get("approx") else "精确",
                        "是" if e.get("ever_recommended") else "否",
                        e.get("max_rec_reads", 0), "", e["title"], "", "", "", "",
                        "", "", "wechat.json 无此条"])
    print()
    print(f"CSV 已写入: {out}")


if __name__ == "__main__":
    main()
