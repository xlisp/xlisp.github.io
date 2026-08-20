# -*- coding: utf-8 -*-
"""
开局诊断：进过推荐池 vs 从未进过，首日渠道曲线有什么不同。

问题背景
--------
find_invisible_articles.py 发现：进过推荐池的文章中位数 1,535，没进过的 166，
9.2 倍且两个分布几乎不重叠。那么问题变成——**开局的哪个信号决定了算法捞不捞你**？

数据来源与口径（重要）
--------------------
导出文件里没有"每篇文章每天"的表：
  - Block A 是账号级日粒度（日期/渠道/阅读）
  - Block C 是文章级但只有**整个窗口的汇总**
所以首日曲线只能靠**跨快照差分**重建：把 45 份不同日期拉取的导出按文件名日期排序，
同一篇文章在各份里的 Block C 值连起来，就是它的累计阅读时间序列。
这对新发文章是准确的——30 天滚动窗口完整覆盖它的生命周期，窗口值 == 累计值。
（对发表超过 30 天的老文会失真，所以本脚本只分析"有发布后早期快照"的文章。）

核心指标
--------
把渠道分成三类，看开局时的构成：
  - 私域  = 公众号消息 + 公众号主页   （订阅者主动看，你的基本盘）
  - 社交  = 聊天会话 + 朋友圈          （真人转发，最强的正反馈信号）
  - 推荐  = 推荐                       （算法分发）
关键检验：**在推荐还没起量的时候（Day 0-1），两组的社交表现是否已经分化？**
如果是，说明社交信号是算法的输入，你就有了一个发布后数小时内可干预的杠杆。

用法：
    python opening_curve_analysis.py
    python opening_curve_analysis.py --exports ~/Downloads/mp_exports --max-day 7
需要：pandas, xlrd（conda `torch` 环境）
"""
import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from datetime import date, datetime
from statistics import median

try:
    import pandas as pd
except ImportError:
    sys.exit("需要 pandas / xlrd，请在 conda `torch` 环境里运行：conda activate torch")

DEFAULT_EXPORTS = os.path.expanduser("~/Downloads/mp_exports")

TOTAL_CHANNEL = "全部"
REC = ["推荐"]
SOCIAL = ["聊天会话", "朋友圈"]
OWNED = ["公众号消息", "公众号主页"]

EARLY_DAY = 1        # "开局"定义：发布后第 N 天以内的第一个可用快照
MAX_DAY = 7          # 曲线画到第几天


def pull_date_of(fn: str):
    m = re.search(r"(\d{4}-\d{2}-\d{2})\.xls[x]?$", fn)
    return datetime.strptime(m.group(1), "%Y-%m-%d").date() if m else None


def parse_pubdate(s):
    m = re.fullmatch(r"(\d{4})(\d{2})(\d{2})", str(s).strip())
    return date(int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


def load_snapshots(exports_dir):
    """-> {title: {pubdate, snaps: {pull_date: {channel: reads}}}}, 以及跳过的文件"""
    files = sorted((fn for fn in os.listdir(exports_dir)
                    if fn.startswith("tendency_") and fn.lower().endswith((".xls", ".xlsx"))),
                   key=lambda f: pull_date_of(f) or date.min)
    arts, skipped = {}, []
    for fn in files:
        pd_date = pull_date_of(fn)
        if pd_date is None:
            skipped.append(f"{fn}: 文件名无日期")
            continue
        try:
            engine = "xlrd" if fn.lower().endswith(".xls") else "openpyxl"
            raw = pd.read_excel(os.path.join(exports_dir, fn), header=None, engine=engine)
        except Exception as e:
            skipped.append(f"{fn}: {type(e).__name__}")
            continue

        c = raw.iloc[:, [11, 12, 13, 14]].copy()
        c.columns = ["channel", "pubdate", "title", "readers"]
        c = c[c["pubdate"].astype(str).str.match(r"\d{8}")].copy()
        c["readers"] = pd.to_numeric(c["readers"], errors="coerce").fillna(0).astype(int)
        c["channel"] = c["channel"].astype(str).str.strip()
        c["title"] = c["title"].astype(str)

        for _, r in c.iterrows():
            if r["channel"] == TOTAL_CHANNEL:
                continue                        # 汇总行不进渠道明细
            a = arts.setdefault(r["title"], {"pubdate": parse_pubdate(r["pubdate"]),
                                             "snaps": defaultdict(dict)})
            a["snaps"][pd_date][r["channel"]] = int(r["readers"])
    return arts, skipped, len(files)


def bucket(chan_map):
    rec = sum(v for k, v in chan_map.items() if k in REC)
    soc = sum(v for k, v in chan_map.items() if k in SOCIAL)
    own = sum(v for k, v in chan_map.items() if k in OWNED)
    tot = sum(chan_map.values())
    return {"total": tot, "rec": rec, "social": soc, "owned": own,
            "other": tot - rec - soc - own}


def build_rows(arts, early_day, max_day):
    rows = []
    for title, a in arts.items():
        pub = a["pubdate"]
        if not pub:
            continue
        series = []
        for pull, chans in sorted(a["snaps"].items()):
            d = (pull - pub).days
            if d < 0:
                continue
            series.append((d, bucket(chans)))
        if not series:
            continue
        ever_rec = any(b["rec"] > 0 for _, b in series)
        final = max(b["total"] for _, b in series)
        early = next((b for d, b in series if d <= early_day), None)
        rows.append({"title": title, "pub": pub, "ever_rec": ever_rec, "final": final,
                     "series": [(d, b) for d, b in series if d <= max_day],
                     "early": early,
                     "early_day": next((d for d, b in series if d <= early_day), None),
                     "first_day": series[0][0]})
    return rows


def pct(a, b):
    return f"{a / b:.0%}" if b else "  - "


def stat_block(label, rows, key):
    vals = [key(r) for r in rows if key(r) is not None]
    if not vals:
        return f"  {label:<16} 无数据"
    return (f"  {label:<16} 中位 {median(vals):>7.0f}   "
            f"范围 {min(vals):>6.0f} ~ {max(vals):>7.0f}   n={len(vals)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exports", default=DEFAULT_EXPORTS)
    ap.add_argument("--early-day", type=int, default=EARLY_DAY)
    ap.add_argument("--max-day", type=int, default=MAX_DAY)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    arts, skipped, n_files = load_snapshots(args.exports)
    rows = build_rows(arts, args.early_day, args.max_day)

    W = 84
    print("=" * W)
    print("开局诊断：进池组 vs 未进池组")
    print("=" * W)
    print(f"导出文件 {n_files} 份，解析出文章 {len(arts)} 篇，可定位发布日 {len(rows)} 篇")
    if skipped:
        print(f"跳过: {'; '.join(skipped[:3])}{' ...' if len(skipped) > 3 else ''}")

    # 只有"发布后早期就有快照"的文章才能谈开局
    usable = [r for r in rows if r["early"] is not None]
    late = [r for r in rows if r["early"] is None]
    print(f"有 Day≤{args.early_day} 快照、可分析开局的: {len(usable)} 篇")
    print(f"首个快照已晚于 Day{args.early_day}（发布早于导出覆盖 / 导出有断档）: {len(late)} 篇，已排除")
    print("⚠️  口径：跨快照差分重建，对新文准确；含推断成分的老文已排除。")

    got = [r for r in usable if r["ever_rec"]]
    nope = [r for r in usable if not r["ever_rec"]]

    # ── 开局对比：这是全脚本的核心
    print()
    print("=" * W)
    print(f"【开局对比】Day≤{args.early_day} 的首个快照")
    print("=" * W)
    for label, grp in (("进过推荐池", got), ("从未进推荐", nope)):
        print(f"\n■ {label}（{len(grp)} 篇）")
        print(stat_block("开局总阅读", grp, lambda r: r["early"]["total"]))
        print(stat_block("开局·社交", grp, lambda r: r["early"]["social"]))
        print(stat_block("开局·私域", grp, lambda r: r["early"]["owned"]))
        print(stat_block("开局·推荐", grp, lambda r: r["early"]["rec"]))
        print(stat_block("开局社交占比%", grp,
                         lambda r: 100 * r["early"]["social"] / r["early"]["total"]
                         if r["early"]["total"] else None))
        print(stat_block("社交/私域 比", grp,
                         lambda r: 100 * r["early"]["social"] / r["early"]["owned"]
                         if r["early"]["owned"] else None))

    # ── 关键检验：剔除开局就已有推荐的，看"纯自然开局"是否分化
    print()
    print("=" * W)
    print("【关键检验】开局时推荐尚未起量的文章，社交信号是否已分化")
    print("=" * W)
    clean = [r for r in usable if r["early"]["rec"] == 0]
    cg = [r for r in clean if r["ever_rec"]]
    cn = [r for r in clean if not r["ever_rec"]]
    print(f"开局推荐=0 的文章共 {len(clean)} 篇：后来进池 {len(cg)} 篇，始终没进 {len(cn)} 篇")
    if cg and cn:
        for label, grp in (("后来进池", cg), ("始终没进", cn)):
            print(f"\n■ {label}（{len(grp)} 篇）")
            print(stat_block("开局总阅读", grp, lambda r: r["early"]["total"]))
            print(stat_block("开局·社交", grp, lambda r: r["early"]["social"]))
            print(stat_block("开局社交占比%", grp,
                             lambda r: 100 * r["early"]["social"] / r["early"]["total"]
                             if r["early"]["total"] else None))
    else:
        print("样本不足，无法做这项检验（某一组为空）。")

    # ── 逐篇曲线
    print()
    print("=" * W)
    print(f"【逐篇开局曲线】Day0~Day{args.max_day}，格式 Day: 总阅读(社交/私域/推荐)")
    print("=" * W)
    for label, grp in (("✅ 进过推荐池", got), ("❌ 从未进推荐", nope)):
        print(f"\n{label}")
        for r in sorted(grp, key=lambda x: -x["final"]):
            print(f"\n  {r['final']:>6} 终值 · {r['pub']} · {r['title'][:44]}")
            for d, b in r["series"]:
                bar = "█" * min(40, int(b["total"] / max(1, r["final"]) * 40))
                print(f"    D{d}: {b['total']:>6} "
                      f"(社{b['social']:>5}/私{b['owned']:>4}/推{b['rec']:>6}) {bar}")

    # ── CSV
    out = args.out or os.path.join(args.exports, "converted", "opening_curves.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["标题", "发表日", "是否进过推荐池", "终值", "开局快照日(Day)",
                    "开局总阅读", "开局社交", "开局私域", "开局推荐", "开局社交占比"])
        for r in sorted(usable, key=lambda x: (-x["ever_rec"], -x["final"])):
            e = r["early"]
            w.writerow([r["title"], r["pub"], "是" if r["ever_rec"] else "否", r["final"],
                        r["early_day"], e["total"], e["social"], e["owned"], e["rec"],
                        f"{e['social'] / e['total']:.3f}" if e["total"] else ""])
    print()
    print(f"CSV 已写入: {out}")


if __name__ == "__main__":
    main()
