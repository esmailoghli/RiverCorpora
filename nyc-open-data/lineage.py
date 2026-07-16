"""
Build a lineage for each series: order its members oldest -> newest and link
each edition to the one before it.

Consumes a grouping in the clusters.json / version_groups.json shape
(`[{size, members:[{id, name, ...}]}]`). Typically the finalized output of
review_clusters.py. For every cluster it sorts the members by the time period
they cover (ascending) and records, per member, its `period`, its position
`seq`, and the `predecessor`/`successor` ids that make the chain explicit.

Time per member, most reliable first:
  1. a split slice's own key, 'abcd-1234_2020' -> 2020, '..._2013-2014' -> 2013
  2. a year or fiscal-year in the title, '... 2004', 'FY2015', 'FY03 - FY12' (min)
  3. the dataset's publication timestamp from metadata.json (createdAt), used as
     a fallback when the title has no year and as the tie-breaker between two
     editions covering the same year.
Members with no parseable period sort last (undated), ordered by timestamp.

Usage
-----
  python lineage.py --clusters clusters.reviewed.json --dir data
  python lineage.py --clusters version_groups.json --out lineage.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SLICE_RE = re.compile(r"^([a-z0-9]{4}-[a-z0-9]{4})_(.+)$", re.I)
YEAR_RE = re.compile(r"(?:19|20|21|22)\d{2}")          # 1900-2299
FY2_RE = re.compile(r"\bfy\s*'?(\d{2})\b", re.I)       # FY03, FY 12, fy'19
INF = 10 ** 9


def parent_of(did: str) -> str:
    m = SLICE_RE.match(did)
    return m.group(1) if m else did


def year_from_key(key: str):
    """Slice key -> start year. '2020' -> 2020, '2013-2014' -> 2013, else None."""
    m = re.match(r"((?:19|20|21|22)\d{2})", key)
    return int(m.group(1)) if m else None


def fy_to_year(dd: str) -> int:
    """Two-digit fiscal year -> calendar year (FY03 -> 2003, FY97 -> 1997)."""
    d = int(dd)
    return (1900 if d > 50 else 2000) + d


def load_timestamps(root: Path) -> dict:
    """dataset id -> publication year from metadata.json (createdAt, else
    rowsUpdatedAt). Keyed by the real (parent) id."""
    ts = {}
    if not root or not root.exists():
        return ts
    from datetime import datetime, timezone
    for md in root.glob("*__*/metadata.json"):
        try:
            m = json.loads(md.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        did = parent_of(m.get("id") or md.parent.name.rsplit("__", 1)[-1])
        epoch = m.get("createdAt") or m.get("rowsUpdatedAt")
        if isinstance(epoch, (int, float)) and did not in ts:
            ts[did] = datetime.fromtimestamp(epoch, timezone.utc).year
    return ts


def member_period(m: dict, ts_by_id: dict, overrides: dict | None = None):
    """Return (year, source), the period a member covers and where it came from.
    year is None for undated members. A human override (from review_lineage.py)
    wins over everything."""
    did = m.get("id", "")
    if overrides and did in overrides:
        return year_from_key(str(overrides[did])), "reviewed"
    sm = SLICE_RE.match(did)
    if sm:
        # a split slice's key is authoritative: a year, a range, or the explicit
        # 'undated' catch-all (which sorts last, don't fall back to a title/date).
        y = year_from_key(sm.group(2))
        return (y, "slice") if y is not None else (None, "undated")
    name = m.get("name", "")
    years = [int(y) for y in YEAR_RE.findall(name)]
    if years:
        return min(years), "title"
    fys = [fy_to_year(d) for d in FY2_RE.findall(name)]
    if fys:
        return min(fys), "title-fy"
    y = ts_by_id.get(parent_of(did))
    if y is not None:
        return y, "published"
    return None, "undated"


def period_label(m: dict) -> str:
    """A short human period tag for the report, the slice key or title year(s)."""
    sm = SLICE_RE.match(m.get("id", ""))
    if sm:
        return sm.group(2)
    name = m.get("name", "")
    ys = YEAR_RE.findall(name)
    if ys:
        return ys[0] if len(ys) == 1 else f"{min(ys)}-{max(ys)}"
    fy = FY2_RE.findall(name)
    if fy:
        return "FY" + fy[0]
    return "undated"


def order_cluster(members: list, ts_by_id: dict, overrides: dict | None = None) -> list:
    """Return members sorted oldest -> newest, each annotated with period/seq/
    predecessor/successor."""
    keyed = []
    for m in members:
        year, src = member_period(m, ts_by_id, overrides)
        ts = ts_by_id.get(parent_of(m.get("id", "")), INF)
        # dated first (by year, then publish ts), undated last (by ts, then name)
        sort_key = (year if year is not None else INF, ts, m.get("name", ""))
        keyed.append((sort_key, year, src, m))
    keyed.sort(key=lambda t: t[0])

    out = []
    for seq, (_, year, src, m) in enumerate(keyed, 1):
        mm = dict(m)
        if src == "reviewed":
            mm["period"] = str((overrides or {}).get(m.get("id", ""), year))  # human-assigned
        elif src == "published":
            mm["period"] = f"~{year}"          # inferred from publication date
        elif year is None:
            mm["period"] = "undated"
        else:
            mm["period"] = period_label(m)
        mm["period_year"] = year
        mm["period_source"] = src
        mm["seq"] = seq
        out.append(mm)
    for i, mm in enumerate(out):
        mm["predecessor"] = out[i - 1]["id"] if i > 0 else None
        mm["successor"] = out[i + 1]["id"] if i < len(out) - 1 else None
    return out


def write_report(clusters: list, path: Path) -> None:
    lines = ["=" * 78, "SERIES LINEAGE (oldest -> newest)", "=" * 78, ""]
    multi = [c for c in clusters if c["size"] > 1]
    for n, c in enumerate(multi, 1):
        span = c.get("span")
        head = f"[{n}] {c.get('label', '')}   ({c['size']} editions"
        head += f", {span[0]}..{span[1]})" if span and span[0] else ")"
        lines.append(head)
        periods = [m["period"] for m in c["members"]]
        if len(periods) > 20:                          # keep the summary readable
            periods = periods[:10] + [f"... ({len(periods) - 20} more) ..."] + periods[-10:]
        lines.append("      " + "  ->  ".join(periods))
        for m in c["members"]:
            lines.append(f"       {m['seq']:>2}. {m['period']:<12} {m['name']} ({m['id']})")
        lines.append("")
    singles = [c for c in clusters if c["size"] == 1]
    if singles:
        lines += ["-" * 78, f"SINGLE-EDITION: {len(singles)}", "-" * 78]
    path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description="Order each series' members oldest -> newest.")
    ap.add_argument("--clusters", default="clusters.reviewed.json",
                    help="grouping to order (clusters.json / version_groups.json / reviewed)")
    ap.add_argument("--dir", default="data",
                    help="metadata dir for publication-date fallback/tie-break (optional)")
    ap.add_argument("--overrides", default=None,
                    help="period_overrides.json {id: 'YYYY'|'YYYY-YYYY'|'undated'} from "
                         "review_lineage.py -- human-assigned periods win over inference")
    ap.add_argument("--out", default="lineage.json")
    ap.add_argument("--report", default="lineage.txt")
    args = ap.parse_args()

    src = Path(args.clusters)
    if not src.exists():
        raise SystemExit(f"{src} not found -- finalize a review first (review_clusters.py --finalize).")
    clusters = json.loads(src.read_text())
    ts_by_id = load_timestamps(Path(args.dir))
    if ts_by_id:
        print(f"Loaded publication years for {len(ts_by_id)} datasets from {args.dir}/")
    overrides = {}
    if args.overrides and Path(args.overrides).exists():
        overrides = json.loads(Path(args.overrides).read_text())
        print(f"Applying {len(overrides)} reviewed period override(s) from {args.overrides}")

    out = []
    for c in clusters:
        ordered = order_cluster(c.get("members", []), ts_by_id, overrides)
        years = [m["period_year"] for m in ordered if m["period_year"] is not None]
        nc = dict(c)
        nc["members"] = ordered
        nc["span"] = [min(years), max(years)] if years else None
        out.append(nc)

    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    write_report(out, Path(args.report))
    multi = sum(1 for c in out if c["size"] > 1)
    undated = sum(1 for c in out for m in c["members"] if m["period_year"] is None)
    print(f"Ordered {len(out)} series ({multi} multi-edition); {undated} undated member(s).")
    print(f"Wrote {args.out}, {args.report}")


if __name__ == "__main__":
    main()
