"""
Review the year-less members of multi-edition series and pin their period, so a
lineage isn't ordered by a guessed publication date.

Some editions have no year in their title. This tool walks exactly the multi-edition
clusters that contain such a member and lets you assign the real period by hand,
seeing the rest of the series for context. Split '_undated' slices are NOT surfaced:
that slice is the no-date catch-all (rows whose split column had no parseable date),
so it is undated by design and already sorts last.

For each year-less member it shows the series' current oldest -> newest ordering
(your assignments so far already applied), marks the member in question, and its
publication year, then you choose:

    y  accept   -- keep the inferred position ('~YYYY', or 'undated')
    e  edit     -- type the real period: a year (2019), a range (2019-2020),
                   or 'undated' to sort it last
    u  undated  -- sort it last
    s  skip     -- decide later
    q  quit     -- save and exit (resume later)
    ?  help

STATEFUL: every decision is written into lineage_plan.csv (a `reviewed` column
marks done rows); stop and resume any time. TEAM REVIEW via --shard / --merge.

APPLY writes period_overrides.json, which lineage.py consumes:
  python review_lineage.py --apply
  python lineage.py --clusters clusters.reviewed.json --dir data \
                    --overrides period_overrides.json

Usage
-----
  python review_lineage.py [PLAN] [--clusters clusters.reviewed.json] [--dir data]
  python review_lineage.py --shard N
  python review_lineage.py --merge [--parts "lineage_plan.part*.csv"]
  python review_lineage.py --apply [--overrides period_overrides.json]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from math import ceil
from pathlib import Path

from lineage import (SLICE_RE, load_timestamps, order_cluster, parent_of,
                     year_from_key)

csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))

FIELDS = ["cluster", "member_id", "member_name", "label", "span",
          "inferred", "pub_year", "assigned", "reviewed"]
DONE = {"y", "yes", "1", "true"}
DATASET_URL = "https://data.cityofnewyork.us/d/{id}/"
YEARLESS = {"published", "undated"}          # period_source values with no title year


def needs_review(m: dict) -> bool:
    """A member worth a human period only if it has no title year AND is not a
    split slice. A split '_undated' slice is the no-date catch-all bucket. It is
    undated by construction (its rows have no date) and already sorts last, so
    there is nothing to assign."""
    return m["period_source"] in YEARLESS and not SLICE_RE.match(m.get("id", ""))


# ---------------------------------------------------------------------------
# Affected clusters: multi-edition series with >= 1 year-less member
# ---------------------------------------------------------------------------

def affected(clusters: list, ts: dict, overrides: dict | None = None) -> list:
    """Deterministic list of (key, label, ordered_members, yearless_members) for
    every multi-edition cluster that still has a member with no title year."""
    out, n = [], 0
    for c in clusters:
        members = c.get("members", [])
        if len(members) < 2:
            continue
        ordered = order_cluster(members, ts, overrides)
        yl = [m for m in ordered if needs_review(m)]
        if not yl:
            continue
        n += 1
        out.append((f"L{n}", c.get("label", ""), ordered, yl))
    return out


# ---------------------------------------------------------------------------
# Plan I/O
# ---------------------------------------------------------------------------

def build_plan(clusters: list, ts: dict) -> list[dict]:
    rows = []
    for key, label, ordered, yl in affected(clusters, ts):
        years = [m["period_year"] for m in ordered if m["period_year"] is not None]
        span = f"{min(years)}-{max(years)}" if years else "?"
        for m in yl:
            rows.append({
                "cluster": key, "member_id": m["id"],
                "member_name": m.get("name", ""), "label": label, "span": span,
                "inferred": m["period"],
                "pub_year": ts.get(parent_of(m["id"]), ""),
                "assigned": "", "reviewed": "",
            })
    return rows


def read_plan(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for fld in FIELDS:
            r.setdefault(fld, "")
    return rows


def save(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def is_reviewed(r: dict) -> bool:
    return str(r.get("reviewed", "")).strip().lower() in DONE


def overrides_from(rows: list[dict]) -> dict:
    """member_id -> assigned period, for rows the reviewer has decided."""
    ov = {}
    for r in rows:
        if is_reviewed(r) and r.get("assigned"):
            ov[r["member_id"]] = r["assigned"]
    return ov


# ---------------------------------------------------------------------------
# Team review: shard / merge
# ---------------------------------------------------------------------------

def shard_plan(path: Path, n: int) -> None:
    rows = read_plan(path)
    n = max(1, min(n, len(rows)))
    size = ceil(len(rows) / n)
    made = []
    for k in range(n):
        chunk = rows[k * size:(k + 1) * size]
        if not chunk:
            continue
        part = path.with_name(f"{path.stem}.part{k + 1}.csv")
        save(part, chunk)
        made.append((part, len(chunk)))
    print(f"Split {len(rows)} member(s) into {len(made)} part(s):")
    for part, cnt in made:
        print(f"  {part.name}  ({cnt} rows)")
    print(f"\nEach reviewer:   python review_lineage.py {made[0][0].name}")
    print(f"Then merge back: python review_lineage.py --merge --plan {path.name}")


def merge_parts(path: Path, part_glob: str, out: Path) -> None:
    parts = sorted(path.parent.glob(part_glob))
    if not parts:
        sys.exit(f"No part files matched {part_glob!r} in {path.parent}/")
    merged, order = {}, []
    if path.exists():
        for r in read_plan(path):
            merged[r["member_id"]] = r
            order.append(r["member_id"])
    conflicts = 0
    for pp in parts:
        for r in read_plan(pp):
            mid = r["member_id"]
            if mid not in merged:
                merged[mid] = r
                order.append(mid)
            elif is_reviewed(r):
                cur = merged[mid]
                if is_reviewed(cur) and cur.get("assigned") != r.get("assigned"):
                    conflicts += 1
                    print(f"  conflict on {mid}: keeping '{cur.get('assigned')}' "
                          f"over '{r.get('assigned')}'")
                elif not is_reviewed(cur):
                    merged[mid] = r
    save(out, [merged[i] for i in order])
    done = sum(1 for i in order if is_reviewed(merged[i]))
    print(f"Merged {len(parts)} part(s) -> {out}  ({done}/{len(order)} reviewed"
          f"{', ' + str(conflicts) + ' conflict(s)' if conflicts else ''})")


# ---------------------------------------------------------------------------
# Apply -> period_overrides.json
# ---------------------------------------------------------------------------

def apply(path: Path, out: Path) -> None:
    rows = read_plan(path)
    ov = overrides_from(rows)
    out.write_text(json.dumps(ov, indent=2, ensure_ascii=False))
    left = sum(1 for r in rows if not is_reviewed(r))
    print(f"Wrote {len(ov)} period override(s) -> {out}"
          + (f"  ({left} member(s) still unreviewed)" if left else ""))
    print(f"Now: python lineage.py --overrides {out.name}")


# ---------------------------------------------------------------------------
# Interactive review
# ---------------------------------------------------------------------------

def render_chain(ordered: list, mark_id: str) -> str:
    parts = []
    for m in ordered:
        p = m["period"]
        parts.append(f"«{p}»" if m["id"] == mark_id else p)
    if len(parts) > 22:
        i = next((k for k, m in enumerate(ordered) if m["id"] == mark_id), 0)
        lo, hi = max(0, i - 6), min(len(parts), i + 7)
        head = "" if lo == 0 else "... -> "
        tail = "" if hi == len(parts) else " -> ..."
        parts = parts[lo:hi]
        return head + "  ->  ".join(parts) + tail
    return "  ->  ".join(parts)


def show(row: dict, pos: int, total: int, clusters_ctx: dict, ts: dict,
         overrides: dict) -> None:
    key = row["cluster"]
    label, members = clusters_ctx.get(key, (row.get("label", ""), []))
    ordered = order_cluster(members, ts, overrides) if members else []
    mid = row["member_id"]
    print("\n" + "=" * 72)
    print(f"[{pos}/{total}]  series: {label}   (span {row.get('span', '?')})")
    if ordered:
        print("  order: " + render_chain(ordered, mid))
    print(f"  member : {row.get('member_name', '')}")
    print(f"       id: {mid}   {DATASET_URL.format(id=parent_of(mid))}")
    pub = row.get("pub_year", "")
    print(f"  no year in title - inferred '{row.get('inferred', '')}'"
          + (f", published ~{pub}" if pub else ", no publication date"))
    if row.get("assigned"):
        print(f"  assigned: {row['assigned']}")
    print("-" * 72)


def review(path: Path, clusters: list, ts: dict, review_all: bool) -> None:
    rows = read_plan(path)
    # cluster key -> (label, full member list), rebuilt deterministically
    ctx = {}
    for key, label, ordered, _yl in affected(clusters, ts):
        ctx[key] = (label, ordered)

    idx = [i for i, r in enumerate(rows) if review_all or not is_reviewed(r)]
    done = len(rows) - len(idx)
    n_clusters = len({r["cluster"] for r in rows})
    print(f"{len(rows)} year-less member(s) across {n_clusters} series; "
          f"{done} reviewed, {len(idx)} to go.")
    if not idx:
        print("Nothing to review. (use --all to revisit; --apply to write overrides.)")
        return

    HELP = ("  y accept inferred   e edit (year / 2019-2020 / undated)   u undated\n"
            "  s skip              q quit & save                        ? help")
    for pos, i in enumerate(idx, 1):
        row = rows[i]
        overrides = overrides_from(rows)                 # live: reflect decisions so far
        show(row, pos, len(idx), ctx, ts, overrides)
        while True:
            try:
                choice = input("  [y]accept [e]dit [u]ndated [s]kip [q]uit [?] > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                save(path, rows)
                print(f"\nStopped. Progress saved to {path}.")
                return
            if choice in ("y", "yes", "accept"):
                inf = row.get("inferred", "")
                row["assigned"] = "undated" if inf == "undated" else inf.lstrip("~")
                row["reviewed"] = "y"
            elif choice in ("u", "undated"):
                row["assigned"], row["reviewed"] = "undated", "y"
            elif choice in ("e", "edit"):
                val = input("     period (e.g. 2019, 2019-2020, undated): ").strip()
                if not val:
                    print("     unchanged.")
                    continue
                if val.lower() != "undated" and year_from_key(val) is None:
                    if input(f"     '{val}' has no parseable year - use anyway? [y/N] ")\
                            .strip().lower() not in DONE:
                        continue
                row["assigned"] = "undated" if val.lower() == "undated" else val
                row["reviewed"] = "y"
                print(f"     -> {row['assigned']}")
            elif choice in ("s", "skip"):
                row["reviewed"] = ""
            elif choice in ("q", "quit", "abort"):
                save(path, rows)
                print(f"\nQuit. Progress saved to {path} - re-run to continue.")
                return
            elif choice in ("?", "h", "help"):
                print(HELP)
                continue
            else:
                print("     (y / e / u / s / q / ?)")
                continue
            save(path, rows)
            break

    left = sum(1 for r in rows if not is_reviewed(r))
    print(f"\nDone. {len(rows) - left}/{len(rows)} reviewed, {left} left.")
    print(f"Saved to {path}. Apply with:  python review_lineage.py --apply")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Assign periods to year-less members for lineage ordering.")
    ap.add_argument("plan_file", nargs="?", default=None,
                    help="plan file to review (positional; overrides --plan, e.g. a shard)")
    ap.add_argument("--clusters", default="clusters.reviewed.json",
                    help="finalized grouping (clusters.reviewed.json / version_groups.json)")
    ap.add_argument("--plan", default="lineage_plan.csv")
    ap.add_argument("--dir", default="data", help="metadata dir for publication dates")
    ap.add_argument("--all", action="store_true", help="revisit already-reviewed rows")
    ap.add_argument("--rebuild", action="store_true", help="regenerate the plan (loses edits)")
    ap.add_argument("--shard", type=int, metavar="N")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--parts", default=None)
    ap.add_argument("--apply", action="store_true", help="write period_overrides.json from the plan")
    ap.add_argument("--overrides", default="period_overrides.json", help="output for --apply")
    args = ap.parse_args()

    plan_path = Path(args.plan_file or args.plan)

    if args.shard is not None:
        if not plan_path.exists():
            sys.exit(f"{plan_path} not found - run review_lineage.py once to build it.")
        shard_plan(plan_path, args.shard)
        return
    if args.merge:
        merge_parts(plan_path, args.parts or f"{plan_path.stem}.part*.csv", plan_path)
        return
    if args.apply:
        if not plan_path.exists():
            sys.exit(f"{plan_path} not found - review first.")
        apply(plan_path, Path(args.overrides))
        return

    src = Path(args.clusters)
    if not src.exists():
        sys.exit(f"{src} not found - finalize a review first (review_clusters.py --finalize).")
    clusters = json.loads(src.read_text())
    ts = load_timestamps(Path(args.dir))

    if not plan_path.exists() or args.rebuild:
        rows = build_plan(clusters, ts)
        if not rows:
            print("No multi-edition series has a year-less member. Nothing to review.")
            return
        save(plan_path, rows)
        print(f"Built {plan_path} ({len(rows)} year-less member(s) across "
              f"{len({r['cluster'] for r in rows})} series).")

    review(plan_path, clusters, ts, args.all)


if __name__ == "__main__":
    main()
