"""
Interactive reviewer for split_plan.csv to eyeball each proposed split and
confirm, skip, or fix the axis, one dataset at a time.

For each candidate it shows the full title, the proposed split column and span,
the other coverage columns it considered, and a few sample values from the data:

    y  yes   -- keep the split on the proposed column
    s  skip  -- don't split this dataset
    e  edit  -- pick a different split column
    a  abort -- save progress and quit (resume later)
    ?  help

Each decision is written straight back into the plan file (a `reviewed` column marks
done rows), so you can stop any time and continue where you left off.

TEAM REVIEW: shard the plan into independent files so several people review
different slices in parallel, then merge the results back:

  python review_splits.py --shard 3            # -> split_plan.part1.csv, .part2, .part3
  #  hand one part to each reviewer; each runs:
  python review_splits.py split_plan.part2.csv --dir data
  #  then recombine the reviewed parts into the master plan:
  python review_splits.py --merge              # folds split_plan.part*.csv back in

Usage
-----
  python review_splits.py [PLAN] [--plan split_plan.csv] [--dir data] [--all]
  python review_splits.py --shard N
  python review_splits.py --merge [--parts "split_plan.part*.csv"]
"""

import argparse
import csv
import re
import shutil
import sys
from itertools import islice
from math import ceil
from pathlib import Path

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

FIELDS = ["id", "name", "action", "split_column", "span", "other_candidates", "reviewed"]
SCAN = 15          # rows sampled from each data.csv for the preview
DONE = {"y", "yes", "1", "true"}
PREVIEW_ROWS = 5   # full table rows shown per dataset (override with --rows)
MAX_COL_W = 16     # max width of a previewed cell in the grid
LINE_W = 120       # fallback table width if the terminal size is unknown
DATASET_URL = "https://data.cityofnewyork.us/d/{id}/"


def render_table(header: list[str], rows: list[list[str]]) -> tuple[list[str], list[int]]:
    """
    Compact fixed-width preview of header + rows, clipped to the terminal width.
    Returns (lines, dropped_column_indices). Dropped columns are shown separately.
    """
    line_w = max(40, shutil.get_terminal_size((LINE_W, 24)).columns - 6)  # -6 for indent
    def cell(v: str) -> str:
        v = " ".join(str(v).split())
        return v if len(v) <= MAX_COL_W else v[:MAX_COL_W - 1] + "…"
    ncols = len(header)
    head = [cell(h) for h in header]
    data = [[cell(r[i]) if i < len(r) else "" for i in range(ncols)] for r in rows]
    widths = [max(len(head[i]), *(len(d[i]) for d in data)) if data else len(head[i])
              for i in range(ncols)]
    shown, total = [], 0
    for i in range(ncols):
        if total + widths[i] + 3 > line_w and shown:
            break
        shown.append(i)
        total += widths[i] + 3
    fmt = lambda cells: " | ".join(cells[i].ljust(widths[i]) for i in shown)
    out = [fmt(head), "-" * min(total, line_w)] + [fmt(d) for d in data]
    dropped = [i for i in range(ncols) if i not in shown]
    return out, dropped


def norm_col(name: str) -> str:
    return "_".join(re.findall(r"[a-z0-9]+", (name or "").lower()))


def candidate_columns(row: dict) -> list[str]:
    """Column names offered for this dataset (chosen + others)."""
    cols = [c for c in re.findall(r"([a-z0-9_]+)\[", row.get("other_candidates", ""))]
    chosen = row.get("split_column", "")
    if chosen and chosen not in cols:
        cols.insert(0, chosen)
    return cols


def load_samples(csv_path: Path, columns: list[str],
                 preview_rows: int = PREVIEW_ROWS) -> tuple[list[str], dict, list[list[str]]]:
    """Return (header, {column: [sample values]}, first few full rows) from data.csv."""
    if not csv_path.exists():
        return [], {}, []
    try:
        with csv_path.open(newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            header = next(reader)
            nidx = {norm_col(h): i for i, h in enumerate(header)}
            samples = {c: [] for c in columns}
            rows = []
            for rec in islice(reader, max(SCAN, preview_rows)):
                if len(rows) < preview_rows:
                    rows.append(rec)
                for c in columns:
                    i = nidx.get(c)
                    if i is not None and i < len(rec):
                        v = rec[i].strip()
                        if v and v not in samples[c] and len(samples[c]) < 6:
                            samples[c].append(v)
            return header, samples, rows
    except (OSError, csv.Error, StopIteration):
        return [], {}, []


def folder_index(root: Path) -> dict:
    idx = {}
    for sub in root.iterdir() if root.exists() else []:
        if sub.is_dir() and (sub / "data.csv").exists():
            idx[sub.name.rsplit("__", 1)[-1]] = sub
    return idx


def read_plan(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r.setdefault("reviewed", "")
    return rows


def save(plan_path: Path, rows: list[dict]) -> None:
    tmp = plan_path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(plan_path)


def is_reviewed(row: dict) -> bool:
    return str(row.get("reviewed", "")).strip().lower() in DONE


def shard_plan(plan_path: Path, n: int) -> None:
    """Split the plan into n contiguous part files for parallel review."""
    rows = read_plan(plan_path)
    n = max(1, min(n, len(rows)))
    size = ceil(len(rows) / n)
    made = []
    for k in range(n):
        chunk = rows[k * size:(k + 1) * size]
        if not chunk:
            continue
        part = plan_path.with_name(f"{plan_path.stem}.part{k + 1}.csv")
        save(part, chunk)
        made.append((part, len(chunk)))
    print(f"Split {len(rows)} row(s) into {len(made)} part(s):")
    for part, cnt in made:
        print(f"  {part.name}  ({cnt} rows)")
    print(f"\nEach reviewer:   python review_splits.py {made[0][0].name} --dir <dir>")
    print(f"Then merge back: python review_splits.py --merge --plan {plan_path.name}")


def merge_parts(plan_path: Path, part_glob: str, out_path: Path) -> None:
    """Fold reviewed decisions from the part files back into the master plan."""
    parts = sorted(plan_path.parent.glob(part_glob))
    if not parts:
        sys.exit(f"No part files matched {part_glob!r} in {plan_path.parent}/")
    # start from the master plan (preserves order); overlay decisions from parts
    merged = {}
    order = []
    if plan_path.exists():
        for r in read_plan(plan_path):
            merged[r["id"]] = r
            order.append(r["id"])
    conflicts = 0
    for pp in parts:
        for r in read_plan(pp):
            rid = r["id"]
            if rid not in merged:
                merged[rid] = r
                order.append(rid)
            elif is_reviewed(r):
                cur = merged[rid]
                if is_reviewed(cur) and (cur.get("action"), cur.get("split_column")) != \
                        (r.get("action"), r.get("split_column")):
                    conflicts += 1
                    print(f"  conflict on {rid}: keeping '{cur.get('action')}/"
                          f"{cur.get('split_column')}' over '{r.get('action')}/{r.get('split_column')}'")
                elif not is_reviewed(cur):
                    merged[rid] = r
    save(out_path, [merged[i] for i in order])
    reviewed = sum(1 for i in order if is_reviewed(merged[i]))
    print(f"Merged {len(parts)} part(s) -> {out_path}  "
          f"({reviewed}/{len(order)} reviewed{', ' + str(conflicts) + ' conflict(s)' if conflicts else ''})")


def show(row: dict, n: int, total: int, folder: Path | None, n_rows: int = PREVIEW_ROWS) -> None:
    cols = candidate_columns(row)
    rid = row.get("id", "")
    print("\n" + "=" * 70)
    print(f"[{n}/{total}]  {row.get('name', '')}")
    print(f"          id: {rid}")
    if rid:
        print(f"        link: {DATASET_URL.format(id=rid)}")
    print(f"  split on  : {row.get('split_column', '')}   (span {row.get('span', '?')})")
    others = row.get("other_candidates", "")
    if others:
        print(f"  candidates: {others}")
    if folder:
        header, samples, rows = load_samples(folder / "data.csv", cols, n_rows)
        if samples:
            print("  sample values:")
            for c in cols:
                vals = ", ".join(samples.get(c, [])) or "(none)"
                mark = " <- proposed" if c == row.get("split_column") else ""
                print(f"     {c:<24} {vals}{mark}")
        if header and rows:
            print(f"  first {len(rows)} row(s):")
            grid, dropped = render_table(header, rows)
            for line in grid:
                print(f"     {line}")
            if dropped:
                print(f"  other columns (row 1 of {len(dropped)} not shown above):")
                first = rows[0]
                for i in dropped:
                    v = " ".join(first[i].split()) if i < len(first) else ""
                    if len(v) > 60:
                        v = v[:59] + "…"
                    print(f"     {header[i]}: {v}")
    else:
        print("  (no local data.csv - preview unavailable)")
    print("-" * 70)


def main() -> None:
    ap = argparse.ArgumentParser(description="Interactively review/correct a split plan.")
    ap.add_argument("plan_file", nargs="?", default=None,
                    help="plan file to review (positional; overrides --plan, e.g. a shard)")
    ap.add_argument("--plan", default="split_plan.csv", help="plan file (default split_plan.csv)")
    ap.add_argument("--dir", default="data")
    ap.add_argument("--all", action="store_true", help="review every row, including already-reviewed")
    ap.add_argument("--rows", type=int, default=PREVIEW_ROWS,
                    help=f"full table rows to preview per dataset (default {PREVIEW_ROWS})")
    ap.add_argument("--shard", type=int, metavar="N",
                    help="split the plan into N part files for parallel review, then exit")
    ap.add_argument("--merge", action="store_true",
                    help="fold reviewed part files back into the master plan, then exit")
    ap.add_argument("--parts", default=None,
                    help="glob for --merge (default '<plan-stem>.part*.csv')")
    ap.add_argument("--out", default=None, help="output path for --merge (default: the plan)")
    args = ap.parse_args()

    plan_path = Path(args.plan_file or args.plan)

    if args.shard is not None:
        if not plan_path.exists():
            sys.exit(f"{plan_path} not found - run split_datasets.py first.")
        shard_plan(plan_path, args.shard)
        return
    if args.merge:
        part_glob = args.parts or f"{plan_path.stem}.part*.csv"
        merge_parts(plan_path, part_glob, Path(args.out) if args.out else plan_path)
        return

    if not plan_path.exists():
        sys.exit(f"{plan_path} not found - run split_datasets.py first to create it.")

    rows = read_plan(plan_path)

    folders = folder_index(Path(args.dir))
    todo = [i for i, r in enumerate(rows) if args.all or not is_reviewed(rows[i])]
    done_already = len(rows) - len(todo)
    print(f"{len(rows)} candidate(s); {done_already} already reviewed, {len(todo)} to go.")
    if not todo:
        print("Nothing left to review. (use --all to revisit.)")
        return

    HELP = ("  y = yes, split on the proposed column   s = skip (don't split)\n"
            "  e = edit the split column                a = abort & save (resume later)")
    for pos, i in enumerate(todo, 1):
        row = rows[i]
        show(row, pos, len(todo), folders.get(row.get("id", "")), args.rows)
        while True:
            try:
                choice = input("  [y]es  [s]kip  [e]dit  [a]bort  [?] > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                save(plan_path, rows)
                print(f"\nStopped. Progress saved to {plan_path}.")
                return
            if choice in ("y", "yes"):
                row["action"], row["reviewed"] = "split", "y"
            elif choice in ("s", "skip"):
                row["action"], row["reviewed"] = "skip", "y"
            elif choice in ("e", "edit"):
                cols = candidate_columns(row)
                hint = (", ".join(cols)) if cols else "(type the column name)"
                new = input(f"     new split column [{hint}]: ").strip()
                if not new:
                    print("     unchanged.")
                    continue
                hdr = []
                if folders.get(row.get("id", "")):
                    hdr, _, _ = load_samples(folders[row["id"]] / "data.csv", [])
                if hdr and norm_col(new) not in {norm_col(h) for h in hdr}:
                    if input(f"     '{new}' not found in the data header - use anyway? [y/N] ").strip().lower() not in DONE:
                        continue
                row["split_column"], row["action"], row["reviewed"] = norm_col(new), "split", "y"
                print(f"     -> will split on {row['split_column']}")
            elif choice in ("a", "abort", "q"):
                save(plan_path, rows)
                print(f"\nAborted. Progress saved to {plan_path} - re-run to continue.")
                return
            elif choice in ("?", "h", "help"):
                print(HELP)
                continue
            else:
                print("     (y / s / e / a / ?)")
                continue
            save(plan_path, rows)          # persist after every decision
            break

    remaining = sum(1 for r in rows if str(r.get("reviewed", "")).strip().lower() not in DONE)
    kept = sum(1 for r in rows if r.get("action") == "split")
    print(f"\nDone. {kept} to split, {len(rows) - kept} skipped, {remaining} unreviewed.")
    print(f"Saved to {plan_path}.")


if __name__ == "__main__":
    main()
