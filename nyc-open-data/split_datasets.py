"""
Year-splitter to turn consolidated multi-year tables into per-year tables.

Some datasets internalize time as rows ("311 Service Requests from 2010 to
Present", one table, many years) while others are published as separate yearly
datasets. This normalizes the first kind into the second, so the whole corpus is
uniform ("one table = one period") before clustering/version matching.

What it splits, and on which column, comes from column_profiles.json's
`split_columns` (the coverage/record-time axis, e.g. created_date; NOT an
attribute date like year_built). A dataset is split only when that axis spans
>= --min-years distinct years in the profile sample. Each output slice is a real
per-year dataset folder (data.csv + metadata.json with split provenance), named
"<original name> (<year>)" so the name matcher regroups the slices into one
series via the year qualifier.

Plan-driven (propose -> correct -> apply): the dry run writes an editable
split_plan.csv with the chosen axis per candidate plus the alternatives it
considered. You correct it before applying:
  - action = skip          -> don't split this dataset
  - split_column = <other> -> split on a different column (even one not
                              auto-detected; add your own row to force a split)
--apply then splits exactly what the plan says, so a wrong axis is a one-cell
edit rather than a blocker.

SAFE BY DEFAULT: a dry run that only writes the plan. With --apply it writes the
slices into <dir> and moves each original consolidated folder into
<dir>/_split_originals/ (reversible), leaving <dir> uniform.

Usage
-----
  python split_datasets.py [--dir data] [--profiles column_profiles.json]   # write plan
  # edit split_plan.csv (skip rows / override split_column)
  python split_datasets.py --dir data --apply                               # apply the plan
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import re

from profile_columns import norm_col, year_of

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

# A year RANGE ("2015-2016", "2015/16", "2015 - 2016"), school years, fiscal-year
# ranges. Kept whole as the slice key so a school year isn't split across two
# calendar years. NOTE: the "YYYY-NN" form is only a range when NN is the
# CONSECUTIVE next year (2015-16). It must NOT match a year-month like "2010-01",
# which is a monthly reporting value that has to collapse to its year (2010)  or
# the table gets split per month, not per year.
RANGE_RE = re.compile(r"^((?:19|20|21|22)\d{2})\s*[-/]\s*((?:19|20|21|22)\d{2}|\d{2})$")


def split_key(value: str) -> str | None:
    """Slice key for a cell: a consecutive-year range stays whole (school/fiscal
    year), everything else collapses to its 4-digit calendar year. So monthly
    values like '2010-01' land in the '2010' slice, not their own month slice."""
    v = " ".join(str(value).split())
    m = RANGE_RE.match(v)
    if m:
        y1, g2 = int(m.group(1)), m.group(2)
        if len(g2) == 4:
            y2 = int(g2)
        else:                                    # 2-digit tail: resolve its century
            y2 = (y1 // 100) * 100 + int(g2)
            if y2 <= y1:                         # century rollover, e.g. 1999-00 -> 2000
                y2 += 100
        if y2 == y1 + 1:                         # a real consecutive-year range
            return f"{y1}-{g2}"
        # otherwise it's a year-month (or a wide span)
    return year_of(v)


def safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in " _-" else "_" for c in (name or "")).strip().replace(" ", "_")[:80]


def choose_split_column(split_columns: list) -> dict | None:
    """Pick the coverage axis to split on: prefer report/fiscal-named, then widest span."""
    def score(c):
        try:
            span = int(c["span"][1]) - int(c["span"][0])
        except (ValueError, KeyError, IndexError):
            span = 0
        toks = set(c["column"].split("_"))
        name_pref = 2 if toks & {"report", "reporting", "fiscal", "fy"} else (1 if "year" in toks else 0)
        return (name_pref, span)
    return max(split_columns, key=score) if split_columns else None


def split_one(csv_path: Path, split_col: str, dir_: Path, orig_dirname: str,
              orig_id: str, orig_name: str, orig_meta: dict) -> Counter:
    """
    Stream csv_path once, routing each row to a per-year slice folder under dir_.
    Rows whose split column has no parseable year go to an 'undated' slice (no data
    is dropped). Returns a Counter of rows per year. Writes slices atomically.
    """
    f = csv_path.open(newline="", encoding="utf-8", errors="replace")
    reader = csv.reader(f)
    header = next(reader)
    norm_header = [norm_col(h) for h in header]
    if split_col not in norm_header:
        f.close()
        return Counter()
    idx = norm_header.index(split_col)

    writers = {}        # year -> (file, csv.writer, folder, tmp_path)
    counts = Counter()

    def open_year(year: str):
        folder = dir_ / f"{safe_name(orig_name)}_{year}__{orig_id}_{year}"
        folder.mkdir(parents=True, exist_ok=True)
        tmp = folder / "data.csv.tmp"
        fh = tmp.open("w", newline="", encoding="utf-8")
        w = csv.writer(fh)
        w.writerow(header)
        writers[year] = (fh, w, folder, tmp)
        return writers[year]

    for row in reader:
        y = split_key(row[idx]) if idx < len(row) else None
        y = y or "undated"
        w = writers[y][1] if y in writers else open_year(y)[1]
        w.writerow(row)
        counts[y] += 1

    f.close()
    for year, (fh, _w, folder, tmp) in writers.items():
        fh.close()
        tmp.replace(folder / "data.csv")
        meta = dict(orig_meta)
        meta["id"] = f"{orig_id}_{year}"
        meta["name"] = f"{orig_name} ({year})"
        meta["split_provenance"] = {"parent_id": orig_id, "year": year,
                                    "split_column": split_col, "rows": counts[year]}
        (folder / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description="Split consolidated multi-year tables into per-year tables.")
    ap.add_argument("--dir", default="data")
    ap.add_argument("--profiles", default="column_profiles.json")
    ap.add_argument("--min-years", type=int, default=2,
                    help="only propose a split if the coverage axis spans >= this many distinct years")
    ap.add_argument("--apply", action="store_true", help="apply the plan (default: write/refresh the plan)")
    ap.add_argument("--plan", default="split_plan.csv", help="editable per-dataset split decisions")
    ap.add_argument("--force", action="store_true", help="overwrite an existing plan when (re)proposing")
    ap.add_argument("--quarantine", default=None,
                    help="where to move split originals (default <dir>/_split_originals)")
    ap.add_argument("--manifest", default="split_manifest.json")
    args = ap.parse_args()

    root = Path(args.dir)
    quarantine = Path(args.quarantine) if args.quarantine else root / "_split_originals"
    plan_path = Path(args.plan)
    profiles = {p["id"]: p for p in json.loads(Path(args.profiles).read_text())}

    # id -> dataset folder
    folder_of = {}
    for sub in root.iterdir():
        if sub.is_dir() and (sub / "data.csv").exists():
            folder_of[sub.name.rsplit("__", 1)[-1]] = sub

    def meta_name(did: str) -> tuple[dict, str]:
        try:
            m = json.loads((folder_of[did] / "metadata.json").read_text())
        except (OSError, json.JSONDecodeError):
            m = {"id": did}
        return m, (m.get("name") or folder_of[did].name)

    # auto-detected candidates: a chosen coverage axis spanning >= min-years
    candidates = []
    for did, prof in profiles.items():
        col = choose_split_column(prof.get("split_columns", []))
        if not col or did not in folder_of:
            continue
        try:
            yrs = int(col["span"][1]) - int(col["span"][0]) + 1
        except (ValueError, KeyError, IndexError):
            yrs = 0
        if yrs >= args.min_years:
            candidates.append((did, prof, col, yrs))
    candidates.sort(key=lambda c: -c[3])

    # -------- propose: write the editable plan --------
    if not args.apply:
        print(f"{len(candidates)} dataset(s) qualify to split "
              f"(coverage axis spanning >= {args.min_years} years):\n")
        for did, prof, col, yrs in candidates:
            _, name = meta_name(did)
            print(f"  {did}  on `{col['column']}`  span {col['span'][0]}-{col['span'][1]} "
                  f"(~{yrs} slices)   {name[:50]}")
        if plan_path.exists() and not args.force:
            print(f"\n{plan_path} already exists - edit it, then run --apply "
                  f"(or --force to regenerate and lose edits).")
            return
        with open(plan_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["id", "name", "action", "split_column",
                                              "span", "other_candidates"])
            w.writeheader()
            for did, prof, col, yrs in candidates:
                _, name = meta_name(did)
                others = "; ".join(f"{c['column']}[{c['span'][0]}-{c['span'][1]}]"
                                   for c in prof.get("split_columns", []))
                w.writerow({"id": did, "name": name, "action": "split",
                            "split_column": col["column"],
                            "span": f"{col['span'][0]}-{col['span'][1]}",
                            "other_candidates": others})
        print(f"\nWrote {plan_path}. Review it (set action=skip to exclude, edit "
              f"split_column to override), then run --apply.")
        return

    # -------- apply: from the plan if present, else auto --------
    if plan_path.exists():
        decisions = []
        for r in csv.DictReader(open(plan_path, newline="")):
            if str(r.get("action", "")).strip().lower() in {"skip", "no", "n", "0", ""}:
                continue
            col = norm_col(r.get("split_column", ""))
            if col and r["id"] in folder_of:
                decisions.append((r["id"], col))
        print(f"Applying {plan_path}: {len(decisions)} dataset(s) to split")
    else:
        decisions = [(did, col["column"]) for did, _p, col, _y in candidates]
        print(f"No {plan_path}; using {len(decisions)} auto-detected axis(es). "
              f"(Run without --apply first to review a plan.)")

    manifest = {}
    for did, split_col in decisions:
        folder = folder_of[did]
        meta, name = meta_name(did)
        counts = split_one(folder / "data.csv", split_col, root, folder.name, did, name, meta)
        if not counts:
            print(f"  {did}: column `{split_col}` not found / no rows - skipped")
            continue
        quarantine.mkdir(exist_ok=True)
        shutil.move(str(folder), str(quarantine / folder.name))
        manifest[did] = {"split_column": split_col,
                         "slices": {y: n for y, n in sorted(counts.items())}}
        yrs_str = ", ".join(f"{y}:{n}" for y, n in sorted(counts.items()))
        print(f"  {did} on `{split_col}` -> {len(counts)} slices, original quarantined  [{yrs_str}]")

    Path(args.manifest).write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\nSplit {len(manifest)} dataset(s); originals in {quarantine}. Manifest -> {args.manifest}")
    print("Re-run profile_columns / cluster_datasets / version_match on the now-uniform corpus.")


if __name__ == "__main__":
    main()
