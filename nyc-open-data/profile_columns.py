"""
Column profiler for NYC Open Data — one scan over each dataset's data, serving
two purposes from the same pass:

  1. PARTITION GUARD (now). A column that is *constant* within a dataset pins
     that dataset to one slice of some dimension. We split those constants:
       - temporal constant (value is a year/FY/date) -> the VERSION axis, pinned
         to one period; this is NOT a partition (two yearly editions SHOULD
         merge), so it is recorded separately and never blocks a merge.
       - categorical constant (borough, class, program, …) -> a PARTITION marker.
         Two datasets that share such a column with DIFFERENT values are
         different tables, not versions — version_match.py uses this to block a
         merge, catching partitions invisible in the name, metadata and agency
         (e.g. a table silently pre-filtered to Class = "Condominium").

  2. SPLIT AXIS (later). A temporal column (date/year) whose value VARIES across
     rows is the axis a multi-year table will be split on. We report it with its
     span, so the planned year-splitter can consume the same profile.

The two are complementary: the qualifier logic that decides "this constant is
temporal (version axis) vs categorical (partition)" is exactly what tells the
future splitter which column is safe to split BY (split on year, never on
borough). One scan feeds both.

Emits column_profiles.json. Consumed by version_match.py via --profiles.

Usage
-----
  python profile_columns.py [--dir data] [--scan-cap 20000] [--jobs 0]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from cluster_datasets import YEAR_RE

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

# A value is temporal if it is a year/FY (YEAR_RE) or a date.
DATE_RE = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}([ t].*)?$|^\d{1,2}/\d{1,2}/\d{2,4}.*$")
TEMPORAL_TYPES = {"calendar_date", "date"}

# A split axis must be a COVERAGE/record-time column (when the row was reported),
# NOT an entity-attribute date. We tell them apart by name tokens plus, decisively,
# publication alignment: a coverage column reaches up to the dataset's own update
# year, whereas an attribute like `year_built` (1900-2007) does not.
COVERAGE_TOKENS = {"report", "reported", "reporting", "fiscal", "fy", "year", "yr",
                   "date", "month", "quarter", "period", "calendar", "as", "of"}
ATTRIBUTE_TOKENS = {"built", "birth", "dob", "vintage", "expiration", "expire",
                    "expires", "effective", "maturity", "demolished", "renovated",
                    "founded", "established", "acquired"}
ALIGN_SLACK = 3        # coverage end-year may lag the update year by a few years
MAX_SPAN = 30          # a coverage axis spans the data era, not a century
MIN_YEAR = 1990        # ... and starts in the platform era, not the deep past

# Number of byte offsets to sample from, so a file physically sorted by a
# partition column (e.g. all-Brooklyn rows first) doesn't read as "constant".
SAMPLE_POINTS = 5
# Min non-blank values before a single-valued column counts as a real constant
# (filters near-empty/degenerate columns from acting as partition markers).
MIN_CONST_ROWS = 10


def norm_col(name: str) -> str:
    """Stable cross-dataset column key: lowercase, alphanumerics joined by '_'."""
    return "_".join(re.findall(r"[a-z0-9]+", (name or "").lower()))


def is_temporal_value(v: str) -> bool:
    v = v.strip().lower()
    if not v:
        return False
    return bool(YEAR_RE.match(v) or re.fullmatch(r"(19|20|21|22)\d{2}", v)
                or DATE_RE.match(v))


def year_of(v: str):
    m = re.search(r"(19|20|21|22)\d{2}", v)
    return m.group(0) if m else None


def sample_rows(path: Path, cap: int):
    """
    Yield up to `cap` rows spread across SAMPLE_POINTS byte offsets (plus the
    header from the top), so constants are judged over the whole file rather
    than a possibly-sorted head. Header is returned first.
    """
    size = path.stat().st_size
    f = path.open(newline="", encoding="utf-8", errors="replace")
    reader = csv.reader(f)
    try:
        header = next(reader)
    except StopIteration:
        f.close()
        return
    yield "HEADER", header
    per = max(1, cap // SAMPLE_POINTS)
    for k in range(SAMPLE_POINTS):
        off = int(size * k / SAMPLE_POINTS)
        if off:
            f.seek(off)
            f.readline()                      # resync to the next full line
        r = csv.reader(f)
        for i, row in enumerate(r):
            if i >= per:
                break
            yield "ROW", row
    f.close()


def col_types_from_metadata(md_path: Path) -> tuple[dict, int | None]:
    """
    (norm_col -> dataTypeName, reference_year) from the dataset's metadata.json.
    reference_year is the data's last-update year (rowsUpdatedAt, else
    viewLastModified/createdAt) — used to test coverage-column alignment.
    """
    try:
        m = json.loads(md_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}, None
    out = {}
    for c in m.get("columns", []):
        key = norm_col(c.get("fieldName") or c.get("name") or "")
        if key:
            out[key] = (c.get("dataTypeName") or "").lower()
    ref_year = None
    for k in ("rowsUpdatedAt", "viewLastModified", "createdAt"):
        v = m.get(k)
        if isinstance(v, (int, float)) and v > 0:
            ref_year = datetime.fromtimestamp(v, timezone.utc).year
            break
    return out, ref_year


def profile_one(args_tuple):
    """Profile a single dataset directory. Runs in a worker process."""
    subdir, scan_cap = args_tuple
    sub = Path(subdir)
    csv_path = sub / "data.csv"
    if not csv_path.exists():
        return None
    did = sub.name.rsplit("__", 1)[-1]
    types, ref_year = col_types_from_metadata(sub / "metadata.json")

    header = None
    vals = []            # per column: distinct values, capped
    n_nonblank = []      # per column: count of non-blank values
    years = []           # per column: set of years seen
    CAP = 16
    try:
        for kind, row in sample_rows(csv_path, scan_cap):
            if kind == "HEADER":
                header = [norm_col(h) for h in row]
                n = len(header)
                vals = [set() for _ in range(n)]
                n_nonblank = [0] * n
                years = [set() for _ in range(n)]
                continue
            for ci in range(min(len(header), len(row))):
                v = " ".join(row[ci].split())
                if not v:
                    continue
                n_nonblank[ci] += 1
                if len(vals[ci]) < CAP:
                    vals[ci].add(v)
                if is_temporal_value(v):
                    y = year_of(v)
                    if y:
                        years[ci].add(y)
    except (OSError, csv.Error):
        return None
    if not header:
        return None

    partition_constants = {}     # categorical constant -> value (a partition)
    temporal_constants = {}      # temporal constant -> pinned period
    split_columns = []           # varying temporal column -> the split axis
    for ci, col in enumerate(header):
        if not col or n_nonblank[ci] == 0:
            continue
        distinct = len(vals[ci])
        toks = set(col.split("_"))
        # A coverage (record-time) column: a date type or a coverage-named column,
        # and NOT an entity-attribute date (year_built, date_of_birth, ...).
        is_coverage = ((types.get(col, "") in TEMPORAL_TYPES or (toks & COVERAGE_TOKENS))
                       and not (toks & ATTRIBUTE_TOKENS))
        if distinct == 1 and n_nonblank[ci] >= MIN_CONST_ROWS:
            only = next(iter(vals[ci]))
            if is_coverage or is_temporal_value(only):
                temporal_constants[col] = only        # version axis, pinned
            else:
                partition_constants[col] = only.lower()  # partition marker
        elif is_coverage and years[ci]:
            ys = sorted(int(y) for y in years[ci])
            # Publication alignment: the axis must reach up to the data's update
            # year, span the data era (not a century), and start in that era.
            # This is what rejects `year_built` (1900-2007) as a split axis.
            aligned = ref_year is None or ys[-1] >= ref_year - ALIGN_SLACK
            if aligned and ys[-1] - ys[0] <= MAX_SPAN and ys[0] >= MIN_YEAR:
                split_columns.append({"column": col, "span": [str(ys[0]), str(ys[-1])],
                                      "type": types.get(col, "")})

    return {
        "id": did,
        "partition_constants": partition_constants,
        "temporal_constants": temporal_constants,
        "split_columns": split_columns,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Profile dataset columns for partition guards + split axes.")
    ap.add_argument("--dir", default="data", help="directory of *__<id>/data.csv folders")
    ap.add_argument("--scan-cap", type=int, default=20_000,
                    help="rows sampled per dataset (spread across the file)")
    ap.add_argument("--jobs", type=int, default=0, help="worker processes (0 = all cores)")
    ap.add_argument("--out", default="column_profiles.json")
    args = ap.parse_args()

    subdirs = [str(p) for p in Path(args.dir).glob("*__*") if (p / "data.csv").exists()]
    print(f"profiling {len(subdirs)} datasets with data.csv")
    jobs = args.jobs if args.jobs and args.jobs > 0 else (os.cpu_count() or 1)

    profiles = []
    tasks = [(s, args.scan_cap) for s in subdirs]
    if jobs > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            for r in ex.map(profile_one, tasks, chunksize=8):
                if r:
                    profiles.append(r)
    else:
        for t in tasks:
            r = profile_one(t)
            if r:
                profiles.append(r)

    with_part = sum(1 for p in profiles if p["partition_constants"])
    with_split = sum(1 for p in profiles if p["split_columns"])
    Path(args.out).write_text(json.dumps(profiles, indent=2, ensure_ascii=False))
    print(f"profiled {len(profiles)} datasets: "
          f"{with_part} have categorical partition constant(s), "
          f"{with_split} have a temporal split axis")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
