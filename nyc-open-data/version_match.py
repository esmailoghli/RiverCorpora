"""
Version matching for NYC Open Data (Stage 2): group temporal versions and
re-publications of the *same logical table* into one series.

Pure content matching (column value-overlap) badly overmerges here.
The reliable discriminator is the NAME: the borough/product tokens separate some series, 
and the year/FY tokens (version qualifiers) mark the versions within a series.

So this stage groups by BASE NAME (the dataset name with version qualifiers
(years, FY, quarters) removed) using IDF-weighted token similarity and
average-linkage (chain-resistant). The within-block similarity is bimodal
(same series ca. 1.0, different borough/product ca. 0.7), so the cut is found
automatically with 1-D k-means. Members of a series differ only in their
qualifiers, which enumerate the versions/periods covered.

For versions whose *names* genuinely changed, name similarity can't help, but
publishers often state the relationship themselves. A dataset's description
frequently links to its own past editions.

These are author-asserted version links (the highest-precision signal we have),
and free (metadata only, no data download). After name grouping, an optional
EXPLICIT-REFERENCE pass parses descriptions for in-corpus dataset IDs that sit
near a version cue (historical/previous/FY/year/...) and unions the linked
series. It is union-only (can only recover missed versions, never split a good
group) and still obeys the geography guard, so a stray cross-borough link can't
merge Brooklyn into Manhattan. This recovers exactly the reworded-title case a
content/entity bridge tried and failed to handle. On by default; --no-links off.

Usage
-----
  python version_match.py [--dir .] [options]
"""

import argparse
import json
import os
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from cluster_datasets import (
    tokenize_name, compute_idf, weighted_overlap, dynamic_clusters,
    kmeans_threshold, UnionFind, singularize,
)

# Partition-dimension tokens: a dataset published once per geography is a
# DIFFERENT logical table per geography, not a version. Two names whose geography
# tokens differ are never the same series, regardless of overall name similarity.
# (This is the inverse of a qualifier: qualifiers must be ignored, partition
# tokens must agree.) Prevents single-linkage from chaining boroughs together via
# a low-IDF hub like "citywide". Run through singularize() so they match the
# (singularized) key tokens
GEO_TOKENS = frozenset(singularize(t) for t in {
    "bronx", "brooklyn", "manhattan", "queens", "staten", "citywide",
})


def geo_compatible(key_a: frozenset, key_b: frozenset) -> bool:
    """False if both names carry geography tokens and they disagree."""
    ga, gb = key_a & GEO_TOKENS, key_b & GEO_TOKENS
    return not (ga and gb and ga != gb)


# Reporting-cadence partition tokens: a monthly table and a quarterly table are
# different tables, never versions of each other. Same logic as geography.
# (Singularize-neutral: these don't end in plural -s.)
FREQ_TOKENS = frozenset({
    "monthly", "quarterly", "annual", "annually", "yearly", "weekly", "daily",
    "biweekly", "biannual", "semiannual",
})


def partition_compatible(key_a: frozenset, key_b: frozenset) -> bool:
    """Geography AND reporting cadence must agree to be the same logical table."""
    if not geo_compatible(key_a, key_b):
        return False
    fa, fb = key_a & FREQ_TOKENS, key_b & FREQ_TOKENS
    return not (fa and fb and fa != fb)


def agency_geo(agency: str) -> frozenset:
    """
    Borough/geography tokens hidden in the *agency* string rather than the name,
    e.g. "Brooklyn Borough President" vs "Manhattan Borough President" both publish
    a dataset literally named "Tourism Grants". Folding these into the geo guard
    keeps such per-office datasets from being mistaken for versions of each other.
    Extracted from the raw agency (not the normalized acronym, which drops the
    borough word). Singularized to match GEO_TOKENS (queens -> queen).
    """
    toks = {singularize(t) for t in re.findall(r"[a-z]+", (agency or "").lower())}
    return frozenset(toks & GEO_TOKENS)


# ---------------------------------------------------------------------------
# Explicit-reference pass: publisher-asserted version links in descriptions.
# ---------------------------------------------------------------------------

# Socrata "4x4" dataset id (e.g. erm2-nwe9). The pattern also matches year ranges
# like 2008-2009, so every hit is validated against the corpus id set before use.
SOCRATA_ID_RE = re.compile(r"\b([a-z0-9]{4}-[a-z0-9]{4})\b", re.I)

# A version cue near the link distinguishes "for historical data, see X" (a real
# version) from "see also our companion table Y". Matches words like historical/
# previous/archived/superseded/FY, plus any explicit year (for 2017, ...).
LINK_CUE_RE = re.compile(
    r"histor|previous|prior|former|archiv|supersed|replac|older|earlier|"
    r"legacy|deprecat|\bpast\b|\bup\s+to\b|\bthrough\b|\bfy\b|\b(?:19|20)\d{2}\b",
    re.I,
)


def load_descriptions(root: Path, ids: set) -> dict:
    """Map in-corpus dataset id -> its description text, read from */metadata.json."""
    desc = {}
    for md in root.glob("*__*/metadata.json"):
        try:
            m = json.loads(md.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        did = m.get("id") or md.parent.name.rsplit("__", 1)[-1]
        if did in ids:
            desc[did] = m.get("description") or ""
    return desc


def extract_link_edges(desc_by_id: dict, id_set: set, require_cue: bool = True):
    """
    Find publisher-asserted cross-references: for each dataset, every link in its
    description to *another in-corpus dataset*, optionally gated by a version cue
    within ~100 chars before the link. Returns (src_id, tgt_id, snippet) tuples.
    """
    edges = []
    for src, desc in desc_by_id.items():
        if not desc:
            continue
        for m in SOCRATA_ID_RE.finditer(desc):
            tgt = m.group(1).lower()
            if tgt == src or tgt not in id_set:
                continue
            if require_cue and not LINK_CUE_RE.search(desc[max(0, m.start() - 100):m.end()]):
                continue
            snippet = " ".join(desc[max(0, m.start() - 60):m.end() + 8].split())
            edges.append((src, tgt, snippet))
    return edges


# ---------------------------------------------------------------------------
# Parallel workers module-level so they pickle for ProcessPoolExecutor.
# Candidate scoring is sharded by token/cluster posting; block refinement is
# sharded by block (both are embarrassingly parallel).
# ---------------------------------------------------------------------------

_KEYS = _IDF = None
_MIN_SHARED = 2
_FLOOR = 0.0


def _cand_init(keys, idf, min_shared, floor):
    global _KEYS, _IDF, _MIN_SHARED, _FLOOR
    _KEYS, _IDF, _MIN_SHARED, _FLOOR = keys, idf, min_shared, floor


def _score_postings(postings_list):
    """Score every intra-posting pair in a chunk; keep only sim >= floor."""
    out = []
    seen = set()
    for members in postings_list:
        m = len(members)
        for x in range(m):
            i = members[x]
            ki = _KEYS[i]
            for y in range(x + 1, m):
                j = members[y]
                pair = (i, j) if i < j else (j, i)
                if pair in seen:
                    continue
                seen.add(pair)
                kj = _KEYS[j]
                if len(ki & kj) < _MIN_SHARED:
                    continue
                s = weighted_overlap(ki, kj, _IDF)
                if s >= _FLOOR:
                    out.append((i, j, s))
    return out


_RFLOOR = _RCUT = 0.0


def _refine_init(floor, cut):
    global _RFLOOR, _RCUT
    _RFLOOR, _RCUT = floor, cut


def _refine_block(task):
    """Average-linkage refine one block into series (runs in a worker)."""
    members, sim_items = task
    d = {frozenset((i, j)): s for i, j, s in sim_items}
    return dynamic_clusters(members, lambda i, j: d.get(frozenset((i, j)), 0.0),
                            _RFLOOR, _RCUT)


def load_from_clusters(path: Path) -> list[dict]:
    """
    Read the datasets from the Stage-1 cluster output (clusters.json), keeping
    each dataset's name-cluster index so within-cluster co-membership can serve
    as one candidate source.
    """
    clusters = json.loads(path.read_text())
    out = []
    for ci, c in enumerate(clusters):
        for m in c["members"]:
            key, qual = tokenize_name(m["name"])
            out.append({"id": m["id"], "name": m["name"],
                        "key": key, "qual": qual, "cluster": ci,
                        "ageo": agency_geo(m.get("agency", ""))})
    return out


def load_from_metadata(root: Path) -> list[dict]:
    """Fallback: read datasets straight from each */metadata.json (no clusters)."""
    out = []
    for md in sorted(root.glob("*__*/metadata.json")):
        try:
            m = json.loads(md.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        name = m.get("name") or md.parent.name
        key, qual = tokenize_name(name)
        cf = (m.get("metadata") or {}).get("custom_fields") or {}
        agency = (cf.get("Dataset Information") or {}).get("Agency") or m.get("attribution") or ""
        out.append({"id": m.get("id") or md.parent.name.rsplit("__", 1)[-1],
                    "name": name, "key": key, "qual": qual, "cluster": -1,
                    "ageo": agency_geo(agency)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Group temporal-version series by base name.")
    ap.add_argument("--clusters", default="clusters.json",
                    help="Stage-1 name-cluster output to consume (default clusters.json); "
                         "its within-cluster pairs are an extra candidate source")
    ap.add_argument("--dir", default=".",
                    help="fallback: read */metadata.json directly if --clusters is missing")
    ap.add_argument("--threshold", type=float, default=0.80,
                    help="base-name similarity cut: a series is tables with a "
                         "(near-)identical base name (default 0.80, strict)")
    ap.add_argument("--auto", action="store_true",
                    help="derive the cut with k-means instead of the strict default "
                         "(targets the dominant split; usually too loose for versions)")
    ap.add_argument("--floor", type=float, default=0.30,
                    help="safety min base-name similarity for a merge (default 0.30)")
    ap.add_argument("--df-cap", type=int, default=300,
                    help="skip tokens occurring in more than this many datasets for "
                         "candidate generation; scale-stable, bounds pairs per token "
                         "(default 300). Within-cluster co-membership still covers "
                         "common-token groups, so nothing is lost.")
    ap.add_argument("--min-shared", type=int, default=2,
                    help="min shared key tokens for two datasets to be candidates")
    ap.add_argument("--cross-cluster", action="store_true",
                    help="also generate candidates across name clusters via shared "
                         "key tokens (catches versions Stage-1 split apart, but scans "
                         "the whole corpus; off by default - within-cluster is enough "
                         "when Stage-1 grouped the versions together)")
    ap.add_argument("--linkage", choices=["single", "average"], default="single",
                    help="single = connected components at the cut (recall-first: "
                         "merges any pair >= cut; the strict cut prevents chaining); "
                         "average = average-linkage (precision, but splits renamed "
                         "versions whose group means dip below the cut)")
    ap.add_argument("--jobs", type=int, default=0,
                    help="parallel worker processes for scoring and refinement "
                         "(0 = all CPU cores)")
    ap.add_argument("--no-links", action="store_true",
                    help="disable the explicit-reference pass (publisher-asserted "
                         "version links in dataset descriptions); on by default")
    ap.add_argument("--link-no-cue", action="store_true",
                    help="union every in-corpus description link, even without a "
                         "nearby version cue (more recall, less precision)")
    ap.add_argument("--link-loose", action="store_true",
                    help="drop the same-period and shared-concept guards on link "
                         "merges (keep only the partition guard); merges sibling "
                         "and cross-reference links too - more recall, less precision")
    ap.add_argument("--link-report", default="version_links.txt",
                    help="where to write the explicit-reference merge log")
    ap.add_argument("--profiles", default="column_profiles.json",
                    help="optional column_profiles.json (from profile_columns.py): "
                         "data-confirmed partition guard - blocks a merge when two "
                         "datasets share a constant column with different categorical "
                         "values (borough/class/program). Ignored if missing.")
    ap.add_argument("--out", default="version_groups.json")
    ap.add_argument("--report", default="version_review.txt")
    args = ap.parse_args()

    clusters_path = Path(args.clusters)
    if clusters_path.exists():
        datasets = load_from_clusters(clusters_path)
        print(f"Consuming {clusters_path} ({len(datasets)} datasets)")
    else:
        datasets = load_from_metadata(Path(args.dir))
        print(f"No {clusters_path}; read {len(datasets)} datasets from {args.dir}/")
    n = len(datasets)
    if n == 0:
        print("No datasets found.")
        return
    idf = compute_idf([d["key"] for d in datasets])

    # Optional data-confirmed partition constants (from profile_columns.py):
    # pconst[i] = {column: categorical_constant_value} for dataset i.
    pconst = [{} for _ in range(n)]
    prof_path = Path(args.profiles)
    if prof_path.exists():
        by_id = {p["id"]: p.get("partition_constants", {})
                 for p in json.loads(prof_path.read_text())}
        for i, d in enumerate(datasets):
            pconst[i] = by_id.get(d["id"], {})
        covered = sum(1 for c in pconst if c)
        print(f"Loaded {prof_path}: partition constants for {covered}/{n} datasets")

    def profiles_ok(i: int, j: int) -> bool:
        """False if a column constant in BOTH datasets holds different values."""
        ca, cb = pconst[i], pconst[j]
        if not ca or not cb:
            return True
        for col, va in ca.items():
            vb = cb.get(col)
            if vb is not None and vb != va:
                return False
        return True

    jobs = args.jobs if args.jobs and args.jobs > 0 else (os.cpu_count() or 1)
    keys = [d["key"] for d in datasets]

    # --- Candidate postings (the blocks within which pairs are enumerated) ---
    postings = []
    # Source 1 (default): co-membership in a Stage-1 name cluster.
    by_cluster = defaultdict(list)
    for i, d in enumerate(datasets):
        if d["cluster"] >= 0:
            by_cluster[d["cluster"]].append(i)
    postings.extend(m for m in by_cluster.values() if len(m) > 1)
    # Source 2 (opt-in): shared key tokens across the whole corpus.
    if args.cross_cluster:
        index = defaultdict(list)
        for i, d in enumerate(datasets):
            for tok in d["key"]:
                index[tok].append(i)
        postings.extend(m for m in index.values() if 1 < len(m) <= args.df_cap)

    # --- Score candidate pairs in parallel, keeping only sim >= floor ---
    scored = []
    if jobs > 1 and len(postings) > 1:
        postings.sort(key=len, reverse=True)          # balance: big blocks first
        chunks = [[] for _ in range(jobs)]
        for idx, p in enumerate(postings):
            chunks[idx % jobs].append(p)
        chunks = [c for c in chunks if c]
        with ProcessPoolExecutor(max_workers=jobs, initializer=_cand_init,
                                 initargs=(keys, idf, args.min_shared, args.floor)) as ex:
            for res in ex.map(_score_postings, chunks):
                scored.extend(res)
    else:
        _cand_init(keys, idf, args.min_shared, args.floor)
        scored = _score_postings(postings)

    uf = UnionFind(n)
    pair_sim = {}
    for i, j, s in scored:
        fs = frozenset((i, j))
        if fs not in pair_sim:
            pair_sim[fs] = s
            uf.union(i, j)
    print(f"scored {len(pair_sim)} candidate pairs (>= floor {args.floor}) "
          f"on {jobs} worker(s)")

    # --- Cut: strict by default (near-identical base name); k-means is opt-in ---
    if args.auto:
        cut = kmeans_threshold(list(pair_sim.values()), 0.80, " (version)")
        print(f"base-name cut: {cut:.3f} (auto/k-means)")
    else:
        cut = args.threshold
        print(f"base-name cut: {cut:.3f} (strict default)")

    # Drop merges that cross a geography partition (Brooklyn vs Manhattan vs
    # Citywide are different tables, not versions). This also severs the
    # single-linkage chain that a low-IDF hub like "citywide" would otherwise form.
    def ok(i, j):
        a, b = datasets[i], datasets[j]
        return (geo_compatible(a["key"] | a["ageo"], b["key"] | b["ageo"])
                and profiles_ok(i, j))

    # --- Group into series ---
    if args.linkage == "single":
        # Single-linkage at the cut: merge any pair scoring >= cut. Recall-first;
        # the strict cut keeps it from chaining unrelated tables into a blob.
        suf = UnionFind(n)
        for fs, s in pair_sim.items():
            if s >= cut:
                i, j = tuple(fs)
                if ok(i, j):
                    suf.union(i, j)
        series = list(suf.components().values())
    else:
        # Average-linkage per block (precision; may split renamed versions whose
        # group mean dips below the cut). Parallel over independent blocks.
        comp = uf.components()
        block_sims = defaultdict(list)
        for fs, s in pair_sim.items():
            i, j = tuple(fs)
            if not ok(i, j):
                continue
            block_sims[uf.find(i)].append((i, j, s))
        series, tasks = [], []
        for root, members in comp.items():
            if len(members) == 1:
                series.append(members)
            else:
                tasks.append((members, block_sims.get(root, [])))
        if jobs > 1 and len(tasks) > 1:
            with ProcessPoolExecutor(max_workers=jobs, initializer=_refine_init,
                                     initargs=(args.floor, cut)) as ex:
                for sub in ex.map(_refine_block, tasks):
                    series.extend(sub)
        else:
            _refine_init(args.floor, cut)
            for t in tasks:
                series.extend(_refine_block(t))

    # --- Explicit-reference pass: union series that link to one another in their
    # descriptions (publisher-asserted versions, e.g. "for historical data see X").
    # Union-only and geo-guarded; recovers reworded titles name matching misses.
    if not args.no_links:
        id_to_idx = {d["id"]: i for i, d in enumerate(datasets)}
        desc_by_id = load_descriptions(Path(args.dir), set(id_to_idx))
        edges = extract_link_edges(desc_by_id, set(id_to_idx),
                                   require_cue=not args.link_no_cue)
        luf = UnionFind(n)
        for s in series:
            for k in range(1, len(s)):
                luf.union(s[0], s[k])
        bridged = []
        rej = defaultdict(int)
        for src, tgt, snip in edges:
            i, j = id_to_idx[src], id_to_idx[tgt]
            if luf.find(i) == luf.find(j):
                continue                                      # already one series
            ki, kj = datasets[i]["key"], datasets[j]["key"]
            qi, qj = datasets[i]["qual"], datasets[j]["qual"]
            # Partition tokens (geography, cadence) may land in the name (key or
            # qual) or in the agency string, so test across all three.
            if not partition_compatible(ki | qi | datasets[i]["ageo"],
                                        kj | qj | datasets[j]["ageo"]):
                rej["partition (geo/cadence)"] += 1           # different borough/cadence
                continue
            if not profiles_ok(i, j):
                rej["partition (data constant)"] += 1         # different borough/class in data
                continue
            if not args.link_loose:
                # A version covers a *different* time period: identical time
                # qualifiers (years/FY, ignoring cadence) mean contemporaneous
                # sibling tables, not versions (Mobility Household vs Person 2024).
                ti, tj = qi - FREQ_TOKENS, qj - FREQ_TOKENS
                if ti and tj and ti == tj:
                    rej["same period (sibling)"] += 1
                    continue
                # A version is about the *same thing*: with no shared concept
                # token the link is an unrelated cross-reference, not a version
                # (e.g. Solid Waste Inspections -> 311 Service Requests).
                if not (ki & kj):
                    rej["no shared concept"] += 1
                    continue
            luf.union(i, j)
            bridged.append((datasets[i]["name"], datasets[j]["name"], snip))
        series = list(luf.components().values())
        write_link_report(bridged, Path(args.link_report))
        rej_str = ", ".join(f"{n} {r}" for r, n in sorted(rej.items())) or "none"
        print(f"explicit-reference pass: {len(edges)} in-corpus link(s) "
              f"({'no-cue' if args.link_no_cue else 'cue-gated'}), "
              f"{len(bridged)} merge(s); rejected: {rej_str}")

    series.sort(key=lambda s: (-len(s), datasets[s[0]]["name"]))
    multi = [s for s in series if len(s) > 1]

    out = []
    for s in series:
        members = sorted(s, key=lambda i: datasets[i]["name"])
        out.append({
            "size": len(s),
            "members": [
                {"id": datasets[i]["id"], "name": datasets[i]["name"],
                 "version": sorted(datasets[i]["qual"])}
                for i in members
            ],
        })
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    write_report(out, Path(args.report))

    print(f"{len(series)} series ({len(multi)} multi-version, "
          f"{len(series) - len(multi)} singletons) over {n} datasets")
    print(f"Wrote {args.out}, {args.report}")


def write_link_report(bridged: list, path: Path) -> None:
    lines = ["=" * 78,
             "EXPLICIT-REFERENCE MERGES (series joined via description cross-links)",
             "=" * 78, ""]
    if not bridged:
        lines.append("  (no in-corpus version links found)")
    for a, b, snip in bridged:
        lines.append(f"  {a}")
        lines.append(f"    -> {b}")
        lines.append(f'       "...{snip}"')
        lines.append("")
    path.write_text("\n".join(lines))


def write_report(series: list[dict], path: Path) -> None:
    lines = ["=" * 78, "TEMPORAL-VERSION SERIES (grouped by base name)", "=" * 78, ""]
    for n, s in enumerate([s for s in series if s["size"] > 1], 1):
        lines.append(f"[{n}] {s['size']} versions")
        for m in s["members"]:
            v = f"  [{', '.join(m['version'])}]" if m["version"] else ""
            lines.append(f"       - {m['name']} ({m['id']}){v}")
        lines.append("")
    singles = [s for s in series if s["size"] == 1]
    if singles:
        lines.append("-" * 78)
        lines.append(f"SINGLE-VERSION (no other version found): {len(singles)}")
        lines.append("-" * 78)
        for s in singles:
            lines.append(f"  - {s['members'][0]['name']} ({s['members'][0]['id']})")
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
