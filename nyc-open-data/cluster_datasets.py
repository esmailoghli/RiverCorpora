"""
Cluster NYC Open Data datasets by name (Stage 1).

Name token blocking over the scraped `*/metadata.json` files:
  Normalize and tokenize each dataset name and pull out version qualifiers
  (years, "historical", "fy2022", ...) so that e.g. "311 Service Requests for
  2004" and "311 Service Requests for 2005" land in the same block. Generic
  tokens are NOT removed by a stopword list; instead candidate pairs are
  generated through an inverted index (real blocking with a document-frequency 
  cap that excludes ubiquitous tokens) and scored with IDF-weighted token overlap, 
  The delivering agency is used only as a boost: it can strengthen a link
  between datasets that already share name tokens, but never creates a
  link on its own.

  Candidate pairs (sharing >= --min-shared-tokens content tokens) form loose
  blocks, which are then refined by UPGMA average-linkage clustering: two
  groups merge only while their MEAN cross-similarity clears the cut. This 
  resists "chaining", where one bridging pair would fuse unrelated datasets.

Usage
-----
  python cluster_datasets.py [--dir .] [options]

Options
-------
  --dir DIR              directory holding the <name>__<id>/ folders (default: .)
  --name-threshold F     OVERRIDE the auto name-similarity cut with a fixed value
  --dynamic-cut          per-block largest-gap cut instead of a fixed/auto cut
  --name-floor F         safety min name similarity for a merge (default: 0.10)
  --min-shared-tokens N  min content tokens two datasets must share to link
  --agency-bonus F       similarity added when two candidates share an agency
  --out FILE             JSON output path (default: clusters.json)
  --report FILE          human-readable report path (default: clusters.txt)

By default the name-similarity cut is derived from the data by clustering the
candidate scores into two groups (k-means, k=2) and cutting at the midpoint
between them, applied as a fixed average-linkage cut.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Vocabulary: what counts as noise vs. a version qualifier
# ---------------------------------------------------------------------------

# Generic tokens dropped from the name blocking key entirely.
# Tokens that indicate a *version* of an otherwise-shared series. Stripped
# from the blocking key but recorded so we can describe how members differ.
QUALIFIER_WORDS = {
    "historical", "historic", "current", "present", "active", "inactive",
    "archived", "archive", "legacy", "annual", "monthly", "weekly", "daily",
    "quarterly", "yearly", "fiscal", "year", "fy", "ytd",
    "q1", "q2", "q3", "q4",
    # Month names (and common abbreviations) are temporal markers. They
    # distinguish versions of the same table, not different tables.
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct",
    "nov", "dec",
}

# Closed-class function words (articles, prepositions, conjunctions). Unlike
# generic *content* words (data, report, nyc), these carry no entity meaning, 
# and IDF actively misweights the rare ones.
FUNCTION_WORDS = {
    "a", "an", "the", "of", "for", "from", "to", "in", "on", "at", "by",
    "and", "or", "with", "as", "per", "into",
}

YEAR_RE = re.compile(r"^(?:fy)?(?:19|20|21|22)\d{2}$")   # 2004, fy2022, and
# projection years/decades up to the 2200s (e.g. 2080s, 2100s sea-level data)
TOKEN_RE = re.compile(r"[a-z0-9]+")

# Agency strings that are too generic to be meaningful for grouping.
GENERIC_AGENCIES = {
    "", "nyc open data", "open data", "data", "nyc", "city of new york",
    "office of technology and innovation", "office of technology and innovation (oti)",
    "department of information technology and telecommunications (doitt)",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_datasets(root: Path) -> list[dict]:
    """Read every <something>__<id>/metadata.json under root."""
    datasets = []
    for meta_path in sorted(root.glob("*/metadata.json")):
        try:
            meta = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  skip {meta_path.parent.name}: {exc}")
            continue

        prov = meta.get("split_provenance") or {}
        datasets.append({
            "dir": meta_path.parent.name,
            "id": meta.get("id"),
            # logical table: split slices share their parent's id, so all slices
            # of one dataset collapse to a single clustering node (and back).
            "logical_id": prov.get("parent_id") or meta.get("id") or meta_path.parent.name,
            "name": meta.get("name") or meta_path.parent.name,
            "agency": extract_agency(meta),
            "category": meta.get("category") or "",
            "schema": extract_schema_sig(meta),
        })
    return datasets


def extract_agency(meta: dict) -> str:
    """
    Best-effort agency name. Prefer the custom 'Agency' field, fall back to
    'attribution'. Returns '' if only a generic value is available.
    """
    cf = (meta.get("metadata") or {}).get("custom_fields") or {}
    ds_info = cf.get("Dataset Information") or {}
    agency = ds_info.get("Agency") or meta.get("attribution") or ""
    agency = agency.strip()
    if agency.lower() in GENERIC_AGENCIES:
        return ""
    return agency


def extract_schema_sig(meta: dict) -> frozenset:
    """Schema as a set of 'fieldName:dataType', excluding system/computed cols."""
    sig = set()
    for col in meta.get("columns", []):
        field = col.get("fieldName") or ""
        if not field or field.startswith(":"):   # :id, :@computed_region_*, etc.
            continue
        sig.add(f"{field}:{col.get('dataTypeName')}")
    return frozenset(sig)


_DIGITS_RE = re.compile(r"\d+")

def normalize_schema_sig(sig: frozenset) -> frozenset:
    """
    Canonicalize a schema signature so structurally-identical schemas match
    despite year/number stamps in column names. Strips digits and separators
    from the fieldName (keeps the data type), e.g.
        '_2006_2007_overall_grade:text'  ->  'overallgrade:text'
    """
    out = set()
    for item in sig:
        field, _, dtype = item.rpartition(":")
        canon = _DIGITS_RE.sub("", field.lower())
        canon = canon.replace("_", "").replace("-", "").replace(" ", "")
        out.add(f"{canon}:{dtype}")
    return frozenset(out)


# ---------------------------------------------------------------------------
# Name tokenization
# ---------------------------------------------------------------------------

def singularize(tok: str) -> str:
    """
    Light, rule-based plural -> singular so morphological variants of a name
    match (e.g. school/schools, list/lists, company/companies). Not linguistically perfect
    but consistent, so both forms map to the same token. Applied to every token, 
    so it never causes a spurious merge unless two genuinely different words collapse together, 
    which these conservative rules avoid.
    """
    if len(tok) <= 3:
        return tok
    if tok.endswith("ies"):                       # companies -> company
        return tok[:-3] + "y"
    if tok.endswith(("ses", "xes", "zes", "ches", "shes")):   # classes -> class
        return tok[:-2]
    if tok.endswith("s") and not tok.endswith("ss"):          # schools -> school
        return tok[:-1]
    return tok


def tokenize_name(name: str) -> tuple[frozenset, frozenset]:
    """
    Return (key_tokens, qualifiers) for a dataset name.

    key_tokens: content tokens used for blocking/similarity
    qualifiers: version markers (years, 'historical', ...) describing how
                  members of a series differ; never used for linking
    """
    key, quals = set(), set()
    for raw in TOKEN_RE.findall(name.lower()):
        tok = singularize(raw)        # school/schools, list/lists, year/years -> one form
        if YEAR_RE.match(tok) or (tok.isdigit() and len(tok) <= 2):
            # 4-digit years, fy2022, and short year fragments (07, 19, quarters)
            quals.add(tok)
        elif tok in QUALIFIER_WORDS:
            quals.add(tok)
        elif tok in FUNCTION_WORDS:   # grammatical glue: not content, IDF mis-weights it
            continue
        elif len(tok) == 1:           # stray single chars
            continue
        else:
            key.add(tok)              # 3+ digit identifiers (311, 911) stay content
    # Note: generic tokens (data, report, nyc, ...) are intentionally NOT removed.
    # IDF weighting down-weights them in similarity and the df cap excludes them
    # from candidate generation, so a hand-maintained stopword list is unneeded.
    return frozenset(key), frozenset(quals)


def normalize_agency(agency: str) -> str:
    """Collapse an agency string to a comparable key (acronym if present)."""
    if not agency:
        return ""
    m = re.search(r"\(([^)]+)\)", agency)      # prefer the (ACRONYM)
    if m:
        return m.group(1).strip().lower()
    return re.sub(r"\s+", " ", agency).strip().lower()


# ---------------------------------------------------------------------------
# Union-Find for connected components
# ---------------------------------------------------------------------------

class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

    def components(self) -> dict[int, list[int]]:
        groups = defaultdict(list)
        for i in range(len(self.parent)):
            groups[self.find(i)].append(i)
        return groups


# ---------------------------------------------------------------------------
# Auto-thresholding
# ---------------------------------------------------------------------------

def kmeans_threshold(values: list[float], default: float, label: str = "",
                     min_separation: float = 0.25) -> float:
    """
    Derive a similarity cut by clustering the scores into two groups with 1-D
    k-means (k=2): the "non-match" group (low scores) and the "match" group
    (high scores). The threshold is the midpoint between the two group centers.

    The guard is a coarse heuristic: k-means always finds a split, so we reject
    cuts where the two centers nearly coincide
    """
    vals = sorted(v for v in values if v is not None)
    if len(vals) < 8 or vals[-1] - vals[0] < 1e-9:
        return default

    c0, c1 = vals[0], vals[-1]              # init centers at the extremes
    for _ in range(100):
        s0 = n0 = s1 = n1 = 0
        for v in vals:
            if abs(v - c0) <= abs(v - c1):
                s0 += v; n0 += 1
            else:
                s1 += v; n1 += 1
        new0 = s0 / n0 if n0 else c0
        new1 = s1 / n1 if n1 else c1
        if abs(new0 - c0) < 1e-9 and abs(new1 - c1) < 1e-9:
            c0, c1 = new0, new1
            break
        c0, c1 = new0, new1

    if c0 > c1:
        c0, c1 = c1, c0

    # Bimodality guard: if the two clusters barely separate, the scores are
    # effectively one mode and any "threshold" would be arbitrary.
    if c1 - c0 < min_separation:
        print(f"  k-means{label}: centers {c0:.3f}/{c1:.3f} too close "
              f"(< {min_separation}) - unimodal, using default {default}")
        return default

    thr = (c0 + c1) / 2
    print(f"  k-means{label}: centers {c0:.3f}/{c1:.3f} - threshold {thr:.3f}")
    return thr


# ---------------------------------------------------------------------------
# Stage 1 - name blocking
# ---------------------------------------------------------------------------

def compute_idf(token_sets: list[frozenset]) -> dict[str, float]:
    """Inverse document frequency for every token in the corpus."""
    n = len(token_sets)
    df = defaultdict(int)
    for ts in token_sets:
        for tok in ts:
            df[tok] += 1
    return {tok: math.log(n / d) for tok, d in df.items()}


def majority_tokens(token_sets: list[frozenset], idf: dict[str, float]) -> list[str]:
    """
    Tokens appearing in at least half of the members, ranked by IDF-weighted
    member count (count x idf). Weighting by IDF keeps generic glue words (now
    that no stopword list removes them) from dominating cluster labels, while
    still favouring distinctive tokens shared by many members.
    """
    if not token_sets:
        return []
    counts = defaultdict(int)
    for ts in token_sets:
        for tok in ts:
            counts[tok] += 1
    cutoff = (len(token_sets) + 1) // 2          # majority (>= half, rounded up)
    keep = [t for t, c in counts.items() if c >= cutoff]
    keep.sort(key=lambda t: -counts[t] * idf.get(t, 0.0))
    return keep


def weighted_overlap(a: frozenset, b: frozenset, idf: dict[str, float]) -> float:
    """IDF-weighted Jaccard between two token sets (0..1)."""
    if not a or not b:
        return 0.0
    inter = a & b
    union = a | b
    num = sum(idf.get(t, 0.0) for t in inter)
    den = sum(idf.get(t, 0.0) for t in union)
    return num / den if den else 0.0


def agglomerate(members: list[int], sim_of) -> list[tuple[float, int, int]]:
    """
    Run full UPGMA average-linkage agglomeration over `members` (dataset
    indices), merging the highest-similarity pair of groups each step until one
    group remains. `sim_of(i, j)` returns the pairwise similarity.

    Returns the merge sequence as (height, rep_i, rep_j) in DESCENDING height
    order, where height is the average-linkage similarity at that merge and
    rep_* are representative member indices (one per merged group). Replaying
    the first t merges as unions reconstructs the clustering at any cut depth.
    """
    k = len(members)
    if k <= 1:
        return []

    groups = {c: [members[c]] for c in range(k)}    # group id -> member indices
    size = {c: 1 for c in range(k)}
    S = defaultdict(float)                           # summed cross-similarity
    for a in range(k):
        for b in range(a + 1, k):
            s = sim_of(members[a], members[b])
            if s:
                S[(a, b)] = s
                S[(b, a)] = s
    active = set(range(k))
    merges = []

    while len(active) > 1:
        best, best_avg = None, -1.0
        act = list(active)
        for x in range(len(act)):
            for y in range(x + 1, len(act)):
                a, b = act[x], act[y]
                avg = S.get((a, b), 0.0) / (size[a] * size[b])
                if avg > best_avg:
                    best_avg, best = avg, (a, b)
        a, b = best
        # representatives (first member of each group) connect the components
        merges.append((best_avg, groups[a][0], groups[b][0]))
        groups[a].extend(groups[b])
        for x in active:
            if x in (a, b):
                continue
            merged = S.get((a, x), 0.0) + S.get((b, x), 0.0)
            if merged:
                S[(a, x)] = merged
                S[(x, a)] = merged
        size[a] += size[b]
        active.discard(b)
        groups.pop(b, None)

    return merges


def choose_cut(heights: list[float], floor: float, fixed: float | None) -> int:
    """
    Decide how many merges to accept from a descending list of merge heights.

    If `fixed` is given, cut at that fixed similarity (the old fixed-threshold
    behaviour, kept as a manual override). Otherwise cut dynamically at the
    LARGEST GAP between consecutive heights, adapting per block. A virtual 0 is
    appended so a fully-coherent block (no internal cliff) stays whole, while a
    chained block is cut at its high-to-low cliff. Merges below floor are never
    accepted, guarding tiny blocks against spurious low-similarity links.
    """
    m = len(heights)
    if m == 0:
        return 0
    if fixed is not None:
        return sum(1 for h in heights if h >= fixed)

    cap = sum(1 for h in heights if h >= floor)      # never accept below floor
    if cap == 0:
        return 0
    best_t, best_gap = cap, -1.0
    for t in range(1, cap + 1):
        lower = heights[t] if t < m else 0.0         # virtual 0 past the last merge
        gap = heights[t - 1] - lower
        if gap > best_gap:
            best_gap, best_t = gap, t
    return best_t


def dynamic_clusters(members: list[int], sim_of, floor: float,
                     fixed: float | None) -> list[list[int]]:
    """Agglomerate `members`, cut the dendrogram, and return the subgroups."""
    if len(members) <= 1:
        return [list(members)]
    merges = agglomerate(members, sim_of)
    t = choose_cut([h for h, _, _ in merges], floor, fixed)

    parent = {mem: mem for mem in members}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for _, ra, rb in merges[:t]:
        parent[find(ra)] = find(rb)

    comps = defaultdict(list)
    for mem in members:
        comps[find(mem)].append(mem)
    return list(comps.values())


def name_clusters(
    datasets: list[dict],
    idf: dict[str, float],
    floor: float,
    agency_bonus: float,
    min_shared: int = 2,
    fixed: float | None = None,
    dynamic: bool = False,
    debug_band: float | None = None,
) -> tuple[list[list[int]], list[dict], float | None]:
    """
    Block on shared name tokens, score candidate pairs, and return
    (clusters, debug_pairs, name_cut).

    clusters:    list of clusters, each a list of dataset indices
    debug_pairs: if debug_band is set, every candidate pair whose score falls
                  within +/- debug_band of the cut (borderline links); else empty
    name_cut:    the similarity cut actually used (None if dynamic gap cutting)

    Pairs sharing >= `min_shared` content tokens form loose candidate blocks,
    then each block is cut by average-linkage. The cut is, in priority order:
    `fixed` if supplied, else a dynamic largest-gap cut if `dynamic`, else a
    k-means (k=2) threshold derived from the candidate-pair similarity scores.
    """
    n = len(datasets)
    token_sets = [d["_key_tokens"] for d in datasets]
    agencies = [normalize_agency(d["agency"]) for d in datasets]

    # Inverted index, skipping uninformative high-frequency tokens for
    # candidate generation (they'd create giant useless blocks).
    df_cap = max(2, int(0.10 * n))
    index = defaultdict(list)
    for i, ts in enumerate(token_sets):
        for tok in ts:
            index[tok].append(i)

    uf = UnionFind(n)
    seen_pairs = set()
    debug_pairs = []
    pair_sim = {}        # frozenset({i, j}) -> similarity, for average-linkage

    for tok, members in index.items():
        if len(members) > df_cap:        # too common to be discriminating
            continue
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                i, j = members[x], members[y]
                key = (i, j) if i < j else (j, i)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)

                shared = token_sets[i] & token_sets[j]
                base = weighted_overlap(token_sets[i], token_sets[j], idf)
                agency_match = bool(agencies[i] and agencies[i] == agencies[j])
                sim = base + (agency_bonus if agency_match else 0.0)

                # Loose blocking: any pair sharing >= min_shared tokens becomes a
                # candidate edge. Average-linkage (below) decides actual cluster
                # boundaries, so the block only needs to over-collect.
                if len(shared) >= min_shared:
                    uf.union(i, j)
                    pair_sim[frozenset((i, j))] = sim

                if debug_band is not None and abs(sim - floor) <= debug_band:
                    debug_pairs.append({
                        "score": round(sim, 3),
                        "base": round(base, 3),
                        "agency_match": agency_match,
                        "shared_count": len(shared),
                        "candidate": len(shared) >= min_shared,
                        "a": datasets[i]["name"],
                        "a_id": datasets[i]["id"],
                        "b": datasets[j]["name"],
                        "b_id": datasets[j]["id"],
                        "shared": sorted(shared, key=lambda t: -idf.get(t, 0.0)),
                    })

    # Resolve the cut: manual override, else dynamic gap, else k-means over the
    # candidate-pair similarity distribution.
    if fixed is not None:
        name_cut = fixed
    elif dynamic:
        name_cut = None
    else:
        name_cut = kmeans_threshold(list(pair_sim.values()), 0.45, " (name)")

    # Refine every candidate block with average-linkage at the resolved cut.
    sim_of = lambda i, j: pair_sim.get(frozenset((i, j)), 0.0)
    final = []
    for block in uf.components().values():
        if len(block) == 1:
            final.append(block)
        else:
            final.extend(dynamic_clusters(block, sim_of, floor, name_cut))

    debug_pairs.sort(key=lambda p: -p["score"])
    return final, debug_pairs, name_cut


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def dominant(values: list[str]) -> str:
    """Most common non-empty value in a list."""
    counts = defaultdict(int)
    for v in values:
        if v:
            counts[v] += 1
    return max(counts, key=counts.get) if counts else ""


def build_clusters(datasets: list[dict], args) -> tuple[list[dict], list[dict]]:
    for d in datasets:
        key, quals = tokenize_name(d["name"])
        d["_key_tokens"] = key
        d["_qualifiers"] = quals
        d["num_columns"] = len(d["schema"])

    # Collapse each split family to one representative for clustering, then
    # expand back. A dataset's per-year slices are the same logical table by
    # construction, so they belong in a single cluster no matter what, and
    # folding them to one node stops the many per-year slices from chaining into
    # a giant block that the O(k^3) average-linkage can't finish. The year is a
    # qualifier (not a key token), so a family's slices share identical key
    # tokens: collapsing them never changes which cluster the family lands in.
    collapse = not getattr(args, "no_collapse_splits", False)
    if collapse:
        groups = defaultdict(list)
        for i, d in enumerate(datasets):
            groups[d["logical_id"]].append(i)
    else:
        groups = {i: [i] for i in range(len(datasets))}
    group_keys = list(groups.keys())
    rep_datasets = [datasets[groups[k][0]] for k in group_keys]
    if collapse:
        families = sum(1 for k in group_keys if len(groups[k]) > 1)
        print(f"  collapsed {len(datasets)} datasets -> {len(rep_datasets)} logical "
              f"tables for clustering ({families} split families)")

    idf = compute_idf([d["_key_tokens"] for d in rep_datasets])
    debug_band = args.debug_band if args.debug_pairs else None
    rep_components, debug_pairs, name_cut = name_clusters(
        rep_datasets, idf, args.name_floor, args.agency_bonus,
        args.min_shared_tokens, args.name_threshold, args.dynamic_cut,
        debug_band
    )
    args._name_cut = name_cut          # stash for reporting in main()

    # expand each representative back to all the slices it stood in for
    components = []
    for comp in rep_components:
        members = []
        for p in comp:
            members.extend(groups[group_keys[p]])
        components.append(members)

    clusters = []
    for members in components:
        members.sort(key=lambda i: datasets[i]["name"])

        # Tokens common to EVERY member (strict intersection). Often empty for
        # transitively-linked clusters, so it's reported but not used to label.
        shared = set.intersection(*[set(datasets[i]["_key_tokens"]) for i in members]) \
            if members else set()
        shared = sorted(shared, key=lambda t: -idf.get(t, 0.0))

        # Label tokens: those appearing in a MAJORITY of members, ranked by how
        # many members carry them, then by idf. Robust to chained clusters where
        # no single token spans all members.
        label_tokens = majority_tokens([datasets[i]["_key_tokens"] for i in members], idf)

        clusters.append({
            "size": len(members),
            "label": " / ".join(label_tokens[:4]) or datasets[members[0]]["name"],
            "shared_name_tokens": shared,
            "agency": dominant([datasets[i]["agency"] for i in members]),
            "category": dominant([datasets[i]["category"] for i in members]),
            "members": [
                {
                    "id": datasets[i]["id"],
                    "name": datasets[i]["name"],
                    "agency": datasets[i]["agency"],
                    "qualifiers": sorted(datasets[i]["_qualifiers"]),
                    "num_columns": datasets[i]["num_columns"],
                }
                for i in members
            ],
        })

    clusters.sort(key=lambda c: (-c["size"], c["label"]))
    return clusters, debug_pairs


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(clusters: list[dict], path: Path) -> None:
    lines = []
    multi = [c for c in clusters if c["size"] > 1]
    singles = [c for c in clusters if c["size"] == 1]

    lines.append("=" * 78)
    lines.append("NYC OPEN DATA - NAME CLUSTERS")
    lines.append("=" * 78)
    lines.append(f"{len(clusters)} clusters total "
                 f"({len(multi)} multi-dataset, {len(singles)} singletons)")
    lines.append("")

    for n, c in enumerate(multi, 1):
        lines.append(f"[{n}] {c['label']}   ({c['size']} datasets)")
        if c["agency"]:
            lines.append(f"     agency   : {c['agency']}")
        if c["category"]:
            lines.append(f"     category : {c['category']}")
        for m in c["members"]:
            q = f"  [{', '.join(m['qualifiers'])}]" if m["qualifiers"] else ""
            lines.append(f"       - {m['name']} ({m['id']}, {m['num_columns']} cols){q}")
        lines.append("")

    if singles:
        lines.append("-" * 78)
        lines.append(f"SINGLETONS ({len(singles)})")
        lines.append("-" * 78)
        for c in singles:
            lines.append(f"  - {c['members'][0]['name']} ({c['members'][0]['id']})")
        lines.append("")

    path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Name-cluster NYC Open Data datasets (Stage 1).")
    parser.add_argument("--dir", default=".")
    # By default the name-similarity cut is derived from the data by clustering
    # the scores into two groups (k-means, k=2) and cutting at the midpoint, then
    # applied as a fixed average-linkage cut. Override with a manual value, or
    # switch to per-block dynamic gap cutting.
    parser.add_argument("--name-threshold", type=float, default=None,
                        help="OVERRIDE the auto (k-means) name-similarity cut with a fixed value")
    parser.add_argument("--dynamic-cut", action="store_true",
                        help="cut each block at its largest dendrogram gap instead of a "
                             "fixed/auto threshold")
    parser.add_argument("--name-floor", type=float, default=0.10,
                        help="min name similarity a merge may have (safety floor; default 0.10)")
    parser.add_argument("--min-shared-tokens", type=int, default=2,
                        help="min content tokens two datasets must share to link; "
                             "2 breaks single-token chaining (set 1 for old behavior)")
    parser.add_argument("--agency-bonus", type=float, default=0.10)
    parser.add_argument("--no-collapse-splits", action="store_true",
                        help="cluster every split slice individually instead of "
                             "collapsing each split family (split_provenance.parent_id) "
                             "to one representative; slower and can blow up when a "
                             "dataset has many per-year slices")
    parser.add_argument("--out", default="clusters.json")
    parser.add_argument("--report", default="clusters.txt")
    parser.add_argument("--debug-pairs", action="store_true",
                        help="print candidate pairs that scored near the threshold")
    parser.add_argument("--debug-band", type=float, default=0.15,
                        help="how far from the threshold counts as borderline (default: 0.15)")
    args = parser.parse_args()

    root = Path(args.dir)
    print(f"Loading datasets from {root}/ ...")
    datasets = load_datasets(root)
    print(f"  loaded {len(datasets)} datasets")

    if not datasets:
        print("No metadata.json files found.")
        return

    clusters, debug_pairs = build_clusters(datasets, args)

    if args._name_cut is None:
        print("\nname cut: dynamic gap")
    else:
        how = "manual" if args.name_threshold is not None else "auto/k-means"
        print(f"\nname cut: {args._name_cut:.3f} ({how})")

    multi = sum(1 for c in clusters if c["size"] > 1)
    print(f"{len(clusters)} clusters ({multi} multi-dataset)")

    Path(args.out).write_text(json.dumps(clusters, indent=2, ensure_ascii=False))
    write_report(clusters, Path(args.report))
    print(f"Wrote {args.out}, {args.report}")
    print("Next: python schema_match.py --dir "
          f"{args.dir} --clusters {args.out}")

    if args.debug_pairs:
        center = args._name_cut if args._name_cut is not None else args.name_floor
        print_debug_pairs(debug_pairs, center, args.debug_band)


def print_debug_pairs(pairs: list[dict], center: float, band: float) -> None:
    """Show borderline candidate pairs so the name floor/threshold can be tuned."""
    print()
    print("=" * 78)
    print(f"BORDERLINE PAIRS  (within +/-{band} of {center})")
    print(f"  {len(pairs)} pairs - 'o' candidate edge (>= min-shared tokens), "
          f"'-' not a candidate. Final clusters set by average-linkage, not per-pair.")
    print("=" * 78)
    if not pairs:
        print("  (none - try a wider --debug-band)")
        return
    for p in pairs:
        mark = "o" if p.get("candidate") else "-"
        bonus = "  +agency" if p["agency_match"] else ""
        shared = ", ".join(p["shared"][:5]) or "(none)"
        print(f" {mark} {p['score']:.3f} (base {p['base']:.3f}{bonus}, {p['shared_count']} shared)")
        print(f"     {p['a']}  ({p['a_id']})")
        print(f"     {p['b']}  ({p['b_id']})")
        print(f"     shared: {shared}")


if __name__ == "__main__":
    main()
