"""
Interactive reviewer for clusters.json. Confirm each name-cluster, fix its
membership by hand, and rescue singletons that should have been grouped.

Two things to review, both from cluster_datasets.py's clusters.json:

  CLUSTERS:    for each multi-dataset cluster: accept it, or add / remove member
               tables by hand (remove a table and, unless you re-add it elsewhere,
               it falls back to its own group).
  SINGLETONS:  for each lone dataset it lists other datasets that share >= 1 name
               token (ranked by how many), so you can attach the singleton to the
               series it belongs to instead of leaving it orphaned.

SPLIT-AWARE: a dataset's per-year split slices are treated as one atomic table.
A cluster that is nothing but a single table's own slices has no decision to make,
so it is auto-accepted and skipped. In any other cluster, each split family is
shown and acted on as ONE line (its slices move together), not one row per year.

For every unit it shows the members and the related datasets (shared name
tokens), each with a pick number, then you choose:

    y  accept   -- keep the current membership as a cluster
    a  add      -- pull in datasets (by (pick) number or raw id)
    r  remove   -- drop members (by [member] number or raw id)
    d  dissolve -- break this cluster up into singletons
    s  skip     -- decide later (leaves it unreviewed)
    q  quit     -- save progress and exit (resume later)
    ?  help

STATEFUL: every edit is written straight back into cluster_plan.csv (a `reviewed`
column marks done rows), so you can stop any time and continue where you left off.

TEAM REVIEW: shard the plan into independent files, review in parallel, merge back
  python review_clusters.py --shard 3          # -> cluster_plan.part1.csv, .part2, .part3
  python review_clusters.py cluster_plan.part2.csv
  python review_clusters.py --merge            # folds cluster_plan.part*.csv back in

APPLY: turn the reviewed plan into a corrected clusters.json:
  python review_clusters.py --finalize --out clusters.reviewed.json

Usage
-----
  python review_clusters.py [PLAN] [--clusters clusters.json] [--plan cluster_plan.csv]
                            [--only cluster|singleton] [--all]
  python review_clusters.py --shard N
  python review_clusters.py --merge [--parts "cluster_plan.part*.csv"] [--out FILE]
  python review_clusters.py --finalize [--out clusters.reviewed.json]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict, deque
from math import ceil
from pathlib import Path

from cluster_datasets import (compute_idf, dominant, majority_tokens,
                              tokenize_name, write_report)

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

FIELDS = ["unit", "kind", "label", "action", "members", "orig_members", "reviewed"]
DONE = {"y", "yes", "1", "true"}
DATASET_URL = "https://data.cityofnewyork.us/d/{id}/"
# a split slice's id is "<parent 4x4>_<year>"; the resolvable page is the parent's
SLICE_RE = re.compile(r"^([a-z0-9]{4}-[a-z0-9]{4})_", re.I)
MAX_CANDIDATES = 12


MIN_SHARED = 1          # a related dataset must share at least this many name tokens


def dataset_url(did: str) -> str:
    m = SLICE_RE.match(did)
    return DATASET_URL.format(id=m.group(1) if m else did)


def parent_of(did: str) -> str:
    """The parent dataset id for a split slice ('abcd-1234_2020' -> 'abcd-1234');
    the id itself for a normal dataset."""
    m = SLICE_RE.match(did)
    return m.group(1) if m else did


def is_slice(did: str) -> bool:
    return bool(SLICE_RE.match(did))


def group_by_parent(ids: list[str]) -> list[tuple]:
    """Collapse ids into display units. One per parent table, as
    [(parent_id, [member ids]), ...] in first-seen order, so a dataset's split
    slices show as a single grouped unit instead of one row per year."""
    groups, order = {}, []
    for i in ids:
        p = parent_of(i)
        if p not in groups:
            groups[p] = []
            order.append(p)
        groups[p].append(i)
    return [(p, groups[p]) for p in order]


# ---------------------------------------------------------------------------
# Corpus index (everything is derivable from clusters.json itself)
# ---------------------------------------------------------------------------

class Corpus:
    """id -> name/metadata/name-tokens, plus a token inverted index, built from
    every member across all clusters in clusters.json."""

    def __init__(self, clusters: list[dict]):
        self.info, self.toks = {}, {}
        for c in clusters:
            for m in c.get("members", []):
                i = m["id"]
                self.info[i] = m
                self.toks[i] = tokenize_name(m.get("name", i))[0]
        self.inv = defaultdict(set)
        for i, ts in self.toks.items():
            for t in ts:
                self.inv[t].add(i)
        # parent table -> all of its ids (a split family's slices, or a lone id)
        self.by_parent = defaultdict(list)
        for i in self.info:
            self.by_parent[parent_of(i)].append(i)

    def name(self, i: str) -> str:
        return self.info.get(i, {}).get("name", i)

    def union_tokens(self, ids: list[str]) -> set:
        out = set()
        for i in ids:
            out |= self.toks.get(i, set())
        return out

    def display_name(self, pkey: str) -> str:
        """Human label for a parent table: its name, or '<base> (N slices)' for a
        split family."""
        fam = self.by_parent.get(pkey, [pkey])
        nm = self.name(fam[0])
        if len(fam) > 1 or is_slice(fam[0]):
            base = re.sub(r"\s*\([^()]*\)\s*$", "", nm) or pkey
            return f"{base}  ({len(fam)} slices)"
        return nm

    def related_families(self, ids: list[str], k: int = MAX_CANDIDATES,
                         min_shared: int = MIN_SHARED) -> list[tuple]:
        """Parent tables outside `ids` sharing >= min_shared name tokens, ranked by
        best per-slice shared-token count. A split family counts once, not per
        slice. Returns [(parent_id, (score, shared_tokens)), ...]."""
        member = set(ids)
        here = {parent_of(i) for i in ids}
        utoks = self.union_tokens(ids)
        score = defaultdict(int)
        for t in utoks:
            for j in self.inv.get(t, ()):
                if j not in member:
                    score[j] += 1
        fam = {}
        for j, s in score.items():
            if s < min_shared:
                continue
            p = parent_of(j)
            if p in here:                         # same family as a current member
                continue
            best, shared = fam.get(p, (0, set()))
            fam[p] = (max(best, s), shared | (self.toks.get(j, set()) & utoks))
        ranked = sorted(fam.items(), key=lambda kv: (-kv[1][0], self.display_name(kv[0])))
        return ranked[:k]


# ---------------------------------------------------------------------------
# Plan I/O
# ---------------------------------------------------------------------------

def members_of(row: dict) -> list[str]:
    return [x for x in (row.get("members", "") or "").split() if x]


def set_members(row: dict, ids: list[str]) -> None:
    seen = []
    for i in ids:
        if i not in seen:
            seen.append(i)
    row["members"] = " ".join(seen)


def fallback_label(members: list[dict]) -> str:
    """A label from the most common name tokens across members, used when the
    input lacks a `label` field (e.g. version_match's version_groups.json)."""
    cnt = Counter()
    for m in members:
        for t in tokenize_name(m.get("name", ""))[0]:
            cnt[t] += 1
    top = [t for t, _ in cnt.most_common(4)]
    return " / ".join(top) or (members[0].get("name", "") if members else "")


def build_plan(clusters: list[dict]) -> list[dict]:
    """One row per cluster (multi-member first, then singletons), mirroring the
    input's ordering. Accepts either clusters.json or version_groups.json. A
    missing `label` is derived from the member names.

    A cluster that is nothing but one table's own split slices is auto-accepted
    (reviewed='auto'). There is no decision to make, so it's skipped in the
    interactive loop but still kept intact by --finalize."""
    rows, ci, si = [], 0, 0
    for c in clusters:
        members = c.get("members", [])
        ids = [m["id"] for m in members]
        parents = {parent_of(i) for i in ids}
        auto = bool(ids) and all(is_slice(i) for i in ids) and len(parents) == 1
        if len(ids) > 1:
            ci += 1
            unit, kind = f"c{ci}", "cluster"
        else:
            si += 1
            unit, kind = f"s{si}", "singleton"
        joined = " ".join(ids)
        rows.append({"unit": unit, "kind": kind,
                     "label": c.get("label") or fallback_label(members),
                     "action": "accept", "members": joined,
                     "orig_members": joined, "reviewed": "auto" if auto else ""})
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


def is_reviewed(row: dict) -> bool:
    return str(row.get("reviewed", "")).strip().lower() in DONE


# ---------------------------------------------------------------------------
# Team review: shard / merge  (same model as review_splits.py)
# ---------------------------------------------------------------------------

def shard_plan(plan_path: Path, n: int) -> None:
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
    print(f"Split {len(rows)} unit(s) into {len(made)} part(s):")
    for part, cnt in made:
        print(f"  {part.name}  ({cnt} units)")
    print(f"\nEach reviewer:   python review_clusters.py {made[0][0].name}")
    print(f"Then merge back: python review_clusters.py --merge --plan {plan_path.name}")


def merge_parts(plan_path: Path, part_glob: str, out_path: Path) -> None:
    parts = sorted(plan_path.parent.glob(part_glob))
    if not parts:
        sys.exit(f"No part files matched {part_glob!r} in {plan_path.parent}/")
    merged, order = {}, []
    if plan_path.exists():
        for r in read_plan(plan_path):
            merged[r["unit"]] = r
            order.append(r["unit"])
    conflicts = 0
    for pp in parts:
        for r in read_plan(pp):
            u = r["unit"]
            if u not in merged:
                merged[u] = r
                order.append(u)
            elif is_reviewed(r):
                cur = merged[u]
                if is_reviewed(cur) and (cur.get("action"), cur.get("members")) != \
                        (r.get("action"), r.get("members")):
                    conflicts += 1
                    print(f"  conflict on {u}: keeping reviewed master over part")
                elif not is_reviewed(cur):
                    merged[u] = r
    save(out_path, [merged[u] for u in order])
    reviewed = sum(1 for u in order if is_reviewed(merged[u]))
    print(f"Merged {len(parts)} part(s) -> {out_path}  ({reviewed}/{len(order)} reviewed"
          f"{', ' + str(conflicts) + ' conflict(s)' if conflicts else ''})")


# ---------------------------------------------------------------------------
# Finalize: reviewed plan -> corrected clusters.json (union-find over members)
# ---------------------------------------------------------------------------

def _member_dict(mo: dict, did: str) -> dict:
    """Normalize a member to the clusters.json member schema, filling the
    qualifiers/num_columns that version_groups.json or hand-made input omit
    (write_report requires them)."""
    return {"id": mo.get("id", did),
            "name": mo.get("name", did),
            "agency": mo.get("agency", ""),
            "qualifiers": mo.get("qualifiers") or mo.get("version") or [],
            "num_columns": mo.get("num_columns", 0)}


def finalize(plan_path: Path, corpus: Corpus, out_path: Path) -> None:
    rows = read_plan(plan_path)
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        parent[find(a)] = find(b)

    # seed every id so a removed/orphaned id becomes its own singleton
    for r in rows:
        for i in (r.get("orig_members", "") or "").split():
            find(i)
    for r in rows:
        # reviewed accept -> keep members together; reviewed dissolve -> no union;
        # not-yet-reviewed -> preserve the original grouping so a partial review
        # still yields a sane clustering.
        rev = str(r.get("reviewed", "")).strip().lower()
        if rev == "placed":
            continue                                   # absorbed into another cluster
        if is_reviewed(r):
            if r.get("action") == "accept":
                ids = members_of(r)
            else:
                continue                               # dissolve: no grouping
        else:
            ids = (r.get("orig_members", "") or "").split()   # incl. auto: keep family whole
        for i in ids[1:]:
            union(ids[0], i)

    # split families are atomic: a table's slices always stay together, so a
    # family removed from a cluster becomes its own series rather than shattering
    # into per-year singletons.
    for sids in corpus.by_parent.values():
        for i in sids[1:]:
            union(sids[0], i)

    groups = defaultdict(list)
    for i in list(parent):
        groups[find(i)].append(i)

    idf = compute_idf(list(corpus.toks.values()))
    clusters = []
    for ids in groups.values():
        ids.sort(key=lambda i: corpus.name(i))
        tok_sets = [corpus.toks.get(i, frozenset()) for i in ids]
        shared = sorted(set.intersection(*[set(t) for t in tok_sets]) if tok_sets else set(),
                        key=lambda t: -idf.get(t, 0.0))
        label = majority_tokens(tok_sets, idf)
        clusters.append({
            "size": len(ids),
            "label": " / ".join(label[:4]) or corpus.name(ids[0]),
            "shared_name_tokens": shared,
            "agency": dominant([corpus.info.get(i, {}).get("agency", "") for i in ids]),
            "category": dominant([corpus.info.get(i, {}).get("category", "") for i in ids]),
            "members": [_member_dict(corpus.info.get(i, {}), i) for i in ids],
        })
    clusters.sort(key=lambda c: (-c["size"], c["label"]))

    out_path.write_text(json.dumps(clusters, indent=2, ensure_ascii=False))
    report = out_path.with_suffix(".txt")
    write_report(clusters, report)
    multi = sum(1 for c in clusters if c["size"] > 1)
    print(f"Finalized -> {out_path} ({len(clusters)} clusters, {multi} multi-dataset). "
          f"Report -> {report}")


# ---------------------------------------------------------------------------
# Interactive review
# ---------------------------------------------------------------------------

def show(row: dict, pos: int, total: int, corpus: Corpus,
         max_cand: int, min_shared: int) -> tuple:
    """Render one unit with split families grouped into single lines. Returns
    (member_units, candidate_units), each a list of (parent_id, [ids]) so add/
    remove pick numbers map back to whole tables."""
    ids = members_of(row)
    munits = group_by_parent(ids)
    print("\n" + "=" * 72)
    tag = row.get("kind", "").upper()
    tables = f"{len(munits)} table{'s' if len(munits) != 1 else ''}"
    extra = f", {len(ids)} datasets" if len(ids) != len(munits) else ""
    print(f"[{pos}/{total}]  {tag}  {row.get('label', '')}   ({tables}{extra})")
    print("  members:")
    for k, (pkey, mids) in enumerate(munits, 1):
        if len(mids) == 1 and not is_slice(mids[0]):
            m = corpus.info.get(mids[0], {})
            print(f"    [{k}] {mids[0]}  {m.get('name', mids[0])}  ({m.get('num_columns', '?')} cols)")
        else:                                          # a split family
            print(f"    [{k}] {pkey}  {corpus.display_name(pkey)}")
        print(f"          {dataset_url(pkey)}")
    cfams = corpus.related_families(ids, max_cand, min_shared)
    if cfams:
        print(f"  related (share >= {min_shared} name token, ranked):")
        for k, (pkey, (sc, shared)) in enumerate(cfams, 1):
            toks = ", ".join(sorted(shared, key=len, reverse=True)[:6])
            print(f"    ({k}) {pkey}  {corpus.display_name(pkey)}   [+{sc}: {toks}]")
    else:
        print("  related: (nothing shares a name token)")
    print("-" * 72)
    cunits = [(pkey, corpus.by_parent.get(pkey, [pkey])) for pkey, _ in cfams]
    return munits, cunits


def parse_picks(text: str, munits: list, cunits: list, mode: str,
                corpus: Corpus) -> list[str]:
    """Turn 'add'/'remove' input into ids, operating on whole tables. A bare
    number is a pick. (n) a related table for add, [n] a member table for
    remove, expanded to all of that table's ids. A raw id is expanded to its
    whole split family too, so you add/remove a table, never a stray year."""
    pool = cunits if mode == "add" else munits
    out = []
    for tok in text.replace(",", " ").split():
        if tok.isdigit():
            k = int(tok) - 1
            if 0 <= k < len(pool):
                out.extend(pool[k][1])
        else:
            out.extend(corpus.by_parent.get(tok, [tok]))   # id or parent -> family
    return out


def review(plan_path: Path, corpus: Corpus, only: str | None,
           review_all: bool, max_cand: int, min_shared: int,
           regroup_dissolved: bool = False) -> None:
    rows = read_plan(plan_path)

    def settled(r: dict) -> bool:                     # decided: reviewed, auto, or absorbed
        return is_reviewed(r) or str(r.get("reviewed", "")).strip().lower() in ("auto", "placed")

    def claimed_by_others(cur_idx: int) -> dict:
        """id -> the accepted/auto row that already groups it (excluding cur_idx),
        so a row whose members are all spoken-for can be skipped."""
        out = {}
        for j, r in enumerate(rows):
            if j == cur_idx:
                continue
            rev = str(r.get("reviewed", "")).strip().lower()
            if (rev in DONE or rev == "auto") and r.get("action") == "accept":
                for m in members_of(r):
                    out.setdefault(m, r)
        return out

    # Pre-pass: rows whose members are ALL already grouped by a settled cluster
    # (e.g. absorbed in a prior session) are marked 'placed' now, so the "to go"
    # count is honest instead of skipping them mid-walk.
    pre_absorbed = 0
    if not review_all:
        for i, r in enumerate(rows):
            if settled(r):
                continue
            mem = members_of(r)
            if mem and all(m in claimed_by_others(i) for m in mem):
                r["reviewed"], r["action"] = "placed", "dissolve"
                pre_absorbed += 1
        if pre_absorbed:
            save(plan_path, rows)

    # existing orig_members strings, so re-queuing a dissolved cluster's members
    # as singletons is idempotent (a second --regroup-dissolved won't duplicate).
    existing_units = {r.get("orig_members", "") for r in rows if r.get("action") != "dissolve"}
    gcount = [0]

    def explode(row: dict) -> list[int]:
        """Turn a dissolved cluster into fresh singleton review rows, one per
        member table (split families stay whole). Returns their row indices."""
        new = []
        for pkey, ids in group_by_parent(members_of(row)):
            joined = " ".join(ids)
            if joined in existing_units:          # already surfaced, don't duplicate
                continue
            existing_units.add(joined)
            gcount[0] += 1
            rows.append({"unit": f"g{gcount[0]}", "kind": "singleton",
                         "label": corpus.display_name(pkey),
                         "action": "accept", "members": joined,
                         "orig_members": joined, "reviewed": ""})
            new.append(len(rows) - 1)
        return new

    work = deque(i for i, r in enumerate(rows)
                 if (review_all or not settled(rows[i]))
                 and (only is None or r.get("kind") == only))
    # Retroactive: pull the member tables of already-dissolved clusters back in
    # as singletons so you can regroup them into new series.
    regrouped = 0
    if regroup_dissolved:
        for r in list(rows):
            if str(r.get("reviewed", "")).strip().lower() in DONE and r.get("action") == "dissolve":
                added = explode(r)
                work.extend(added)
                regrouped += len(added)

    done = sum(1 for r in rows if is_reviewed(r))
    auto = sum(1 for r in rows if str(r.get("reviewed", "")).strip().lower() == "auto")
    scope = f" {only}s" if only else ""
    tags = []
    if auto:
        tags.append(f"{auto} split-only auto-accepted")
    if pre_absorbed:
        tags.append(f"{pre_absorbed} already grouped elsewhere")
    if regrouped:
        tags.append(f"{regrouped} dissolved table(s) re-queued")
    note = f" ({'; '.join(tags)})" if tags else ""
    print(f"{len(rows)} unit(s); {done} reviewed{note}. {len(work)}{scope} to go.")
    if regrouped:
        save(plan_path, rows)
    if not work:
        print("Nothing to review. (use --all to revisit, --only to pick a kind, "
              "--regroup-dissolved to rebuild from dissolved clusters.)")
        return

    HELP = ("  y accept    a add tables    r remove tables   d dissolve to singletons\n"
            "  x explode   - dissolve AND re-queue its tables as singletons to regroup\n"
            "  s skip      q quit & save   ? help\n"
            "  add/remove take pick numbers - (n) for related, [n] for members - or raw ids;\n"
            "  a split family is one table: acting on it moves all its year-slices together")
    pos, total = 0, len(work)
    while work:
        i = work.popleft()
        row = rows[i]
        if settled(row):
            continue
        mem = members_of(row)
        claimed = claimed_by_others(i)
        pos += 1
        total = pos + len(work)
        if mem and all(m in claimed for m in mem):
            host = claimed[mem[0]]
            row["reviewed"], row["action"] = "placed", "dissolve"
            save(plan_path, rows)
            print(f"  · [{pos}/{total}] '{row.get('label') or mem[0]}' already grouped "
                  f"into '{host.get('label') or host.get('unit')}' - skipping")
            continue
        munits, cunits = show(row, pos, total, corpus, max_cand, min_shared)
        while True:
            try:
                choice = input("  [y]accept [a]dd [r]emove [d]issolve [x]plode [s]kip [q]uit [?] > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                save(plan_path, rows)
                print(f"\nStopped. Progress saved to {plan_path}.")
                return
            if choice in ("y", "yes", "accept"):
                row["action"], row["reviewed"] = "accept", "y"
            elif choice in ("d", "dissolve"):
                row["action"], row["reviewed"] = "dissolve", "y"
            elif choice in ("x", "explode", "regroup"):
                # dissolve, then re-queue the member tables as singletons to rebuild
                row["action"], row["reviewed"] = "dissolve", "y"
                added = explode(row)
                work.extend(added)
                print(f"     exploded into {len(added)} singleton(s) - they'll come up to regroup")
            elif choice in ("s", "skip"):
                row["reviewed"] = ""            # explicitly leave for later
            elif choice in ("a", "add", "r", "remove"):
                mode = "add" if choice[0] == "a" else "remove"
                raw = input(f"     {mode} (numbers or ids): ").strip()
                if not raw:
                    print("     unchanged.")
                    continue
                picks = parse_picks(raw, munits, cunits, mode, corpus)
                unknown = [p for p in picks if p not in corpus.info]
                if unknown and input(f"     unknown id(s) {', '.join(unknown)} - use anyway? [y/N] ")\
                        .strip().lower() not in DONE:
                    picks = [p for p in picks if p not in unknown]
                cur = members_of(row)
                if mode == "add":
                    set_members(row, cur + picks)
                else:
                    set_members(row, [c for c in cur if c not in picks])
                save(plan_path, rows)           # persist the edit; still unreviewed
                munits, cunits = show(row, pos, total, corpus, max_cand, min_shared)
                continue
            elif choice in ("q", "abort", "quit"):
                save(plan_path, rows)
                print(f"\nQuit. Progress saved to {plan_path} - re-run to continue.")
                return
            elif choice in ("?", "h", "help"):
                print(HELP)
                continue
            else:
                print("     (y / a / r / d / x / s / q / ?)")
                continue
            save(plan_path, rows)
            break

    left = sum(1 for r in rows if not settled(r))
    print(f"\nDone. {len(rows) - left}/{len(rows)} settled, {left} left.")
    print(f"Saved to {plan_path}. Apply with:  python review_clusters.py --finalize")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Interactively review/correct name clusters.")
    ap.add_argument("plan_file", nargs="?", default=None,
                    help="plan file to review (positional; overrides --plan, e.g. a shard)")
    ap.add_argument("--clusters", default="clusters.json", help="cluster_datasets.py output")
    ap.add_argument("--plan", default="cluster_plan.csv", help="stateful review plan")
    ap.add_argument("--only", choices=["cluster", "singleton"], default=None,
                    help="review only clusters or only singletons")
    ap.add_argument("--all", action="store_true", help="revisit already-reviewed units")
    ap.add_argument("--regroup-dissolved", action="store_true",
                    help="re-queue the member tables of already-dissolved clusters as "
                         "singletons, so you can rebuild them into new series")
    ap.add_argument("--max-candidates", type=int, default=MAX_CANDIDATES,
                    help=f"related datasets to list per unit (default {MAX_CANDIDATES})")
    ap.add_argument("--min-shared-tokens", type=int, default=MIN_SHARED,
                    help=f"min shared name tokens to list a related dataset (default {MIN_SHARED})")
    ap.add_argument("--rebuild", action="store_true", help="regenerate the plan from clusters.json (loses edits)")
    ap.add_argument("--shard", type=int, metavar="N", help="split the plan into N parts, then exit")
    ap.add_argument("--merge", action="store_true", help="fold reviewed part files back in, then exit")
    ap.add_argument("--parts", default=None, help="glob for --merge (default '<plan-stem>.part*.csv')")
    ap.add_argument("--finalize", action="store_true", help="write a corrected clusters.json from the plan")
    ap.add_argument("--out", default=None, help="output path for --merge / --finalize")
    args = ap.parse_args()

    plan_path = Path(args.plan_file or args.plan)

    if args.shard is not None:
        if not plan_path.exists():
            sys.exit(f"{plan_path} not found - run review_clusters.py once to build it.")
        shard_plan(plan_path, args.shard)
        return
    if args.merge:
        merge_parts(plan_path, args.parts or f"{plan_path.stem}.part*.csv",
                    Path(args.out) if args.out else plan_path)
        return

    clusters_path = Path(args.clusters)
    if not clusters_path.exists():
        sys.exit(f"{clusters_path} not found - run cluster_datasets.py first.")
    corpus = Corpus(json.loads(clusters_path.read_text()))

    if args.finalize:
        if not plan_path.exists():
            sys.exit(f"{plan_path} not found - review first.")
        finalize(plan_path, corpus, Path(args.out) if args.out else Path("clusters.reviewed.json"))
        return

    # build the plan on first run (or --rebuild), then review
    if not plan_path.exists() or args.rebuild:
        rows = build_plan(json.loads(clusters_path.read_text()))
        save(plan_path, rows)
        print(f"Built {plan_path} from {clusters_path} ({len(rows)} units).")

    review(plan_path, corpus, args.only, args.all, args.max_candidates,
           args.min_shared_tokens, args.regroup_dissolved)


if __name__ == "__main__":
    main()
