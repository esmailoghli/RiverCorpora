"""
Column matching across the editions of a series: decide which column
of one edition is which column of the next, so schema drift can be read off a
series directly.

Consumes lineage.json and emits column_matched_lineage.json: the same
structure, with every column of every member assigned a `column_id` UUID that
is STABLE ACROSS THE SERIES. One UUID spans all matched occurrences of a
column, so a column keeps its identity through renames and through the
addition or removal of other columns; a column with no match gets a fresh UUID.

Two columns match when they agree EXACTLY on either:
  1. their Socrata fieldName: survives a display-name rename, so this is
     what catches "Complaint Type" -> "Problem (formerly Complaint Type)"
  2. their normalized header: lowercase alphanumerics, singularized, so
     "Base License Number" == "base_license_number" and "Police Precincts"
     == "Police Precinct"

NEVER FORCED: editions are routinely supersets/subsets of each other, and a
column that matches nothing is simply reported as an addition or a removal.
Position/index is NOT a fallback (unlike the Wikipedia matcher, which can
lean on it for header-less tables): NYC publishers reorder columns freely
between editions, so column 3 of one edition has no reason to be column 3 of
the next, and matching on it would invent relationships rather than find them.

IT SKIPS:
  SINGLETONS      a series of one edition has nothing to match against; its
                  columns just get their UUIDs.
  SPLIT FAMILIES  the per-year slices of one dataset are cut from a single
                  table by split_datasets.py and share its schema by
                  construction, so the slices of one parent are matched by
                  identity. A cluster that is nothing but one parent's slices
                  needs no matching at all; in a mixed cluster each split
                  family is collapsed to ONE representative, and its result is
                  handed back to every slice.

Usage
-----
  python match_columns.py --lineage lineage.json --dir data
  python match_columns.py --out column_matched_lineage.json
"""

from __future__ import annotations

import argparse
import json
import uuid
from collections import defaultdict
from pathlib import Path

from cluster_datasets import singularize
from lineage import parent_of
from profile_columns import norm_col


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

def index_folders(root: Path) -> dict:
    """dataset id -> folder. The id is the segment after the last '__'."""
    out = {}
    for sub in root.iterdir():
        if sub.is_dir() and not sub.name.startswith("_") and "__" in sub.name:
            out[sub.name.rsplit("__", 1)[-1]] = sub
    return out


def load_schema(did: str, folder_of: dict) -> list[dict] | None:
    """
    Columns of a dataset, as [{name, field_name, type}]. None if unavailable.

    A split slice ('abcd-1234_2020') resolves to its own folder when the split
    has been applied, and otherwise falls back to its parent: split_datasets.py
    copies the parent's metadata verbatim into each slice, so the schema is the
    same either way.
    """
    folder = folder_of.get(did) or folder_of.get(parent_of(did))
    if not folder:
        return None
    try:
        meta = json.loads((folder / "metadata.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None
    cols = []
    for c in meta.get("columns") or []:
        name = (c.get("name") or "").strip()
        field = (c.get("fieldName") or "").strip()
        if not name and not field:
            continue
        cols.append({"name": name or field,
                     "field_name": field,
                     "type": c.get("dataTypeName") or ""})
    return cols or None


def name_key(name: str) -> str:
    """
    Normalized header key: norm_col() with each token singularized, so a header
    that only drifts in number ('Police Precincts' -> 'Police Precinct') stays
    the same column. Still an exact key, singularize() is a normalization
    step, the same one cluster_datasets.py applies to dataset names, not a
    similarity measure.
    """
    return "_".join(singularize(t) for t in norm_col(name).split("_") if t)


def keys_of(col: dict) -> tuple:
    """The two exact keys a column can be matched on: (fieldName, normalized name)."""
    return col["field_name"] or None, name_key(col["name"]) or None


# ---------------------------------------------------------------------------
# Series
# ---------------------------------------------------------------------------

def classify(members: list[dict]) -> str:
    if len(members) == 1:
        return "singleton"
    if len({parent_of(m["id"]) for m in members}) == 1:
        return "split-family"
    return "matched"


def split_families(members: list[dict]) -> list[list[dict]]:
    """
    Group members by parent id, each group ordered by seq, groups ordered by
    their earliest seq. Every group shares one schema, so matching runs over
    one member per group.
    """
    fams = defaultdict(list)
    for m in members:
        fams[parent_of(m["id"])].append(m)
    groups = [sorted(v, key=lambda m: m.get("seq") or 0) for v in fams.values()]
    groups.sort(key=lambda g: g[0].get("seq") or 0)
    return groups


def match_cluster(cluster: dict, folder_of: dict) -> dict:
    out = dict(cluster)
    members = sorted(cluster["members"], key=lambda m: m.get("seq") or 0)
    out["match_kind"] = classify(members)

    groups = split_families(members)
    schemas = {g[0]["id"]: load_schema(g[0]["id"], folder_of) for g in groups}
    if any(s is None for s in schemas.values()):
        out["match_kind"] = "unavailable"

    # Series-wide key -> UUID. A column matches any earlier edition of the
    # series that used the same key, so one UUID covers every occurrence,
    # including a column that skips an edition and comes back.
    by_field, by_name = {}, {}
    series = {}
    new_members, prev_id = [], None

    for g in groups:
        rid = g[0]["id"]
        cols = schemas.get(rid) or []
        prev_cols = schemas.get(prev_id) if prev_id else None
        prev_keys = {k for c in prev_cols for k in keys_of(c) if k} if prev_cols else set()

        rendered, added, renamed, retyped = [], [], [], []
        for c in cols:
            fk, nk = keys_of(c)
            cid = (by_field.get(fk) if fk else None) or (by_name.get(nk) if nk else None)
            first_seen = cid is None
            if first_seen:
                cid = str(uuid.uuid4())
            if fk:
                by_field.setdefault(fk, cid)
            if nk:
                by_name.setdefault(nk, cid)

            matched = (not first_seen) and bool(prev_keys & {k for k in (fk, nk) if k})
            entry = {"column_id": cid, "name": c["name"],
                     "field_name": c["field_name"], "type": c["type"],
                     "status": "initial" if not prev_cols
                     else "kept" if matched
                     else "returned" if not first_seen
                     else "new"}
            rendered.append(entry)

            s = series.setdefault(cid, {"column_id": cid, "name": c["name"],
                                        "field_name": c["field_name"],
                                        "type": c["type"], "members": [],
                                        "aliases": []})
            s["members"].extend(m["id"] for m in g)
            if c["name"] not in s["aliases"]:
                s["aliases"].append(c["name"])

            if prev_cols and entry["status"] == "kept":
                p = next((x for x in prev_cols
                          if cid in (by_field.get(x["field_name"]),
                                     by_name.get(name_key(x["name"])))), None)
                if p and name_key(p["name"]) != name_key(c["name"]):
                    renamed.append({"column_id": cid, "from": p["name"],
                                    "to": c["name"]})
                if p and p["type"] != c["type"]:
                    retyped.append({"column_id": cid, "from": p["type"],
                                    "to": c["type"]})
            elif prev_cols and entry["status"] in ("new", "returned"):
                added.append({"column_id": cid, "name": c["name"]})

        removed = []
        if prev_cols:
            curr_keys = {k for c in cols for k in keys_of(c) if k}
            for p in prev_cols:
                pf, pn = keys_of(p)
                if not (curr_keys & {k for k in (pf, pn) if k}):
                    removed.append({"column_id": (by_field.get(pf) or by_name.get(pn)),
                                    "name": p["name"]})

        for m in g:
            mm = dict(m)
            mm["columns"] = rendered if cols else None
            # Inside a split family every slice is the same table, so drift is
            # only meaningful for the family's first slice.
            mm["schema_change"] = None
            if m is g[0] and prev_cols:
                mm["schema_change"] = {"added": added, "removed": removed,
                                       "renamed": renamed, "retyped": retyped}
            new_members.append(mm)
        prev_id = rid

    new_members.sort(key=lambda m: m.get("seq") or 0)
    out["members"] = new_members
    for v in series.values():
        v["n_members"] = len(v["members"])
    out["column_series"] = list(series.values())
    if out["match_kind"] != "singleton":
        out["column_stability"] = summarize(out["column_series"], len(members))
    return out


def summarize(series: list[dict], n_members: int) -> dict:
    """How much of the schema is shared by the whole series."""
    if not series:
        return {"columns": 0, "in_all": 0, "in_some": 0, "renamed": 0}
    in_all = sum(1 for c in series if c["n_members"] == n_members)
    return {"columns": len(series), "in_all": in_all,
            "in_some": len(series) - in_all,
            "renamed": sum(1 for c in series if len(c["aliases"]) > 1)}


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Match column headers across the editions of each series.")
    ap.add_argument("--lineage", default="lineage.json",
                    help="ordered series to match (lineage.py output)")
    ap.add_argument("--dir", default="data", help="directory of dataset folders")
    ap.add_argument("--out", default="column_matched_lineage.json")
    ap.add_argument("--report", default="column_matches.txt")
    args = ap.parse_args()

    clusters = json.loads(Path(args.lineage).read_text())
    root = Path(args.dir)
    folder_of = index_folders(root) if root.exists() else {}
    if not folder_of:
        print(f"warning: no dataset folders under {root}; nothing to match")

    out = [match_cluster(c, folder_of) for c in clusters]
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))

    kinds = defaultdict(int)
    for c in out:
        kinds[c["match_kind"]] += 1
    lines = [f"series: {len(out)}"]
    for k, v in sorted(kinds.items()):
        lines.append(f"  {k:<14} {v}")
    lines.append("")
    for c in out:
        if c["match_kind"] != "matched":
            continue
        st = c.get("column_stability") or {}
        lines.append(f"{c.get('label', '')}  [{c['size']} editions]")
        lines.append("  columns %s | in every edition %s | partial %s | renamed %s"
                     % (st.get("columns"), st.get("in_all"), st.get("in_some"),
                        st.get("renamed")))
        for m in c["members"]:
            ch = m.get("schema_change")
            if not ch:
                continue
            lines.append("    %-12s %-24s +%d -%d ~%d T%d"
                         % (m["period"], m["id"], len(ch["added"]), len(ch["removed"]),
                            len(ch["renamed"]), len(ch["retyped"])))
            for r in ch["renamed"]:
                lines.append(f"        rename {r['from']!r} -> {r['to']!r}")
        lines.append("")
    Path(args.report).write_text("\n".join(lines))

    print(f"wrote {args.out} and {args.report}")
    for k, v in sorted(kinds.items()):
        print(f"  {k:<14} {v}")


if __name__ == "__main__":
    main()
