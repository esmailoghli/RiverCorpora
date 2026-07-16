# Wikipedia Tables

Extracts and tracks tables across all revisions of every page in a MediaWiki XML dump (e.g. English Wikipedia). Output is one JSONL file per dump file, with one JSON object per (page, logical table) pair containing all revisions of that table.

Only **article namespace (ns 0)** pages are processed. The placeholder table that the VisualEditor / wikitext "insert table" button inserts (caption *Caption text*, headers *Header text*, cells *Example*) is ignored.

## Revision filtering

To suppress short-lived noise, each logical table's revision history is post-processed:

- **Rollback exclusion** — when a table's content returns to an earlier state and the round-trip (from the start of the excursion to the return) happens within **7 days**, the intervening excursion *and* the returning revision are dropped (`A→B→A ⇒ A`, `A→B→C→D→A ⇒ A`). A return after 7 days is treated as a genuine reappearance and kept.
- **Same-user grouping** — a run of consecutive revisions by the same user, each within **1 hour** of the previous, is collapsed to its last revision (`A→B→C→D` with `B,C` same user ⇒ `A→C→D`). The window is the `sameUserGroupWindow` constant in `page.go`.

## Matching

Tables are linked across revisions with the three-stage temporal object matching of Bleifuß et al., *"Structured Object Matching across Web Page Revisions"* (2021): a **local search** over position-neighbours, then a **strict** stage using the Ruzicka similarity `sim_strict`, then a **relaxed** stage using the containment similarity `sim_relaxed`. Each stage computes a maximum-weight bipartite matching (Hungarian) over the still-unmatched tables, with a rear-view window over the last `k=5` versions (decayed by `φ`) and IDF-weighted content tokens. The tuning constants (`kWindow`, `posWindow`, `decayFactor`, `thetaStrict`, `thetaRelaxed`) live at the top of `match.go`.

## Build

```
go build .
```

## Usage

```
# Single file
wikitables file.xml.bz2

# All *.xml.bz2 in the current directory
wikitables --all

# From a list of paths
wikitables --input-list files.txt
```

## Flags

| Flag | Description |
|---|---|
| `--jobs N` | Process N files in parallel (default: 1) |
| `--done-list FILE` | Track completed files for safe resume after interruption |
| `--exclude-page ID` | Skip a page by ID; may be repeated |
| `--tmp-dir DIR` | Directory for intermediate spool files (default: OS temp) |

## Output format

Each line is a JSON object:

```json
{
  "pageID": 123,
  "pageTitle": "Example",
  "tableID": "456-0",
  "deletedAt": null,
  "tables": [
    {
      "revisionID": 789,
      "revisionDate": "Mon Jan 02 15:04:05 CET 2006",
      "contributor": {"username": "Alice", "ip": null, "id": "42"},
      "format": "wikitable",
      "artificialColumnHeaders": ["uuid-1", "uuid-2"],
      "matchScore": 0.95,
      "cells": [[{"content": "A", "wasExpanded": false}, ...], ...]
    }
  ]
}
```

`matchScore` is absent for the first revision of a table. `deletedAt` is present when the table was removed before the final revision of the page.

`artificialColumnHeaders` gives each column a synthetic UUID that stays stable **across revisions of the same table by column identity**: from one version to the next, columns are matched first by equal header text (`!` / `<th>` cells) and then by position, so a column keeps its UUID through insertions, deletions, and reorderings of *other* columns. Newly added columns get a fresh UUID. Tables without headers fall back to position-based (index) stability. (Column matching is header+position only — it does not use cell content — so a renamed header or a header-less table with a mid-column insertion is not tracked semantically.)