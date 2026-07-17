# NYC Open Data

Scrapes every table dataset on [data.cityofnewyork.us](https://data.cityofnewyork.us) and reconstructs the version series and lineage the portal does not record.

Unlike the [Wikipedia lake](../wikipedia-tables/), NYC publishes each edition as an independent dataset with no link to its predecessors, so versions and lineage are *proposed* by the pipeline and then *corrected by hand*. Every proposing step writes a plan or grouping that a matching `review_*` tool lets you fix before it is applied. The review tools are stateful (stop and resume any time) and shardable (`--shard N` / `--merge` for several annotators in parallel).

## Requirements

Python ≥ 3.14, `requests`. A [Socrata app token](https://data.cityofnewyork.us/profile/edit/developer_settings) raises the rate limit:

```
export NYC_APP_TOKEN=your_token
```

## Pipeline

Run in this order from the folder. Each step's output feeds the next.

### 1. Scrape

```
python scrape_nyc_open_data.py --out-dir data
```

Discovers every table dataset and downloads it into `data/<name>__<id>/` (`data.csv`, `metadata.json`, …). Re-run with `--resume` after an interruption, or `--skip-unchanged` to refresh only datasets whose data actually changed.

### 2. De-duplicate

```
python dedup_data.py --dir data              # dry run: show collisions
python dedup_data.py --dir data --apply      # move dupes to data/_duplicates/
```

Keeps one folder per dataset id when a rename-in-place scraped the same dataset twice.

### 3. Profile columns

```
python profile_columns.py --dir data
```

Writes `column_profiles.json`, consumed by steps 4 and 6.

### 4. Split multi-year tables

Brings the corpus to one table per period before matching.

```
python split_datasets.py --dir data                 # writes split_plan.csv (dry run)
python review_splits.py --plan split_plan.csv --dir data   # confirm / skip / fix the axis
python split_datasets.py --dir data --apply         # apply the reviewed plan
```

`--apply` writes the per-year slices into `data/` and moves each original consolidated folder into `data/_split_originals/` (reversible).

### 5. Cluster by name

```
python cluster_datasets.py --dir data
```

Writes `clusters.json` and a readable `clusters.txt`.

### 6. Match versions

```
python version_match.py --dir data --clusters clusters.json --profiles column_profiles.json
```

Writes `version_groups.json` and `version_review.txt`.

### 7. Review the series

```
python review_clusters.py --clusters version_groups.json      # accept / add / remove / dissolve
python review_clusters.py --finalize --out clusters.reviewed.json
```

`clusters.reviewed.json` is the curated ground truth for the next step.

### 8. Build the lineage

```
python lineage.py --clusters clusters.reviewed.json --dir data
```

Writes `lineage.json` (members ordered oldest → newest, with `predecessor` / `successor` links) and `lineage.txt`.

### 9. Review undated editions

Only needed where an edition carries no year in its title and was ordered by a guessed publication date.

```
python review_lineage.py --clusters clusters.reviewed.json --dir data
python review_lineage.py --apply                              # writes period_overrides.json
python lineage.py --clusters clusters.reviewed.json --dir data --overrides period_overrides.json
```

The re-run of `lineage.py` produces the final `lineage.json`.

### 10. Match columns across editions

```
python match_columns.py --lineage lineage.json --dir data
```

Writes `column_matched_lineage.json`: `lineage.json` again, with every column of
every edition assigned a `column_id` UUID that is stable across the series, plus
a per-edition `schema_change` (added / removed / renamed / retyped). Columns are
matched only on an exactly equal Socrata `fieldName` or normalized header; a
column that matches nothing keeps its own UUID and is reported as an addition or
a removal rather than being forced onto a neighbour.

## Files

| Script | Writes |
|---|---|
| `scrape_nyc_open_data.py` | `data/<name>__<id>/`, `catalog.json`, `snapshots.json`, `downloads.json` |
| `dedup_data.py` | `data/_duplicates/` |
| `profile_columns.py` | `column_profiles.json` |
| `split_datasets.py` | `split_plan.csv`, slices in `data/`, `data/_split_originals/` |
| `review_splits.py` | `split_plan.csv` (in place) |
| `cluster_datasets.py` | `clusters.json`, `clusters.txt` |
| `version_match.py` | `version_groups.json`, `version_review.txt` |
| `review_clusters.py` | `cluster_plan.csv`, `clusters.reviewed.json` |
| `lineage.py` | `lineage.json`, `lineage.txt` |
| `review_lineage.py` | `lineage_plan.csv`, `period_overrides.json` |
| `match_columns.py` | `column_matched_lineage.json`, `column_matches.txt` |

Every script takes `--help`, and its module docstring documents the options and the reasoning behind the step.

## What is in this repository

Only the pipeline and the **hand-curated review results** are committed:

- `split_plan.csv` — the reviewed split axes (step 4)
- `clusters.reviewed.json` — the reviewed version series (step 7)

These are the one part of the pipeline no script can reproduce, and they let you re-run everything downstream without repeating the manual review. Everything else — the scraped corpus and the generated `lineage.json` — is regenerated by the steps above, and the finished lake is published on Zenodo (see the [main README](../README.md)).
