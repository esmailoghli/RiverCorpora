"""
NYC Open Data tables scraper

Features
--------
- Discovers every table-type dataset on data.cityofnewyork.us
- Downloads all rows to CSV via the SODA3 export endpoint
- Records a download manifest per dataset (start, finish, duration, row count,
  bytes, and the source's rowsUpdatedAt) so a later run can skip datasets that
  have not changed (--skip-unchanged)
- Fetches full view metadata (columns, timestamps, owner)
- Logs dataset revisions when the publishing API permits
- Diffs schema and row-count against the previous run's snapshot
- Downloads any file attachments declared in dataset metadata

Output layout
-------------
<out-dir>/
  catalog.json                         # Discovery API summary of all datasets
  snapshots.json                       # per-run schema/count snapshots (appended)
  downloads.json                       # per-dataset download manifest (keyed by id)
  <name>__<id>/
    data.csv                           # all rows (export.csv stream)
    download.json                      # this dataset's download manifest entry
    metadata.json                      # full /api/views/{id} response
    revisions.json                     # revision history (if accessible)
    diff.json                          # schema + count diff vs previous snapshot
    attachments/                       # downloaded attachments

Usage
-----
  export NYC_APP_TOKEN=your_token
  python scrape_nyc_open_data.py [options]

Options
-------
  --out-dir DIR        output directory (default: ./data)
  --catalog-only       only write catalog.json, skip downloads
  --resume             skip datasets whose data.csv already exists
  --skip-unchanged     skip download if data.csv exists AND the source's
                       rowsUpdatedAt matches the recorded manifest (i.e. the
                       data has not changed since last download)
  --no-attachments     skip attachment downloads
  --no-revisions       skip revision API calls
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DOMAIN = "data.cityofnewyork.us"
DISCOVERY_URL = "https://api.us.socrata.com/api/catalog/v1"
VIEWS_URL = f"https://{DOMAIN}/api/views"
EXPORT_BASE = f"https://{DOMAIN}/api/v3/views"          # SODA3 export endpoint
PUBLISHING_URL = f"https://{DOMAIN}/api/publishing/v1/revision"

CATALOG_LIMIT = 100
RATE_LIMIT_SLEEP = 0.2

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


# ---------------------------------------------------------------------------
# HTTP session
# ---------------------------------------------------------------------------

def make_session(app_token: str) -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "X-App-Token": app_token,
        "Accept": "application/json",
    })
    retry = Retry(
        total=5,
        backoff_factor=1.5,
        status_forcelist=[429, 502, 503, 504],
        allowed_methods=["GET"],
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_tables(session: requests.Session) -> list[dict]:
    """Return all table-type dataset entries from the Socrata Discovery API."""
    datasets = []
    offset = 0
    print("Discovering datasets...", flush=True)

    while True:
        resp = session.get(
            DISCOVERY_URL,
            params={
                "domains": DOMAIN,
                "only": "dataset",
                "limit": CATALOG_LIMIT,
                "offset": offset,
            },
            timeout=30,
        )
        resp.raise_for_status()
        body = resp.json()
        results = body.get("results", [])
        if not results:
            break

        datasets.extend(results)
        total = body.get("resultSetSize", "?")
        print(f"  {len(datasets)} / {total}", flush=True)

        if len(results) < CATALOG_LIMIT:
            break
        offset += CATALOG_LIMIT
        time.sleep(RATE_LIMIT_SLEEP)

    return datasets


def extract_catalog_entry(entry: dict) -> dict:
    resource = entry.get("resource", {})
    classification = entry.get("classification", {})
    return {
        "id": resource.get("id"),
        "name": resource.get("name"),
        "description": resource.get("description", ""),
        "category": classification.get("domain_category", ""),
        "tags": resource.get("tags") or [],
        "updated_at": resource.get("updatedAt", ""),
        "permalink": entry.get("permalink", ""),
    }


# ---------------------------------------------------------------------------
# Full view metadata
# ---------------------------------------------------------------------------

def fetch_view_metadata(session: requests.Session, dataset_id: str) -> dict | None:
    """
    Fetch the full /api/views/{id} response.
    Returns None on 403/404 (private or removed dataset).
    """
    resp = session.get(f"{VIEWS_URL}/{dataset_id}", timeout=30)
    if resp.status_code in (403, 404):
        return None
    resp.raise_for_status()
    return resp.json()


def extract_schema(metadata: dict) -> dict:
    """Pull column names/types out of a view metadata blob."""
    columns = []
    for col in metadata.get("columns", []):
        columns.append({
            "fieldName": col.get("fieldName"),
            "name": col.get("name"),
            "dataTypeName": col.get("dataTypeName"),
            "renderTypeName": col.get("renderTypeName"),
            "position": col.get("position"),
        })
    return {
        "columns": columns,
        "row_count": metadata.get("rowsUpdatedAt"),
        "created_at": metadata.get("createdAt"),
        "updated_at": metadata.get("rowsUpdatedAt"),
        "view_last_modified": metadata.get("viewLastModified"),
    }


# ---------------------------------------------------------------------------
# Revisions
# ---------------------------------------------------------------------------

def fetch_revisions(session: requests.Session, dataset_id: str) -> list[dict] | None:
    """
    Fetch the revision history via the Socrata Publishing API.
    Returns None if the endpoint is inaccessible or errors.
    """
    try:
        resp = session.get(f"{PUBLISHING_URL}/{dataset_id}", timeout=30)
    except requests.RequestException:
        return None
    if resp.status_code in (403, 404, 500):
        return None
    resp.raise_for_status()
    body = resp.json()

    if isinstance(body, list):
        raw = body
    else:
        raw = body.get("results", [body] if "revision_seq" in body else [])

    revisions = []
    for rev in raw:
        revisions.append({
            "revision_seq": rev.get("revision_seq"),
            "action": rev.get("action", {}).get("type"),
            "created_at": rev.get("created_at"),
            "created_by": (rev.get("created_by") or {}).get("display_name"),
            "published": rev.get("published", False),
        })

    return sorted(revisions, key=lambda r: r.get("revision_seq") or 0)


# ---------------------------------------------------------------------------
# Schema + count diffing
# ---------------------------------------------------------------------------

def diff_schemas(prev: dict, curr: dict) -> dict:
    """
    Compare two schema snapshots and return a structured diff.
    Tracks added/removed/changed columns and row-count changes.
    """
    def col_map(schema: dict) -> dict[str, dict]:
        return {c["fieldName"]: c for c in schema.get("columns", []) if c.get("fieldName")}

    prev_cols = col_map(prev)
    curr_cols = col_map(curr)

    added = [c for name, c in curr_cols.items() if name not in prev_cols]
    removed = [c for name, c in prev_cols.items() if name not in curr_cols]
    changed = []
    for name in prev_cols:
        if name in curr_cols:
            p, c = prev_cols[name], curr_cols[name]
            field_diffs = {}
            for field in ("dataTypeName", "renderTypeName", "name"):
                if p.get(field) != c.get(field):
                    field_diffs[field] = {"from": p.get(field), "to": c.get(field)}
            if field_diffs:
                changed.append({"fieldName": name, "changes": field_diffs})

    return {
        "columns_added": added,
        "columns_removed": removed,
        "columns_changed": changed,
        "updated_at_changed": prev.get("updated_at") != curr.get("updated_at"),
        "view_modified_changed": prev.get("view_last_modified") != curr.get("view_last_modified"),
        "has_changes": bool(added or removed or changed),
    }


def load_snapshots(snapshots_path: Path) -> dict[str, dict]:
    """Load the existing snapshots index keyed by dataset id."""
    if not snapshots_path.exists():
        return {}
    return json.loads(snapshots_path.read_text())


def save_snapshots(snapshots_path: Path, snapshots: dict) -> None:
    snapshots_path.write_text(json.dumps(snapshots, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Attachments
# ---------------------------------------------------------------------------

def download_attachments(
    session: requests.Session,
    dataset_id: str,
    metadata: dict,
    out_dir: Path,
) -> list[str]:
    """
    Download any file attachments declared in the view metadata.
    Returns list of saved filenames.
    """
    attachments = (metadata.get("metadata") or {}).get("attachments") or []
    if not attachments:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    for att in attachments:
        asset_id = att.get("assetId") or att.get("blobId")
        filename = att.get("filename") or att.get("name") or asset_id
        if not asset_id or not filename:
            continue

        dest = out_dir / filename
        if dest.exists():
            saved.append(filename)
            continue

        url = f"{VIEWS_URL}/{dataset_id}/files/{asset_id}?filename={filename}"
        try:
            resp = session.get(url, timeout=60, stream=True)
            if resp.status_code in (403, 404):
                continue
            resp.raise_for_status()
            with dest.open("wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
            saved.append(filename)
        except requests.RequestException:
            pass
        time.sleep(RATE_LIMIT_SLEEP)

    return saved


# ---------------------------------------------------------------------------
# Row data download
# ---------------------------------------------------------------------------

DOWNLOAD_RETRIES = 5

def count_csv_rows(path: Path) -> int:
    """Accurate data-row count (handles quoted embedded newlines), header excluded."""
    try:
        with path.open(newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            next(reader)                       # header
            return sum(1 for _ in reader)
    except (OSError, StopIteration):
        return 0


def download_table(session: requests.Session, dataset_id: str, out_path: Path,
                   prefix: str = " ") -> tuple[int, int]:
    """
    Download a dataset's rows via the SODA3 export endpoint
    (/api/v3/views/{id}/export.csv). No offsetpagination, so it scales to 
    arbitrarily large tables.

    Streams to a .tmp file and only renames it on a clean, complete download
    (atomic), so a dropped connection never leaves a half file. Retries transient
    read/connection failures with capped exponential backoff.

    Returns (rows_written, bytes_written); (0, 0) if inaccessible (403/404).
    """
    url = f"{EXPORT_BASE}/{dataset_id}/export.csv"
    tmp_path = out_path.with_suffix(".csv.tmp")

    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            with session.get(url, stream=True, timeout=(30, 600)) as resp:
                if resp.status_code in (403, 404):
                    return 0, 0
                resp.raise_for_status()
                nbytes = 0
                with tmp_path.open("wb") as f:
                    for chunk in resp.iter_content(chunk_size=1 << 20):  # 1 MiB
                        if chunk:
                            f.write(chunk)
                            nbytes += len(chunk)
            rows = count_csv_rows(tmp_path)
            tmp_path.replace(out_path)          # atomic: only a complete stream is kept
            return rows, nbytes
        except (
            requests.exceptions.ReadTimeout,
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.ConnectionError,
        ) as exc:
            tmp_path.unlink(missing_ok=True)
            if attempt < DOWNLOAD_RETRIES:
                wait = min(60, 2 ** attempt)
                print(f"{prefix} data: {type(exc).__name__}, retry {attempt}/{DOWNLOAD_RETRIES} "
                      f"in {wait}s", flush=True)
                time.sleep(wait)
                continue
            raise
    return 0, 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_dirname(name: str, dataset_id: str) -> str:
    safe = "".join(c if c.isalnum() or c in " _-" else "_" for c in (name or ""))
    safe = safe.strip().replace(" ", "_")[:80]
    return f"{safe}__{dataset_id}"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape all tables from NYC Open Data.")
    parser.add_argument("--out-dir", default="./data")
    parser.add_argument("--catalog-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip datasets with existing data.csv")
    parser.add_argument("--skip-unchanged", action="store_true",
                        help="Skip download if data.csv exists and the source's rowsUpdatedAt "
                             "matches the recorded manifest (data unchanged since last run)")
    parser.add_argument("--no-attachments", action="store_true")
    parser.add_argument("--no-revisions", action="store_true")
    parser.add_argument("--workers", type=int, default=1, metavar="N",
                        help="parallel dataset workers (default: 1)")
    args = parser.parse_args()

    app_token = os.environ.get("NYC_APP_TOKEN")
    if not app_token:
        sys.exit("Error: set the NYC_APP_TOKEN environment variable before running.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    session = make_session(app_token)
    snapshots_path = out_dir / "snapshots.json"
    snapshots = load_snapshots(snapshots_path)
    downloads_path = out_dir / "downloads.json"
    downloads = load_snapshots(downloads_path)

    # --- Discover ---
    raw_datasets = discover_tables(session)
    catalog = [extract_catalog_entry(e) for e in raw_datasets]
    print(f"\nFound {len(catalog)} table datasets.\n")
    (out_dir / "catalog.json").write_text(
        json.dumps(catalog, indent=2, ensure_ascii=False)
    )

    if args.catalog_only:
        return

    failed = []
    failed_lock = threading.Lock()
    snapshots_lock = threading.Lock()
    downloads_lock = threading.Lock()
    total = len(catalog)

    def process(i: int, meta: dict) -> None:
        dataset_id = meta["id"]
        if not dataset_id:
            return

        name = meta["name"] or dataset_id
        tag = f"[{i}/{total}] {name} ({dataset_id})"
        prefix = f"  [{dataset_id}]"

        dirname = safe_dirname(name, dataset_id)
        ds_dir = out_dir / dirname
        ds_dir.mkdir(parents=True, exist_ok=True)

        print(tag, flush=True)

        # -- Full metadata --
        view_meta = fetch_view_metadata(session, dataset_id)
        if view_meta is None:
            print(f"{prefix} metadata: inaccessible, skipping", flush=True)
            return
        (ds_dir / "metadata.json").write_text(
            json.dumps(view_meta, indent=2, ensure_ascii=False)
        )
        time.sleep(RATE_LIMIT_SLEEP)

        # -- Schema snapshot + diff --
        current_schema = extract_schema(view_meta)
        current_schema["snapshot_time"] = now_iso()

        with snapshots_lock:
            if dataset_id in snapshots:
                diff = diff_schemas(snapshots[dataset_id], current_schema)
                (ds_dir / "diff.json").write_text(
                    json.dumps(diff, indent=2, ensure_ascii=False)
                )
                if diff["has_changes"]:
                    added = len(diff["columns_added"])
                    removed = len(diff["columns_removed"])
                    changed = len(diff["columns_changed"])
                    print(f"{prefix} schema diff: +{added} col, -{removed} col, ~{changed} changed", flush=True)
                else:
                    print(f"{prefix} schema diff: no changes", flush=True)
            else:
                print(f"{prefix} schema diff: first snapshot", flush=True)
            snapshots[dataset_id] = current_schema
            save_snapshots(snapshots_path, snapshots)

        # -- Revisions --
        if not args.no_revisions:
            revisions = fetch_revisions(session, dataset_id)
            if revisions is not None:
                (ds_dir / "revisions.json").write_text(
                    json.dumps(revisions, indent=2, ensure_ascii=False)
                )
                print(f"{prefix} revisions: {len(revisions)} logged", flush=True)
            else:
                print(f"{prefix} revisions: not accessible (403/404)", flush=True)
            time.sleep(RATE_LIMIT_SLEEP)

        # -- Attachments --
        if not args.no_attachments:
            saved = download_attachments(
                session, dataset_id, view_meta, ds_dir / "attachments"
            )
            if saved:
                print(f"{prefix} attachments: {len(saved)} saved ({', '.join(saved)})", flush=True)

        # -- Row data --
        csv_path = ds_dir / "data.csv"
        if args.resume and csv_path.exists():
            print(f"{prefix} data: skipped (exists)", flush=True)
            return

        # Change detection: rowsUpdatedAt is the source's last-data-change marker.
        # If it matches what we recorded and the file is still there, the data has
        # not changed since the last download, so we can skip it.
        cur_updated = view_meta.get("rowsUpdatedAt")
        prev = downloads.get(dataset_id)
        if (args.skip_unchanged and csv_path.exists() and prev
                and cur_updated is not None
                and prev.get("rows_updated_at") == cur_updated
                and prev.get("rows", 0) > 0):
            print(f"{prefix} data: unchanged since last run "
                  f"(rowsUpdatedAt={cur_updated}), skipping", flush=True)
            return

        started = now_iso()
        t0 = time.time()
        try:
            rows, nbytes = download_table(session, dataset_id, csv_path, prefix)
            if rows or nbytes:
                duration = time.time() - t0
                entry = {
                    "id": dataset_id,
                    "name": name,
                    "download_started": started,
                    "download_finished": now_iso(),
                    "duration_seconds": round(duration, 1),
                    "rows": rows,
                    "bytes": nbytes,
                    "rows_updated_at": cur_updated,          # the change-detection key
                    "view_last_modified": view_meta.get("viewLastModified"),
                    "source": "export.csv",
                }
                (ds_dir / "download.json").write_text(
                    json.dumps(entry, indent=2, ensure_ascii=False))
                with downloads_lock:
                    downloads[dataset_id] = entry
                    save_snapshots(downloads_path, downloads)
                print(f"{prefix} data: done - {rows:,} rows, {nbytes/1e6:.1f} MB "
                      f"in {duration:.0f}s", flush=True)
            else:
                print(f"{prefix} data: inaccessible (403/404)", flush=True)
        except Exception as exc:
            print(f"{prefix} data: ERROR {exc}", flush=True)
            with failed_lock:
                failed.append({"id": dataset_id, "name": name, "error": str(exc)})

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process, i, meta): meta
            for i, meta in enumerate(catalog, 1)
            if meta.get("id")
        }
        for future in as_completed(futures):
            exc = future.exception()
            if exc:
                meta = futures[future]
                with failed_lock:
                    failed.append({"id": meta.get("id"), "name": meta.get("name"), "error": str(exc)})

    if failed:
        fail_path = out_dir / "failed.json"
        fail_path.write_text(json.dumps(failed, indent=2))
        print(f"\n{len(failed)} datasets failed - see {fail_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
