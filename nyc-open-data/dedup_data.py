"""
De-duplicate dataset folders that share a Socrata id.

The same dataset can be scraped into two folders (usually a rename-in-place), 
giving two  directories with the same trailing id. That double-counts everywhere 
downstream. This keeps the most complete folder per id and sets the duplicates aside.

"Most complete" = largest data.csv; ties broken by newest metadata.json. A folder
with data.csv always beats one without (a failed/partial scrape).

SAFE BY DEFAULT: a dry run that only prints. With --apply it MOVES duplicates
into a quarantine folder (reversible). Add --delete to remove them outright.

Usage
-----
  python dedup_data.py [--dir data]            # dry run: show collisions
  python dedup_data.py --dir data --apply      # move duplicates to data/_duplicates/
  python dedup_data.py --dir data --apply --delete   # delete duplicates instead
"""

import argparse
import re
import shutil
from collections import defaultdict
from pathlib import Path

ID_RE = re.compile(r"([a-z0-9]{4}-[a-z0-9]{4})$")


def completeness(folder: Path) -> tuple:
    """Sort key: (has data.csv, data.csv size, metadata.json mtime). Higher = keep."""
    csv = folder / "data.csv"
    size = csv.stat().st_size if csv.exists() else -1
    md = folder / "metadata.json"
    mt = md.stat().st_mtime if md.exists() else 0.0
    return (size >= 0, size, mt)


def main() -> None:
    ap = argparse.ArgumentParser(description="Keep one folder per dataset id; set duplicates aside.")
    ap.add_argument("--dir", default="data")
    ap.add_argument("--quarantine", default=None,
                    help="where to move duplicates (default <dir>/_duplicates)")
    ap.add_argument("--apply", action="store_true", help="actually act (default: dry run)")
    ap.add_argument("--delete", action="store_true", help="delete duplicates instead of moving")
    args = ap.parse_args()

    root = Path(args.dir)
    quarantine = Path(args.quarantine) if args.quarantine else root / "_duplicates"

    by_id = defaultdict(list)
    for sub in root.iterdir():
        if not sub.is_dir() or sub == quarantine:
            continue
        m = ID_RE.search(sub.name)
        if m:
            by_id[m.group(1)].append(sub)

    collisions = {i: fs for i, fs in by_id.items() if len(fs) > 1}
    if not collisions:
        print(f"No id collisions among {len(by_id)} dataset folder(s). Nothing to do.")
        return

    print(f"{len(collisions)} id(s) with duplicate folders "
          f"(of {len(by_id)} unique ids):\n")
    moved = 0
    for did, folders in sorted(collisions.items()):
        folders.sort(key=completeness, reverse=True)
        keep, dupes = folders[0], folders[1:]
        kc = keep / "data.csv"
        ksz = f"{kc.stat().st_size/1e6:.1f} MB" if kc.exists() else "no data.csv"
        print(f"  {did}")
        print(f"     KEEP  {keep.name}  ({ksz})")
        for d in dupes:
            dc = d / "data.csv"
            dsz = f"{dc.stat().st_size/1e6:.1f} MB" if dc.exists() else "no data.csv"
            action = "would move"
            if args.apply:
                if args.delete:
                    shutil.rmtree(d); action = "DELETED"
                else:
                    quarantine.mkdir(exist_ok=True)
                    shutil.move(str(d), str(quarantine / d.name)); action = "moved"
            print(f"     {action}: {d.name}  ({dsz})")
            moved += 1

    if args.apply:
        dest = "deleted" if args.delete else f"moved to {quarantine}"
        print(f"\n{moved} duplicate folder(s) {dest}.")
        print("Regenerate clusters.json / version_groups.json so they reflect the dedup.")
    else:
        print(f"\nDRY RUN - {moved} folder(s) would be set aside. "
              f"Re-run with --apply (moves to quarantine) or --apply --delete.")


if __name__ == "__main__":
    main()
