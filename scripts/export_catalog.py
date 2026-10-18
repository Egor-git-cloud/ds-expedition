"""Export only shared relative metadata from a previously generated inventory."""
import argparse
import csv
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ("region", "relative_path", "extension", "size_bytes", "modality", "is_annotation", "inferred_class")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=ROOT / "data/manifests/inventory.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "configs/catalog/files.csv")
    args = parser.parse_args()
    with args.inventory.open(encoding="utf-8-sig", newline="") as stream:
        rows = [{k: r[k] for k in COLUMNS} for r in csv.DictReader(stream)]
    for row in rows:
        path = PurePosixPath(row["relative_path"])
        if path.is_absolute() or ".." in path.parts or "\\" in str(path) or ":" in str(path):
            raise ValueError(f"Non-relative catalog path: {path}")
        if path.parts[0] != row["region"]:
            raise ValueError(f"Region/path mismatch: {path}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["relative_path"]))
    print(f"Exported {len(rows)} metadata rows. Dataset was not accessed.")


if __name__ == "__main__":
    main()
