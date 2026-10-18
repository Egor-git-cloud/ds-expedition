#!/usr/bin/env python3
"""Inventory the KOZ-3 training dataset without opening large raster files.

The default run reads file names and filesystem metadata only. Pass
``--count-annotations`` to open GeoJSON annotation files and count features by
class; this may cause Yandex.Disk to download those files on demand.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .paths import load_paths, resolve_path


MODALITY_MARKERS = {
    "La": "la",
    "Li": "li",
    "Ae": "ae",
    "Ае": "ae",
    "Or": "or",
    "SpOr": "sp_or",
    "MagGr": "maggr",
    "EtGr": "etgr",
    "TerFl": "terfl",
}

CLASS_MARKERS = (
    ("курганы_поврежденные", "kurgany_povrezhdennye"),
    ("курганы_повреждённые", "kurgany_povrezhdennye"),
    ("курганы_целые", "kurgany_tselye"),
    ("городища", "gorodishcha"),
    ("фортификации", "fortifikatsii"),
    ("архитектуры", "arkhitektury"),
    ("finds_points", "finds_points"),
    ("object_poly", "object_poly"),
    ("findspoints", "finds_points"),
    ("objectpoly", "object_poly"),
    ("находки", "finds_points"),
    ("объекты", "object_poly"),
)


def infer_modality(path: Path) -> str:
    """Guess the source modality from folder and filename tokens."""
    text = f"{path.parent.name}_{path.stem}".casefold()
    for marker, modality in MODALITY_MARKERS.items():
        if re.search(rf"(?:^|_){re.escape(marker.casefold())}(?:_|$)", text):
            return modality
    return "unknown"


def infer_class(path: Path) -> str:
    """Guess a canonical class from an annotation filename."""
    text = path.stem.casefold()
    for marker, class_name in CLASS_MARKERS:
        if marker.casefold() in text:
            return class_name
    return "unknown"


def read_utm(region_dir: Path) -> str:
    path = region_dir / "UTM.json"
    if not path.exists():
        return ""
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        return str(data.get("crs", data.get("utm", "")))
    except (OSError, json.JSONDecodeError):
        return "unreadable"


def annotation_feature_count(path: Path) -> tuple[int | None, str]:
    """Read a GeoJSON file when explicitly requested and return feature count."""
    try:
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8-sig"))
        return len(data.get("features", [])), ""
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        return None, str(exc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, nargs="?", help="Корневая папка train-koz-3 (по умолчанию из настроек)")
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Куда записать inventory.csv, regions.csv и summary.json",
    )
    parser.add_argument(
        "--count-annotations", action="store_true",
        help="Прочитать GeoJSON и посчитать объекты; файлы могут скачаться с Яндекс Диска",
    )
    parser.add_argument(
        "--read-utm", action="store_true",
        help="Открыть UTM.json каждой территории; синхронизируемый диск может загружать файлы",
    )
    args = parser.parse_args()

    paths = load_paths()
    root = resolve_path(args.dataset) if args.dataset is not None else paths["dataset_root"]
    args.output_dir = resolve_path(args.output_dir) if args.output_dir is not None else paths["manifests_dir"]
    if not root.is_dir():
        parser.error(f"Папка датасета не найдена: {root}")

    files = sorted(path for path in root.rglob("*") if path.is_file())
    regions = sorted(path for path in root.iterdir() if path.is_dir())

    inventory: list[dict[str, Any]] = []
    region_stats: dict[str, dict[str, Any]] = {
        region.name: {
            "region": region.name,
            "utm_crs": read_utm(region) if args.read_utm else "",
            "file_count": 0,
            "size_bytes": 0,
            "las_count": 0,
            "tif_count": 0,
            "geojson_count": 0,
            "annotation_count": 0,
        }
        for region in regions
    }
    annotations_by_class: Counter[str] = Counter()
    feature_counts_by_class: Counter[str] = Counter()
    extension_counts: Counter[str] = Counter()
    modality_counts: Counter[str] = Counter()
    total_bytes = 0

    for path in files:
        relative_path = path.relative_to(root)
        region_name = relative_path.parts[0] if len(relative_path.parts) > 1 else ""
        extension = path.suffix.casefold()
        size_bytes = path.stat().st_size
        is_annotation = extension == ".geojson" and any(
            "разметка" in part.casefold() or "markup" in part.casefold()
            for part in relative_path.parts[:-1]
        )
        class_name = infer_class(path) if is_annotation else ""
        modality = infer_modality(path) if extension in {".tif", ".las", ".geojson"} else ""
        row: dict[str, Any] = {
            "region": region_name,
            "relative_path": relative_path.as_posix(),
            "extension": extension,
            "size_bytes": size_bytes,
            "modality": modality,
            "is_annotation": is_annotation,
            "inferred_class": class_name,
            "feature_count": "",
            "read_error": "",
        }

        stats = region_stats.get(region_name)
        if stats is not None:
            stats["file_count"] += 1
            stats["size_bytes"] += size_bytes
            if extension == ".las":
                stats["las_count"] += 1
            elif extension == ".tif":
                stats["tif_count"] += 1
            elif is_annotation:
                stats["geojson_count"] += 1
                stats["annotation_count"] += 1

        if is_annotation:
            annotations_by_class[class_name] += 1
            if args.count_annotations:
                feature_count, read_error = annotation_feature_count(path)
                row["feature_count"] = "" if feature_count is None else feature_count
                row["read_error"] = read_error
                if feature_count is not None:
                    feature_counts_by_class[class_name] += feature_count

        inventory.append(row)
        extension_counts[extension or "[no extension]"] += 1
        if modality:
            modality_counts[modality] += 1
        total_bytes += size_bytes

    args.output_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = args.output_dir / "inventory.csv"
    regions_path = args.output_dir / "regions.csv"
    summary_path = args.output_dir / "summary.json"

    with inventory_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(inventory[0]) if inventory else [])
        if inventory:
            writer.writeheader()
            writer.writerows(inventory)

    region_rows = list(region_stats.values())
    with regions_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(region_rows[0]) if region_rows else [])
        if region_rows:
            writer.writeheader()
            writer.writerows(region_rows)

    summary = {
        "dataset_root": str(root),
        "regions": len(regions),
        "files": len(files),
        "total_size_bytes": total_bytes,
        "total_size_gib": round(total_bytes / (1024 ** 3), 3),
        "extensions": dict(sorted(extension_counts.items())),
        "files_by_modality": dict(sorted(modality_counts.items())),
        "annotation_files_by_inferred_class": dict(sorted(annotations_by_class.items())),
        "annotation_features_by_inferred_class": (
            dict(sorted(feature_counts_by_class.items())) if args.count_annotations else None
        ),
        "annotation_feature_counts_requested": args.count_annotations,
        "outputs": {
            "inventory_csv": str(inventory_path),
            "regions_csv": str(regions_path),
            "summary_json": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Territories: {len(regions)}")
    print(f"Files: {len(files)} ({summary['total_size_gib']:.2f} GiB by filesystem metadata)")
    print("Extensions:", ", ".join(f"{key}={value}" for key, value in sorted(extension_counts.items())))
    print("Annotations by inferred class:", json.dumps(summary["annotation_files_by_inferred_class"], ensure_ascii=False))
    print(f"Wrote: {inventory_path}, {regions_path}, {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
