"""Plan selected files without opening or downloading the dataset."""
import argparse
import csv
import hashlib
import json
from pathlib import PurePosixPath

from .common import read_json, write_json
from .paths import load_paths, resolve_path
from .splits import PARTITIONS, selected_regions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", choices=PARTITIONS, default="train")
    parser.add_argument("--split", default="configs/splits/v1.json")
    parser.add_argument("--catalog", default="configs/catalog/files.csv")
    parser.add_argument("--modalities", default="la,li,ae,or,sp_or,maggr,etgr,terfl")
    parser.add_argument("--region", action="append", help="Optional subset of regions within the selected partition")
    parser.add_argument("--output", help="Optional JSON file; otherwise print a summary")
    args = parser.parse_args(argv)
    split, catalog = resolve_path(args.split), resolve_path(args.catalog)
    if hashlib.sha256(catalog.read_bytes()).hexdigest() != read_json(split)["catalog_sha256"]:
        raise ValueError("Catalog differs from the split snapshot; coordinate an updated version with the team")
    regions = set(selected_regions(split, args.partition))
    if args.region:
        if set(args.region) - regions:
            raise ValueError("Requested region is outside the selected partition")
        regions = set(args.region)
    modalities = set(args.modalities.split(","))
    if modalities - {"la", "li", "ae", "or", "sp_or", "maggr", "etgr", "terfl"}:
        raise ValueError("Unknown modality")
    with catalog.open(encoding="utf-8-sig", newline="") as stream:
        files = [r for r in csv.DictReader(stream) if r["region"] in regions and
                 (r["modality"] in modalities or PurePosixPath(r["relative_path"]).name == "UTM.json")]
    result = {"partition": args.partition, "split_sha256": hashlib.sha256(split.read_bytes()).hexdigest(),
              "dataset_root": str(load_paths()["dataset_root"]),
              "regions": sorted(regions), "file_count": len(files),
              "logical_size_gib": round(sum(int(r["size_bytes"]) for r in files) / 2**30, 3), "files": files}
    if args.output:
        write_json(resolve_path(args.output), result)
    print(json.dumps({k: v for k, v in result.items() if k != "files"}, ensure_ascii=False, indent=2))
    return 0
