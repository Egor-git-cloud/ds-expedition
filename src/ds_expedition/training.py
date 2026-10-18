"""Training contract template. No dataset preparation or model training yet."""
import argparse
import json

from .common import read_json
from .paths import resolve_path
from .splits import selected_regions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.json")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    config = read_json(resolve_path(args.config))
    if config["train_partition"] != "train" or config["tuning_partition"] != "validation":
        raise ValueError("Training uses train and tuning uses validation; test is held out")
    split = resolve_path(config["split_file"])
    print(json.dumps({"status": "template", "config": config,
                      "train_regions": len(selected_regions(split, "train")),
                      "validation_regions": len(selected_regions(split, "validation"))}, ensure_ascii=False, indent=2))
    if args.execute:
        parser.error("Training backend and georeferenced tile/label preparation are not implemented yet")
    return 0
