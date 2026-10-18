"""Platform entry: input directory and output GeoJSON FILE (not directory)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ds_expedition.prediction import main as predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir")
    parser.add_argument("output_file")
    parser.add_argument("--weights")
    parser.add_argument("--config", default="configs/baseline.json")
    args = parser.parse_args()
    command = ["--input-dir", args.input_dir, "--output", args.output_file,
               "--all-regions", "--execute", "--config", args.config]
    if args.weights:
        command.extend(["--weights", args.weights])
    return predict(command)


if __name__ == "__main__":
    raise SystemExit(main())
