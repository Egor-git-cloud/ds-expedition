"""Unified checkout command; no package installation required for plans/help."""
import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("split", "plan", "predict", "evaluate", "train"))
    args = parser.parse_args(sys.argv[1:2])
    rest = sys.argv[2:]
    module = {"split": "splits", "plan": "preparation", "predict": "prediction",
              "evaluate": "evaluation", "train": "training"}[args.command]
    return importlib.import_module(f"ds_expedition.{module}").main(rest)


if __name__ == "__main__":
    raise SystemExit(main())
