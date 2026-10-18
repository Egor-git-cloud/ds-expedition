#!/usr/bin/env python3
"""Print settings without accessing the dataset."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ds_expedition.paths import main

if __name__ == "__main__":
    raise SystemExit(main())
