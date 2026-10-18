#!/usr/bin/env python3
"""Run the inventory tool directly from a checkout."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ds_expedition.audit import main

if __name__ == "__main__":
    raise SystemExit(main())
