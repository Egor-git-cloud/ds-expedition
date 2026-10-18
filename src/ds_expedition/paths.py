"""Resolve shared defaults, local overrides and environment paths.

All relative paths are anchored at the project root, never the current working
directory. Loading settings does not create directories or open the dataset.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

PATH_KEYS = ("dataset_root", "manifests_dir", "prepared_data_dir", "models_dir", "results_dir")


def project_root() -> Path:
    """Locate the checkout (editable install) or an explicitly selected project."""
    override = os.environ.get("DS_EXPEDITION_PROJECT_ROOT")
    root = Path(override).expanduser().resolve() if override else Path(__file__).resolve().parents[2]
    if not (root / "configs" / "paths.json").is_file():
        raise FileNotFoundError(
            f"Project settings not found at {root / 'configs' / 'paths.json'}. "
            "Use an editable install or set DS_EXPEDITION_PROJECT_ROOT to the checkout."
        )
    return root


def resolve_path(value: str | Path, root: Path | None = None) -> Path:
    """Resolve an absolute or project-relative path, including ~ and env vars."""
    path = Path(os.path.expandvars(str(value))).expanduser()
    if not path.is_absolute():
        path = (root if root is not None else project_root()) / path
    return path.resolve()


def load_paths() -> dict[str, Path]:
    """Load defaults < local JSON < DS_EXPEDITION_<KEY> environment variables."""
    root = project_root()
    settings: dict[str, str] = {}
    for path in (root / "configs" / "paths.json", root / "configs" / "paths.local.json"):
        if not path.exists():
            continue
        with path.open(encoding="utf-8-sig") as stream:
            values = json.load(stream)
        if not isinstance(values, dict) or set(values) - set(PATH_KEYS):
            raise ValueError(f"Expected a JSON object with known path keys in {path}")
        settings.update(values)
    result = {}
    for key in PATH_KEYS:
        value = os.environ.get(f"DS_EXPEDITION_{key.upper()}", settings.get(key))
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Missing or empty path setting: {key}")
        result[key] = resolve_path(value, root)
    return result


def main() -> int:
    """Print effective settings without accessing the data."""
    print(json.dumps({key: str(value) for key, value in load_paths().items()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
