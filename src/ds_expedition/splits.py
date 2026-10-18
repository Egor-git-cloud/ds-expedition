"""Create and consume a shared, metadata-only territorial split."""
import argparse
import csv
import hashlib
import random
import re
import unicodedata
from collections import Counter, defaultdict

from .common import read_json, write_json
from .paths import resolve_path

PARTITIONS = ("train", "validation", "test")


def safe_region(name: str) -> str:
    if not name or name in {".", ".."} or "/" in name or "\\" in name or ":" in name:
        raise ValueError(f"Invalid region identifier: {name!r}")
    return name


def selected_regions(split_path, partition):
    """Validate the entire split before returning selected identifiers."""
    data = read_json(split_path)
    seen = set()
    groups = set()
    for part in PARTITIONS:
        names = data["partitions"][part]
        local_groups = set()
        for name in names:
            safe_region(name)
            if name in seen:
                raise ValueError(f"Repeated region in split: {name}")
            seen.add(name)
            local_groups.add(data["groups"][name])
        if groups & local_groups:
            raise ValueError("Related territories occur in multiple partitions")
        groups.update(local_groups)
    if set(data["groups"]) != seen:
        raise ValueError("Split groups and partition membership differ")
    return data["partitions"][partition]


def group_name(region, policy):
    """Conservative name-based grouping; it does not prove spatial independence."""
    name = unicodedata.normalize("NFC", region.split("_", 1)[1]).casefold().replace("ё", "е")
    for token in policy["family_tokens"]:
        if token in name:
            return token
    name = re.sub(r"_\d+(?:\.\d+)?(?:км)?$", "", name)
    if name in policy["uncertain_series"]:
        return "unverified_named_series"
    return name


def create_split(rows, policy):
    """Balance region counts and class presence using only catalog metadata."""
    by_region = defaultdict(set)
    for row in rows:
        region = safe_region(row["region"])
        by_region[region]
        if row["is_annotation"].lower() == "true" and row["inferred_class"] not in ("", "unknown"):
            by_region[region].add(row["inferred_class"])
    groups = {r: group_name(r, policy) for r in sorted(by_region)}
    members = defaultdict(list)
    for r, g in groups.items():
        members[g].append(r)
    class_total = Counter(c for values in by_region.values() for c in values)
    ratios = policy["ratios"]
    if set(ratios) != set(PARTITIONS) or abs(sum(ratios.values()) - 1) > 1e-8 or min(ratios.values()) <= 0:
        raise ValueError("Ratios must be positive and sum to 1")
    pins = policy["pinned_groups"]
    if set(pins) - set(members) or set(pins.values()) - set(PARTITIONS):
        raise ValueError("Invalid pinned group or partition")

    def score(parts):
        value = 0.0
        for part in PARTITIONS:
            fraction = ratios[part]
            value += 4 * ((len(parts[part]) / len(groups)) - fraction) ** 2
            counts = Counter(c for r in parts[part] for c in by_region[r])
            value += sum((counts[c] / total - fraction) ** 2 for c, total in sorted(class_total.items()))
        return value

    rng = random.Random(policy["seed"])
    best = None
    best_score = float("inf")
    for _ in range(300):
        parts = {p: [] for p in PARTITIONS}
        for group, part in pins.items():
            parts[part].extend(members[group])
        pending = sorted(set(members) - set(pins))
        rng.shuffle(pending)
        for group in pending:
            choices = []
            for part in PARTITIONS:
                parts[part].extend(members[group])
                choices.append((score(parts), part))
                del parts[part][-len(members[group]):]
            chosen = min(choices)[1]
            parts[chosen].extend(members[group])
        # Prefer covering each class, where the pinned groups permit it.
        missing = sum(c not in {c for r in parts[p] for c in by_region[r]} for p in PARTITIONS for c in class_total)
        loss = score(parts) + missing * 10
        if loss < best_score:
            best_score, best = loss, {p: sorted(v) for p, v in parts.items()}
    return {
        "version": policy["version"], "seed": policy["seed"],
        "status": "provisional_name_groups_spatial_review_pending",
        "target_ratios": ratios, "partitions": best, "groups": groups,
        "class_presence_by_partition": {
            p: dict(sorted(Counter(c for r in best[p] for c in by_region[r]).items())) for p in PARTITIONS
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="configs/catalog/files.csv")
    parser.add_argument("--policy", default="configs/splits/policy.json")
    parser.add_argument("--output", default="configs/splits/v1.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    catalog, policy_path, output = map(resolve_path, (args.catalog, args.policy, args.output))
    if output.exists() and not args.overwrite:
        parser.error("Split exists. Use a new version/output, or explicitly --overwrite.")
    with catalog.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = create_split(rows, read_json(policy_path))
    result["catalog_sha256"] = hashlib.sha256(catalog.read_bytes()).hexdigest()
    result["policy_sha256"] = hashlib.sha256(policy_path.read_bytes()).hexdigest()
    write_json(output, result)
    for part in PARTITIONS:
        print(part, len(selected_regions(output, part)), result["class_presence_by_partition"][part])
    return 0
