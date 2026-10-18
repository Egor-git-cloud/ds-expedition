"""Calculate the supplied metric on already prepared, selected GeoJSON files."""
import argparse

from .common import load_vendor, read_json, write_json
from .paths import project_root, resolve_path
from .splits import PARTITIONS, selected_regions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--ground-truth", required=True, help="Merged GT in EPSG:3857 with canonical class names")
    parser.add_argument("--output", default="results/baseline/metrics.json")
    parser.add_argument("--split", default="configs/splits/v1.json")
    parser.add_argument("--partition", choices=PARTITIONS, default="validation")
    args = parser.parse_args(argv)
    names = set(selected_regions(resolve_path(args.split), args.partition))
    prediction = read_json(resolve_path(args.predictions))
    truth = read_json(resolve_path(args.ground_truth))
    classes = {"kurgany_tselye", "kurgany_povrezhdennye", "gorodishcha", "arkhitektury",
               "fortifikatsii", "finds_points", "object_poly"}
    for label, data in (("predictions", prediction), ("ground truth", truth)):
        if data.get("type") != "FeatureCollection" or not isinstance(data.get("features"), list):
            raise ValueError(f"Invalid FeatureCollection: {label}")
        for feature in data["features"]:
            props = feature.get("properties") or {}
            if props.get("region_name") not in names:
                raise ValueError(f"{label}: missing region_name or region outside selected partition")
            if props.get("class_name") not in classes:
                raise ValueError(f"{label}: unknown/missing canonical class_name")
            allowed = {"Polygon"} if label == "predictions" else {"Polygon", "MultiPolygon", "Point"}
            if (feature.get("geometry") or {}).get("type") not in allowed:
                raise ValueError(f"{label}: unsupported geometry")
    if not truth["features"]:
        raise ValueError("Ground truth is empty")
    metric = load_vendor(project_root() / "vendor/organizer/compute_metrics.py", "_organizer_metric")
    result = metric.F1Calculator().calculate_f1(prediction, truth)
    write_json(resolve_path(args.output), result)
    print(f"Final score: {result['final_score']:.6f}")
    return 0
