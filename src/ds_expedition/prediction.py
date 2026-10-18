"""Run the provided baseline on selected regions; default action is a plan."""
import argparse
import json
import tempfile
from pathlib import Path

from .common import load_vendor, read_json
from .paths import load_paths, project_root, resolve_path
from .splits import PARTITIONS, safe_region, selected_regions



def prefer_embedded_raster_crs(baseline):
    """Give raster coordinates their own CRS, using region UTM only as fallback.

    The organizer implementation prefers region_crs over src.crs. That is wrong
    when a Web Mercator raster and UTM.json coexist. Override this one call on
    the freshly loaded module; the original vendor file remains unchanged.
    """
    original = baseline.process_extra_raster

    def process(model, class_names, raster_path, modality, idx, total,
                region_crs, region_name, infer, window_pixels=48_000_000):
        with baseline.rasterio.open(raster_path) as source:
            effective_crs = source.crs.to_string() if source.crs else region_crs
        if not effective_crs:
            raise ValueError(f"No raster CRS or region CRS: {raster_path}")
        return original(model, class_names, raster_path, modality, idx, total,
                        effective_crs, region_name, infer, window_pixels)

    baseline.process_extra_raster = process


def run_baseline(dataset, output, regions, weights, config):
    """Process regions sequentially and stream features into an atomic result.

    This adapter uses the supplied RunConfig/processing functions directly so
    selecting a split never requires copying data or creating directory links.
    Original baseline internals may still allocate large arrays per region.
    """
    if not weights.is_file():
        raise FileNotFoundError(f"Baseline weights not found: {weights}")
    for region in regions:
        safe_region(region)
        if not (dataset / region).is_dir():
            raise FileNotFoundError(f"Missing region: {dataset / region}")
    baseline = load_vendor(project_root() / "vendor/organizer/baseline_solution/solution.py", "_organizer_baseline")
    prefer_embedded_raster_crs(baseline)
    infer = baseline.InferConfig(**{k: config[k] for k in (
        "conf_threshold", "tile_height", "tile_width", "overlap_ratio")})
    cfg = baseline.RunConfig(infer=infer, **{k: config[k] for k in (
        "hillshade_res", "hillshade_max_pixels", "window_pixels", "extra_modalities", "seed")})
    baseline.configure_determinism(cfg.seed)
    model, classes = baseline.load_model(str(weights))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                         suffix=".json.tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write('{"type":"FeatureCollection","features":[')
            count = 0
            for region in regions:
                features = baseline._collect_features(model, classes, [dataset / region], cfg)
                features, _ = baseline._postprocess(features, cfg)
                for feature in features:
                    feature["properties"]["fid"] = count
                    if count:
                        stream.write(",")
                    json.dump(feature, stream, ensure_ascii=False, allow_nan=False)
                    count += 1
            stream.write("]}\n")
        temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return count


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir")
    parser.add_argument("--output", default="results/baseline/predictions.geojson")
    parser.add_argument("--weights", help="Defaults to models_dir/baseline.pt; obtain weights separately")
    parser.add_argument("--config", default="configs/baseline.json")
    parser.add_argument("--split", default="configs/splits/v1.json")
    parser.add_argument("--partition", choices=PARTITIONS, default="validation")
    parser.add_argument("--region", action="append")
    parser.add_argument("--all-regions", action="store_true", help="Platform input: ignore development split")
    parser.add_argument("--execute", action="store_true", help="Actually read data and run the model")
    args = parser.parse_args(argv)
    paths = load_paths()
    dataset = resolve_path(args.input_dir) if args.input_dir else paths["dataset_root"]
    output = resolve_path(args.output)
    weights = resolve_path(args.weights) if args.weights else paths["models_dir"] / "baseline.pt"
    config = read_json(resolve_path(args.config))
    if args.all_regions:
        if args.region:
            parser.error("--all-regions cannot be combined with --region")
        regions = sorted(p.name for p in dataset.iterdir() if p.is_dir()) if args.execute else []
    else:
        regions = selected_regions(resolve_path(args.split), args.partition)
        if args.region:
            if set(args.region) - set(regions):
                parser.error("Requested region is outside the selected partition")
            regions = sorted(set(args.region))
    if not args.execute:
        print(json.dumps({"action": "plan_only", "input": str(dataset), "output": str(output),
                          "weights": str(weights), "regions": regions,
                          "all_input_regions_on_execution": args.all_regions, "config": config}, ensure_ascii=False, indent=2))
        return 0
    if not regions:
        parser.error("No regions selected")
    print(f"Saved {run_baseline(dataset, output, regions, weights, config)} features to {output}")
    return 0
