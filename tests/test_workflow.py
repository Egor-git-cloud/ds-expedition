"""Small contract tests; no raw data, network, GPU, or external libraries."""
import csv
import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ds_expedition.common import read_json, write_json
from ds_expedition import preparation, prediction, splits


class WorkflowTests(unittest.TestCase):
    def test_frozen_split_is_complete_disjoint_and_grouped(self):
        path = ROOT / "configs/splits/v1.json"
        data = read_json(path)
        parts = [set(splits.selected_regions(path, p)) for p in splits.PARTITIONS]
        with (ROOT / "configs/catalog/files.csv").open(encoding="utf-8", newline="") as stream:
            expected = {r["region"] for r in csv.DictReader(stream)}
        self.assertEqual(set.union(*parts), expected)
        self.assertEqual(sum(map(len, parts)), len(expected))
        self.assertEqual(data["catalog_sha256"], hashlib.sha256((ROOT / "configs/catalog/files.csv").read_bytes()).hexdigest())

    def test_shared_group_across_partitions_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "split.json"
            write_json(file, {"partitions": {"train": ["001_A"], "validation": ["002_A"], "test": []},
                              "groups": {"001_A": "same", "002_A": "same"}})
            with self.assertRaises(ValueError):
                splits.selected_regions(file, "train")

    def test_name_families(self):
        policy = read_json(ROOT / "configs/splits/policy.json")
        self.assertEqual(splits.group_name("001_Гнёздово_Нивленское", policy),
                         splits.group_name("217_Гнёздово_Селище_3", policy))
        self.assertEqual(splits.group_name("006_Осечки_1.1км", policy),
                         splits.group_name("191_Осечки_1.3км", policy))

    def test_plan_works_with_nonexistent_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(preparation, "load_paths", return_value={"dataset_root": Path(directory) / "absent"}):
                with redirect_stdout(io.StringIO()) as out:
                    preparation.main(["--partition", "train", "--modalities", "sp_or", "--region", "161_Курганы_32"])
                result = json.loads(out.getvalue())
                self.assertGreater(result["file_count"], 0)
                self.assertEqual(result["regions"], ["161_Курганы_32"])

    def test_plan_rejects_region_from_other_partition(self):
        region = read_json(ROOT / "configs/splits/v1.json")["partitions"]["test"][0]
        with self.assertRaises(ValueError):
            preparation.main(["--partition", "train", "--region", region])

    def test_raster_crs_precedes_region_fallback(self):
        from unittest.mock import MagicMock
        for embedded, expected in (("EPSG:3857", "EPSG:3857"), (None, "EPSG:32637")):
            source = SimpleNamespace(crs=SimpleNamespace(to_string=lambda: embedded) if embedded else None)
            opened = MagicMock()
            opened.__enter__.return_value = source
            original = MagicMock(return_value=[])
            module = SimpleNamespace(rasterio=SimpleNamespace(open=MagicMock(return_value=opened)),
                                     process_extra_raster=original)
            prediction.prefer_embedded_raster_crs(module)
            module.process_extra_raster(None, {}, Path("raster.tif"), "sp_or", 1, 1,
                                        "EPSG:32637", "region", None)
            self.assertEqual(original.call_args.args[6], expected)

    def test_adapter_streams_distinct_fids_and_preserves_original_on_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("001_A", "002_B"):
                (root / name).mkdir()
            weight = root / "weights.pt"
            weight.write_bytes(b"fake")
            output = root / "prediction.json"
            counter = []

            def collect(model, classes, paths, config):
                counter.extend(paths)
                return [{"type": "Feature", "properties": {"fid": 0}, "geometry": None}]

            vendor = SimpleNamespace(process_extra_raster=lambda *args: [], InferConfig=lambda **kw: kw, RunConfig=lambda **kw: SimpleNamespace(**kw),
                                     configure_determinism=lambda seed: None, load_model=lambda path: (None, None),
                                     _collect_features=collect, _postprocess=lambda features, cfg: (features, {}))
            config = read_json(ROOT / "configs/baseline.json")
            with patch.object(prediction, "load_vendor", return_value=vendor):
                self.assertEqual(prediction.run_baseline(root, output, ["001_A", "002_B"], weight, config), 2)
                self.assertEqual([f["properties"]["fid"] for f in read_json(output)["features"]], [0, 1])
                self.assertEqual(counter, [root / "001_A", root / "002_B"])
                original = output.read_bytes()
                vendor._collect_features = lambda *args: (_ for _ in ()).throw(RuntimeError("simulated"))
                with self.assertRaises(RuntimeError):
                    prediction.run_baseline(root, output, ["001_A"], weight, config)
                self.assertEqual(output.read_bytes(), original)
                self.assertEqual(list(root.glob("*.json.tmp")), [])


if __name__ == "__main__":
    unittest.main()
