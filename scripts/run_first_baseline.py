import os, sys, time, json, logging, threading, hashlib, platform
import importlib.metadata
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'src'))
for key, value in {'YOLO_CONFIG_DIR':'tmp/ultralytics', 'TORCH_HOME':'tmp/torch', 'MPLCONFIGDIR':'tmp/matplotlib', 'TEMP':'tmp/runtime', 'TMP':'tmp/runtime'}.items():
    path = root / value
    path.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(path)
os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='')
import psutil
from ds_expedition.paths import load_paths
from ds_expedition.common import read_json, write_json
from ds_expedition.prediction import run_baseline
out = root / 'results/first_run_161'
out.mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename=out/'run.log', encoding='utf-8', filemode='w', level=logging.INFO)
errors = []
class CaptureErrors(logging.Handler):
    def emit(self, record):
        if record.levelno >= logging.ERROR:
            errors.append(record.getMessage())
logging.getLogger().addHandler(CaptureErrors())
process = psutil.Process()
peak = [0]
stop = threading.Event()
def monitor():
    while not stop.is_set():
        info = process.memory_info()
        peak[0] = max(peak[0], info.rss, getattr(info, 'peak_wset', 0))
        stop.wait(0.2)
thread = threading.Thread(target=monitor, daemon=True)
thread.start()
started = time.perf_counter()
report = {'region':'161_Курганы_32', 'device':'cpu', 'threads':2, 'python':platform.python_version(), 'status':'running'}
try:
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    weights = root / 'vendor/organizer/baseline_solution/model.pt'
    config = read_json(root / 'configs/baseline.json')
    report['config'] = config
    report['weights_sha256'] = hashlib.sha256(weights.read_bytes()).hexdigest()
    report['versions'] = {n:importlib.metadata.version(n) for n in ['torch','ultralytics','numpy','rasterio','shapely','pyproj','affine','psutil']}
    print('Starting CPU inference...', flush=True)
    start_inference = time.perf_counter()
    count = run_baseline(load_paths()['dataset_root'], out/'predictions.geojson', ['161_Курганы_32'], weights, config)
    report['baseline_elapsed_seconds'] = round(time.perf_counter()-start_inference, 3)
    if errors:
        raise RuntimeError('Baseline logged errors: ' + '; '.join(errors))
    from shapely.geometry import shape
    from collections import Counter
    features = read_json(out/'predictions.geojson')['features']
    if any(not shape(f['geometry']).is_valid or shape(f['geometry']).is_empty for f in features):
        raise ValueError('Invalid or empty predicted geometry')
    report.update(feature_count=count, classes=dict(Counter(f['properties']['class_name'] for f in features)), status='success')
except Exception as exc:
    report.update(status='failed', error=repr(exc))
    logging.exception('First run failed')
    raise
finally:
    stop.set()
    thread.join()
    report['elapsed_seconds'] = round(time.perf_counter()-started, 3)
    info = process.memory_info()
    report['peak_process_working_set_mib'] = round(max(peak[0],info.rss,getattr(info,'peak_wset',0))/2**20,1)
    write_json(out/'report.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
