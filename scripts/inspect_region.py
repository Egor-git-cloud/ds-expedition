"""Inspect one raster and its annotations, then score supplied predictions locally.

Reads only selected catalog entries. No GT merging across sources is performed;
positive-area overlaps within a class stop the score for manual review.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from ds_expedition.common import read_json, write_json, load_vendor
from ds_expedition.paths import load_paths, resolve_path
from ds_expedition.splits import safe_region


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--region', required=True)
    parser.add_argument('--modality', required=True)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--classes', help='Optional comma-separated class subset; partial GT will not be scored')
    parser.add_argument('--annotation-cache', help='Optional local root with replacement annotations at their original relative paths')
    args = parser.parse_args()
    safe_region(args.region)
    output = resolve_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cache = ROOT / 'tmp/matplotlib'
    cache.mkdir(parents=True, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(cache)
    import numpy as np
    import rasterio
    from pyproj import CRS, Transformer
    from shapely.geometry import shape, mapping, box
    from shapely.ops import transform
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    dataset = load_paths()['dataset_root']
    with (ROOT / 'configs/catalog/files.csv').open(encoding='utf-8', newline='') as stream:
        rows = [r for r in csv.DictReader(stream) if r['region'] == args.region and r['modality'] == args.modality]
    rasters = [r for r in rows if r['extension'] == '.tif']
    annotations = [r for r in rows if r['is_annotation'].lower() == 'true']
    if len(rasters) != 1 or not annotations:
        raise ValueError('This diagnostic requires exactly one raster and nonempty annotation file list')
    all_annotation_count = len(annotations)
    if args.classes:
        selected = set(args.classes.split(','))
        annotations = [r for r in annotations if r['inferred_class'] in selected]
        if not annotations:
            raise ValueError('No annotations match requested classes')
    partial = len(annotations) != all_annotation_count
    features, sources = [], []
    for row in annotations:
        path = dataset / row['relative_path']
        if args.annotation_cache:
            cached = resolve_path(args.annotation_cache) / row['relative_path']
            if cached.is_file():
                path = cached
        print('Reading annotation:', path.name, flush=True)
        raw = path.read_bytes()
        data = json.loads(raw.decode('utf-8-sig'))
        crs_name = (data.get('crs') or {}).get('properties', {}).get('name')
        if not crs_name:
            raise ValueError(f'Missing explicit CRS: {path.name}')
        source_crs = CRS.from_user_input(crs_name)
        converter = Transformer.from_crs(source_crs, 3857, always_xy=True)
        if row['inferred_class'] in ('', 'unknown'):
            raise ValueError('Unknown annotation class')
        sources.append({'relative_path': row['relative_path'], 'sha256': hashlib.sha256(raw).hexdigest(),
                        'source_crs': source_crs.to_string(), 'feature_count': len(data['features'])})
        for source_index, feature in enumerate(data['features']):
            geom = shape(feature['geometry'])
            if geom.geom_type not in ('Polygon', 'Point') or geom.is_empty or not geom.is_valid:
                raise ValueError(f'Unsupported/invalid geometry: {path.name}, feature {source_index}')
            geom = transform(converter.transform, geom)
            features.append({'type':'Feature', 'geometry':mapping(geom), 'properties': {
                'fid':len(features), 'class_name':row['inferred_class'], 'region_name':args.region,
                'markup_type':args.modality, 'crs':'EPSG:3857', 'source_file':row['relative_path'],
                'source_fid':(feature.get('properties') or {}).get('fid', source_index)}})
    truth = {'type':'FeatureCollection', 'features':features}
    write_json(output / ('ground_truth.partial.geojson' if partial else 'ground_truth.geojson'), truth)
    predictions = read_json(resolve_path(args.predictions))
    if predictions.get('type') != 'FeatureCollection':
        raise ValueError('Expected prediction FeatureCollection')
    for feature in predictions['features']:
        props = feature['properties']
        if props.get('region_name') != args.region or props.get('markup_type') != args.modality:
            raise ValueError('Prediction is outside diagnostic scope')
        geom = shape(feature['geometry'])
        if geom.geom_type != 'Polygon' or not geom.is_valid or geom.is_empty:
            raise ValueError('Invalid prediction geometry')
    overlaps = []
    for i, a in enumerate(features):
        for j in range(i+1, len(features)):
            b = features[j]
            if a['properties']['class_name'] == b['properties']['class_name']:
                if shape(a['geometry']).intersection(shape(b['geometry'])).area > 0:
                    overlaps.append([i, j])
    print('Reading raster...', flush=True)
    with rasterio.open(dataset / rasters[0]['relative_path']) as raster:
        if not raster.crs:
            raise ValueError('Missing raster CRS')
        if raster.width * raster.height > 4_000_000:
            raise ValueError('Small-raster diagnostic is limited to 4 million pixels')
        image = raster.read()
        if image.dtype != np.uint8 or raster.count not in (3,4):
            raise ValueError('Preview currently expects uint8 RGB/RGBA')
        raster_crs = raster.crs.to_string()
        inverse = ~raster.transform
        bounds = tuple(raster.bounds)
        width, height = raster.width, raster.height
    converter = Transformer.from_crs(3857, raster_crs, always_xy=True)
    geometries = [transform(converter.transform, shape(f['geometry'])) for f in features]
    footprint = box(*bounds)
    inside = [footprint.covers(g) for g in geometries]
    intersects = [footprint.intersects(g) for g in geometries]
    report = {'region':args.region, 'modality':args.modality, 'scope':'single region diagnostic, not full validation', 'partial_gt':partial,
              'annotation_files_used':len(annotations), 'annotation_files_expected':all_annotation_count,
              'raster':{'width':width,'height':height,'crs':raster_crs,'bounds':bounds},
              'gt_counts':dict(Counter(f['properties']['class_name'] for f in features)),
              'prediction_counts':dict(Counter(f['properties']['class_name'] for f in predictions['features'])),
              'gt_inside_raster':sum(inside),'gt_intersecting_raster':sum(intersects),
              'within_class_positive_area_overlaps':overlaps, 'sources':sources}
    colors = {'kurgany_tselye':'#16e870', 'kurgany_povrezhdennye':'#ffae28'}
    fig, axes = plt.subplots(1,2, figsize=(14,7))
    for ax in axes:
        ax.imshow(np.moveaxis(image,0,-1))
        ax.set_xlim(0,width)
        ax.set_ylim(height,0)
        ax.axis('off')
    axes[0].set_title('Satellite image')
    axes[1].set_title(f'Ground truth: {len(features)} | Predictions: {len(predictions["features"])}')
    for feature, geom in zip(features, geometries):
        color = colors.get(feature['properties']['class_name'],'cyan')
        if geom.geom_type == 'Polygon':
            for ring in [geom.exterior, *geom.interiors]:
                points = np.array([inverse * xy for xy in ring.coords])
                axes[1].plot(points[:,0],points[:,1],color=color,linewidth=1.4)
        else:
            x,y = inverse * (geom.x,geom.y)
            axes[1].scatter([x],[y],color=color,s=25)
    for feature in predictions['features']:
        geom = transform(converter.transform,shape(feature['geometry']))
        points = np.array([inverse*xy for xy in geom.exterior.coords])
        axes[1].plot(points[:,0],points[:,1],color='cyan',linestyle='--')
    fig.legend(handles=[Line2D([0],[0],color=colors['kurgany_tselye'],label='GT: intact mound'),
                        Line2D([0],[0],color=colors['kurgany_povrezhdennye'],label='GT: damaged mound'),
                        Line2D([0],[0],color='cyan',linestyle='--',label='Prediction')],loc='lower center',ncol=3)
    fig.suptitle(f'Region {args.region.split(chr(95))[0]} | ' + ('PARTIAL GT: damaged mounds only' if partial else 'Ground truth check'))
    fig.tight_layout(rect=(0,0.05,1,0.96))
    fig.savefig(output/'ground_truth_preview.png',dpi=140)
    plt.close(fig)
    if partial:
        report['metric_status'] = 'not_calculated_incomplete_ground_truth'
        write_json(output/'gt_report.json',report)
        print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
        return
    if overlaps or not all(intersects):
        report['metric_status'] = 'blocked_pending_overlap_or_coverage_review'
        write_json(output/'gt_report.json',report)
        raise ValueError('Review GT overlaps/coverage before calculating the local metric')
    metric = load_vendor(ROOT/'vendor/organizer/compute_metrics.py','_organizer_metric_diagnostic')
    result = metric.F1Calculator().calculate_f1(predictions,truth)
    split_data = read_json(ROOT/'configs/splits/v1.json')
    partitions = [part for part, names in split_data['partitions'].items() if args.region in names]
    result['scope'] = {'region':args.region,'modality':args.modality,'split':partitions,'full_validation':False}
    write_json(output/'metrics.json',result)
    report['final_score'] = result['final_score']
    report['metric_status'] = 'local_diagnostic_only'
    write_json(output/'gt_report.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)

if __name__ == '__main__':
    main()
