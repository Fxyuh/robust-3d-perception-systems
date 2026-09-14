#!/usr/bin/env python3
"""Merge per-scene ROS 2 tracks and run official nuScenes mini evaluation."""

from __future__ import annotations

import argparse
import collections
import collections.abc
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

# motmetrics 1.1.3 predates the Python 3.10 collections namespace change.
if not hasattr(collections, 'Iterable'):
    collections.Iterable = collections.abc.Iterable  # type: ignore[attr-defined]

import pandas as pd
from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.tracking.evaluate import TrackingEval
from nuscenes.eval.tracking.mot import MOTAccumulatorCustom
from nuscenes.utils.splits import mini_val
from motmetrics.mot import MOTAccumulator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--scene-results-root',
        default='work_dirs/centerpoint_deployment/tracking')
    parser.add_argument(
        '--combined-result',
        default='work_dirs/centerpoint_deployment/tracking/'
        'nuscenes_mini_tracking.json')
    parser.add_argument(
        '--output-dir',
        default='work_dirs/centerpoint_deployment/tracking/official_metrics')
    parser.add_argument('--dataroot', default='data/nuscenes')
    parser.add_argument('--version', default='v1.0-mini')
    parser.add_argument('--eval-set', default='mini_val')
    parser.add_argument('--render-curves', action='store_true')
    return parser.parse_args()


def expected_sample_tokens(nusc: NuScenes) -> List[str]:
    scene_by_name = {scene['name']: scene for scene in nusc.scene}
    tokens: List[str] = []
    for scene_name in mini_val:
        scene = scene_by_name[scene_name]
        token = scene['first_sample_token']
        scene_tokens = []
        while token:
            scene_tokens.append(token)
            token = nusc.get('sample', token)['next']
        if len(scene_tokens) != scene['nbr_samples']:
            raise RuntimeError(f'{scene_name} sample chain is incomplete')
        tokens.extend(scene_tokens)
    return tokens


def load_and_merge(root: Path, expected: List[str]) -> Dict[str, Any]:
    results: Dict[str, list] = {}
    expected_set = set(expected)
    for scene_name in mini_val:
        path = root / f'{scene_name}.json'
        if not path.is_file():
            raise FileNotFoundError(f'missing scene result: {path}')
        payload = json.loads(path.read_text(encoding='utf-8'))
        meta = payload.get('meta', {})
        if not meta.get('use_lidar') or meta.get('use_camera'):
            raise ValueError(f'unexpected sensor metadata in {path}')
        for token, tracks in payload.get('results', {}).items():
            if token in results:
                raise ValueError(f'duplicate sample token {token}')
            if token not in expected_set:
                raise ValueError(f'{path} contains non-mini-val token {token}')
            results[token] = tracks
    missing = expected_set - set(results)
    extra = set(results) - expected_set
    if missing or extra:
        raise ValueError(
            f'sample mismatch: missing={len(missing)}, extra={len(extra)}')
    ordered_results = {token: results[token] for token in expected}
    return {
        'meta': {
            'use_camera': False,
            'use_lidar': True,
            'use_radar': False,
            'use_map': False,
            'use_external': False,
        },
        'results': ordered_results,
    }


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    temporary = path.with_name(path.name + '.tmp')
    if path.exists():
        existing = json.loads(path.read_text(encoding='utf-8'))
        if existing != payload:
            raise FileExistsError(
                f'existing combined result differs; refusing to overwrite {path}')
        print(f'Reusing identical combined result {path}')
        return
    if temporary.exists():
        raise FileExistsError(f'refusing to overwrite {temporary}')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8')
    temporary.rename(path)


def write_summary_csv(path: Path, summary: Dict[str, Any]) -> None:
    label_metrics = summary['label_metrics']
    labels = list(next(iter(label_metrics.values())).keys())
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['metric', 'overall', *labels])
        for metric, values in label_metrics.items():
            writer.writerow(
                [metric, summary.get(metric), *[values[label] for label in labels]])


def compatible_event_dataframe(indices: Any, events: Any) -> pd.DataFrame:
    """Build events for both legacy and current motmetrics internals."""
    if not isinstance(indices, dict):
        index = pd.MultiIndex.from_tuples(
            indices, names=['FrameId', 'Event'])
        return pd.DataFrame(
            events,
            index=index,
            columns=['Type', 'OId', 'HId', 'D'])
    if len(events) == 0:
        return MOTAccumulator.new_event_dataframe()
    event_type = pd.Categorical(
        events['Type'],
        categories=[
            'RAW', 'FP', 'MISS', 'SWITCH', 'MATCH', 'TRANSFER', 'ASCEND',
            'MIGRATE'
        ],
        ordered=False)
    frame = pd.concat(
        [
            pd.Series(event_type, name='Type'),
            pd.Series(events['OId'], dtype=object, name='OId'),
            pd.Series(events['HId'], dtype=object, name='HId'),
            pd.Series(events['D'], dtype=float, name='D'),
        ],
        axis=1)
    frame.index = pd.MultiIndex.from_arrays(
        [indices['FrameId'], indices['Event']], names=['FrameId', 'Event'])
    return frame


def main() -> None:
    args = parse_args()
    if args.version != 'v1.0-mini' or args.eval_set != 'mini_val':
        raise ValueError('this tool intentionally evaluates v1.0-mini/mini_val')
    output_dir = Path(args.output_dir)
    summary_path = output_dir / 'metrics_summary.json'
    csv_path = output_dir / 'metrics_summary.csv'
    if (not summary_path.is_file() and output_dir.exists() and
            any(path.is_file() for path in output_dir.rglob('*'))):
        raise FileExistsError(
            f'refusing to overwrite files already present in {output_dir}')

    # nuScenes-devkit 1.1.11 overrides this constructor using legacy
    # motmetrics internals. Keep both list- and dictionary-based representations
    # compatible without changing the metric definitions.
    MOTAccumulatorCustom.new_event_dataframe_with_data = staticmethod(
        compatible_event_dataframe)

    nusc = NuScenes(
        version=args.version, dataroot=args.dataroot, verbose=False)
    expected = expected_sample_tokens(nusc)
    payload = load_and_merge(Path(args.scene_results_root), expected)
    combined_result = Path(args.combined_result)
    write_json_atomic(combined_result, payload)
    prediction_count = sum(len(items) for items in payload['results'].values())
    print(f'Merged {len(payload["results"])} samples and '
          f'{prediction_count} track boxes into {combined_result}')
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        if not csv_path.exists():
            write_summary_csv(csv_path, summary)
        print(f'Reusing official metrics in {summary_path}')
        print(f'Compact metrics table: {csv_path}')
        return

    evaluator = TrackingEval(
        config=config_factory('tracking_nips_2019'),
        result_path=str(combined_result),
        eval_set=args.eval_set,
        output_dir=str(output_dir),
        nusc_version=args.version,
        nusc_dataroot=args.dataroot,
        verbose=True)
    summary = evaluator.main(render_curves=args.render_curves)
    write_summary_csv(csv_path, summary)
    print(f'Wrote compact metrics table to {csv_path}')


if __name__ == '__main__':
    main()
