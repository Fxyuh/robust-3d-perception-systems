#!/usr/bin/env python3
"""Export one ordered nuScenes scene for deterministic ROS 2 replay.

Each frame contains the five-channel tensor produced by the official
CenterPoint test pipeline after loading and aligning the current LiDAR sweep
plus nine historical sweeps. Labels and predictions are not modified or
exported.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from mmengine.config import Config
from mmengine.registry import init_default_scope

from mmdet3d.registry import DATASETS


ARRAY_MAGIC = b'CPDARR1\0'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default='configs/centerpoint/'
        'centerpoint_pillar02_nuscenes_mini_1gpu.py')
    parser.add_argument('--scene-name', default='scene-0103')
    parser.add_argument(
        '--output-root',
        default='work_dirs/centerpoint_deployment/ros2-scenes')
    return parser.parse_args()


def write_float_array(path: Path, value: np.ndarray) -> None:
    value = np.ascontiguousarray(value, dtype=np.float32)
    if value.ndim == 0 or value.ndim > 4:
        raise ValueError(f'unsupported array rank {value.ndim} for {path}')
    shape = tuple(value.shape) + (0, ) * (4 - value.ndim)
    header = struct.pack('<8sII4Q', ARRAY_MAGIC, 1, value.ndim, *shape)
    with path.open('wb') as stream:
        stream.write(header)
        stream.write(value.tobytes(order='C'))


def rotation_to_quaternion(rotation: np.ndarray) -> Sequence[float]:
    """Return normalized (qx, qy, qz, qw) from a 3x3 rotation matrix."""
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (rotation[2, 1] - rotation[1, 2]) / scale
        qy = (rotation[0, 2] - rotation[2, 0]) / scale
        qz = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        diagonal = np.diag(rotation)
        axis = int(np.argmax(diagonal))
        if axis == 0:
            scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] -
                              rotation[2, 2]) * 2.0
            qw = (rotation[2, 1] - rotation[1, 2]) / scale
            qx = 0.25 * scale
            qy = (rotation[0, 1] + rotation[1, 0]) / scale
            qz = (rotation[0, 2] + rotation[2, 0]) / scale
        elif axis == 1:
            scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] -
                              rotation[2, 2]) * 2.0
            qw = (rotation[0, 2] - rotation[2, 0]) / scale
            qx = (rotation[0, 1] + rotation[1, 0]) / scale
            qy = 0.25 * scale
            qz = (rotation[1, 2] + rotation[2, 1]) / scale
        else:
            scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] -
                              rotation[1, 1]) * 2.0
            qw = (rotation[1, 0] - rotation[0, 1]) / scale
            qx = (rotation[0, 2] + rotation[2, 0]) / scale
            qy = (rotation[1, 2] + rotation[2, 1]) / scale
            qz = 0.25 * scale
    quaternion = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    return quaternion.tolist()


def load_json_records(path: Path) -> Dict[str, dict]:
    records = json.loads(path.read_text(encoding='utf-8'))
    return {record['token']: record for record in records}


def main() -> None:
    args = parse_args()
    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))
    dataset_cfg = cfg.test_dataloader.dataset
    dataset = DATASETS.build(dataset_cfg)

    data_root = Path(dataset_cfg.data_root)
    version = dataset_cfg.metainfo.get('version', 'v1.0-mini')
    if version != 'v1.0-mini':
        raise ValueError(f'expected v1.0-mini, got {version}')
    table_root = data_root / version
    samples = load_json_records(table_root / 'sample.json')
    scenes = load_json_records(table_root / 'scene.json')
    matching_scene_tokens = [
        token for token, record in scenes.items()
        if record['name'] == args.scene_name
    ]
    if len(matching_scene_tokens) != 1:
        raise ValueError(
            f'expected one scene named {args.scene_name}, found '
            f'{len(matching_scene_tokens)}')
    scene_token = matching_scene_tokens[0]

    selected: List[dict] = []
    for dataset_index in range(len(dataset)):
        data_info = dataset.get_data_info(dataset_index)
        token = data_info['token']
        sample_record = samples[token]
        if sample_record['scene_token'] == scene_token:
            selected.append({
                'dataset_index': dataset_index,
                'data_info': data_info,
                'sample': sample_record,
            })
    selected.sort(key=lambda item: item['sample']['timestamp'])
    if not selected:
        raise RuntimeError(f'{args.scene_name} is absent from the test dataset')
    if len(selected) != scenes[scene_token]['nbr_samples']:
        raise RuntimeError('dataset scene length differs from nuScenes metadata')
    for current, following in zip(selected, selected[1:]):
        if (current['sample']['next'] != following['sample']['token'] or
                following['sample']['prev'] != current['sample']['token']):
            raise RuntimeError('scene frames do not form a valid prev/next chain')

    output_dir = Path(args.output_root) / args.scene_name
    temporary_dir = output_dir.with_name(output_dir.name + '.tmp')
    if output_dir.exists() or temporary_dir.exists():
        raise FileExistsError(
            f'refusing to overwrite existing export: {output_dir} '
            f'(or temporary directory {temporary_dir})')
    frames_dir = temporary_dir / 'frames'
    frames_dir.mkdir(parents=True)

    first_timestamp_ns = int(selected[0]['sample']['timestamp']) * 1000
    frame_records = []
    csv_records = []
    for sequence_index, item in enumerate(selected):
        dataset_index = item['dataset_index']
        data_info = item['data_info']
        sample_record = item['sample']
        points = dataset[dataset_index]['inputs']['points']
        points = points.detach().cpu().numpy().astype(np.float32, copy=False)
        if points.ndim != 2 or points.shape[1] != 5 or points.shape[0] == 0:
            raise RuntimeError(
                f'dataset index {dataset_index} has invalid shape {points.shape}')
        if not np.isfinite(points).all():
            raise RuntimeError(f'dataset index {dataset_index} has non-finite points')

        relative_path = Path('frames') / f'{sequence_index:06d}_points_f32.cpd'
        write_float_array(temporary_dir / relative_path, points)
        timestamp_ns = int(sample_record['timestamp']) * 1000
        lidar_to_global = np.asarray(data_info['ego2global'], dtype=np.float64) @ \
            np.asarray(data_info['lidar_points']['lidar2ego'], dtype=np.float64)
        qx, qy, qz, qw = rotation_to_quaternion(lidar_to_global[:3, :3])
        pose = {
            'translation': lidar_to_global[:3, 3].tolist(),
            'quaternion_xyzw': [qx, qy, qz, qw],
            'matrix_row_major': lidar_to_global.reshape(-1).tolist(),
        }
        record = {
            'sequence_index': sequence_index,
            'dataset_index': dataset_index,
            'sample_token': sample_record['token'],
            'timestamp_ns': timestamp_ns,
            'relative_time_s': (timestamp_ns - first_timestamp_ns) / 1e9,
            'point_count': int(points.shape[0]),
            'relative_path': relative_path.as_posix(),
            'lidar_to_global': pose,
        }
        frame_records.append(record)
        csv_records.append({
            'sequence_index': sequence_index,
            'dataset_index': dataset_index,
            'sample_token': sample_record['token'],
            'timestamp_ns': timestamp_ns,
            'point_count': int(points.shape[0]),
            'relative_path': relative_path.as_posix(),
            'tx': pose['translation'][0],
            'ty': pose['translation'][1],
            'tz': pose['translation'][2],
            'qx': qx,
            'qy': qy,
            'qz': qz,
            'qw': qw,
        })
        print(f'[{sequence_index + 1:02d}/{len(selected):02d}] '
              f'index={dataset_index} points={points.shape[0]} '
              f'token={sample_record["token"]}')

    timestamp_differences = np.diff(
        [record['timestamp_ns'] for record in frame_records]) / 1e9
    manifest = {
        'format_version': 1,
        'scene_name': args.scene_name,
        'scene_token': scene_token,
        'dataset_version': version,
        'config': args.config,
        'frame_count': len(frame_records),
        'duration_s': frame_records[-1]['relative_time_s'],
        'median_frame_interval_s': float(np.median(timestamp_differences)),
        'input_semantics': (
            'official CenterPoint nuScenes test-pipeline points after loading '
            'and aligning the current frame plus nine historical sweeps'),
        'point_fields': ['x', 'y', 'z', 'intensity', 'time_lag'],
        'coordinate_frame': 'current LIDAR_TOP frame',
        'frames': frame_records,
    }
    (temporary_dir / 'manifest.json').write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8')
    with (temporary_dir / 'frames.csv').open(
            'w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_records[0]))
        writer.writeheader()
        writer.writerows(csv_records)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir.rename(output_dir)
    print(f'Exported {len(frame_records)} frames to {output_dir}')
    print(f'Duration: {manifest["duration_s"]:.3f} s; median interval: '
          f'{manifest["median_frame_interval_s"]:.3f} s')


if __name__ == '__main__':
    main()
