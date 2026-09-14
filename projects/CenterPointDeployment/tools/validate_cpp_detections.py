#!/usr/bin/env python3
"""Compare final C++ decoded detections with the PyTorch reference."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
from mmengine.config import Config
from mmengine.registry import init_default_scope

from validate_decoded_predictions import (compare_at_threshold, decode,
                                            load_npz, select)


ARRAY_HEADER = struct.Struct('<8sII4Q')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default='configs/centerpoint/centerpoint_pillar02_nuscenes_mini_1gpu.py')
    parser.add_argument(
        '--reference',
        default='work_dirs/centerpoint_deployment/onnx/reference_sample_fp32.npz')
    parser.add_argument(
        '--candidate',
        default='work_dirs/centerpoint_deployment/tensorrt/'
        'detections_cpp_f32.cpd')
    parser.add_argument('--sample-index', type=int, default=0)
    parser.add_argument(
        '--report',
        default='work_dirs/centerpoint_deployment/tensorrt/'
        'decoded_parity_cpp_end_to_end_sample0.json')
    parser.add_argument(
        '--score-thresholds', type=float, nargs='+', default=[0.1, 0.2, 0.3])
    parser.add_argument('--acceptance-threshold', type=float, default=0.2)
    parser.add_argument('--min-matched-fraction', type=float, default=0.99)
    parser.add_argument('--max-mean-center-error-m', type=float, default=0.01)
    return parser.parse_args()


def read_detections(path: Path):
    with path.open('rb') as stream:
        header = stream.read(ARRAY_HEADER.size)
        if len(header) != ARRAY_HEADER.size:
            raise RuntimeError(f'truncated CPD header: {path}')
        magic, dtype_code, ndim, *shape = ARRAY_HEADER.unpack(header)
        if magic != b'CPDARR1\0' or dtype_code != 1 or ndim != 2:
            raise RuntimeError(f'invalid detection CPD array: {path}')
        dimensions = tuple(shape[:ndim])
        values = np.fromfile(stream, dtype=np.float32)
    if len(dimensions) != 2 or dimensions[1] != 11:
        raise RuntimeError(f'expected [N, 11] detections, got {dimensions}')
    values = values.reshape(dimensions)
    return {
        'boxes': values[:, :9],
        'scores': values[:, 9],
        'labels': values[:, 10].astype(np.int64),
    }


def main() -> None:
    args = parse_args()
    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))
    reference = decode(cfg, load_npz(args.reference), args.sample_index)
    candidate = read_detections(Path(args.candidate))
    class_count = sum(
        len(task['class_names']) for task in cfg.model.pts_bbox_head.tasks)

    comparisons = {}
    for threshold in args.score_thresholds:
        comparisons[str(threshold)] = compare_at_threshold(
            select(reference, threshold), {
                'boxes': candidate['boxes'][candidate['scores'] >= threshold],
                'scores': candidate['scores'][candidate['scores'] >= threshold],
                'labels': candidate['labels'][candidate['scores'] >= threshold],
            }, range(class_count))

    report = {
        'config': args.config,
        'sample_index': args.sample_index,
        'reference': args.reference,
        'candidate': args.candidate,
        'thresholds': comparisons,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(comparisons, indent=2))
    print(f'Report: {report_path}')

    acceptance = comparisons[str(args.acceptance_threshold)]
    if acceptance['matched_fraction'] < args.min_matched_fraction:
        raise RuntimeError('C++ decoded prediction match fraction is too low')
    if (acceptance['mean_center_error_m'] is None or
            acceptance['mean_center_error_m'] >
            args.max_mean_center_error_m):
        raise RuntimeError('C++ decoded prediction center error is too high')
    print('C++ end-to-end decoded prediction parity passed')


if __name__ == '__main__':
    main()
