#!/usr/bin/env python3
"""Compare decoded PyTorch and TensorRT CenterPoint detections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from scipy.optimize import linear_sum_assignment

from mmdet3d.registry import DATASETS, MODELS


HEAD_ORDER = ('heatmap', 'reg', 'height', 'dim', 'rot', 'vel')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default='configs/centerpoint/centerpoint_pillar02_nuscenes_mini_1gpu.py')
    parser.add_argument(
        '--reference',
        default='work_dirs/centerpoint_deployment/onnx/'
        'reference_sample_fp32.npz')
    parser.add_argument(
        '--candidate',
        default='work_dirs/centerpoint_deployment/tensorrt/'
        'centerpoint_dense_mixed_fp16_deep.outputs.npz')
    parser.add_argument('--sample-index', type=int, default=0)
    parser.add_argument(
        '--report',
        default='work_dirs/centerpoint_deployment/tensorrt/'
        'decoded_parity_sample0.json')
    parser.add_argument(
        '--score-thresholds', type=float, nargs='+', default=[0.1, 0.2, 0.3])
    parser.add_argument('--acceptance-threshold', type=float, default=0.2)
    parser.add_argument('--min-matched-fraction', type=float, default=0.99)
    parser.add_argument('--max-mean-center-error-m', type=float, default=0.01)
    return parser.parse_args()


def load_npz(path: str) -> Dict[str, np.ndarray]:
    with np.load(path) as loaded:
        return {key: loaded[key] for key in loaded.files}


def rebuild_head_outputs(arrays: Dict[str, np.ndarray], task_count: int):
    tasks = []
    for task_index in range(task_count):
        prediction = {
            head: torch.from_numpy(
                np.ascontiguousarray(arrays[f'task{task_index}_{head}']))
            for head in HEAD_ORDER
        }
        tasks.append([prediction])
    return tuple(tasks)


def decode(cfg: Config, arrays: Dict[str, np.ndarray], sample_index: int):
    model = MODELS.build(cfg.model).eval()
    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    sample = dataset[sample_index]
    metadata = sample['data_samples'].metainfo
    task_count = len(model.pts_bbox_head.class_names)
    outputs = rebuild_head_outputs(arrays, task_count)
    with torch.no_grad():
        return model.pts_bbox_head.predict_by_feat(outputs, [metadata])[0]


def select(prediction, score_threshold: float):
    mask = prediction.scores_3d >= score_threshold
    return {
        'boxes': prediction.bboxes_3d.tensor[mask].cpu().numpy(),
        'scores': prediction.scores_3d[mask].cpu().numpy(),
        'labels': prediction.labels_3d[mask].cpu().numpy(),
    }


def compare_at_threshold(reference, candidate,
                         class_ids: Iterable[int]) -> Dict[str, object]:
    matched_center_errors = []
    matched_box_errors = []
    matched_score_errors = []
    matched = 0
    class_counts = {}
    for class_id in class_ids:
        ref_indices = np.flatnonzero(reference['labels'] == class_id)
        cand_indices = np.flatnonzero(candidate['labels'] == class_id)
        class_counts[str(class_id)] = {
            'pytorch': int(len(ref_indices)),
            'tensorrt': int(len(cand_indices)),
        }
        if len(ref_indices) == 0 or len(cand_indices) == 0:
            continue
        ref_boxes = reference['boxes'][ref_indices]
        cand_boxes = candidate['boxes'][cand_indices]
        cost = np.linalg.norm(
            ref_boxes[:, None, :2] - cand_boxes[None, :, :2], axis=2)
        rows, cols = linear_sum_assignment(cost)
        matched += len(rows)
        matched_center_errors.extend(cost[rows, cols].tolist())
        matched_box_errors.extend(
            np.abs(ref_boxes[rows] - cand_boxes[cols]).mean(axis=1).tolist())
        matched_score_errors.extend(
            np.abs(reference['scores'][ref_indices[rows]] -
                   candidate['scores'][cand_indices[cols]]).tolist())

    denominator = max(len(reference['boxes']), len(candidate['boxes']), 1)
    return {
        'pytorch_count': int(len(reference['boxes'])),
        'tensorrt_count': int(len(candidate['boxes'])),
        'matched_count': int(matched),
        'matched_fraction': float(matched / denominator),
        'mean_center_error_m': float(np.mean(matched_center_errors))
        if matched_center_errors else None,
        'max_center_error_m': float(np.max(matched_center_errors))
        if matched_center_errors else None,
        'mean_box_parameter_abs_error': float(np.mean(matched_box_errors))
        if matched_box_errors else None,
        'mean_score_abs_error': float(np.mean(matched_score_errors))
        if matched_score_errors else None,
        'class_counts': class_counts,
    }


def main() -> None:
    args = parse_args()
    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))
    reference_arrays = load_npz(args.reference)
    candidate_arrays = load_npz(args.candidate)
    reference_prediction = decode(cfg, reference_arrays, args.sample_index)
    candidate_prediction = decode(cfg, candidate_arrays, args.sample_index)
    class_count = sum(
        len(task['class_names']) for task in cfg.model.pts_bbox_head.tasks)

    comparisons = {}
    for threshold in args.score_thresholds:
        reference = select(reference_prediction, threshold)
        candidate = select(candidate_prediction, threshold)
        comparisons[str(threshold)] = compare_at_threshold(
            reference, candidate, range(class_count))

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
    acceptance_key = str(args.acceptance_threshold)
    if acceptance_key not in comparisons:
        raise RuntimeError(
            f'acceptance threshold {acceptance_key} was not evaluated')
    acceptance = comparisons[acceptance_key]
    if acceptance['matched_fraction'] < args.min_matched_fraction:
        raise RuntimeError('decoded prediction match fraction is too low')
    if (acceptance['mean_center_error_m'] is None or
            acceptance['mean_center_error_m'] >
            args.max_mean_center_error_m):
        raise RuntimeError('decoded prediction center error is too high')
    print('Decoded prediction parity passed')


if __name__ == '__main__':
    main()
