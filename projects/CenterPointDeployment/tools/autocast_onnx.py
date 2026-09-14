#!/usr/bin/env python3
"""Convert the FP32 dense CenterPoint ONNX model to mixed FP16."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import onnx
import onnxruntime as ort
from modelopt.onnx.autocast import convert_to_mixed_precision


DEFAULT_EXCLUDED_NODES = [
    r'^/backbone/blocks\.0/.*',
    r'^/backbone/blocks\.1/.*',
    r'^/shared_conv/.*',
    r'^/task_heads\..*',
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--onnx',
        default='work_dirs/centerpoint_deployment/onnx/'
        'centerpoint_dense_fp32.onnx')
    parser.add_argument(
        '--reference',
        default='work_dirs/centerpoint_deployment/onnx/'
        'reference_sample_fp32.npz')
    parser.add_argument(
        '--output',
        default='work_dirs/centerpoint_deployment/onnx/'
        'centerpoint_dense_mixed_fp16_deep.onnx')
    parser.add_argument('--max-abs-error', type=float, default=0.25)
    parser.add_argument('--max-mean-error', type=float, default=0.02)
    parser.add_argument('--data-max', type=float, default=512.0)
    parser.add_argument('--max-depth-of-reduction', type=int, default=100000)
    parser.add_argument(
        '--exclude-node',
        action='append',
        default=DEFAULT_EXCLUDED_NODES,
        help='Regex for ONNX node names that must remain FP32; repeatable.')
    return parser.parse_args()


def compare_onnx(model_path: Path,
                 reference: Dict[str, np.ndarray]) -> Dict[str, object]:
    session = ort.InferenceSession(
        str(model_path), providers=['CPUExecutionProvider'])
    inputs = {item.name: reference[item.name] for item in session.get_inputs()}
    output_values = session.run(None, inputs)
    comparison: Dict[str, Dict[str, float]] = {}
    for metadata, actual in zip(session.get_outputs(), output_values):
        expected = reference[metadata.name].astype(np.float32)
        difference = np.abs(actual.astype(np.float32) - expected)
        comparison[metadata.name] = {
            'max_abs_error': float(difference.max()),
            'mean_abs_error': float(difference.mean()),
        }
    return {
        'onnxruntime_version': ort.__version__,
        'outputs': comparison,
        'global_max_abs_error': max(
            item['max_abs_error'] for item in comparison.values()),
        'global_max_mean_error': max(
            item['mean_abs_error'] for item in comparison.values()),
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.onnx)
    reference_path = Path(args.reference)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    calibration_path = output_path.with_name('autocast_input_fp32.npz')
    report_path = output_path.with_suffix('.json')

    with np.load(reference_path) as loaded:
        reference = {key: loaded[key] for key in loaded.files}
    np.savez_compressed(calibration_path, pseudo_image=reference['pseudo_image'])

    converted_model = convert_to_mixed_precision(
        onnx_path=str(input_path),
        low_precision_type='fp16',
        keep_io_types=True,
        calibration_data=str(calibration_path),
        providers=['cpu'],
        data_max=args.data_max,
        max_depth_of_reduction=args.max_depth_of_reduction,
        nodes_to_exclude=args.exclude_node or None)
    onnx.checker.check_model(converted_model)
    onnx.save(converted_model, str(output_path))

    graph_stats = {
        'nodes': len(converted_model.graph.node),
        'cast_nodes': sum(node.op_type == 'Cast'
                          for node in converted_model.graph.node),
        'fp16_initializers': sum(
            initializer.data_type == onnx.TensorProto.FLOAT16
            for initializer in converted_model.graph.initializer),
        'fp32_initializers': sum(
            initializer.data_type == onnx.TensorProto.FLOAT
            for initializer in converted_model.graph.initializer),
    }
    parity = compare_onnx(output_path, reference)
    report = {
        'source': str(input_path),
        'output': str(output_path),
        'calibration_input': str(calibration_path),
        'keep_io_types': True,
        'data_max': args.data_max,
        'max_depth_of_reduction': args.max_depth_of_reduction,
        'excluded_node_patterns': args.exclude_node,
        'graph': graph_stats,
        'parity': parity,
    }
    report_path.write_text(
        json.dumps(report, indent=2) + '\n', encoding='utf-8')

    print(json.dumps(graph_stats, indent=2))
    print(json.dumps({
        'global_max_abs_error': parity['global_max_abs_error'],
        'global_max_mean_error': parity['global_max_mean_error'],
    }, indent=2))
    print(f'Mixed precision ONNX: {output_path}')
    print(f'Report: {report_path}')
    if parity['global_max_abs_error'] > args.max_abs_error:
        raise RuntimeError('mixed precision maximum absolute error is too high')
    if parity['global_max_mean_error'] > args.max_mean_error:
        raise RuntimeError('mixed precision mean error is too high')


if __name__ == '__main__':
    main()
