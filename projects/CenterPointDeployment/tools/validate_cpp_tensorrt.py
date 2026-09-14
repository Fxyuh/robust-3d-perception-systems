#!/usr/bin/env python3
"""Run the dense TensorRT engine from the C++/CUDA pseudo image."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from build_and_validate_tensorrt import (compare_outputs, load_engine,
                                         run_engine)


ARRAY_HEADER = struct.Struct('<8sII4Q')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pseudo-image',
        default='work_dirs/centerpoint_deployment/preprocess/'
        'pseudo_image_cpp_f32.cpd')
    parser.add_argument(
        '--reference',
        default='work_dirs/centerpoint_deployment/onnx/'
        'reference_sample_fp32.npz')
    parser.add_argument(
        '--engine',
        default='work_dirs/centerpoint_deployment/tensorrt/'
        'centerpoint_dense_mixed_fp16_deep.engine')
    parser.add_argument(
        '--output',
        default='work_dirs/centerpoint_deployment/tensorrt/'
        'centerpoint_cpp_preprocess_mixed_fp16.outputs.npz')
    parser.add_argument(
        '--report',
        default='work_dirs/centerpoint_deployment/tensorrt/'
        'centerpoint_cpp_preprocess_mixed_fp16.json')
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--atol', type=float, default=0.1)
    parser.add_argument('--rtol', type=float, default=0.05)
    return parser.parse_args()


def read_float_cpd(path: Path) -> np.ndarray:
    with path.open('rb') as stream:
        header = stream.read(ARRAY_HEADER.size)
        if len(header) != ARRAY_HEADER.size:
            raise RuntimeError(f'truncated CPD header: {path}')
        magic, dtype_code, ndim, *shape = ARRAY_HEADER.unpack(header)
        if magic != b'CPDARR1\0' or dtype_code != 1 or not 1 <= ndim <= 4:
            raise RuntimeError(f'invalid float32 CPD array: {path}')
        dimensions = tuple(shape[:ndim])
        value = np.fromfile(stream, dtype=np.float32)
    if value.size != int(np.prod(dimensions)):
        raise RuntimeError(f'CPD payload size does not match shape: {path}')
    return np.ascontiguousarray(value.reshape(dimensions))


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as loaded:
        return {name: loaded[name] for name in loaded.files}


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for TensorRT validation')

    reference = load_npz(Path(args.reference))
    cpp_pseudo_image = read_float_cpd(Path(args.pseudo_image))
    if cpp_pseudo_image.shape != reference['pseudo_image'].shape:
        raise RuntimeError(
            f'pseudo image shape mismatch: {cpp_pseudo_image.shape} vs '
            f'{reference["pseudo_image"].shape}')
    inference_inputs = dict(reference)
    inference_inputs['pseudo_image'] = cpp_pseudo_image

    torch.cuda.reset_peak_memory_stats()
    engine = load_engine(Path(args.engine))
    outputs, timing = run_engine(engine, inference_inputs, args.warmup,
                                 args.iterations)
    comparison = compare_outputs(reference, outputs, args.atol, args.rtol)

    output_path = Path(args.output)
    report_path = Path(args.report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **outputs)
    pseudo_difference = np.abs(
        cpp_pseudo_image - reference['pseudo_image'].astype(np.float32))
    report = {
        'pseudo_image': args.pseudo_image,
        'reference': args.reference,
        'engine': args.engine,
        'raw_outputs': str(output_path),
        'timing': timing,
        'preprocess_parity': {
            'max_abs_error': float(pseudo_difference.max()),
            'mean_abs_error': float(pseudo_difference.mean()),
        },
        'dense_output_comparison': comparison,
    }
    report_path.write_text(
        json.dumps(report, indent=2) + '\n', encoding='utf-8')

    failures = [
        name for name, values in comparison.items() if not values['passed']
    ]
    worst = sorted(
        comparison.items(),
        key=lambda item: item[1]['max_abs_error'],
        reverse=True)[:5]
    print(json.dumps(timing, indent=2))
    print('Largest end-to-end raw tensor errors:')
    for name, values in worst:
        print(
            f"  {name}: max={values['max_abs_error']:.6g}, "
            f"mean={values['mean_abs_error']:.6g}")
    print(f'Outputs: {output_path}')
    print(f'Report: {report_path}')
    if failures:
        raise RuntimeError('C++ preprocessing + TensorRT parity failed for: ' +
                           ', '.join(failures))
    print('C++ preprocessing + TensorRT raw-output parity passed')


if __name__ == '__main__':
    main()
