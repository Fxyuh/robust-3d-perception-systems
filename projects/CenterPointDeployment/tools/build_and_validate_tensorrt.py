#!/usr/bin/env python3
"""Build a TensorRT engine and compare it with PyTorch reference tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import tensorrt as trt
import torch


LOGGER = trt.Logger(trt.Logger.WARNING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--onnx',
        default='work_dirs/centerpoint_deployment/onnx/'
        'centerpoint_dense_mixed_fp16_deep.onnx')
    parser.add_argument(
        '--reference',
        default='work_dirs/centerpoint_deployment/onnx/'
        'reference_sample_fp32.npz')
    parser.add_argument(
        '--engine',
        default='work_dirs/centerpoint_deployment/tensorrt/'
        'centerpoint_dense_mixed_fp16_deep.engine')
    parser.add_argument('--precision', choices=('fp16', 'fp32'), default='fp16')
    parser.add_argument('--workspace-gib', type=float, default=4.0)
    parser.add_argument(
        '--allow-tf32',
        action='store_true',
        help='Allow TensorRT to use TF32 for tensors declared as FP32.')
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--atol', type=float, default=0.1)
    parser.add_argument('--rtol', type=float, default=0.05)
    return parser.parse_args()


def build_engine(onnx_path: Path, engine_path: Path, precision: str,
                 workspace_gib: float, allow_tf32: bool) -> None:
    builder = trt.Builder(LOGGER)
    # TensorRT 11 removed EXPLICIT_BATCH because explicit dimensions are now
    # the only network mode.  Keep the branch for TensorRT 10 compatibility.
    explicit_batch = getattr(trt.NetworkDefinitionCreationFlag,
                             'EXPLICIT_BATCH', None)
    network_flags = 0 if explicit_batch is None else 1 << int(explicit_batch)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, LOGGER)
    model_bytes = onnx_path.read_bytes()
    if not parser.parse(model_bytes):
        errors = '\n'.join(str(parser.get_error(i))
                           for i in range(parser.num_errors))
        raise RuntimeError(f'failed to parse {onnx_path}:\n{errors}')

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(workspace_gib * (1 << 30)))
    if not allow_tf32:
        config.clear_flag(trt.BuilderFlag.TF32)
    if precision == 'fp16':
        fp16_tensors = sum(
            layer.get_output(index).dtype == trt.float16
            for layer_index in range(network.num_layers)
            for layer in [network.get_layer(layer_index)]
            for index in range(layer.num_outputs)
            if layer.get_output(index) is not None)
        if fp16_tensors == 0:
            raise RuntimeError(
                'FP16 build requested but the strongly typed ONNX graph has '
                'no FP16 tensors')
        # TensorRT 10 uses a weak-typing builder flag. TensorRT 11 removed it
        # and follows the explicit FP16 tensor types in the ONNX graph.
        legacy_fp16_flag = getattr(trt.BuilderFlag, 'FP16', None)
        if legacy_fp16_flag is not None:
            config.set_flag(legacy_fp16_flag)

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError('TensorRT engine build returned None')
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(serialized_engine)


def torch_dtype(trt_dtype: trt.DataType) -> torch.dtype:
    return torch.from_numpy(np.empty((), dtype=trt.nptype(trt_dtype))).dtype


def load_engine(engine_path: Path) -> trt.ICudaEngine:
    runtime = trt.Runtime(LOGGER)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError(f'failed to deserialize {engine_path}')
    return engine


def run_engine(engine: trt.ICudaEngine, reference: Dict[str, np.ndarray],
               warmup: int, iterations: int) -> tuple[Dict[str, np.ndarray],
                                                       Dict[str, float]]:
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError('failed to create TensorRT execution context')

    input_names: List[str] = []
    output_names: List[str] = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            input_names.append(name)
        else:
            output_names.append(name)
    if input_names != ['pseudo_image']:
        raise RuntimeError(f'unexpected engine inputs: {input_names}')

    input_tensor = torch.from_numpy(
        np.ascontiguousarray(reference['pseudo_image'])).cuda()
    if -1 in tuple(engine.get_tensor_shape('pseudo_image')):
        if not context.set_input_shape('pseudo_image', tuple(input_tensor.shape)):
            raise RuntimeError('failed to set dynamic input shape')
    context.set_tensor_address('pseudo_image', input_tensor.data_ptr())

    device_outputs: Dict[str, torch.Tensor] = {}
    for name in output_names:
        shape = tuple(context.get_tensor_shape(name))
        if any(dim < 0 for dim in shape):
            raise RuntimeError(f'unresolved output shape for {name}: {shape}')
        output = torch.empty(
            shape, dtype=torch_dtype(engine.get_tensor_dtype(name)), device='cuda')
        context.set_tensor_address(name, output.data_ptr())
        device_outputs[name] = output

    # Use a non-default stream so TensorRT does not insert conservative
    # synchronization around enqueueV3.
    torch.cuda.current_stream().synchronize()
    stream = torch.cuda.Stream()
    for _ in range(warmup):
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError('TensorRT warmup inference failed')
    stream.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(stream)
    for _ in range(iterations):
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError('TensorRT timed inference failed')
    end.record(stream)
    end.synchronize()
    total_ms = start.elapsed_time(end)

    outputs = {
        name: tensor.float().cpu().numpy()
        for name, tensor in device_outputs.items()
    }
    timing = {
        'iterations': iterations,
        'mean_latency_ms_dense_network': total_ms / iterations,
        'fps_dense_network': iterations * 1000.0 / total_ms,
        'torch_peak_allocated_mib':
        torch.cuda.max_memory_allocated() / (1 << 20),
    }
    return outputs, timing


def compare_outputs(reference: Dict[str, np.ndarray],
                    outputs: Dict[str, np.ndarray], atol: float,
                    rtol: float) -> Dict[str, Dict[str, float | bool]]:
    comparison: Dict[str, Dict[str, float | bool]] = {}
    for name, actual in outputs.items():
        expected = reference[name].astype(np.float32)
        actual = actual.astype(np.float32)
        difference = np.abs(actual - expected)
        passed = bool(np.allclose(actual, expected, atol=atol, rtol=rtol))
        comparison[name] = {
            'passed': passed,
            'max_abs_error': float(difference.max()),
            'mean_abs_error': float(difference.mean()),
        }
    return comparison


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for TensorRT validation')

    onnx_path = Path(args.onnx)
    reference_path = Path(args.reference)
    engine_path = Path(args.engine)
    report_path = engine_path.with_suffix('.json')
    if not engine_path.exists():
        build_engine(onnx_path, engine_path, args.precision, args.workspace_gib,
                     args.allow_tf32)

    with np.load(reference_path) as loaded:
        reference = {key: loaded[key] for key in loaded.files}

    torch.cuda.reset_peak_memory_stats()
    engine = load_engine(engine_path)
    outputs, timing = run_engine(engine, reference, args.warmup,
                                 args.iterations)
    output_path = engine_path.with_suffix('.outputs.npz')
    np.savez_compressed(output_path, **outputs)
    comparison = compare_outputs(reference, outputs, args.atol, args.rtol)
    report = {
        'onnx': str(onnx_path),
        'engine': str(engine_path),
        'precision': args.precision,
        'allow_tf32': args.allow_tf32,
        'tensorrt_version': trt.__version__,
        'gpu': torch.cuda.get_device_name(0),
        'timing': timing,
        'raw_outputs': str(output_path),
        'comparison': comparison,
    }
    report_path.write_text(
        json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(timing, indent=2))
    print(f'Report: {report_path}')
    failures = [
        name for name, values in comparison.items() if not values['passed']
    ]
    worst = sorted(
        comparison.items(),
        key=lambda item: item[1]['max_abs_error'],
        reverse=True)[:10]
    print('Largest absolute errors:')
    for name, values in worst:
        print(
            f"  {name}: max={values['max_abs_error']:.6g}, "
            f"mean={values['mean_abs_error']:.6g}")
    if failures:
        raise RuntimeError(
            'TensorRT parity failed for: ' + ', '.join(failures))
    print('TensorRT parity passed for all outputs')


if __name__ == '__main__':
    main()
