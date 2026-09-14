#!/usr/bin/env python3
"""Export one real PointPillars preprocessing sample for C++/CUDA parity.

The exported points are the five-channel tensor produced by the official
nuScenes test pipeline after loading and aligning the current frame plus nine
sweeps. Labels are neither loaded nor written.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.registry import init_default_scope
from mmengine.runner.checkpoint import load_checkpoint

from mmdet3d.registry import DATASETS, MODELS


ARRAY_MAGIC = b'CPDARR1\0'
DTYPE_CODES = {np.dtype(np.float32): 1, np.dtype(np.int32): 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default='configs/centerpoint/centerpoint_pillar02_nuscenes_mini_1gpu.py')
    parser.add_argument(
        '--checkpoint', default='checkpoints/centerpoint_pillar02_nuscenes.pth')
    parser.add_argument('--sample-index', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument(
        '--output-dir', default='work_dirs/centerpoint_deployment/preprocess')
    return parser.parse_args()


def write_array(path: Path, value: np.ndarray) -> None:
    value = np.ascontiguousarray(value)
    if value.dtype not in DTYPE_CODES:
        raise TypeError(f'unsupported dtype {value.dtype} for {path}')
    if value.ndim > 4:
        raise ValueError('arrays with more than four dimensions are unsupported')
    shape = tuple(value.shape) + (0, ) * (4 - value.ndim)
    header = struct.pack('<8sII4Q', ARRAY_MAGIC, DTYPE_CODES[value.dtype],
                         value.ndim, *shape)
    with path.open('wb') as stream:
        stream.write(header)
        stream.write(value.tobytes(order='C'))


def stats(value: np.ndarray) -> Dict[str, object]:
    return {
        'shape': list(value.shape),
        'dtype': str(value.dtype),
        'min': float(value.min()),
        'max': float(value.max()),
        'mean': float(value.astype(np.float64).mean()),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')

    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))
    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.to(device).eval()

    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    if not 0 <= args.sample_index < len(dataset):
        raise IndexError(
            f'sample index {args.sample_index} outside dataset length {len(dataset)}')
    batch = pseudo_collate([dataset[args.sample_index]])

    with torch.no_grad():
        processed = model.data_preprocessor(batch, training=False)
        points = processed['inputs']['points'][0].contiguous()
        voxel_dict = processed['inputs']['voxels']
        voxel_features = model.pts_voxel_encoder(
            voxel_dict['voxels'], voxel_dict['num_points'],
            voxel_dict['coors'])
        pseudo_image = model.pts_middle_encoder(
            voxel_features, voxel_dict['coors'], batch_size=1).contiguous()

    pfn = model.pts_voxel_encoder.pfn_layers[0]
    batch_norm = pfn.norm
    scale = batch_norm.weight / torch.sqrt(
        batch_norm.running_var + batch_norm.eps)
    folded_weight = pfn.linear.weight * scale[:, None]
    folded_bias = batch_norm.bias - batch_norm.running_mean * scale

    arrays = {
        'points_f32.cpd': points.detach().cpu().numpy().astype(np.float32),
        'voxels_f32.cpd': voxel_dict['voxels'].detach().cpu().numpy().astype(
            np.float32),
        'coors_i32.cpd': voxel_dict['coors'].detach().cpu().numpy().astype(
            np.int32),
        'num_points_i32.cpd': voxel_dict['num_points'].detach().cpu().numpy(
        ).astype(np.int32),
        'pfn_features_f32.cpd': voxel_features.detach().cpu().numpy().astype(
            np.float32),
        'pseudo_image_f32.cpd': pseudo_image.detach().cpu().numpy().astype(
            np.float32),
        'pfn_weight_f32.cpd': folded_weight.detach().cpu().numpy().astype(
            np.float32),
        'pfn_bias_f32.cpd': folded_bias.detach().cpu().numpy().astype(
            np.float32),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in arrays.items():
        write_array(output_dir / name, value)

    voxel_layer = model.data_preprocessor.voxel_layer
    encoder = model.pts_voxel_encoder
    manifest = {
        'config': args.config,
        'checkpoint': args.checkpoint,
        'sample_index': args.sample_index,
        'dataset_length': len(dataset),
        'input_semantics': (
            'official nuScenes test-pipeline points after multi-sweep loading '
            'and coordinate alignment'),
        'voxel_size': list(voxel_layer.voxel_size),
        'point_cloud_range': list(voxel_layer.point_cloud_range),
        'grid_shape': list(voxel_layer.grid_shape),
        'max_num_points': voxel_layer.max_num_points,
        'max_voxels_test': voxel_layer.max_voxels[1],
        'pfn': {
            'input_channels_after_decoration': encoder.in_channels,
            'output_channels': pfn.units,
            'with_cluster_center': encoder._with_cluster_center,
            'with_voxel_center': encoder._with_voxel_center,
            'with_distance': encoder._with_distance,
            'legacy': encoder.legacy,
            'batch_norm_eps': batch_norm.eps,
            'batch_norm_folded': True,
        },
        'arrays': {name: stats(value) for name, value in arrays.items()},
    }
    (output_dir / 'manifest.json').write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8')

    print(f'Exported sample {args.sample_index}: {points.shape[0]} points')
    print(f'Hard voxels: {voxel_dict["voxels"].shape[0]}')
    print(f'Pseudo image: {tuple(pseudo_image.shape)}')
    print(f'Output: {output_dir}')


if __name__ == '__main__':
    main()
