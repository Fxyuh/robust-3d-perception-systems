#!/usr/bin/env python3
"""Export the dense BEV part of the pillar CenterPoint model to ONNX.

The deployment boundary is intentionally placed after pillar feature encoding
and scatter.  This keeps the ONNX graph limited to standard dense operators,
while voxelization, pillar encoding and scatter remain the next C++/CUDA
milestone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.registry import init_default_scope
from mmengine.runner.checkpoint import load_checkpoint
from torch import Tensor, nn

from mmdet3d.registry import DATASETS, MODELS


HEAD_ORDER = ('heatmap', 'reg', 'height', 'dim', 'rot', 'vel')


class DenseCenterPoint(nn.Module):
    """SECOND backbone, SECONDFPN and CenterHead with flat tensor outputs."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.backbone = model.pts_backbone
        self.neck = model.pts_neck
        self.shared_conv = model.pts_bbox_head.shared_conv
        self.task_heads = model.pts_bbox_head.task_heads

    def forward(self, pseudo_image: Tensor) -> Tuple[Tensor, ...]:
        features = self.neck(self.backbone(pseudo_image))[0]
        shared = self.shared_conv(features)
        outputs: List[Tensor] = []
        for task_head in self.task_heads:
            task_outputs = task_head(shared)
            for head_name in HEAD_ORDER:
                outputs.append(task_outputs[head_name])
        return tuple(outputs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default='configs/centerpoint/centerpoint_pillar02_nuscenes_mini_1gpu.py')
    parser.add_argument(
        '--checkpoint', default='checkpoints/centerpoint_pillar02_nuscenes.pth')
    parser.add_argument('--sample-index', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--precision', choices=('fp16', 'fp32'), default='fp16')
    parser.add_argument(
        '--allow-tf32',
        action='store_true',
        help='Allow CUDA TF32 while generating the PyTorch reference.')
    parser.add_argument('--opset', type=int, default=17)
    parser.add_argument(
        '--output-dir', default='work_dirs/centerpoint_deployment/onnx')
    return parser.parse_args()


def build_model(config_path: str, checkpoint_path: str,
                device: torch.device) -> Tuple[Config, nn.Module]:
    cfg = Config.fromfile(config_path)
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))
    model = MODELS.build(cfg.model)
    load_checkpoint(model, checkpoint_path, map_location='cpu')
    model.to(device).eval()
    return cfg, model


def get_one_sample(cfg: Config, sample_index: int) -> Dict:
    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    if not 0 <= sample_index < len(dataset):
        raise IndexError(
            f'sample index {sample_index} outside dataset of length {len(dataset)}')
    return pseudo_collate([dataset[sample_index]])


def extract_pseudo_image(model: nn.Module, batch: Dict) -> Tensor:
    processed = model.data_preprocessor(batch, training=False)
    voxel_dict = processed['inputs']['voxels']
    batch_data_samples = processed['data_samples']
    batch_input_metas = [sample.metainfo for sample in batch_data_samples]

    voxel_features = model.pts_voxel_encoder(
        voxel_dict['voxels'], voxel_dict['num_points'], voxel_dict['coors'],
        None, batch_input_metas)
    batch_size = int(voxel_dict['coors'][-1, 0].item()) + 1
    pseudo_image = model.pts_middle_encoder(
        voxel_features, voxel_dict['coors'], batch_size)
    return pseudo_image.contiguous()


def output_names(task_classes: Sequence[Sequence[str]]) -> List[str]:
    return [
        f'task{task_index}_{head_name}'
        for task_index in range(len(task_classes))
        for head_name in HEAD_ORDER
    ]


def tensor_stats(tensor: Tensor) -> Dict[str, object]:
    return {
        'shape': list(tensor.shape),
        'dtype': str(tensor.dtype).removeprefix('torch.'),
        'min': float(tensor.min().item()),
        'max': float(tensor.max().item()),
        'mean': float(tensor.float().mean().item()),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available')
    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    torch.backends.cudnn.allow_tf32 = args.allow_tf32

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / f'centerpoint_dense_{args.precision}.onnx'
    reference_path = output_dir / f'reference_sample_{args.precision}.npz'
    manifest_path = output_dir / f'manifest_{args.precision}.json'

    cfg, model = build_model(args.config, args.checkpoint, device)
    batch = get_one_sample(cfg, args.sample_index)
    # ONNX's legacy tracer may save convolution inputs internally.  Tensors
    # created under inference_mode forbid that in PyTorch 2.7, while no_grad
    # still avoids autograd work and leaves ordinary tensors for export.
    with torch.no_grad():
        pseudo_image = extract_pseudo_image(model, batch)
        dense_model = DenseCenterPoint(model).to(device).eval()
        if args.precision == 'fp16':
            # TensorRT 11 requires precision to be explicit in the graph.
            # Exporting half weights and tensors produces a strongly typed
            # FP16 ONNX model without adding ModelOpt as another dependency.
            dense_model.half()
            pseudo_image = pseudo_image.half()
        reference_outputs = dense_model(pseudo_image)

    task_classes = model.pts_bbox_head.class_names
    names = output_names(task_classes)
    if len(names) != len(reference_outputs):
        raise RuntimeError(
            f'expected {len(names)} outputs, got {len(reference_outputs)}')

    torch.onnx.export(
        dense_model,
        (pseudo_image, ),
        str(onnx_path),
        input_names=['pseudo_image'],
        output_names=names,
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False)

    import onnx
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    reference_arrays = {
        'pseudo_image': pseudo_image.detach().cpu().numpy(),
    }
    reference_arrays.update({
        name: value.detach().cpu().numpy()
        for name, value in zip(names, reference_outputs)
    })
    np.savez_compressed(reference_path, **reference_arrays)

    manifest = {
        'deployment_boundary': (
            'PointPillarsScatter output -> SECOND -> SECONDFPN -> CenterHead '
            'raw task tensors'),
        'config': args.config,
        'checkpoint': args.checkpoint,
        'sample_index': args.sample_index,
        'precision': args.precision,
        'allow_tf32': args.allow_tf32,
        'opset': args.opset,
        'input': {
            'name': 'pseudo_image',
            **tensor_stats(pseudo_image),
        },
        'outputs': {
            name: tensor_stats(value)
            for name, value in zip(names, reference_outputs)
        },
        'task_classes': task_classes,
        'head_order': list(HEAD_ORDER),
        'onnx_file': onnx_path.name,
        'reference_file': reference_path.name,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8')

    print(f'ONNX check passed: {onnx_path}')
    print(f'Input shape: {tuple(pseudo_image.shape)}')
    print(f'Output tensors: {len(reference_outputs)}')
    print(f'Reference sample: {reference_path}')
    print(f'Manifest: {manifest_path}')


if __name__ == '__main__':
    main()
