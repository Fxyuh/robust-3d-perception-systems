# Copyright (c) OpenMMLab. All rights reserved.
"""Deterministic input degradation for BEVFusion evaluation and training."""

import hashlib
import os.path as osp
from typing import Any, Dict, Optional, Sequence

import numpy as np
from mmcv.transforms import BaseTransform

from mmdet3d.registry import TRANSFORMS


def _sample_seed(seed: int, sample_key: str, stream: str) -> int:
    """Derive an order-independent RNG seed for one sample and stream."""
    payload = f'{seed}:{stream}:{sample_key}'.encode('utf-8')
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder='little', signed=False)


@TRANSFORMS.register_module(name='V1FrozenSensorInputDegradation')
class SensorInputDegradation(BaseTransform):
    """Apply deterministic camera and LiDAR degradation to model inputs.

    Camera subsets and point masks are derived from the sample identifier, so
    results do not depend on dataloader worker count or iteration order. For a
    fixed seed, increasing a degradation level produces a nested subset.

    Args:
        camera_drop_count: Number of camera views replaced by ``camera_mean``.
        lidar_drop_ratio: Fraction of input points to remove.
        seed: Global experiment seed.
        camera_mean: Per-channel value which becomes zero after the model's
            image normalization.
    """

    def __init__(
        self,
        camera_drop_count: int = 0,
        lidar_drop_ratio: float = 0.0,
        seed: int = 0,
        camera_mean: Sequence[float] = (123.675, 116.28, 103.53)
    ) -> None:
        if camera_drop_count < 0:
            raise ValueError('camera_drop_count cannot be negative')
        if not 0.0 <= lidar_drop_ratio < 1.0:
            raise ValueError('lidar_drop_ratio must be in [0, 1)')
        if len(camera_mean) != 3:
            raise ValueError('camera_mean must contain three values')

        self.camera_drop_count = camera_drop_count
        self.lidar_drop_ratio = lidar_drop_ratio
        self.seed = seed
        self.camera_mean = np.asarray(camera_mean, dtype=np.float32)

    @staticmethod
    def _sample_key(results: Dict[str, Any]) -> str:
        for key in ('sample_idx', 'token', 'lidar_path'):
            if key in results:
                return str(results[key])
        raise KeyError('A stable sample identifier is required')

    def _drop_cameras(
        self,
        results: Dict[str, Any],
        sample_key: str,
        camera_drop_count: Optional[int] = None
    ) -> list:
        camera_drop_count = (self.camera_drop_count if camera_drop_count is None
                             else camera_drop_count)
        images = results.get('img')
        if images is None or camera_drop_count == 0:
            return []

        num_views = len(images)
        if camera_drop_count > num_views:
            raise ValueError(
                f'Cannot drop {camera_drop_count} of {num_views} views')
        rng = np.random.default_rng(
            _sample_seed(self.seed, sample_key, 'camera'))
        dropped_indices = sorted(
            rng.permutation(num_views)[:camera_drop_count].tolist())
        for index in dropped_indices:
            images[index] = np.full_like(images[index], self.camera_mean)
        results['img'] = images
        return dropped_indices

    def _drop_lidar(
        self,
        results: Dict[str, Any],
        sample_key: str,
        lidar_drop_ratio: Optional[float] = None,
        exact_count: bool = False
    ) -> tuple:
        lidar_drop_ratio = (self.lidar_drop_ratio
                            if lidar_drop_ratio is None else lidar_drop_ratio)
        points = results.get('points')
        if points is None:
            return 0, 0

        points_before = len(points)
        if points_before == 0:
            raise ValueError('Sensor degradation received an empty point cloud')
        if lidar_drop_ratio == 0.0:
            return points_before, points_before
        rng = np.random.default_rng(
            _sample_seed(self.seed, sample_key, 'lidar'))
        if exact_count:
            drop_count = min(points_before - 1,
                             int(np.floor(points_before * lidar_drop_ratio)))
            keep_indices = np.sort(rng.permutation(points_before)[drop_count:])
        else:
            keep_scores = rng.random(points_before)
            keep_indices = np.flatnonzero(keep_scores >= lidar_drop_ratio)
            if len(keep_indices) == 0:
                keep_indices = np.asarray([np.argmax(keep_scores)])
        results['points'] = points[keep_indices]
        return points_before, len(keep_indices)

    @staticmethod
    def _record(
        results: Dict[str, Any],
        seed: int,
        camera_drop_count: int,
        dropped_indices: list,
        lidar_drop_ratio: float,
        points_before: int,
        points_after: int
    ) -> None:
        image_paths = results.get('img_path', [])
        dropped_names = [
            osp.basename(osp.dirname(image_paths[index]))
            for index in dropped_indices if index < len(image_paths)
        ]
        results['sensor_degradation'] = dict(
            seed=seed,
            camera_drop_count=camera_drop_count,
            dropped_camera_indices=dropped_indices,
            dropped_camera_names=dropped_names,
            lidar_drop_ratio=lidar_drop_ratio,
            points_before=points_before,
            points_after=points_after)

    def transform(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Degrade sensor inputs without reading or changing annotations."""
        sample_key = self._sample_key(results)
        dropped_indices = self._drop_cameras(results, sample_key)
        points_before, points_after = self._drop_lidar(results, sample_key)

        self._record(results, self.seed, self.camera_drop_count,
                     dropped_indices, self.lidar_drop_ratio, points_before,
                     points_after)
        return results
