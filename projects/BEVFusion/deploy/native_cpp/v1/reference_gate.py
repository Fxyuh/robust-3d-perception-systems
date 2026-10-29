"""Geometry- and density-conditioned BEV fusion for degraded sensors."""

from typing import List, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from mmdet3d.registry import MODELS
from projects.BEVFusion.bevfusion.transfusion_head import ConvFuser


class ReliabilityConvFuser(ConvFuser):
    """Gate camera and LiDAR BEV features using input reliability cues.

    The inherited convolution, batch norm and activation retain the exact
    ``ConvFuser`` state-dict keys. The final gate convolution starts at zero,
    making clean-input fusion identical to the official checkpoint at load
    time. Sensor cues use only inputs and calibration, never annotations.
    """

    needs_sensor_context = True

    def __init__(self, in_channels: Sequence[int], out_channels: int,
                 xbound: Sequence[float], ybound: Sequence[float],
                 hidden_channels: int = 16) -> None:
        super().__init__(in_channels=in_channels, out_channels=out_channels)
        if len(in_channels) != 2:
            raise ValueError('ReliabilityConvFuser requires camera and LiDAR')
        if len(xbound) < 2 or len(ybound) < 2:
            raise ValueError('BEV bounds must contain minimum and maximum')
        self.xbound = (float(xbound[0]), float(xbound[1]))
        self.ybound = (float(ybound[0]), float(ybound[1]))
        self.reliability_net = nn.Sequential(
            nn.Conv2d(5, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 2, 1))
        nn.init.zeros_(self.reliability_net[-1].weight)
        nn.init.zeros_(self.reliability_net[-1].bias)

    def _bev_centers(self, height: int, width: int, device: torch.device,
                     dtype: torch.dtype) -> Tensor:
        x = self.xbound[0] + (torch.arange(height, device=device, dtype=dtype)
                              + 0.5) * (self.xbound[1] - self.xbound[0]) / height
        y = self.ybound[0] + (torch.arange(width, device=device, dtype=dtype)
                              + 0.5) * (self.ybound[1] - self.ybound[0]) / width
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        return torch.stack((xx, yy, torch.zeros_like(xx),
                            torch.ones_like(xx)), dim=-1).reshape(-1, 4).T

    @torch.no_grad()
    def _density_map(self, points: List[Tensor], height: int,
                     width: int, device: torch.device) -> Tensor:
        maps = []
        for cloud in points:
            xy = cloud[:, :2].to(device=device, dtype=torch.float32)
            ix = torch.floor((xy[:, 0] - self.xbound[0]) * height /
                             (self.xbound[1] - self.xbound[0])).long()
            iy = torch.floor((xy[:, 1] - self.ybound[0]) * width /
                             (self.ybound[1] - self.ybound[0])).long()
            valid = (ix >= 0) & (ix < height) & (iy >= 0) & (iy < width)
            indices = ix[valid] * width + iy[valid]
            counts = torch.bincount(indices, minlength=height * width)
            density = torch.log1p(counts.reshape(height, width).float())
            maps.append((density / 3.0).clamp(max=1.0))
        return torch.stack(maps, dim=0).unsqueeze(1)

    @staticmethod
    def _camera_availability(imgs: Tensor, metas: list) -> Tensor:
        # A missing view is mean-filled by the existing degradation transform
        # and therefore zero after image normalization. Metadata takes priority.
        valid = imgs.detach().abs().flatten(start_dim=2).amax(dim=-1) > 1e-6
        for batch_index, meta in enumerate(metas):
            degradation = meta.get('sensor_degradation')
            if degradation is not None:
                valid[batch_index] = True
                for camera_index in degradation['dropped_camera_indices']:
                    valid[batch_index, camera_index] = False
        return valid

    @torch.no_grad()
    def _camera_coverage(self, imgs: Tensor, metas: list, height: int,
                         width: int) -> tuple:
        batch_size, num_cameras, _, image_height, image_width = imgs.shape
        centers = self._bev_centers(height, width, imgs.device, torch.float32)
        available = self._camera_availability(imgs, metas)
        coverage = []
        for batch_index in range(batch_size):
            meta = metas[batch_index]
            lidar_aug = self._as_matrix(
                meta.get('lidar_aug_matrix', torch.eye(4)),
                imgs.device).reshape(4, 4)
            original_points = torch.linalg.solve(lidar_aug, centers)
            lidar2img = self._as_matrix(meta['lidar2img'], imgs.device)
            img_aug = self._as_matrix(
                meta.get('img_aug_matrix', torch.eye(4)), imgs.device)
            if img_aug.ndim == 2:
                img_aug = img_aug.unsqueeze(0).expand(num_cameras, -1, -1)
            projected = lidar2img @ original_points
            depth = projected[:, 2, :]
            uv_depth = torch.stack((projected[:, 0, :] /
                                    depth.clamp(min=1e-5),
                                    projected[:, 1, :] /
                                    depth.clamp(min=1e-5), depth), dim=1)
            augmented = (img_aug[:, :3, :3] @ uv_depth +
                         img_aug[:, :3, 3:4])
            visible = ((depth > 1e-5) & (augmented[:, 0, :] >= 0) &
                       (augmented[:, 0, :] < image_width) &
                       (augmented[:, 1, :] >= 0) &
                       (augmented[:, 1, :] < image_height))
            all_views = visible.sum(dim=0).clamp(min=1)
            valid_views = (visible & available[batch_index, :, None]).sum(dim=0)
            coverage.append((valid_views / all_views).reshape(height, width))
        return torch.stack(coverage, dim=0).unsqueeze(1), available

    @staticmethod
    def _as_matrix(value, device: torch.device) -> Tensor:
        if isinstance(value, Tensor):
            return value.to(device=device, dtype=torch.float32)
        return torch.as_tensor(
            np.asarray(value), device=device, dtype=torch.float32)

    def forward(self, inputs: List[Tensor], points: List[Tensor], imgs: Tensor,
                batch_input_metas: list) -> Tensor:
        camera, lidar = inputs
        if camera.shape[0] != len(points) or camera.shape[-2:] != lidar.shape[-2:]:
            raise ValueError('Camera, LiDAR and point batch shapes must match')
        height, width = camera.shape[-2:]
        with torch.no_grad():
            density = self._density_map(points, height, width, camera.device)
            coverage, available = self._camera_coverage(
                imgs, batch_input_metas, height, width)
            centers = self._bev_centers(height, width, camera.device,
                                        torch.float32)
            distance = centers[:2].square().sum(dim=0).sqrt().reshape(
                1, 1, height, width)
            distance = (distance / 76.4).clamp(max=1.0).expand(
                camera.shape[0], -1, -1, -1)
        camera_energy = torch.log1p(camera.float().abs().mean(1, keepdim=True))
        lidar_energy = torch.log1p(lidar.float().abs().mean(1, keepdim=True))
        cues = torch.cat((density, coverage, distance,
                          camera_energy, lidar_energy), dim=1)
        log_scales = torch.tanh(self.reliability_net(cues))
        camera_scale = torch.exp(log_scales[:, :1]).to(camera.dtype)
        lidar_scale = torch.exp(log_scales[:, 1:]).to(lidar.dtype)
        present = available.any(dim=1).to(camera.dtype).view(-1, 1, 1, 1)
        weighted = torch.cat((camera * camera_scale * present,
                              lidar * lidar_scale), dim=1)
        return self[2](self[1](self[0](weighted)))
