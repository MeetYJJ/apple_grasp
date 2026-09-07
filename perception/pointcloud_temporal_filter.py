"""Motion-adaptive temporal filtering for instance segmentation masks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True)
class TemporalPointCloudConfig:
    """IoU thresholds and EMA weights for dynamic mask tracking.

    ``alpha`` is always the historical weight in
    ``alpha * previous + (1 - alpha) * current``. A large mask displacement
    therefore selects a small alpha and follows the current detector quickly.
    """

    high_iou_threshold: float = 0.7
    low_iou_threshold: float = 0.3
    high_iou_alpha: float = 0.5
    medium_iou_alpha: float = 0.3
    low_iou_alpha: float = 0.1
    mask_threshold: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.low_iou_threshold < self.high_iou_threshold <= 1.0:
            raise ValueError(
                "IoU thresholds must satisfy 0 <= low < high <= 1"
            )
        for name in (
            "high_iou_alpha",
            "medium_iou_alpha",
            "low_iou_alpha",
        ):
            value = getattr(self, name)
            if not 0.0 <= value < 1.0:
                raise ValueError("{} must be in [0, 1)".format(name))
        if not 0.0 < self.mask_threshold <= 1.0:
            raise ValueError("mask_threshold must be in (0, 1]")


@dataclass
class MaskTemporalStats:
    raw_area: int = 0
    previous_area: int = 0
    stable_area: int = 0
    iou: float = 1.0
    alpha: float = 0.0

    @property
    def current_weight(self) -> float:
        return 1.0 - self.alpha


@dataclass
class PointCloudTemporalStats:
    """Compatibility stats for the now stateless cloud pass-through."""

    raw_points: int = 0
    fused_points: int = 0


class TemporalPointCloudFilter:
    """Track masks with motion-adaptive EMA without cloud accumulation.

    Historical point clouds and ICP are intentionally absent: every cloud
    passed to GraspNet must be generated from the current filtered depth and
    current stable mask. ``filter_point_cloud`` remains as a stateless
    compatibility method and returns the input cloud unchanged.
    """

    def __init__(
        self, config: Optional[TemporalPointCloudConfig] = None, **overrides
    ) -> None:
        if config is not None and overrides:
            raise ValueError("Pass either config or keyword overrides, not both")
        self.config = config or TemporalPointCloudConfig(**overrides)
        self.last_mask: Optional[np.ndarray] = None
        self.previous_mask: Optional[np.ndarray] = None
        self._mask_probability: Optional[np.ndarray] = None
        self.last_mask_stats = MaskTemporalStats()
        self.last_pointcloud_stats = PointCloudTemporalStats()

    def reset(self) -> None:
        """Forget mask history."""

        self.last_mask = None
        self.previous_mask = None
        self._mask_probability = None
        self.last_mask_stats = MaskTemporalStats()
        self.last_pointcloud_stats = PointCloudTemporalStats()

    def filter_mask(self, current_mask: np.ndarray) -> np.ndarray:
        """Return a binary mask using IoU-selected EMA history weight."""

        current = np.asarray(current_mask)
        if current.ndim != 2:
            raise ValueError(
                "current_mask must have shape (H, W), got {}".format(
                    current.shape
                )
            )
        current = np.ascontiguousarray(current, dtype=bool)

        if (
            self.last_mask is None
            or self._mask_probability is None
            or self.last_mask.shape != current.shape
        ):
            return self._initialize(current)

        previous_area = int(self.last_mask.sum())
        iou = self.mask_iou(current, self.last_mask)
        alpha = self._alpha_for_iou(iou)
        probability = (
            alpha * self._mask_probability
            + (1.0 - alpha) * current.astype(np.float32)
        )
        stable = np.ascontiguousarray(
            probability >= self.config.mask_threshold, dtype=bool
        )

        self.last_mask = current.copy()
        self.previous_mask = stable.copy()
        self._mask_probability = np.ascontiguousarray(
            np.clip(probability, 0.0, 1.0), dtype=np.float32
        )
        self.last_mask_stats = MaskTemporalStats(
            raw_area=int(current.sum()),
            previous_area=previous_area,
            stable_area=int(stable.sum()),
            iou=iou,
            alpha=alpha,
        )
        return stable

    stabilize_mask = filter_mask

    def filter_point_cloud(self, current_point_cloud: Any) -> Any:
        """Return only the current cloud; no ICP or historical fusion."""

        if current_point_cloud is None or not hasattr(
            current_point_cloud, "points"
        ):
            raise TypeError("point cloud must provide a points array")
        point_count = len(current_point_cloud.points)
        self.last_pointcloud_stats = PointCloudTemporalStats(
            raw_points=point_count,
            fused_points=point_count,
        )
        return current_point_cloud

    process = filter_point_cloud

    def _initialize(self, current: np.ndarray) -> np.ndarray:
        stable = np.ascontiguousarray(current, dtype=bool)
        self.last_mask = stable.copy()
        self.previous_mask = stable.copy()
        self._mask_probability = stable.astype(np.float32)
        area = int(stable.sum())
        self.last_mask_stats = MaskTemporalStats(
            raw_area=area,
            previous_area=0,
            stable_area=area,
            iou=1.0,
            alpha=0.0,
        )
        return stable

    def _alpha_for_iou(self, iou: float) -> float:
        if iou > self.config.high_iou_threshold:
            return self.config.high_iou_alpha
        if iou > self.config.low_iou_threshold:
            return self.config.medium_iou_alpha
        return self.config.low_iou_alpha

    @staticmethod
    def mask_iou(current: np.ndarray, previous: np.ndarray) -> float:
        union = int(np.count_nonzero(current | previous))
        if union == 0:
            return 1.0
        intersection = int(np.count_nonzero(current & previous))
        return float(intersection) / float(union)
