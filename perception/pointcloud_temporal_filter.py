"""Temporal stabilization for segmentation masks and RGB-D point clouds."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class TemporalPointCloudConfig:
    """Configuration for mask EMA, ICP alignment, and cloud fusion."""

    mask_alpha: float = 0.7
    mask_threshold: float = 0.5
    area_ratio_min: float = 0.67
    area_ratio_max: float = 1.5
    area_anomaly_current_weight_scale: float = 0.25
    min_mask_iou: float = 0.3
    low_iou_current_weight_scale: float = 0.5
    max_empty_mask_frames: int = 8

    point_beta: float = 0.5
    low_point_threshold: int = 15000
    target_points: int = 20000
    target_min_points: int = 18000
    target_max_points: int = 22000

    enable_icp: bool = True
    icp_voxel_size: float = 0.004
    icp_max_correspondence_distance: float = 0.02
    icp_max_iterations: int = 30
    icp_min_points: int = 100
    icp_min_fitness: float = 0.20
    icp_max_inlier_rmse: float = 0.02
    icp_max_translation: float = 0.08
    icp_max_rotation_degrees: float = 35.0
    supplement_max_distance: float = 0.04
    sampling_seed: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.mask_alpha < 1.0:
            raise ValueError("mask_alpha must be in [0, 1)")
        if not 0.0 < self.mask_threshold <= 1.0:
            raise ValueError("mask_threshold must be in (0, 1]")
        if not 0.0 < self.area_ratio_min < 1.0:
            raise ValueError("area_ratio_min must be in (0, 1)")
        if self.area_ratio_max <= 1.0:
            raise ValueError("area_ratio_max must be greater than 1")
        if self.area_ratio_min >= self.area_ratio_max:
            raise ValueError("area ratio minimum must be below its maximum")
        if not 0.0 < self.area_anomaly_current_weight_scale <= 1.0:
            raise ValueError(
                "area_anomaly_current_weight_scale must be in (0, 1]"
            )
        if not 0.0 <= self.min_mask_iou <= 1.0:
            raise ValueError("min_mask_iou must be in [0, 1]")
        if not 0.0 < self.low_iou_current_weight_scale <= 1.0:
            raise ValueError("low_iou_current_weight_scale must be in (0, 1]")
        if self.max_empty_mask_frames < 1:
            raise ValueError("max_empty_mask_frames must be positive")
        if not 0.0 <= self.point_beta <= 1.0:
            raise ValueError("point_beta must be in [0, 1]")
        if self.low_point_threshold < 1:
            raise ValueError("low_point_threshold must be positive")
        if not (
            1
            <= self.target_min_points
            <= self.target_points
            <= self.target_max_points
        ):
            raise ValueError(
                "target points must satisfy min <= target <= max"
            )
        if self.icp_voxel_size < 0:
            raise ValueError("icp_voxel_size must be non-negative")
        if self.icp_max_correspondence_distance <= 0:
            raise ValueError("icp_max_correspondence_distance must be positive")
        if self.icp_max_iterations < 1 or self.icp_min_points < 3:
            raise ValueError("ICP iteration/point limits are invalid")
        if not 0.0 <= self.icp_min_fitness <= 1.0:
            raise ValueError("icp_min_fitness must be in [0, 1]")
        if self.icp_max_inlier_rmse <= 0:
            raise ValueError("icp_max_inlier_rmse must be positive")
        if self.icp_max_translation <= 0:
            raise ValueError("icp_max_translation must be positive")
        if not 0.0 < self.icp_max_rotation_degrees <= 180.0:
            raise ValueError("icp_max_rotation_degrees must be in (0, 180]")
        if self.supplement_max_distance <= 0:
            raise ValueError("supplement_max_distance must be positive")


@dataclass
class MaskTemporalStats:
    raw_area: int = 0
    previous_area: int = 0
    stable_area: int = 0
    area_ratio: float = 1.0
    iou: float = 1.0
    current_weight: float = 1.0
    area_anomaly: bool = False
    iou_anomaly: bool = False
    reset_after_empty: bool = False


@dataclass
class PointCloudTemporalStats:
    raw_points: int = 0
    previous_points: int = 0
    matched_points: int = 0
    supplemented_points: int = 0
    fused_points: int = 0
    low_point_fallback: bool = False
    icp_attempted: bool = False
    icp_accepted: bool = False
    icp_fitness: float = 0.0
    icp_inlier_rmse: float = float("inf")
    effective_beta: float = 1.0
    history_retained: bool = False
    transformation: np.ndarray = field(
        default_factory=lambda: np.eye(4, dtype=np.float64)
    )
    warning_message: Optional[str] = None


class TemporalPointCloudFilter:
    """Stabilize an apple mask and its point cloud across adjacent frames.

    ``filter_mask`` performs a probability EMA. Area or IoU anomalies reduce
    the current-frame weight instead of replacing the history immediately.
    ``filter_point_cloud`` registers the previous stable cloud to the current
    cloud, blends matched points, and uses aligned historical points to fill a
    temporary point-count deficit.
    """

    def __init__(
        self, config: Optional[TemporalPointCloudConfig] = None, **overrides
    ) -> None:
        if config is not None and overrides:
            raise ValueError("Pass either config or keyword overrides, not both")
        self.config = config or TemporalPointCloudConfig(**overrides)
        self.previous_mask: Optional[np.ndarray] = None
        self.previous_point_cloud: Optional[Any] = None
        self._mask_probability: Optional[np.ndarray] = None
        self._empty_mask_frames = 0
        self.last_mask_stats = MaskTemporalStats()
        self.last_pointcloud_stats = PointCloudTemporalStats()

    def reset(self) -> None:
        """Forget all temporal history."""

        self.previous_mask = None
        self.previous_point_cloud = None
        self._mask_probability = None
        self._empty_mask_frames = 0
        self.last_mask_stats = MaskTemporalStats()
        self.last_pointcloud_stats = PointCloudTemporalStats()

    def filter_mask(self, current_mask: np.ndarray) -> np.ndarray:
        """Return a binary mask stabilized with EMA and consistency gates."""

        current = np.asarray(current_mask)
        if current.ndim != 2:
            raise ValueError(
                "current_mask must have shape (H, W), got {}".format(
                    current.shape
                )
            )
        current = np.ascontiguousarray(current, dtype=bool)
        current_area = int(current.sum())

        if (
            self.previous_mask is None
            or self._mask_probability is None
            or self.previous_mask.shape != current.shape
        ):
            if (
                self.previous_mask is not None
                and self.previous_mask.shape != current.shape
            ):
                self.previous_point_cloud = None
            return self._initialize_mask(current)

        previous = self.previous_mask
        previous_area = int(previous.sum())
        if previous_area == 0 and current_area > 0:
            return self._initialize_mask(current)

        area_ratio = self._area_ratio(current_area, previous_area)
        iou = self._mask_iou(current, previous)
        area_anomaly = (
            area_ratio < self.config.area_ratio_min
            or area_ratio > self.config.area_ratio_max
        )
        iou_anomaly = iou < self.config.min_mask_iou

        current_weight = 1.0 - self.config.mask_alpha
        if area_anomaly:
            current_weight *= self.config.area_anomaly_current_weight_scale
        if iou_anomaly:
            current_weight *= self.config.low_iou_current_weight_scale

        if current_area == 0:
            self._empty_mask_frames += 1
        else:
            self._empty_mask_frames = 0

        if self._empty_mask_frames >= self.config.max_empty_mask_frames:
            stable = np.zeros_like(current)
            self.previous_mask = None
            self._mask_probability = None
            self.previous_point_cloud = None
            self.last_mask_stats = MaskTemporalStats(
                raw_area=0,
                previous_area=previous_area,
                stable_area=0,
                area_ratio=0.0,
                iou=iou,
                current_weight=current_weight,
                area_anomaly=True,
                iou_anomaly=True,
                reset_after_empty=True,
            )
            return stable

        probability = (
            (1.0 - current_weight) * self._mask_probability
            + current_weight * current.astype(np.float32)
        )
        stable = np.ascontiguousarray(
            probability >= self.config.mask_threshold, dtype=bool
        )
        self._mask_probability = np.ascontiguousarray(
            np.clip(probability, 0.0, 1.0), dtype=np.float32
        )
        self.previous_mask = stable.copy()
        if not np.any(stable):
            self.previous_point_cloud = None
        self.last_mask_stats = MaskTemporalStats(
            raw_area=current_area,
            previous_area=previous_area,
            stable_area=int(stable.sum()),
            area_ratio=area_ratio,
            iou=iou,
            current_weight=current_weight,
            area_anomaly=area_anomaly,
            iou_anomaly=iou_anomaly,
        )
        return stable

    stabilize_mask = filter_mask

    def filter_point_cloud(self, current_point_cloud: Any) -> Any:
        """Return an ICP-aligned, nearest-neighbor fused Open3D point cloud."""

        current_points, current_colors = self._cloud_arrays(current_point_cloud)
        if len(current_points) == 0:
            raise ValueError("current_point_cloud must not be empty")

        o3d = self._load_open3d()
        raw_count = len(current_points)
        previous_count = (
            0
            if self.previous_point_cloud is None
            else len(self.previous_point_cloud.points)
        )
        stats = PointCloudTemporalStats(
            raw_points=raw_count,
            previous_points=previous_count,
            low_point_fallback=raw_count < self.config.low_point_threshold,
        )

        if self.previous_point_cloud is None:
            output_points, output_colors = self._cap_current_points(
                current_points, current_colors
            )
            output = self._make_cloud(o3d, output_points, output_colors)
            stats.fused_points = len(output_points)
            self._set_range_warning(stats)
            self.previous_point_cloud = self._clone_cloud(o3d, output)
            self.last_pointcloud_stats = stats
            return output

        previous_points, previous_colors = self._cloud_arrays(
            self.previous_point_cloud
        )
        transformation, alignment_ok, icp_values = self._align_previous(
            o3d, previous_points, current_points
        )
        stats.icp_attempted = icp_values[0]
        stats.icp_accepted = alignment_ok
        stats.icp_fitness = icp_values[1]
        stats.icp_inlier_rmse = icp_values[2]
        stats.transformation = transformation.copy()

        update_history = True
        if alignment_ok:
            aligned_previous = self._transform_points(
                previous_points, transformation
            )
            fused_points, fused_colors, matched = self._blend_current(
                current_points,
                current_colors,
                aligned_previous,
                previous_colors,
            )
            stats.matched_points = matched
            stats.effective_beta = self.config.point_beta
            (
                fused_points,
                fused_colors,
                supplemented,
            ) = self._supplement_points(
                fused_points,
                fused_colors,
                aligned_previous,
                previous_colors,
            )
            stats.supplemented_points = supplemented
        else:
            fused_points, fused_colors = self._cap_current_points(
                current_points, current_colors
            )
            # A sparse frame that cannot be registered must not poison the
            # historical model used to recover the next frame.  It is still
            # returned honestly (and warned about) rather than using an
            # unaligned stale cloud as if it were current geometry.
            update_history = raw_count >= self.config.low_point_threshold
            stats.history_retained = not update_history

        output = self._make_cloud(o3d, fused_points, fused_colors)
        stats.fused_points = len(fused_points)
        self._set_range_warning(stats)
        if update_history:
            self.previous_point_cloud = self._clone_cloud(o3d, output)
        self.last_pointcloud_stats = stats
        return output

    process = filter_point_cloud

    def _initialize_mask(self, current: np.ndarray) -> np.ndarray:
        stable = np.ascontiguousarray(current, dtype=bool)
        self.previous_mask = stable.copy()
        self._mask_probability = stable.astype(np.float32)
        self._empty_mask_frames = 1 if not np.any(stable) else 0
        area = int(stable.sum())
        self.last_mask_stats = MaskTemporalStats(
            raw_area=area,
            previous_area=0,
            stable_area=area,
            current_weight=1.0,
        )
        return stable

    @staticmethod
    def _area_ratio(current_area: int, previous_area: int) -> float:
        if previous_area == 0:
            return 1.0 if current_area == 0 else float("inf")
        return float(current_area) / float(previous_area)

    @staticmethod
    def _mask_iou(current: np.ndarray, previous: np.ndarray) -> float:
        union = int(np.count_nonzero(current | previous))
        if union == 0:
            return 1.0
        intersection = int(np.count_nonzero(current & previous))
        return float(intersection) / float(union)

    def _align_previous(
        self, o3d: Any, previous: np.ndarray, current: np.ndarray
    ) -> Tuple[np.ndarray, bool, Tuple[bool, float, float]]:
        identity = np.eye(4, dtype=np.float64)
        if not self.config.enable_icp:
            return identity, True, (False, 1.0, 0.0)
        if min(len(previous), len(current)) < self.config.icp_min_points:
            return identity, False, (False, 0.0, float("inf"))

        source = self._make_cloud(o3d, previous, None)
        target = self._make_cloud(o3d, current, None)
        if self.config.icp_voxel_size > 0:
            source = source.voxel_down_sample(self.config.icp_voxel_size)
            target = target.voxel_down_sample(self.config.icp_voxel_size)
        if min(len(source.points), len(target.points)) < 3:
            return identity, False, (True, 0.0, float("inf"))

        # Camera/object motion can exceed the correspondence distance between
        # frames.  A centroid translation gives point-to-point ICP a useful
        # initial estimate while the acceptance gates below still reject jumps.
        initial = identity.copy()
        initial[:3, 3] = (
            np.asarray(target.points).mean(axis=0)
            - np.asarray(source.points).mean(axis=0)
        )
        if np.linalg.norm(initial[:3, 3]) > self.config.icp_max_translation:
            return identity, False, (True, 0.0, float("inf"))

        try:
            registration = o3d.pipelines.registration.registration_icp(
                source,
                target,
                self.config.icp_max_correspondence_distance,
                initial,
                (
                    o3d.pipelines.registration
                    .TransformationEstimationPointToPoint()
                ),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=self.config.icp_max_iterations
                ),
            )
        except RuntimeError:
            return identity, False, (True, 0.0, float("inf"))

        transformation = np.asarray(
            registration.transformation, dtype=np.float64
        )
        fitness = float(registration.fitness)
        rmse = float(registration.inlier_rmse)
        if transformation.shape != (4, 4) or not np.all(
            np.isfinite(transformation)
        ):
            return identity, False, (True, fitness, rmse)
        translation = float(np.linalg.norm(transformation[:3, 3]))
        angle = self._rotation_angle_degrees(transformation[:3, :3])
        accepted = (
            fitness >= self.config.icp_min_fitness
            and rmse <= self.config.icp_max_inlier_rmse
            and translation <= self.config.icp_max_translation
            and angle <= self.config.icp_max_rotation_degrees
        )
        return transformation if accepted else identity, accepted, (
            True,
            fitness,
            rmse,
        )

    def _blend_current(
        self,
        current: np.ndarray,
        current_colors: Optional[np.ndarray],
        aligned_previous: np.ndarray,
        previous_colors: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray], int]:
        tree = cKDTree(aligned_previous)
        distances, indices = tree.query(
            current,
            k=1,
            distance_upper_bound=self.config.icp_max_correspondence_distance,
        )
        matched = np.isfinite(distances) & (indices < len(aligned_previous))
        points = current.copy()
        beta = self.config.point_beta
        points[matched] = (
            beta * current[matched]
            + (1.0 - beta) * aligned_previous[indices[matched]]
        )

        colors = None if current_colors is None else current_colors.copy()
        if colors is not None and previous_colors is not None:
            colors[matched] = (
                beta * current_colors[matched]
                + (1.0 - beta) * previous_colors[indices[matched]]
            )
        return points, colors, int(matched.sum())

    def _supplement_points(
        self,
        current: np.ndarray,
        current_colors: Optional[np.ndarray],
        aligned_previous: np.ndarray,
        previous_colors: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray], int]:
        needed = self.config.target_points - len(current)
        if needed <= 0:
            points, colors = self._cap_current_points(current, current_colors)
            return points, colors, 0

        tree = cKDTree(current)
        distances, _ = tree.query(aligned_previous, k=1)
        candidates = np.flatnonzero(
            np.isfinite(distances)
            & (distances <= self.config.supplement_max_distance)
        )
        if len(candidates) == 0:
            return current, current_colors, 0

        # Prefer historical samples that restore missing surface regions.
        order = candidates[np.argsort(distances[candidates])[::-1]]
        selected = order[:needed]
        points = np.concatenate((current, aligned_previous[selected]), axis=0)

        colors = current_colors
        if current_colors is not None and previous_colors is not None:
            colors = np.concatenate(
                (current_colors, previous_colors[selected]), axis=0
            )
        elif len(selected) > 0:
            # Avoid an invalid Open3D cloud with fewer colors than points.
            colors = None
        return points, colors, len(selected)

    def _cap_current_points(
        self, points: np.ndarray, colors: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if len(points) <= self.config.target_points:
            return points.copy(), None if colors is None else colors.copy()
        rng = np.random.default_rng(self.config.sampling_seed)
        indices = rng.choice(
            len(points), self.config.target_points, replace=False
        )
        selected_colors = None if colors is None else colors[indices]
        return points[indices], selected_colors

    def _set_range_warning(self, stats: PointCloudTemporalStats) -> None:
        if stats.fused_points >= self.config.target_min_points:
            return
        stats.warning_message = (
            "Temporal cloud has {} points, below target minimum {}; "
            "insufficient aligned history was available, so fixed-size model "
            "sampling may still repeat points".format(
                stats.fused_points, self.config.target_min_points
            )
        )
        warnings.warn(stats.warning_message, RuntimeWarning, stacklevel=3)

    @staticmethod
    def _transform_points(
        points: np.ndarray, transformation: np.ndarray
    ) -> np.ndarray:
        return points @ transformation[:3, :3].T + transformation[:3, 3]

    @staticmethod
    def _rotation_angle_degrees(rotation: np.ndarray) -> float:
        cosine = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
        return float(np.degrees(np.arccos(cosine)))

    @staticmethod
    def _cloud_arrays(cloud: Any) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if cloud is None or not hasattr(cloud, "points"):
            raise TypeError("point cloud must provide a points array")
        points = np.asarray(cloud.points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("point cloud points must have shape (N, 3)")
        if not np.all(np.isfinite(points)):
            raise ValueError("point cloud contains NaN or infinite coordinates")

        colors = None
        has_colors = getattr(cloud, "has_colors", None)
        if callable(has_colors) and has_colors():
            colors = np.asarray(cloud.colors, dtype=np.float64)
            if colors.shape != points.shape:
                raise ValueError("point cloud colors must match point shape")
        return np.ascontiguousarray(points), (
            None if colors is None else np.ascontiguousarray(colors)
        )

    @staticmethod
    def _make_cloud(
        o3d: Any, points: np.ndarray, colors: Optional[np.ndarray]
    ) -> Any:
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(
            np.ascontiguousarray(points, dtype=np.float64)
        )
        if colors is not None:
            cloud.colors = o3d.utility.Vector3dVector(
                np.ascontiguousarray(np.clip(colors, 0.0, 1.0), dtype=np.float64)
            )
        return cloud

    @classmethod
    def _clone_cloud(cls, o3d: Any, cloud: Any) -> Any:
        points, colors = cls._cloud_arrays(cloud)
        return cls._make_cloud(o3d, points, colors)

    @staticmethod
    def _load_open3d() -> Any:
        try:
            import open3d as o3d
        except ImportError as exc:
            raise ImportError(
                "Open3D is required for temporal point-cloud filtering"
            ) from exc
        return o3d
