"""RGB-D projection, staged point-cloud filtering, and GraspNet sampling."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import numpy as np

from .depth_filter import DepthFilter, DepthFilterConfig

if TYPE_CHECKING:
    import open3d


class InsufficientPointCloudError(ValueError):
    """Raised when too few real points remain to safely run GraspNet."""


@dataclass(frozen=True)
class PointCloudConfig:
    """Point-cloud processing switches and tunable parameters.

    Destructive filters are disabled by default. With a valid apple mask this
    preserves nearly all deprojected depth samples and avoids the previous
    large point-count collapse. ``workspace_roi`` is
    ``(x_min, y_min, x_max, y_max)`` in pixels; maximum bounds are exclusive.
    Depth-filter color deltas are expressed in millimetres because both current
    RGB-D adapters normalize depth to ``uint16`` millimetres.
    """

    workspace_roi: Optional[Tuple[int, int, int, int]] = None
    min_depth_m: Optional[float] = None
    max_depth_m: Optional[float] = None

    enable_depth_preprocessing: bool = True
    enable_median_filter: bool = True
    median_kernel: int = 3
    enable_spatial_smoothing: bool = True
    spatial_diameter: int = 5
    spatial_sigma_color: float = 30.0
    spatial_sigma_space: float = 3.0
    enable_hole_filling: bool = True
    hole_fill_kernel: int = 3
    hole_fill_min_neighbors: int = 5
    hole_fill_iterations: int = 1
    hole_fill_max_depth_delta_mm: float = 80.0

    enable_outlier_filter: bool = False
    outlier_nb_neighbors: int = 20
    outlier_std_ratio: float = 2.0
    enable_radius_outlier_filter: bool = False
    radius_outlier_nb_points: int = 8
    radius_outlier_radius: float = 0.01

    enable_voxel_downsample: bool = False
    voxel_size: float = 0.001
    enable_normal_estimation: bool = False
    normal_radius: float = 0.01
    normal_max_nn: int = 30

    min_real_points_warning: int = 5000
    min_real_points_reject: int = 1000

    def __post_init__(self) -> None:
        if self.workspace_roi is not None and len(self.workspace_roi) != 4:
            raise ValueError("workspace_roi must contain x_min, y_min, x_max, y_max")
        if self.min_depth_m is not None and self.min_depth_m < 0:
            raise ValueError("min_depth_m must be non-negative")
        if self.max_depth_m is not None and self.max_depth_m <= 0:
            raise ValueError("max_depth_m must be positive")
        if (
            self.min_depth_m is not None
            and self.max_depth_m is not None
            and self.min_depth_m >= self.max_depth_m
        ):
            raise ValueError("min_depth_m must be smaller than max_depth_m")
        if self.median_kernel not in (1, 3, 5):
            raise ValueError("median_kernel must be one of 1, 3, or 5")
        if self.spatial_diameter <= 0 or self.spatial_diameter % 2 == 0:
            raise ValueError("spatial_diameter must be a positive odd integer")
        if self.spatial_sigma_color <= 0 or self.spatial_sigma_space <= 0:
            raise ValueError("spatial smoothing sigmas must be positive")
        if self.hole_fill_kernel < 3 or self.hole_fill_kernel % 2 == 0:
            raise ValueError("hole_fill_kernel must be an odd integer >= 3")
        max_hole_neighbors = self.hole_fill_kernel ** 2 - 1
        if not 1 <= self.hole_fill_min_neighbors <= max_hole_neighbors:
            raise ValueError("hole_fill_min_neighbors is outside the neighborhood")
        if self.hole_fill_iterations < 0:
            raise ValueError("hole_fill_iterations must be non-negative")
        if self.hole_fill_max_depth_delta_mm <= 0:
            raise ValueError("hole_fill_max_depth_delta_mm must be positive")
        if self.outlier_nb_neighbors < 0:
            raise ValueError("outlier_nb_neighbors must be non-negative")
        if self.enable_outlier_filter and self.outlier_nb_neighbors == 0:
            raise ValueError(
                "outlier_nb_neighbors must be positive when its filter is enabled"
            )
        if self.outlier_std_ratio <= 0:
            raise ValueError("outlier_std_ratio must be positive")
        if self.radius_outlier_nb_points < 0:
            raise ValueError("radius_outlier_nb_points must be non-negative")
        if self.radius_outlier_radius < 0:
            raise ValueError("radius_outlier_radius must be non-negative")
        if self.enable_radius_outlier_filter and (
            self.radius_outlier_nb_points == 0 or self.radius_outlier_radius == 0
        ):
            raise ValueError(
                "radius outlier parameters must be positive when enabled"
            )
        if self.voxel_size < 0:
            raise ValueError("voxel_size must be non-negative")
        if self.enable_voxel_downsample and self.voxel_size == 0:
            raise ValueError("voxel_size must be positive when enabled")
        if self.normal_radius < 0:
            raise ValueError("normal_radius must be non-negative")
        if self.enable_normal_estimation and self.normal_radius == 0:
            raise ValueError("normal_radius must be positive when enabled")
        if self.normal_max_nn <= 0:
            raise ValueError("normal_max_nn must be positive")
        if self.min_real_points_reject < 1:
            raise ValueError("min_real_points_reject must be positive")
        if self.min_real_points_warning <= self.min_real_points_reject:
            raise ValueError(
                "min_real_points_warning must exceed min_real_points_reject"
            )

    def make_depth_filter(self) -> DepthFilter:
        """Build one reusable depth processor from this point-cloud config."""

        return DepthFilter(
            DepthFilterConfig(
                enable_median_filter=self.enable_median_filter,
                median_kernel=self.median_kernel,
                enable_spatial_smoothing=self.enable_spatial_smoothing,
                spatial_diameter=self.spatial_diameter,
                spatial_sigma_color=self.spatial_sigma_color,
                spatial_sigma_space=self.spatial_sigma_space,
                enable_hole_filling=self.enable_hole_filling,
                hole_fill_kernel=self.hole_fill_kernel,
                hole_fill_min_neighbors=self.hole_fill_min_neighbors,
                hole_fill_iterations=self.hole_fill_iterations,
                hole_fill_max_depth_delta_mm=(
                    self.hole_fill_max_depth_delta_mm
                ),
            )
        )


# Backward-compatible name used by the previous pipeline revision.
PointCloudFilterConfig = PointCloudConfig


@dataclass
class PointCloudStats:
    """Point counts captured at every stage of cloud construction."""

    mask_pixels: int = 0
    valid_depth_pixels: int = 0
    after_roi_pixels: int = 0
    hole_filled_pixels: int = 0
    xyz_generated_points: int = 0
    after_range_filter: int = 0
    after_outlier_removal: int = 0
    after_radius_outlier_removal: int = 0
    after_voxel_downsample: int = 0
    final_points: int = 0
    real_points: int = 0
    sampled_points: int = 0
    warning_message: Optional[str] = None

    def reset(self) -> None:
        for field_name in (
            "mask_pixels",
            "valid_depth_pixels",
            "after_roi_pixels",
            "hole_filled_pixels",
            "xyz_generated_points",
            "after_range_filter",
            "after_outlier_removal",
            "after_radius_outlier_removal",
            "after_voxel_downsample",
            "final_points",
            "real_points",
            "sampled_points",
        ):
            setattr(self, field_name, 0)
        self.warning_message = None

    def as_dict(self) -> Dict[str, int]:
        return {
            "mask_pixels": self.mask_pixels,
            "valid_depth_pixels": self.valid_depth_pixels,
            "after_roi_pixels": self.after_roi_pixels,
            "hole_filled_pixels": self.hole_filled_pixels,
            "xyz_generated_points": self.xyz_generated_points,
            "after_range_filter": self.after_range_filter,
            "after_outlier_removal": self.after_outlier_removal,
            "after_radius_outlier_removal": self.after_radius_outlier_removal,
            "after_voxel_downsample": self.after_voxel_downsample,
            "final_points": self.final_points,
            "real_points": self.real_points,
            "sampled_points": self.sampled_points,
        }

    def print(self, frame_index: Optional[int] = None) -> None:
        heading = (
            "Point cloud statistics:"
            if frame_index is None
            else "Frame {} point cloud statistics:".format(frame_index)
        )
        print(heading)
        print("mask pixels: {}".format(self.mask_pixels))
        print("valid depth pixels: {}".format(self.valid_depth_pixels))
        print("after ROI: {}".format(self.after_roi_pixels))
        print("hole-filled pixels: {}".format(self.hole_filled_pixels))
        print("xyz generated points: {}".format(self.xyz_generated_points))
        print("after range filter: {}".format(self.after_range_filter))
        print("after outlier removal: {}".format(self.after_outlier_removal))
        print(
            "after radius outlier removal: {}".format(
                self.after_radius_outlier_removal
            )
        )
        print("after voxel downsample: {}".format(self.after_voxel_downsample))
        print("final points: {}".format(self.final_points))
        print("real_points: {}".format(self.real_points))
        print("sampled_points: {}".format(self.sampled_points))
        if self.warning_message:
            print("WARNING: {}".format(self.warning_message))

    # Compatibility aliases for code written against the previous counters.
    @property
    def raw_mask_pixels(self) -> int:
        return self.mask_pixels

    @property
    def filtered_points(self) -> int:
        return self.final_points

    @property
    def graspnet_input_points(self) -> int:
        return self.sampled_points


@dataclass(frozen=True)
class PointCloudData:
    """Fixed-size model input with optional RGB colors in ``[0, 1]``."""

    points: np.ndarray
    colors: Optional[np.ndarray] = None


def create_point_cloud(
    depth: np.ndarray,
    intrinsic: np.ndarray,
    depth_scale: float,
    mask: Optional[np.ndarray] = None,
    color: Optional[np.ndarray] = None,
    config: Optional[PointCloudConfig] = None,
    stats: Optional[PointCloudStats] = None,
    depth_filter: Optional[DepthFilter] = None,
) -> open3d.geometry.PointCloud:
    """Create a filtered Open3D cloud while preserving existing inputs."""

    cloud_config = config or PointCloudConfig()
    depth_array, intrinsic_array, color_array, object_mask = _validate_inputs(
        depth, intrinsic, depth_scale, mask, color
    )
    height, width = depth_array.shape
    stage_stats = stats or PointCloudStats()
    stage_stats.reset()
    stage_stats.mask_pixels = int(object_mask.sum())

    roi_mask = _create_roi_mask(
        image_shape=(height, width), workspace_roi=cloud_config.workspace_roi
    )
    measured_valid_before_roi = (
        object_mask & np.isfinite(depth_array) & (depth_array > 0)
    )
    stage_stats.valid_depth_pixels = int(measured_valid_before_roi.sum())
    measured_valid_mask = measured_valid_before_roi & roi_mask
    stage_stats.after_roi_pixels = int(measured_valid_mask.sum())

    # Preprocess only the segmented workspace. This prevents median/spatial
    # neighbors from pulling background or occluder depths across the apple
    # silhouette while keeping DepthFilter.process(depth) source-independent.
    masked_depth = depth_array.copy()
    masked_depth[~(object_mask & roi_mask)] = 0
    processed_depth = _preprocess_depth(masked_depth, cloud_config, depth_filter)
    valid_depth_mask = (
        object_mask
        & roi_mask
        & np.isfinite(processed_depth)
        & (processed_depth > 0)
    )
    stage_stats.hole_filled_pixels = int(
        np.count_nonzero(valid_depth_mask & ~measured_valid_mask)
    )
    if not np.any(valid_depth_mask):
        raise InsufficientPointCloudError(
            "No valid depth pixels remain after preprocessing, mask, and ROI"
        )

    rows, cols = np.nonzero(valid_depth_mask)
    depth_metres = processed_depth.astype(np.float32) * np.float32(depth_scale)
    fx = float(intrinsic_array[0, 0])
    fy = float(intrinsic_array[1, 1])
    cx = float(intrinsic_array[0, 2])
    cy = float(intrinsic_array[1, 2])
    z = depth_metres[rows, cols]
    x = (cols.astype(np.float32) - cx) * z / fx
    y = (rows.astype(np.float32) - cy) * z / fy
    points = np.ascontiguousarray(np.column_stack((x, y, z)), dtype=np.float64)
    colors = _extract_colors(color_array, rows, cols)
    stage_stats.xyz_generated_points = len(points)

    range_mask = np.ones(len(points), dtype=bool)
    if cloud_config.min_depth_m is not None:
        range_mask &= points[:, 2] >= cloud_config.min_depth_m
    if cloud_config.max_depth_m is not None:
        range_mask &= points[:, 2] <= cloud_config.max_depth_m
    points = points[range_mask]
    if colors is not None:
        colors = colors[range_mask]
    stage_stats.after_range_filter = len(points)
    if len(points) == 0:
        raise InsufficientPointCloudError(
            "No XYZ points remain inside the configured depth range"
        )

    o3d = _load_open3d()
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        cloud.colors = o3d.utility.Vector3dVector(colors)

    cloud = filter_point_cloud(cloud, cloud_config, stats=stage_stats, o3d=o3d)
    stage_stats.final_points = len(cloud.points)
    stage_stats.real_points = stage_stats.final_points
    if stage_stats.final_points == 0:
        raise InsufficientPointCloudError(
            "All points were removed by point-cloud filtering"
        )
    return cloud


def filter_point_cloud(
    cloud: Any,
    config: Optional[PointCloudConfig] = None,
    o3d: Optional[Any] = None,
    stats: Optional[PointCloudStats] = None,
) -> open3d.geometry.PointCloud:
    """Apply optional outlier filters, voxelization, and normal estimation."""

    cloud_config = config or PointCloudConfig()
    open3d_module = o3d or _load_open3d()
    filtered = cloud

    if (
        cloud_config.enable_outlier_filter
        and len(filtered.points) > cloud_config.outlier_nb_neighbors
    ):
        filtered, _ = filtered.remove_statistical_outlier(
            nb_neighbors=cloud_config.outlier_nb_neighbors,
            std_ratio=cloud_config.outlier_std_ratio,
        )
    if stats is not None:
        stats.after_outlier_removal = len(filtered.points)

    if (
        cloud_config.enable_radius_outlier_filter
        and len(filtered.points) >= cloud_config.radius_outlier_nb_points
    ):
        filtered, _ = filtered.remove_radius_outlier(
            nb_points=cloud_config.radius_outlier_nb_points,
            radius=cloud_config.radius_outlier_radius,
        )
    if stats is not None:
        stats.after_radius_outlier_removal = len(filtered.points)

    if cloud_config.enable_voxel_downsample and len(filtered.points) > 0:
        filtered = filtered.voxel_down_sample(cloud_config.voxel_size)
    if stats is not None:
        stats.after_voxel_downsample = len(filtered.points)

    if cloud_config.enable_normal_estimation and len(filtered.points) >= 3:
        filtered.estimate_normals(
            search_param=open3d_module.geometry.KDTreeSearchParamHybrid(
                radius=cloud_config.normal_radius,
                max_nn=cloud_config.normal_max_nn,
            )
        )
        filtered.orient_normals_towards_camera_location(
            camera_location=np.zeros(3, dtype=np.float64)
        )
        filtered.normalize_normals()
    return filtered


def median_filter_depth(depth: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Compatibility helper for callers that only require median filtering."""

    config = DepthFilterConfig(
        enable_median_filter=kernel_size > 1,
        median_kernel=kernel_size,
        enable_spatial_smoothing=False,
        enable_hole_filling=False,
    )
    return DepthFilter(config).process(depth)


def sample_point_cloud(
    cloud: Any,
    num_points: int = 20000,
    seed: Optional[int] = None,
    stats: Optional[PointCloudStats] = None,
    min_real_points_warning: int = 5000,
    min_real_points_reject: int = 1000,
) -> PointCloudData:
    """Create fixed-size input while warning/rejecting sparse real clouds."""

    if num_points <= 0:
        raise ValueError("num_points must be positive")
    if min_real_points_reject < 1:
        raise ValueError("min_real_points_reject must be positive")
    if min_real_points_warning <= min_real_points_reject:
        raise ValueError(
            "min_real_points_warning must exceed min_real_points_reject"
        )

    points = np.asarray(cloud.points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Point cloud must have shape (N, 3), got {}".format(
            points.shape
        ))
    if not np.all(np.isfinite(points)):
        raise ValueError("Point cloud contains NaN or infinite coordinates")

    real_point_count = len(points)
    if stats is not None:
        stats.real_points = real_point_count
        stats.sampled_points = 0
    if real_point_count < min_real_points_reject:
        message = (
            "Rejecting frame with {} real points; minimum is {}".format(
                real_point_count, min_real_points_reject
            )
        )
        if stats is not None:
            stats.warning_message = message
        raise InsufficientPointCloudError(message)
    if real_point_count < min_real_points_warning:
        message = (
            "Sparse cloud has {} real points; repeated sampling to {} may "
            "reduce grasp stability".format(real_point_count, num_points)
        )
        if stats is not None:
            stats.warning_message = message
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    rng = np.random.default_rng(seed)
    if real_point_count >= num_points:
        indices = rng.choice(real_point_count, num_points, replace=False)
    else:
        # Repeat every real point evenly before adding a non-repeating
        # remainder. This avoids the highly uneven duplicate distribution of
        # one large random choice when the cloud is sparse.
        full_repeats, remainder = divmod(num_points, real_point_count)
        indices = np.tile(np.arange(real_point_count), full_repeats)
        if remainder:
            tail = rng.choice(real_point_count, remainder, replace=False)
            indices = np.concatenate((indices, tail))
        rng.shuffle(indices)

    sampled_colors = None
    has_colors_method = getattr(cloud, "has_colors", None)
    color_data = getattr(cloud, "colors", None)
    has_colors = (
        bool(has_colors_method())
        if callable(has_colors_method)
        else color_data is not None and len(color_data) > 0
    )
    if has_colors:
        colors = np.asarray(color_data, dtype=np.float32)
        if len(colors) != real_point_count:
            raise ValueError("Point and color counts must match")
        sampled_colors = np.ascontiguousarray(colors[indices], dtype=np.float32)

    result = PointCloudData(
        points=np.ascontiguousarray(points[indices], dtype=np.float32),
        colors=sampled_colors,
    )
    if stats is not None:
        stats.sampled_points = len(result.points)
    return result


def _preprocess_depth(
    depth: np.ndarray,
    config: PointCloudConfig,
    depth_filter: Optional[DepthFilter],
) -> np.ndarray:
    if not config.enable_depth_preprocessing:
        return np.ascontiguousarray(depth.copy())
    processor = depth_filter or config.make_depth_filter()
    return processor.process(depth)


def _extract_colors(
    color: Optional[np.ndarray], rows: np.ndarray, cols: np.ndarray
) -> Optional[np.ndarray]:
    if color is None:
        return None
    colors = color[rows, cols].astype(np.float64)
    if colors.size and float(colors.max()) > 1.0:
        colors /= 255.0
    return np.ascontiguousarray(np.clip(colors, 0.0, 1.0))


def _validate_inputs(
    depth: np.ndarray,
    intrinsic: np.ndarray,
    depth_scale: float,
    mask: Optional[np.ndarray],
    color: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
    depth_array = np.asarray(depth)
    intrinsic_array = np.asarray(intrinsic, dtype=np.float32)
    if depth_array.ndim != 2:
        raise ValueError("Depth must have shape (H, W), got {}".format(
            depth_array.shape
        ))
    if not np.issubdtype(depth_array.dtype, np.number):
        raise TypeError("Depth must contain numeric values")
    if intrinsic_array.shape != (3, 3):
        raise ValueError("Intrinsic matrix must have shape (3, 3)")
    if not np.all(np.isfinite(intrinsic_array)):
        raise ValueError("Intrinsic matrix contains NaN or infinite values")
    if not np.isfinite(depth_scale) or depth_scale <= 0:
        raise ValueError("depth_scale must be a positive finite value")
    if intrinsic_array[0, 0] <= 0 or intrinsic_array[1, 1] <= 0:
        raise ValueError("Camera focal lengths fx and fy must be positive")

    if mask is None:
        object_mask = np.ones(depth_array.shape, dtype=bool)
    else:
        object_mask = np.asarray(mask, dtype=bool)
        if object_mask.shape != depth_array.shape:
            raise ValueError(
                "Mask/depth resolution mismatch: {} versus {}".format(
                    object_mask.shape, depth_array.shape
                )
            )

    color_array = None
    if color is not None:
        color_array = np.asarray(color)
        if color_array.ndim != 3 or color_array.shape[:2] != depth_array.shape:
            raise ValueError("Color must have shape (H, W, C) matching depth")
        if color_array.shape[2] != 3:
            raise ValueError("Color must contain exactly three RGB channels")

    return (
        np.ascontiguousarray(depth_array),
        intrinsic_array,
        None if color_array is None else np.ascontiguousarray(color_array),
        np.ascontiguousarray(object_mask),
    )


def _create_roi_mask(
    image_shape: Tuple[int, int],
    workspace_roi: Optional[Tuple[int, int, int, int]],
) -> np.ndarray:
    height, width = image_shape
    if workspace_roi is None:
        return np.ones(image_shape, dtype=bool)

    x_min, y_min, x_max, y_max = (int(value) for value in workspace_roi)
    if not (0 <= x_min < x_max <= width and 0 <= y_min < y_max <= height):
        raise ValueError(
            "workspace_roi {} is outside image size {}x{}".format(
                workspace_roi, width, height
            )
        )
    roi_mask = np.zeros(image_shape, dtype=bool)
    roi_mask[y_min:y_max, x_min:x_max] = True
    return roi_mask


def _load_open3d() -> Any:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError(
            "Open3D is required for point-cloud creation. Install it with: "
            "pip install open3d"
        ) from exc
    return o3d
