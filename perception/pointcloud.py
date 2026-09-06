"""RGB-D projection, geometric filtering, and GraspNet point sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Tuple

import cv2
import numpy as np

if TYPE_CHECKING:
    import open3d


@dataclass(frozen=True)
class PointCloudFilterConfig:
    """Configurable filters applied before a cloud is sent to GraspNet.

    ``workspace_roi`` uses image coordinates ``(x_min, y_min, x_max, y_max)``
    with an exclusive maximum bound. Depth limits are expressed in metres.
    Set ``median_kernel=1``, ``outlier_nb_neighbors=0``, ``voxel_size=0`` or
    ``normal_radius=0`` to disable the corresponding operation.
    """

    median_kernel: int = 3
    workspace_roi: Optional[Tuple[int, int, int, int]] = None
    min_depth_m: Optional[float] = None
    max_depth_m: Optional[float] = None
    outlier_nb_neighbors: int = 20
    outlier_std_ratio: float = 2.0
    voxel_size: float = 0.002
    normal_radius: float = 0.01
    normal_max_nn: int = 30

    def __post_init__(self) -> None:
        if self.median_kernel not in (1, 3, 5):
            raise ValueError("median_kernel must be one of 1, 3, or 5")
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
        if self.outlier_nb_neighbors < 0:
            raise ValueError("outlier_nb_neighbors must be non-negative")
        if self.outlier_std_ratio <= 0:
            raise ValueError("outlier_std_ratio must be positive")
        if self.voxel_size < 0:
            raise ValueError("voxel_size must be non-negative")
        if self.normal_radius < 0:
            raise ValueError("normal_radius must be non-negative")
        if self.normal_max_nn <= 0:
            raise ValueError("normal_max_nn must be positive")


@dataclass
class PointCloudStats:
    """Point counts at the important quality-control stages."""

    raw_mask_pixels: int = 0
    valid_depth_pixels: int = 0
    filtered_points: int = 0
    graspnet_input_points: int = 0

    def print(self) -> None:
        """Print the counters requested by the realtime pipeline."""

        print("原始mask像素: {}".format(self.raw_mask_pixels))
        print("有效depth: {}".format(self.valid_depth_pixels))
        print("滤波后点数: {}".format(self.filtered_points))
        print("最终输入GraspNet点数: {}".format(self.graspnet_input_points))


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
    config: Optional[PointCloudFilterConfig] = None,
    stats: Optional[PointCloudStats] = None,
) -> open3d.geometry.PointCloud:
    """Create a filtered Open3D point cloud from aligned RGB-D and a mask.

    The existing positional inputs remain unchanged. ``config`` and ``stats``
    are optional so offline images and realtime D435i frames share the same
    interface. Output points are expressed in metres in the camera frame.
    """

    filter_config = config or PointCloudFilterConfig()
    depth_array, intrinsic_array, color_array, object_mask = _validate_inputs(
        depth, intrinsic, depth_scale, mask, color
    )
    height, width = depth_array.shape

    if stats is not None:
        stats.raw_mask_pixels = int(object_mask.sum())
        stats.valid_depth_pixels = 0
        stats.filtered_points = 0
        stats.graspnet_input_points = 0

    roi_mask = _create_roi_mask(
        image_shape=(height, width), workspace_roi=filter_config.workspace_roi
    )
    filtered_depth = median_filter_depth(depth_array, filter_config.median_kernel)
    valid_mask = (
        object_mask
        & roi_mask
        & np.isfinite(filtered_depth)
        & (filtered_depth > 0)
    )

    depth_metres = filtered_depth.astype(np.float32) * np.float32(depth_scale)
    if filter_config.min_depth_m is not None:
        valid_mask &= depth_metres >= np.float32(filter_config.min_depth_m)
    if filter_config.max_depth_m is not None:
        valid_mask &= depth_metres <= np.float32(filter_config.max_depth_m)

    valid_depth_pixels = int(valid_mask.sum())
    if stats is not None:
        stats.valid_depth_pixels = valid_depth_pixels
    if valid_depth_pixels == 0:
        raise ValueError("No valid depth pixels remain after mask and ROI filtering")

    rows, cols = np.nonzero(valid_mask)
    fx = float(intrinsic_array[0, 0])
    fy = float(intrinsic_array[1, 1])
    cx = float(intrinsic_array[0, 2])
    cy = float(intrinsic_array[1, 2])
    z = depth_metres[rows, cols]
    x = (cols.astype(np.float32) - cx) * z / fx
    y = (rows.astype(np.float32) - cy) * z / fy
    points = np.ascontiguousarray(np.column_stack((x, y, z)), dtype=np.float64)

    o3d = _load_open3d()
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    if color_array is not None:
        colors = color_array[rows, cols].astype(np.float64)
        if colors.size and float(colors.max()) > 1.0:
            colors /= 255.0
        cloud.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))

    cloud = filter_point_cloud(cloud, filter_config, o3d=o3d)
    filtered_point_count = len(cloud.points)
    if filtered_point_count == 0:
        raise ValueError("All points were removed by point-cloud filtering")
    if stats is not None:
        stats.filtered_points = filtered_point_count
    return cloud


def median_filter_depth(depth: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply a fast median filter while preserving the depth array's units."""

    depth_array = np.asarray(depth)
    if depth_array.ndim != 2:
        raise ValueError("Depth must have shape (H, W), got {}".format(
            depth_array.shape
        ))
    if kernel_size not in (1, 3, 5):
        raise ValueError("kernel_size must be one of 1, 3, or 5")
    if kernel_size == 1:
        return np.ascontiguousarray(depth_array.copy())

    supported_depth = depth_array
    if depth_array.dtype not in (np.uint8, np.uint16, np.float32):
        supported_depth = depth_array.astype(np.float32)
    return np.ascontiguousarray(
        cv2.medianBlur(np.ascontiguousarray(supported_depth), kernel_size)
    )


def filter_point_cloud(
    cloud: Any,
    config: Optional[PointCloudFilterConfig] = None,
    o3d: Optional[Any] = None,
) -> open3d.geometry.PointCloud:
    """Remove statistical outliers, voxelize, and estimate normals."""

    filter_config = config or PointCloudFilterConfig()
    open3d_module = o3d or _load_open3d()
    filtered = cloud

    point_count = len(filtered.points)
    if (
        filter_config.outlier_nb_neighbors > 0
        and point_count > filter_config.outlier_nb_neighbors
    ):
        filtered, _ = filtered.remove_statistical_outlier(
            nb_neighbors=filter_config.outlier_nb_neighbors,
            std_ratio=filter_config.outlier_std_ratio,
        )

    if filter_config.voxel_size > 0 and len(filtered.points) > 0:
        filtered = filtered.voxel_down_sample(filter_config.voxel_size)

    if filter_config.normal_radius > 0 and len(filtered.points) >= 3:
        filtered.estimate_normals(
            search_param=open3d_module.geometry.KDTreeSearchParamHybrid(
                radius=filter_config.normal_radius,
                max_nn=filter_config.normal_max_nn,
            )
        )
        # A single RGB-D view observes the surface from the camera origin.
        filtered.orient_normals_towards_camera_location(
            camera_location=np.zeros(3, dtype=np.float64)
        )
        filtered.normalize_normals()

    return filtered


def sample_point_cloud(
    cloud: Any,
    num_points: int = 20000,
    seed: Optional[int] = None,
    stats: Optional[PointCloudStats] = None,
) -> PointCloudData:
    """Sample exactly ``num_points`` using the policy from the official demo."""

    points = np.asarray(cloud.points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Point cloud must have shape (N, 3), got {}".format(
            points.shape
        ))
    if len(points) == 0:
        raise ValueError("Cannot sample an empty point cloud")
    if num_points <= 0:
        raise ValueError("num_points must be positive")

    rng = np.random.default_rng(seed)
    if len(points) >= num_points:
        indices = rng.choice(len(points), num_points, replace=False)
    else:
        extra = rng.choice(len(points), num_points - len(points), replace=True)
        indices = np.concatenate((np.arange(len(points)), extra))
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
        if len(colors) != len(points):
            raise ValueError("Point and color counts must match")
        sampled_colors = np.ascontiguousarray(colors[indices], dtype=np.float32)

    result = PointCloudData(
        points=np.ascontiguousarray(points[indices], dtype=np.float32),
        colors=sampled_colors,
    )
    if stats is not None:
        stats.graspnet_input_points = len(result.points)
    return result


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
