"""Pure NumPy RGB-D projection and GraspNet point sampling."""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class PointCloudData:
    """Unorganized point cloud in the camera coordinate system."""

    points: np.ndarray
    colors: Optional[np.ndarray] = None


def create_point_cloud(
    depth: np.ndarray,
    intrinsic: np.ndarray,
    depth_scale: float,
    mask: Optional[np.ndarray] = None,
    color: Optional[np.ndarray] = None,
) -> PointCloudData:
    """Project a depth image into an unorganized metric point cloud.

    The returned XYZ coordinates are expressed in the camera frame and metres.
    """

    depth_array = np.asarray(depth)
    intrinsic_array = np.asarray(intrinsic, dtype=np.float32)
    if depth_array.ndim != 2:
        raise ValueError("Depth must have shape (H, W), got {}".format(depth_array.shape))
    if intrinsic_array.shape != (3, 3):
        raise ValueError("Intrinsic matrix must have shape (3, 3)")
    if not np.isfinite(depth_scale) or depth_scale <= 0:
        raise ValueError("depth_scale must be a positive finite value")

    fx = float(intrinsic_array[0, 0])
    fy = float(intrinsic_array[1, 1])
    cx = float(intrinsic_array[0, 2])
    cy = float(intrinsic_array[1, 2])
    if fx <= 0 or fy <= 0:
        raise ValueError("Camera focal lengths fx and fy must be positive")

    if mask is None:
        valid_mask = np.isfinite(depth_array) & (depth_array > 0)
    else:
        valid_mask = np.asarray(mask, dtype=bool)
        if valid_mask.shape != depth_array.shape:
            raise ValueError(
                "Mask/depth resolution mismatch: {} versus {}".format(
                    valid_mask.shape, depth_array.shape
                )
            )
        valid_mask = valid_mask & np.isfinite(depth_array) & (depth_array > 0)

    rows, cols = np.nonzero(valid_mask)
    if rows.size == 0:
        raise ValueError("No valid depth pixels remain after mask processing")

    z = depth_array[rows, cols].astype(np.float32) * np.float32(depth_scale)
    x = (cols.astype(np.float32) - cx) * z / fx
    y = (rows.astype(np.float32) - cy) * z / fy
    points = np.ascontiguousarray(np.column_stack((x, y, z)), dtype=np.float32)

    colors = None
    if color is not None:
        color_array = np.asarray(color)
        if color_array.shape[:2] != depth_array.shape or color_array.ndim != 3:
            raise ValueError("Color must have shape (H, W, C) matching depth")
        colors = np.ascontiguousarray(color_array[rows, cols], dtype=np.float32)

    return PointCloudData(points=points, colors=colors)


def sample_point_cloud(
    cloud: PointCloudData,
    num_points: int = 20000,
    seed: Optional[int] = None,
) -> PointCloudData:
    """Sample exactly ``num_points`` using the policy from the official demo."""

    points = np.asarray(cloud.points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Point cloud must have shape (N, 3), got {}".format(points.shape))
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
    if cloud.colors is not None:
        colors = np.asarray(cloud.colors)
        if len(colors) != len(points):
            raise ValueError("Point and color counts must match")
        sampled_colors = np.ascontiguousarray(colors[indices], dtype=np.float32)

    return PointCloudData(
        points=np.ascontiguousarray(points[indices], dtype=np.float32),
        colors=sampled_colors,
    )
