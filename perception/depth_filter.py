"""Depth-image preprocessing for aligned RGB-D point-cloud generation."""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class DepthFilterConfig:
    """Settings for edge-aware depth smoothing and conservative hole filling."""

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

    def __post_init__(self) -> None:
        if self.median_kernel not in (1, 3, 5):
            raise ValueError("median_kernel must be one of 1, 3, or 5")
        if self.spatial_diameter <= 0 or self.spatial_diameter % 2 == 0:
            raise ValueError("spatial_diameter must be a positive odd integer")
        if self.spatial_sigma_color <= 0 or self.spatial_sigma_space <= 0:
            raise ValueError("spatial smoothing sigmas must be positive")
        if self.hole_fill_kernel < 3 or self.hole_fill_kernel % 2 == 0:
            raise ValueError("hole_fill_kernel must be an odd integer >= 3")
        neighborhood_size = self.hole_fill_kernel * self.hole_fill_kernel - 1
        if not 1 <= self.hole_fill_min_neighbors <= neighborhood_size:
            raise ValueError(
                "hole_fill_min_neighbors must be within the hole neighborhood"
            )
        if self.hole_fill_iterations < 0:
            raise ValueError("hole_fill_iterations must be non-negative")
        if self.hole_fill_max_depth_delta_mm <= 0:
            raise ValueError("hole_fill_max_depth_delta_mm must be positive")


class DepthFilter:
    """Preprocess a depth image without changing its shape, dtype, or units.

    The default configuration targets D435i ``uint16`` millimetre depth. Hole
    filling is deliberately conservative: only zero pixels surrounded by at
    least five valid pixels in a 3x3 neighborhood are filled, preventing the
    foreground from expanding freely into the background.
    """

    def __init__(self, config: Optional[DepthFilterConfig] = None) -> None:
        self.config = config or DepthFilterConfig()

    def process(self, depth: np.ndarray) -> np.ndarray:
        """Return filtered depth with the same ``(H, W)`` shape and dtype."""

        depth_array = np.asarray(depth)
        if depth_array.ndim != 2:
            raise ValueError("depth must have shape (H, W), got {}".format(
                depth_array.shape
            ))
        if not np.issubdtype(depth_array.dtype, np.number):
            raise TypeError("depth must contain numeric values")

        original_dtype = depth_array.dtype
        working = depth_array.astype(np.float32, copy=True)
        working[~np.isfinite(working)] = 0.0
        working[working < 0.0] = 0.0

        if self.config.enable_median_filter and self.config.median_kernel > 1:
            measured_valid = working > 0.0
            median = cv2.medianBlur(
                np.ascontiguousarray(working), self.config.median_kernel
            )
            # Median filtering must not double as hole filling. Preserve the
            # original invalid-pixel map and only denoise measured depths.
            replaceable = measured_valid & (median > 0.0)
            working[replaceable] = median[replaceable]
            working[~measured_valid] = 0.0

        if self.config.enable_spatial_smoothing:
            valid = working > 0.0
            smoothed = cv2.bilateralFilter(
                np.ascontiguousarray(working),
                d=self.config.spatial_diameter,
                sigmaColor=self.config.spatial_sigma_color,
                sigmaSpace=self.config.spatial_sigma_space,
            )
            # Bilateral filtering must not silently turn the entire zero-depth
            # background into geometry; dedicated hole filling handles zeros.
            replaceable = valid & (smoothed > 0.0)
            working[replaceable] = smoothed[replaceable]
            working[~valid] = 0.0

        if self.config.enable_hole_filling:
            working = self._fill_small_holes(working)

        return self._restore_dtype(working, original_dtype)

    def _fill_small_holes(self, depth: np.ndarray) -> np.ndarray:
        result = depth.copy()
        kernel = np.ones(
            (self.config.hole_fill_kernel, self.config.hole_fill_kernel),
            dtype=np.uint8,
        )
        center = self.config.hole_fill_kernel // 2
        kernel[center, center] = 0.0

        for _ in range(self.config.hole_fill_iterations):
            valid = result > 0.0
            if np.all(valid):
                break
            valid_float = valid.astype(np.float32)
            neighbor_count = cv2.filter2D(
                valid_float,
                ddepth=-1,
                kernel=kernel,
                borderType=cv2.BORDER_CONSTANT,
            )
            neighbor_sum = cv2.filter2D(
                np.where(valid, result, 0.0),
                ddepth=-1,
                kernel=kernel,
                borderType=cv2.BORDER_CONSTANT,
            )
            fillable = (~valid) & (
                neighbor_count >= float(self.config.hole_fill_min_neighbors)
            )
            # Require support across the hole in at least one image direction.
            # This stops the foreground silhouette from growing into empty
            # background even when several diagonal neighbors are present.
            vertical_support = np.zeros_like(valid)
            horizontal_support = np.zeros_like(valid)
            vertical_support[1:-1, :] = valid[:-2, :] & valid[2:, :]
            horizontal_support[:, 1:-1] = valid[:, :-2] & valid[:, 2:]

            max_neighbor = cv2.dilate(
                np.where(valid, result, 0.0),
                kernel,
                borderType=cv2.BORDER_CONSTANT,
                borderValue=0.0,
            )
            sentinel = float(np.finfo(np.float32).max)
            min_neighbor = cv2.erode(
                np.where(valid, result, sentinel),
                kernel,
                borderType=cv2.BORDER_CONSTANT,
                borderValue=sentinel,
            )
            locally_continuous = (
                max_neighbor - min_neighbor
                <= self.config.hole_fill_max_depth_delta_mm
            )
            fillable &= (
                (vertical_support | horizontal_support) & locally_continuous
            )
            if not np.any(fillable):
                break
            result[fillable] = (
                neighbor_sum[fillable] / neighbor_count[fillable]
            )
        return result

    @staticmethod
    def _restore_dtype(depth: np.ndarray, dtype: np.dtype) -> np.ndarray:
        if np.issubdtype(dtype, np.integer):
            limits = np.iinfo(dtype)
            restored = np.rint(np.clip(depth, limits.min, limits.max)).astype(dtype)
        else:
            restored = depth.astype(dtype)
        return np.ascontiguousarray(restored)
