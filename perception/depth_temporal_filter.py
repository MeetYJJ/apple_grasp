"""Temporal smoothing for aligned depth inside the current object mask."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class DepthTemporalFilterConfig:
    previous_weight: float = 0.6
    max_depth_delta_mm: float = 80.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.previous_weight < 1.0:
            raise ValueError("previous_weight must be in [0, 1)")
        if self.max_depth_delta_mm <= 0:
            raise ValueError("max_depth_delta_mm must be positive")


class DepthTemporalFilter:
    """Apply an edge-gated EMA only where consecutive masks overlap.

    New mask pixels and large depth discontinuities use the current D435i
    measurement immediately. This prevents a moving apple from inheriting
    background depth or stale depth from its previous image position.
    """

    def __init__(
        self, config: Optional[DepthTemporalFilterConfig] = None
    ) -> None:
        self.config = config or DepthTemporalFilterConfig()
        self.previous_depth: Optional[np.ndarray] = None
        self.previous_mask: Optional[np.ndarray] = None
        self.last_blended_pixels = 0

    def reset(self) -> None:
        self.previous_depth = None
        self.previous_mask = None
        self.last_blended_pixels = 0

    def process(self, depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Return depth with EMA smoothing inside valid mask overlap."""

        current = np.asarray(depth)
        current_mask = np.asarray(mask, dtype=bool)
        if current.ndim != 2:
            raise ValueError("depth must have shape (H, W)")
        if not np.issubdtype(current.dtype, np.number):
            raise TypeError("depth must contain numeric values")
        if current_mask.shape != current.shape:
            raise ValueError("mask and depth shapes must match")

        output = current.copy()
        if (
            self.previous_depth is not None
            and self.previous_mask is not None
            and self.previous_depth.shape == current.shape
        ):
            previous = self.previous_depth
            overlap = (
                current_mask
                & self.previous_mask
                & np.isfinite(current)
                & np.isfinite(previous)
                & (current > 0)
                & (previous > 0)
            )
            depth_delta = np.abs(
                current.astype(np.float64) - previous.astype(np.float64)
            )
            blend_mask = overlap & (
                depth_delta <= self.config.max_depth_delta_mm
            )
            previous_weight = self.config.previous_weight
            blended = (
                previous_weight * previous[blend_mask].astype(np.float64)
                + (1.0 - previous_weight)
                * current[blend_mask].astype(np.float64)
            )
            if np.issubdtype(output.dtype, np.integer):
                blended = np.rint(blended)
            output[blend_mask] = blended.astype(output.dtype)
            self.last_blended_pixels = int(blend_mask.sum())
        else:
            self.last_blended_pixels = 0

        self.previous_depth = np.ascontiguousarray(output.copy())
        self.previous_mask = np.ascontiguousarray(current_mask.copy())
        return np.ascontiguousarray(output)

    __call__ = process
