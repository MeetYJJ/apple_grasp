"""Replaceable apple-segmentation interface.

The default implementation is a lightweight red-region heuristic used only to
exercise the pipeline before a trained apple detector is integrated. A YOLO,
SAM, or combined predictor can be injected without changing downstream code.
"""

from typing import Any, Callable, Optional

import numpy as np
from scipy import ndimage


MaskPredictor = Callable[[np.ndarray], np.ndarray]


class AppleMaskDetector:
    """Produce a boolean apple-region mask from an RGB image.

    Args:
        predictor: Optional callable implementing ``predictor(rgb) -> mask``.
            Its output must have shape ``(H, W)``. When omitted, a temporary
            red-color heuristic is used; it is not an apple recognition model.
        min_red: Minimum normalized red-channel intensity for the heuristic.
        min_dominance: Minimum red advantage over both green and blue.
        min_saturation: Minimum RGB saturation for rejecting gray/white pixels.
        min_area: Minimum connected-component area in pixels.
    """

    def __init__(
        self,
        predictor: Optional[Any] = None,
        min_red: float = 0.35,
        min_dominance: float = 0.08,
        min_saturation: float = 0.20,
        min_area: int = 200,
    ) -> None:
        if not 0.0 <= min_red <= 1.0:
            raise ValueError("min_red must be in [0, 1]")
        if not 0.0 <= min_dominance <= 1.0:
            raise ValueError("min_dominance must be in [0, 1]")
        if not 0.0 <= min_saturation <= 1.0:
            raise ValueError("min_saturation must be in [0, 1]")
        if min_area < 0:
            raise ValueError("min_area must be non-negative")

        if predictor is not None:
            has_detect = callable(getattr(predictor, "detect", None))
            if not has_detect and not callable(predictor):
                raise TypeError("predictor must be callable or provide detect(rgb)")
        self.predictor = predictor
        self.min_red = float(min_red)
        self.min_dominance = float(min_dominance)
        self.min_saturation = float(min_saturation)
        self.min_area = int(min_area)

    def detect(self, rgb: np.ndarray) -> np.ndarray:
        """Return a contiguous boolean mask with shape ``(H, W)``."""

        rgb_array = self._validate_rgb(rgb)
        if self.predictor is None:
            mask = self._detect_red_regions(rgb_array)
        elif callable(getattr(self.predictor, "detect", None)):
            mask = self.predictor.detect(rgb_array)
        else:
            mask = self.predictor(rgb_array)
        return self._normalize_mask(mask, rgb_array.shape[:2])

    @staticmethod
    def _validate_rgb(rgb: np.ndarray) -> np.ndarray:
        rgb_array = np.asarray(rgb)
        if rgb_array.ndim != 3 or rgb_array.shape[2] != 3:
            raise ValueError("rgb must have shape (H, W, 3), got {}".format(rgb_array.shape))
        if not np.issubdtype(rgb_array.dtype, np.number):
            raise TypeError("rgb must contain numeric values")
        if not np.all(np.isfinite(rgb_array)):
            raise ValueError("rgb contains NaN or infinite values")
        return rgb_array

    def _detect_red_regions(self, rgb: np.ndarray) -> np.ndarray:
        rgb_float = rgb.astype(np.float32, copy=False)
        if rgb_float.size and float(rgb_float.max()) > 1.0:
            rgb_float = rgb_float / 255.0
        rgb_float = np.clip(rgb_float, 0.0, 1.0)

        red = rgb_float[..., 0]
        green = rgb_float[..., 1]
        blue = rgb_float[..., 2]
        channel_max = rgb_float.max(axis=2)
        channel_min = rgb_float.min(axis=2)
        saturation = (channel_max - channel_min) / np.maximum(channel_max, 1e-6)

        mask = (
            (red >= self.min_red)
            & ((red - green) >= self.min_dominance)
            & ((red - blue) >= self.min_dominance)
            & (saturation >= self.min_saturation)
        )

        if self.min_area > 1 and np.any(mask):
            labels, component_count = ndimage.label(mask)
            component_sizes = np.bincount(labels.ravel())
            keep = component_sizes >= self.min_area
            keep[0] = False
            mask = keep[labels] if component_count else mask

        return mask

    @staticmethod
    def _normalize_mask(mask: np.ndarray, image_shape: tuple) -> np.ndarray:
        mask_array = np.asarray(mask)
        if mask_array.shape != image_shape:
            raise ValueError(
                "Apple mask must have shape {}, got {}".format(image_shape, mask_array.shape)
            )
        return np.ascontiguousarray(mask_array.astype(bool))
