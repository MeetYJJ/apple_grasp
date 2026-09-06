"""Mask loading and normalization independent of the mask producer."""

from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image


PathLike = Union[str, Path]


def load_mask(mask_path: PathLike) -> np.ndarray:
    """Load a single-channel mask and convert all non-zero pixels to True."""

    path = Path(mask_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError("Mask file not found: {}".format(path))

    with Image.open(str(path)) as image:
        mask = np.asarray(image)

    if mask.ndim == 3:
        # Accept RGB masks only when all channels describe the same support.
        mask = np.any(mask != 0, axis=2)
    elif mask.ndim != 2:
        raise ValueError("Mask must be a 2-D image, got {}".format(mask.shape))
    return np.ascontiguousarray(mask.astype(bool))


def build_valid_mask(depth: np.ndarray, object_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Combine valid depth with an optional workspace/object segmentation mask.

    ``object_mask`` may come from a file, a YOLO bounding-box rasterization, or
    SAM. This function deliberately has no dependency on any detector.
    """

    depth_array = np.asarray(depth)
    if depth_array.ndim != 2:
        raise ValueError("Depth must have shape (H, W), got {}".format(depth_array.shape))

    valid_mask = np.isfinite(depth_array) & (depth_array > 0)
    if object_mask is not None:
        mask_array = np.asarray(object_mask)
        if mask_array.shape != depth_array.shape:
            raise ValueError(
                "Mask/depth resolution mismatch: {} versus {}".format(
                    mask_array.shape, depth_array.shape
                )
            )
        valid_mask &= mask_array.astype(bool)

    return np.ascontiguousarray(valid_mask)
