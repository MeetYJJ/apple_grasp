"""Load an RGB-D frame from the GraspNet offline folder format."""

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import scipy.io as scio
from PIL import Image


PathLike = Union[str, Path]


@dataclass(frozen=True)
class RGBDFrame:
    """Source-independent RGB-D data consumed by the perception pipeline.

    A future RealSense adapter can construct the same object from live frames,
    so downstream point-cloud and grasp modules do not depend on the camera SDK.
    """

    color: np.ndarray
    depth: np.ndarray
    intrinsic: np.ndarray
    depth_scale: float


def load_rgbd(data_dir: PathLike) -> RGBDFrame:
    """Load ``color.png``, ``depth.png`` and ``meta.mat`` from *data_dir*.

    Color is returned as float32 RGB in [0, 1]. Depth values remain in their
    original units; ``depth_scale`` is metres per raw depth unit, matching the
    convention used by the RealSense SDK.
    """

    data_path = Path(data_dir).expanduser().resolve()
    if not data_path.is_dir():
        raise FileNotFoundError("RGB-D data directory not found: {}".format(data_path))

    color_path = data_path / "color.png"
    depth_path = data_path / "depth.png"
    meta_path = data_path / "meta.mat"
    for required_path in (color_path, depth_path, meta_path):
        if not required_path.is_file():
            raise FileNotFoundError("Required RGB-D file not found: {}".format(required_path))

    with Image.open(str(color_path)) as image:
        color = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    with Image.open(str(depth_path)) as image:
        depth = np.asarray(image)

    if depth.ndim != 2:
        raise ValueError("depth.png must be a single-channel image, got {}".format(depth.shape))
    if color.shape[:2] != depth.shape:
        raise ValueError(
            "Color/depth resolution mismatch: {} versus {}".format(color.shape[:2], depth.shape)
        )

    meta = scio.loadmat(str(meta_path))
    if "intrinsic_matrix" not in meta or "factor_depth" not in meta:
        raise KeyError("meta.mat must contain intrinsic_matrix and factor_depth")

    intrinsic = np.asarray(meta["intrinsic_matrix"], dtype=np.float32)
    if intrinsic.shape != (3, 3):
        raise ValueError("intrinsic_matrix must have shape (3, 3), got {}".format(intrinsic.shape))

    factor_depth = np.asarray(meta["factor_depth"]).squeeze()
    if factor_depth.size != 1:
        raise ValueError("factor_depth must contain one scalar value")
    factor_depth_value = float(factor_depth)
    if not np.isfinite(factor_depth_value) or factor_depth_value <= 0:
        raise ValueError("factor_depth must be a positive finite value")
    depth_scale = 1.0 / factor_depth_value

    return RGBDFrame(
        color=np.ascontiguousarray(color),
        depth=np.ascontiguousarray(depth),
        intrinsic=intrinsic,
        depth_scale=depth_scale,
    )
