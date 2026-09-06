"""YOLOv8-seg adapter for producing a single apple-region mask."""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
from PIL import Image


PathLike = Union[str, Path]


class YOLOAppleDetector:
    """Load an Ultralytics segmentation model and merge all apple instances.

    ``model_path`` must point to an existing YOLO segmentation checkpoint. The
    model may be a COCO-pretrained YOLOv8-seg checkpoint or a future custom
    checkpoint whose class-name mapping contains ``apple``.
    """

    def __init__(
        self,
        model_path: PathLike,
        confidence: float = 0.25,
        iou_threshold: float = 0.7,
        device: Optional[str] = None,
        class_name: str = "apple",
    ) -> None:
        """Load a YOLOv8-seg model from ``model_path``."""

        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError("YOLO segmentation model not found: {}".format(
                self.model_path
            ))
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in [0, 1]")
        if not class_name.strip():
            raise ValueError("class_name must not be empty")

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "Ultralytics is required for YOLOAppleDetector. "
                "Install it with: pip install ultralytics"
            ) from exc

        self.confidence = float(confidence)
        self.iou_threshold = float(iou_threshold)
        self.device = device
        self.class_name = class_name.strip().lower()
        self.model = YOLO(str(self.model_path))

        model_task = getattr(self.model, "task", None)
        if model_task is not None and model_task != "segment":
            raise ValueError(
                "YOLO model must be an instance-segmentation model, got task={!r}".format(
                    model_task
                )
            )

        self.apple_class_ids = self._find_class_ids(getattr(self.model, "names", {}))
        if not self.apple_class_ids:
            raise ValueError(
                "YOLO model has no class named {!r}; available classes: {}".format(
                    self.class_name, self._format_class_names(getattr(self.model, "names", {}))
                )
            )

    def detect(self, rgb: np.ndarray) -> np.ndarray:
        """Infer all apple instances and return their union as ``(H, W)`` bool."""

        rgb_array = self._normalize_rgb(rgb)
        height, width = rgb_array.shape[:2]

        # Ultralytics NumPy sources follow the OpenCV BGR convention.
        bgr_array = np.ascontiguousarray(rgb_array[..., ::-1])
        predict_args = {
            "source": bgr_array,
            "conf": self.confidence,
            "iou": self.iou_threshold,
            "classes": self.apple_class_ids,
            "retina_masks": True,
            "verbose": False,
        }
        if self.device is not None:
            predict_args["device"] = self.device

        results = self.model.predict(**predict_args)
        if not results:
            return np.zeros((height, width), dtype=bool)

        result = results[0]
        if result.masks is None or result.boxes is None or len(result.boxes) == 0:
            return np.zeros((height, width), dtype=bool)

        class_ids = result.boxes.cls.detach().cpu().numpy().astype(np.int64)
        instance_masks = result.masks.data.detach().cpu().numpy()
        if len(class_ids) != len(instance_masks):
            raise RuntimeError(
                "YOLO returned {} classes but {} instance masks".format(
                    len(class_ids), len(instance_masks)
                )
            )

        apple_indices = np.isin(class_ids, self.apple_class_ids)
        if not np.any(apple_indices):
            return np.zeros((height, width), dtype=bool)

        apple_mask = np.any(instance_masks[apple_indices] > 0.5, axis=0)
        if apple_mask.shape != (height, width):
            # retina_masks should already preserve the source shape. Nearest
            # resizing is retained for compatibility with older releases.
            nearest = getattr(Image, "Resampling", Image).NEAREST
            apple_mask = np.asarray(
                Image.fromarray(apple_mask.astype(np.uint8)).resize(
                    (width, height), resample=nearest
                ),
                dtype=bool,
            )

        return np.ascontiguousarray(apple_mask, dtype=bool)

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        """Allow this detector to be used as a callable predictor."""

        return self.detect(rgb)

    @staticmethod
    def _normalize_rgb(rgb: np.ndarray) -> np.ndarray:
        rgb_array = np.asarray(rgb)
        if rgb_array.ndim != 3 or rgb_array.shape[2] != 3:
            raise ValueError("rgb must have shape (H, W, 3), got {}".format(rgb_array.shape))
        if not np.issubdtype(rgb_array.dtype, np.number):
            raise TypeError("rgb must contain numeric values")
        if not np.all(np.isfinite(rgb_array)):
            raise ValueError("rgb contains NaN or infinite values")

        if np.issubdtype(rgb_array.dtype, np.floating):
            rgb_float = rgb_array.astype(np.float32, copy=False)
            if rgb_float.size and float(rgb_float.max()) <= 1.0:
                rgb_float = rgb_float * 255.0
            rgb_array = np.clip(rgb_float, 0.0, 255.0).astype(np.uint8)
        elif rgb_array.dtype != np.uint8:
            rgb_array = np.clip(rgb_array, 0, 255).astype(np.uint8)

        return np.ascontiguousarray(rgb_array)

    def _find_class_ids(
        self, names: Optional[Union[Dict[int, str], Sequence[str]]]
    ) -> List[int]:
        if names is None:
            return []
        if isinstance(names, dict):
            items = names.items()
        else:
            items = enumerate(names)
        return [
            int(class_id)
            for class_id, name in items
            if str(name).strip().lower() == self.class_name
        ]

    @staticmethod
    def _format_class_names(
        names: Optional[Union[Dict[int, str], Sequence[str]]]
    ) -> str:
        if names is None:
            return "<unavailable>"
        values = names.values() if isinstance(names, dict) else names
        return ", ".join(str(value) for value in values)
