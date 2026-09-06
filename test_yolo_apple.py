"""Smoke test for YOLOv8-seg apple-mask inference."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from perception.apple_mask import AppleMaskDetector
from perception.yolo_apple_detector import DEFAULT_YOLO_MODEL, YOLOAppleDetector


PROJECT_ROOT = Path(__file__).resolve().parent
COLOR_PATH = PROJECT_ROOT.parent / "graspnet-baseline" / "doc" / "example_data" / "color.png"
OUTPUT_PATH = PROJECT_ROOT / "output" / "apple_mask_yolo.png"


def load_rgb(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError("Example RGB image not found: {}".format(path))
    with Image.open(str(path)) as image:
        return np.asarray(image.convert("RGB"))


def main(model_path: str, device: str = None) -> None:
    rgb = load_rgb(COLOR_PATH)
    yolo_predictor = YOLOAppleDetector(model_path=model_path, device=device)
    detector = AppleMaskDetector(predictor=yolo_predictor)
    apple_mask = detector.detect(rgb)

    if apple_mask.shape != rgb.shape[:2] or apple_mask.dtype != np.bool_:
        raise TypeError("YOLO apple detector must return a bool mask with shape (H, W)")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(apple_mask.astype(np.uint8) * 255, mode="L").save(str(OUTPUT_PATH))
    print("YOLO model loaded: {}".format(yolo_predictor.model_path))
    print("Apple class id(s): {}".format(yolo_predictor.apple_class_ids))
    print("Apple mask saved: {}".format(OUTPUT_PATH))
    print("Mask shape: {}; dtype: {}; pixels: {}".format(
        apple_mask.shape, apple_mask.dtype, int(apple_mask.sum())
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test YOLOv8-seg apple mask inference")
    parser.add_argument(
        "--model-path",
        default=DEFAULT_YOLO_MODEL,
        help=(
            "Local segmentation checkpoint or official Ultralytics model name "
            "(default: yolov8n-seg.pt)"
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional Ultralytics device, for example 0 or cpu",
    )
    arguments = parser.parse_args()
    main(arguments.model_path, arguments.device)
