"""Visual smoke test for the replaceable apple-mask interface."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from perception.apple_mask import AppleMaskDetector


PROJECT_ROOT = Path(__file__).resolve().parent
COLOR_PATH = PROJECT_ROOT.parent / "graspnet-baseline" / "doc" / "example_data" / "color.png"
OUTPUT_PATH = PROJECT_ROOT / "output" / "apple_mask.png"


def load_rgb(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError("Example RGB image not found: {}".format(path))
    with Image.open(str(path)) as image:
        return np.asarray(image.convert("RGB"))


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(str(path))


def display_result(rgb: np.ndarray, mask: np.ndarray) -> None:
    """Display RGB and mask side by side, with a Pillow fallback."""

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        Image.fromarray(rgb).show(title="RGB image")
        Image.fromarray(mask.astype(np.uint8) * 255, mode="L").show(title="Apple mask")
        return

    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].imshow(rgb)
    axes[0].set_title("RGB image")
    axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Apple mask (temporary red-region heuristic)")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    plt.show()


def main(display: bool = True) -> None:
    rgb = load_rgb(COLOR_PATH)
    detector = AppleMaskDetector()
    mask = detector.detect(rgb)

    if mask.shape != rgb.shape[:2] or mask.dtype != np.bool_:
        raise TypeError("AppleMaskDetector must return a bool mask with shape (H, W)")
    if not np.any(mask):
        raise RuntimeError("AppleMaskDetector returned an empty mask for the example image")

    save_mask(mask, OUTPUT_PATH)
    print("Apple mask saved: {}".format(OUTPUT_PATH))
    print("Mask shape: {}; dtype: {}; pixels: {}".format(
        mask.shape, mask.dtype, int(mask.sum())
    ))

    if display:
        display_result(rgb, mask)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the apple mask interface")
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Generate and verify the mask without opening a display window",
    )
    args = parser.parse_args()
    main(display=not args.no_display)
