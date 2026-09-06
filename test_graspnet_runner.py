"""File/camera end-to-end smoke test for the apple grasp pipeline."""

import argparse
from pathlib import Path
from typing import Optional, Tuple

import torch

from grasp.grasp_selector import GraspSelector
from grasp.graspnet_runner import GraspNetRunner
from perception.apple_mask import AppleMaskDetector
from perception.mask_process import build_valid_mask, load_mask
from perception.pointcloud import create_point_cloud, sample_point_cloud
from perception.realsense_camera import RealSenseCamera
from perception.rgbd_loader import RGBDFrame, load_rgbd
from perception.yolo_apple_detector import YOLOAppleDetector


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
BASELINE_DIR = WORKSPACE_ROOT / "graspnet-baseline"
CHECKPOINT_PATH = BASELINE_DIR / "checkpoint-rs.tar"
DATA_DIR = BASELINE_DIR / "doc" / "example_data"
OUTPUT_PATH = PROJECT_ROOT / "output" / "best_grasp.npy"


def load_source(source: str) -> Tuple[RGBDFrame, Optional[Path]]:
    """Return a source-independent RGBDFrame and optional workspace mask path."""

    if source == "file":
        return load_rgbd(DATA_DIR), DATA_DIR / "workspace_mask.png"
    if source == "camera":
        with RealSenseCamera() as camera:
            rgb, depth = camera.get_frame()
            frame = RGBDFrame(
                color=rgb,
                depth=depth,
                intrinsic=camera.intrinsic.copy(),
                depth_scale=camera.depth_scale,
            )
            print("RealSense: {} (serial={})".format(
                camera.device_name, camera.serial_number
            ))
        return frame, None
    raise ValueError("Unsupported source: {}".format(source))


def main(
    source: str = "file",
    yolo_model: Optional[str] = None,
    yolo_device: Optional[str] = None,
) -> None:
    frame, workspace_mask_path = load_source(source)
    print("[0/4] RGB-D source: {}; shape={}".format(source, frame.depth.shape))

    predictor = None
    if yolo_model is not None:
        predictor = YOLOAppleDetector(yolo_model, device=yolo_device)
    apple_mask = AppleMaskDetector(predictor=predictor).detect(frame.color)
    if workspace_mask_path is not None and workspace_mask_path.is_file():
        apple_mask = apple_mask & load_mask(workspace_mask_path)
    valid_mask = build_valid_mask(frame.depth, apple_mask)

    full_cloud = create_point_cloud(
        depth=frame.depth,
        intrinsic=frame.intrinsic,
        depth_scale=frame.depth_scale,
        mask=valid_mask,
        color=frame.color,
    )
    model_cloud = sample_point_cloud(full_cloud, num_points=20000, seed=0)

    runner = GraspNetRunner(CHECKPOINT_PATH, baseline_dir=BASELINE_DIR)
    print("[1/4] Checkpoint loaded: {} (epoch={})".format(
        runner.checkpoint_path, runner.checkpoint_epoch
    ))
    print(
        "[2/4] CUDA available: {}; inference device: {}".format(
            torch.cuda.is_available(), runner.device
        )
    )
    model_device = next(runner.model.parameters()).device
    if model_device != runner.device:
        raise RuntimeError("Model device {} does not match runner device {}".format(
            model_device, runner.device
        ))
    if torch.cuda.is_available() and runner.device.type != "cuda":
        raise RuntimeError("CUDA is available but GraspNetRunner did not select it")

    grasp_group = runner.predict(model_cloud.points)
    if not isinstance(grasp_group, runner._grasp_group_type):
        raise TypeError("GraspNetRunner did not return a GraspGroup")
    print("[3/4] GraspGroup returned with {} candidates".format(len(grasp_group)))

    selector = GraspSelector(OUTPUT_PATH)
    best_grasp = selector.get_best_grasp(grasp_group)
    print("[4/4] Best grasp saved: {}".format(selector.output_path))
    print("position:\n{}".format(best_grasp["position"]))
    print("rotation:\n{}".format(best_grasp["rotation"]))
    print("score:\n{}".format(best_grasp["score"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test GraspNet with file or D435i input")
    parser.add_argument(
        "--source",
        choices=("file", "camera"),
        default="file",
        help="RGB-D input source",
    )
    parser.add_argument(
        "--yolo-model",
        default=None,
        help="Optional YOLOv8-seg checkpoint; omit to use the temporary mask heuristic",
    )
    parser.add_argument(
        "--yolo-device",
        default=None,
        help="Optional Ultralytics device, for example 0 or cpu",
    )
    arguments = parser.parse_args()
    main(arguments.source, arguments.yolo_model, arguments.yolo_device)
