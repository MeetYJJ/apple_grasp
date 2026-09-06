"""Realtime D435i + YOLOv8-seg + GraspNet inference pipeline."""

import argparse
import time
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from camera.realsense_camera import RealSenseCamera
from grasp.grasp_selector import GraspSelector
from grasp.graspnet_runner import GraspNetRunner
from perception.apple_mask import AppleMaskDetector
from perception.mask_process import build_valid_mask
from perception.pointcloud import PointCloudData, create_point_cloud, sample_point_cloud
from perception.yolo_apple_detector import DEFAULT_YOLO_MODEL, YOLOAppleDetector
from visualization.grasp_visualizer import GraspVisualizer


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_BASELINE_DIR = WORKSPACE_ROOT / "graspnet-baseline"
DEFAULT_CHECKPOINT_PATH = DEFAULT_BASELINE_DIR / "checkpoint-rs.tar"


def update_views(
    visualizer: GraspVisualizer,
    enabled: bool,
    rgb: np.ndarray,
    apple_mask: np.ndarray,
    cloud: Optional[PointCloudData],
    best_grasp: Optional[Dict[str, object]],
    status: str,
) -> bool:
    """Update the RGB overlay and modular Open3D grasp scene."""

    rgb_window_alive = True
    if enabled:
        display_rgb = np.asarray(rgb, dtype=np.uint8).copy()
        if apple_mask.shape == display_rgb.shape[:2] and np.any(apple_mask):
            green = np.asarray([0, 255, 0], dtype=np.float32)
            selected = display_rgb[apple_mask].astype(np.float32)
            display_rgb[apple_mask] = np.clip(
                selected * 0.65 + green * 0.35, 0, 255
            ).astype(np.uint8)
        display_bgr = cv2.cvtColor(display_rgb, cv2.COLOR_RGB2BGR)
        cv2.putText(
            display_bgr,
            status,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("Realtime RGB and apple mask", display_bgr)
        key = cv2.waitKey(1) & 0xFF
        rgb_window_alive = key not in (ord("q"), 27)

    position = None if best_grasp is None else best_grasp["position"]
    rotation = None if best_grasp is None else best_grasp["rotation"]
    scene_window_alive = visualizer.update(cloud, position, rotation)
    return rgb_window_alive and scene_window_alive


def print_best_grasp(best_grasp: Dict[str, object]) -> None:
    """Print the selected camera-frame grasp pose."""

    position = np.asarray(best_grasp["position"])
    rotation = np.asarray(best_grasp["rotation"])
    print(
        "position: ({:.6f}, {:.6f}, {:.6f})".format(
            float(position[0]), float(position[1]), float(position[2])
        )
    )
    print("rotation:\n{}".format(rotation))
    print("score: {:.6f}".format(float(best_grasp["score"])))


def run(args: argparse.Namespace) -> None:
    if args.num_points < 2048:
        raise ValueError("--num-points must be at least 2048")
    if args.max_frames < 0:
        raise ValueError("--max-frames must be non-negative")
    if args.coordinate_size <= 0:
        raise ValueError("--coordinate-size must be positive")
    if args.camera_coordinate_size <= 0:
        raise ValueError("--camera-coordinate-size must be positive")
    if args.approach_length <= 0:
        raise ValueError("--approach-length must be positive")

    yolo_backend = YOLOAppleDetector(
        model_path=args.yolo_model,
        confidence=args.confidence,
        device=args.yolo_device,
    )
    mask_detector = AppleMaskDetector(predictor=yolo_backend)
    grasp_runner = GraspNetRunner(
        checkpoint_path=args.checkpoint_path,
        baseline_dir=args.baseline_dir,
    )
    grasp_selector = GraspSelector()
    visualizer = GraspVisualizer(
        enabled=not args.no_visualization,
        camera_frame_size=args.camera_coordinate_size,
        grasp_frame_size=args.coordinate_size,
        approach_length=args.approach_length,
    )

    print("YOLO model: {}".format(yolo_backend.model_path))
    print("YOLO apple class id(s): {}".format(yolo_backend.apple_class_ids))
    print("GraspNet checkpoint: {}".format(grasp_runner.checkpoint_path))
    print("GraspNet device: {}".format(grasp_runner.device))

    captured_frames = 0
    try:
        with RealSenseCamera(
            width=args.width,
            height=args.height,
            fps=args.fps,
            serial_number=args.serial,
        ) as camera:
            intrinsic = camera.intrinsic.copy()
            print("RealSense: {} (serial={})".format(
                camera.device_name, camera.serial_number
            ))
            print("Intrinsic:\n{}".format(intrinsic))
            print("Press q, Esc, or Ctrl+C to stop")

            while args.max_frames == 0 or captured_frames < args.max_frames:
                iteration_start = time.perf_counter()
                rgb, depth_mm = camera.get_frame()
                captured_frames += 1

                apple_mask = mask_detector.detect(rgb)
                mask_pixels = int(apple_mask.sum())
                if mask_pixels == 0:
                    print("Frame {}: no apple detected".format(captured_frames))
                    if not update_views(
                        visualizer,
                        not args.no_visualization,
                        rgb,
                        apple_mask,
                        None,
                        None,
                        "No apple detected",
                    ):
                        break
                    continue

                valid_mask = build_valid_mask(depth_mm, apple_mask)
                valid_depth_pixels = int(valid_mask.sum())
                if valid_depth_pixels == 0:
                    print("Frame {}: apple mask has no valid depth".format(
                        captured_frames
                    ))
                    if not update_views(
                        visualizer,
                        not args.no_visualization,
                        rgb,
                        apple_mask,
                        None,
                        None,
                        "Apple has no valid depth",
                    ):
                        break
                    continue

                apple_cloud = create_point_cloud(
                    depth=depth_mm,
                    intrinsic=intrinsic,
                    depth_scale=camera.depth_scale,
                    mask=valid_mask,
                    color=rgb,
                )
                model_cloud = sample_point_cloud(
                    apple_cloud, num_points=args.num_points
                )
                grasp_group = grasp_runner.predict(model_cloud.points)
                if len(grasp_group) == 0:
                    print("Frame {}: GraspNet returned no candidates".format(
                        captured_frames
                    ))
                    if not update_views(
                        visualizer,
                        not args.no_visualization,
                        rgb,
                        apple_mask,
                        apple_cloud,
                        None,
                        "No grasp candidates",
                    ):
                        break
                    continue

                best_grasp = grasp_selector.get_best_grasp(grasp_group)
                elapsed_ms = (time.perf_counter() - iteration_start) * 1000.0
                print("\nFrame {}: mask={} depth={} points={} grasps={} time={:.1f} ms".format(
                    captured_frames,
                    mask_pixels,
                    valid_depth_pixels,
                    len(apple_cloud.points),
                    len(grasp_group),
                    elapsed_ms,
                ))
                print_best_grasp(best_grasp)

                status = "score={:.3f}  {:.0f} ms".format(
                    float(best_grasp["score"]), elapsed_ms
                )
                if not update_views(
                    visualizer,
                    not args.no_visualization,
                    rgb,
                    apple_mask,
                    apple_cloud,
                    best_grasp,
                    status,
                ):
                    break
    except KeyboardInterrupt:
        print("Realtime grasp pipeline stopped by user")
    finally:
        if not args.no_visualization:
            cv2.destroyAllWindows()
        visualizer.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Realtime D435i + YOLOv8-seg + GraspNet inference"
    )
    parser.add_argument(
        "--yolo-model",
        default=DEFAULT_YOLO_MODEL,
        help=(
            "Local segmentation checkpoint or official Ultralytics model name "
            "(default: yolov8n-seg.pt)"
        ),
    )
    parser.add_argument(
        "--checkpoint-path",
        default=str(DEFAULT_CHECKPOINT_PATH),
        help="Path to checkpoint-rs.tar",
    )
    parser.add_argument(
        "--baseline-dir",
        default=str(DEFAULT_BASELINE_DIR),
        help="Path to graspnet-baseline",
    )
    parser.add_argument("--yolo-device", default=None, help="For example 0 or cpu")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--serial", default=None, help="Optional RealSense serial number")
    parser.add_argument("--num-points", type=int, default=20000)
    parser.add_argument(
        "--coordinate-size",
        type=float,
        default=0.06,
        help="Best-grasp coordinate-frame size in metres",
    )
    parser.add_argument(
        "--camera-coordinate-size",
        type=float,
        default=0.10,
        help="Camera coordinate-frame size in metres",
    )
    parser.add_argument(
        "--approach-length",
        type=float,
        default=0.10,
        help="Displayed grasp approach-arrow length in metres",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after N frames; 0 runs until the user exits",
    )
    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="Run inference without OpenCV/Open3D windows",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
