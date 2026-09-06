"""Realtime D435i + YOLOv8-seg + GraspNet inference pipeline."""

import argparse
from collections import deque
import time
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch

from camera.realsense_camera import RealSenseCamera
from grasp.grasp_selector import GraspSelector
from grasp.graspnet_runner import GraspNetRunner
from perception.apple_mask import AppleMaskDetector
from perception.mask_process import build_valid_mask
from perception.pointcloud import (
    InsufficientPointCloudError,
    PointCloudConfig,
    PointCloudStats,
    create_point_cloud,
    sample_point_cloud,
)
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
    cloud: Optional[Any],
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


def update_stability(real_point_history, real_points: int) -> Optional[float]:
    """Track peak-to-peak point-count variation over ten valid frames."""

    real_point_history.append(int(real_points))
    if len(real_point_history) < real_point_history.maxlen:
        return None
    counts = np.asarray(real_point_history, dtype=np.float64)
    variation = float((counts.max() - counts.min()) / max(counts.mean(), 1.0))
    print(
        "10-frame real-point stability: min={} max={} mean={:.1f} "
        "std={:.1f} variation={:.1f}% ({})".format(
            int(counts.min()),
            int(counts.max()),
            float(counts.mean()),
            float(counts.std()),
            variation * 100.0,
            "PASS" if variation < 0.30 else "WARNING >= 30%",
        )
    )
    return variation


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

    pointcloud_config = PointCloudConfig(
        median_kernel=args.depth_median_kernel,
        workspace_roi=(
            None if args.workspace_roi is None else tuple(args.workspace_roi)
        ),
        min_depth_m=args.min_depth,
        max_depth_m=args.max_depth,
        enable_depth_preprocessing=not args.disable_depth_preprocessing,
        enable_median_filter=not args.disable_median_filter,
        enable_spatial_smoothing=not args.disable_spatial_smoothing,
        spatial_diameter=args.spatial_diameter,
        spatial_sigma_color=args.spatial_sigma_color,
        spatial_sigma_space=args.spatial_sigma_space,
        enable_hole_filling=not args.disable_hole_filling,
        hole_fill_kernel=args.hole_fill_kernel,
        hole_fill_min_neighbors=args.hole_fill_min_neighbors,
        hole_fill_iterations=args.hole_fill_iterations,
        hole_fill_max_depth_delta_mm=args.hole_fill_max_depth_delta_mm,
        enable_outlier_filter=args.enable_outlier_filter,
        outlier_nb_neighbors=args.outlier_neighbors,
        outlier_std_ratio=args.outlier_std_ratio,
        enable_radius_outlier_filter=args.enable_radius_outlier_filter,
        radius_outlier_nb_points=args.radius_outlier_nb_points,
        radius_outlier_radius=args.radius_outlier_radius,
        enable_voxel_downsample=args.enable_voxel_downsample,
        voxel_size=args.voxel_size,
        enable_normal_estimation=args.enable_normal_estimation,
        normal_radius=args.normal_radius,
        normal_max_nn=args.normal_max_nn,
        min_real_points_warning=args.min_real_points_warning,
        min_real_points_reject=args.min_real_points_reject,
    )

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
    print("Point-cloud filters: {}".format(pointcloud_config))
    depth_filter = (
        pointcloud_config.make_depth_filter()
        if pointcloud_config.enable_depth_preprocessing
        else None
    )

    captured_frames = 0
    real_point_history = deque(maxlen=10)
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

                cloud_stats = PointCloudStats()
                pointcloud_start = time.perf_counter()
                try:
                    apple_cloud = create_point_cloud(
                        depth=depth_mm,
                        intrinsic=intrinsic,
                        depth_scale=camera.depth_scale,
                        mask=apple_mask,
                        color=rgb,
                        config=pointcloud_config,
                        stats=cloud_stats,
                        depth_filter=depth_filter,
                    )
                    model_cloud = sample_point_cloud(
                        apple_cloud,
                        num_points=args.num_points,
                        seed=args.sampling_seed,
                        stats=cloud_stats,
                        min_real_points_warning=(
                            pointcloud_config.min_real_points_warning
                        ),
                        min_real_points_reject=(
                            pointcloud_config.min_real_points_reject
                        ),
                    )
                except InsufficientPointCloudError as exc:
                    print("Frame {}: point cloud rejected: {}".format(
                        captured_frames, exc
                    ))
                    cloud_stats.print(frame_index=captured_frames)
                    if not update_views(
                        visualizer,
                        not args.no_visualization,
                        rgb,
                        apple_mask,
                        None,
                        None,
                        "Point cloud rejected",
                    ):
                        break
                    continue

                pointcloud_ms = (time.perf_counter() - pointcloud_start) * 1000.0
                cloud_stats.print(frame_index=captured_frames)
                update_stability(real_point_history, cloud_stats.real_points)

                if grasp_runner.device.type == "cuda":
                    torch.cuda.synchronize(grasp_runner.device)
                graspnet_start = time.perf_counter()
                grasp_group = grasp_runner.predict(model_cloud.points)
                if grasp_runner.device.type == "cuda":
                    torch.cuda.synchronize(grasp_runner.device)
                graspnet_ms = (time.perf_counter() - graspnet_start) * 1000.0
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
                fps = 1000.0 / max(elapsed_ms, 1e-6)
                print(
                    "Frame {}: grasps={} pointcloud={:.1f} ms "
                    "GraspNet={:.1f} ms total={:.1f} ms FPS={:.2f}".format(
                        captured_frames,
                        len(grasp_group),
                        pointcloud_ms,
                        graspnet_ms,
                        elapsed_ms,
                        fps,
                    )
                )
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
        "--workspace-roi",
        type=int,
        nargs=4,
        metavar=("X_MIN", "Y_MIN", "X_MAX", "Y_MAX"),
        default=None,
        help="Optional pixel ROI with exclusive maximum bounds",
    )
    parser.add_argument(
        "--min-depth",
        type=float,
        default=None,
        help="Optional minimum workspace depth in metres",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=None,
        help="Optional maximum workspace depth in metres",
    )
    parser.add_argument("--disable-depth-preprocessing", action="store_true")
    parser.add_argument("--disable-median-filter", action="store_true")
    parser.add_argument("--depth-median-kernel", type=int, default=3)
    parser.add_argument("--disable-spatial-smoothing", action="store_true")
    parser.add_argument("--spatial-diameter", type=int, default=5)
    parser.add_argument("--spatial-sigma-color", type=float, default=30.0)
    parser.add_argument("--spatial-sigma-space", type=float, default=3.0)
    parser.add_argument("--disable-hole-filling", action="store_true")
    parser.add_argument("--hole-fill-kernel", type=int, default=3)
    parser.add_argument("--hole-fill-min-neighbors", type=int, default=5)
    parser.add_argument("--hole-fill-iterations", type=int, default=1)
    parser.add_argument(
        "--hole-fill-max-depth-delta-mm",
        type=float,
        default=80.0,
        help="Maximum local depth span for conservative hole filling",
    )
    parser.add_argument(
        "--enable-outlier-filter",
        action="store_true",
        help="Enable statistical outlier removal (disabled by default)",
    )
    parser.add_argument("--outlier-neighbors", type=int, default=20)
    parser.add_argument("--outlier-std-ratio", type=float, default=2.0)
    parser.add_argument(
        "--enable-radius-outlier-filter",
        action="store_true",
        help="Enable radius outlier removal (disabled by default)",
    )
    parser.add_argument("--radius-outlier-nb-points", type=int, default=8)
    parser.add_argument("--radius-outlier-radius", type=float, default=0.01)
    parser.add_argument(
        "--enable-voxel-downsample",
        action="store_true",
        help="Enable voxel downsampling (disabled by default)",
    )
    parser.add_argument("--voxel-size", type=float, default=0.001)
    parser.add_argument(
        "--enable-normal-estimation",
        action="store_true",
        help="Estimate normals in the realtime cloud (disabled by default)",
    )
    parser.add_argument("--normal-radius", type=float, default=0.01)
    parser.add_argument("--normal-max-nn", type=int, default=30)
    parser.add_argument("--min-real-points-warning", type=int, default=5000)
    parser.add_argument("--min-real-points-reject", type=int, default=1000)
    parser.add_argument(
        "--sampling-seed",
        type=int,
        default=0,
        help="Fixed sampling seed for stable realtime input",
    )
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
