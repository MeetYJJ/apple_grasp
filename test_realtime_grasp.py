"""Realtime D435i + YOLOv8-seg + GraspNet inference pipeline."""

import argparse
import time
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch

from camera.realsense_camera import RealSenseCamera
from grasp.grasp_pose_filter import GraspPoseFilter
from grasp.grasp_selector import GraspSelector
from grasp.graspnet_runner import GraspNetRunner
from perception.apple_mask import AppleMaskDetector
from perception.depth_temporal_filter import (
    DepthTemporalFilter,
    DepthTemporalFilterConfig,
)
from perception.mask_process import build_valid_mask
from perception.pointcloud import (
    InsufficientPointCloudError,
    PointCloudConfig,
    PointCloudStats,
    calculate_valid_depth_ratio,
    create_point_cloud,
    sample_point_cloud,
)
from perception.pointcloud_temporal_filter import (
    TemporalPointCloudConfig,
    TemporalPointCloudFilter,
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


def print_frame_summary(
    frame_index: int,
    fps: float,
    mask_pixels: int,
    valid_depth_ratio: float,
    real_points: int,
    model_points: int,
    best_grasp: Dict[str, object],
    latency_ms: float,
) -> None:
    print("Frame: {}".format(frame_index))
    print("FPS: {:.2f}".format(fps))
    print("mask pixels: {}".format(mask_pixels))
    print("valid depth ratio: {:.1%}".format(valid_depth_ratio))
    print(
        "cloud points: {} real / {} GraspNet input".format(
            real_points, model_points
        )
    )
    print("grasp score: {:.6f}".format(float(best_grasp["score"])))
    print("position: {}".format(np.asarray(best_grasp["position"])))
    print("rotation:\n{}".format(np.asarray(best_grasp["rotation"])))
    print("latency: {:.1f} ms".format(latency_ms))


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
    if args.log_interval < 1:
        raise ValueError("--log-interval must be positive")
    if not 0.0 <= args.min_valid_depth_ratio <= 1.0:
        raise ValueError("--min-valid-depth-ratio must be in [0, 1]")

    pointcloud_config = PointCloudConfig(
        median_kernel=args.depth_median_kernel,
        workspace_roi=(
            None if args.workspace_roi is None else tuple(args.workspace_roi)
        ),
        min_depth_m=args.min_depth,
        max_depth_m=args.max_depth,
        enable_depth_preprocessing=args.enable_cpu_depth_preprocessing,
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
    temporal_config = TemporalPointCloudConfig(
        high_iou_threshold=args.mask_high_iou_threshold,
        low_iou_threshold=args.mask_low_iou_threshold,
        high_iou_alpha=args.mask_high_iou_alpha,
        medium_iou_alpha=args.mask_medium_iou_alpha,
        low_iou_alpha=args.mask_low_iou_alpha,
    )
    depth_temporal_config = DepthTemporalFilterConfig(
        previous_weight=args.depth_temporal_previous_weight,
        max_depth_delta_mm=args.depth_temporal_max_delta_mm,
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
    temporal_filter = TemporalPointCloudFilter(temporal_config)
    depth_temporal_filter = DepthTemporalFilter(depth_temporal_config)
    pose_filter = GraspPoseFilter(
        previous_weight=args.pose_previous_weight,
        position_window_size=args.pose_position_window,
    )
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
    depth_filter = (
        pointcloud_config.make_depth_filter()
        if pointcloud_config.enable_depth_preprocessing
        else None
    )

    captured_frames = 0
    last_valid_cloud = None
    last_valid_grasp = None
    try:
        with RealSenseCamera(
            width=args.width,
            height=args.height,
            fps=args.fps,
            serial_number=args.serial,
            enable_depth_postprocessing=(
                not args.disable_realsense_depth_filters
            ),
            decimation_magnitude=args.decimation_magnitude,
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

                raw_apple_mask = mask_detector.detect(rgb)
                if args.disable_temporal_filter:
                    apple_mask = raw_apple_mask
                else:
                    apple_mask = temporal_filter.filter_mask(raw_apple_mask)

                if not np.any(apple_mask):
                    depth_temporal_filter.reset()
                    pose_filter.reset()
                    if captured_frames % args.log_interval == 0:
                        print("Frame {}: no apple mask".format(captured_frames))
                    if not update_views(
                        visualizer,
                        not args.no_visualization,
                        rgb,
                        apple_mask,
                        None,
                        None,
                        "No stable apple mask",
                    ):
                        break
                    continue

                if (
                    args.disable_temporal_filter
                    or args.disable_depth_temporal_filter
                ):
                    filtered_depth_mm = depth_mm
                else:
                    filtered_depth_mm = depth_temporal_filter.process(
                        depth_mm, apple_mask
                    )

                mask_pixels, valid_depth_pixels, valid_ratio = (
                    calculate_valid_depth_ratio(
                        filtered_depth_mm,
                        apple_mask,
                        min_depth_mm=args.min_depth * 1000.0,
                        max_depth_mm=args.max_depth * 1000.0,
                    )
                )
                if valid_ratio < args.min_valid_depth_ratio:
                    if captured_frames % args.log_interval == 0:
                        print(
                            "Frame {}: depth invalid ratio {:.1%} "
                            "({}/{}) - holding last valid grasp".format(
                                captured_frames,
                                valid_ratio,
                                valid_depth_pixels,
                                mask_pixels,
                            )
                        )
                    if not update_views(
                        visualizer,
                        not args.no_visualization,
                        rgb,
                        apple_mask,
                        last_valid_cloud,
                        last_valid_grasp,
                        "Low depth ratio - holding last grasp",
                    ):
                        break
                    continue

                valid_mask = build_valid_mask(filtered_depth_mm, apple_mask)
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
                try:
                    raw_apple_cloud = create_point_cloud(
                        depth=filtered_depth_mm,
                        intrinsic=intrinsic,
                        depth_scale=camera.depth_scale,
                        mask=apple_mask,
                        color=rgb,
                        config=pointcloud_config,
                        stats=cloud_stats,
                        depth_filter=depth_filter,
                    )
                    model_cloud = sample_point_cloud(
                        raw_apple_cloud,
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
                except InsufficientPointCloudError:
                    print(
                        "Frame {}: cloud too sparse ({} real points)".format(
                            captured_frames,
                            cloud_stats.real_points,
                        )
                    )
                    if not update_views(
                        visualizer,
                        not args.no_visualization,
                        rgb,
                        apple_mask,
                        last_valid_cloud,
                        last_valid_grasp,
                        "Cloud too sparse - holding last grasp",
                    ):
                        break
                    continue

                if grasp_runner.device.type == "cuda":
                    torch.cuda.synchronize(grasp_runner.device)
                grasp_group = grasp_runner.predict(model_cloud.points)
                if grasp_runner.device.type == "cuda":
                    torch.cuda.synchronize(grasp_runner.device)
                if len(grasp_group) == 0:
                    print("Frame {}: GraspNet returned no candidates".format(
                        captured_frames
                    ))
                    if not update_views(
                        visualizer,
                        not args.no_visualization,
                        rgb,
                        apple_mask,
                        raw_apple_cloud,
                        None,
                        "No grasp candidates",
                    ):
                        break
                    continue

                raw_best_grasp = grasp_selector.get_best_grasp(grasp_group)
                if args.disable_pose_filter:
                    filtered_best_grasp = raw_best_grasp
                else:
                    filtered_best_grasp = pose_filter.process(raw_best_grasp)
                # GraspSelector initially persists the raw network result;
                # overwrite it with the pose actually exposed downstream.
                grasp_selector.save(filtered_best_grasp)
                last_valid_cloud = raw_apple_cloud
                last_valid_grasp = filtered_best_grasp
                elapsed_ms = (time.perf_counter() - iteration_start) * 1000.0
                fps = 1000.0 / max(elapsed_ms, 1e-6)
                if captured_frames % args.log_interval == 0:
                    print_frame_summary(
                        captured_frames,
                        fps,
                        int(apple_mask.sum()),
                        valid_ratio,
                        cloud_stats.real_points,
                        len(model_cloud.points),
                        filtered_best_grasp,
                        elapsed_ms,
                    )

                status = "score={:.3f}  {:.0f} ms".format(
                    float(filtered_best_grasp["score"]), elapsed_ms
                )
                if not update_views(
                    visualizer,
                    not args.no_visualization,
                    rgb,
                    apple_mask,
                    raw_apple_cloud,
                    filtered_best_grasp,
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
    parser.add_argument(
        "--serial", default=None, help="Optional RealSense serial number"
    )
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
        default=0.25,
        help="Minimum valid workspace depth in metres (default: 0.25)",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=2.0,
        help="Maximum valid workspace depth in metres (default: 2.0)",
    )
    parser.add_argument(
        "--min-valid-depth-ratio",
        type=float,
        default=0.20,
        help="Skip inference below this valid-depth/mask ratio",
    )
    parser.add_argument(
        "--disable-realsense-depth-filters",
        action="store_true",
        help="Disable SDK spatial/temporal/hole-filling filters",
    )
    parser.add_argument(
        "--decimation-magnitude",
        type=int,
        default=1,
        choices=range(1, 9),
        help="RealSense decimation factor; 1 preserves all depth pixels",
    )
    cpu_depth_group = parser.add_mutually_exclusive_group()
    cpu_depth_group.add_argument(
        "--enable-depth-preprocessing",
        dest="enable_cpu_depth_preprocessing",
        action="store_true",
        help="Enable additional CPU depth filtering after RealSense filters",
    )
    cpu_depth_group.add_argument(
        "--disable-depth-preprocessing",
        dest="enable_cpu_depth_preprocessing",
        action="store_false",
        help="Compatibility option; CPU depth filtering is disabled by default",
    )
    parser.set_defaults(enable_cpu_depth_preprocessing=False)
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
    parser.add_argument("--min-real-points-warning", type=int, default=8000)
    parser.add_argument("--min-real-points-reject", type=int, default=2048)
    parser.add_argument(
        "--sampling-seed",
        type=int,
        default=0,
        help="Fallback seed when Open3D FPS is unavailable",
    )
    parser.add_argument(
        "--disable-temporal-filter",
        action="store_true",
        help="Disable adaptive mask and depth temporal filters",
    )
    parser.add_argument("--mask-high-iou-threshold", type=float, default=0.7)
    parser.add_argument("--mask-low-iou-threshold", type=float, default=0.3)
    parser.add_argument("--mask-high-iou-alpha", type=float, default=0.5)
    parser.add_argument("--mask-medium-iou-alpha", type=float, default=0.3)
    parser.add_argument("--mask-low-iou-alpha", type=float, default=0.1)
    parser.add_argument(
        "--disable-depth-temporal-filter", action="store_true"
    )
    parser.add_argument(
        "--depth-temporal-previous-weight", type=float, default=0.6
    )
    parser.add_argument(
        "--depth-temporal-max-delta-mm", type=float, default=80.0
    )
    parser.add_argument(
        "--disable-pose-filter",
        action="store_true",
        help="Disable position moving average and rotation Slerp",
    )
    parser.add_argument(
        "--pose-previous-weight",
        type=float,
        default=0.7,
        help="Previous-rotation weight used by incremental Slerp",
    )
    parser.add_argument(
        "--pose-position-window",
        type=int,
        default=5,
        help="Moving-average window for valid grasp positions",
    )
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument(
        "--coordinate-size",
        type=float,
        default=0.18,
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
        default=0.20,
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
