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
from grasp.grasp_pose_filter import GraspPoseFilter, GraspPoseFilterStats
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
from perception.pointcloud_temporal_filter import (
    MaskTemporalStats,
    PointCloudTemporalStats,
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


def print_mask_temporal_stats(
    frame_index: int, stats: MaskTemporalStats
) -> None:
    print("Frame {}:".format(frame_index))
    print("raw mask pixels: {}".format(stats.raw_area))
    print("stable mask pixels: {}".format(stats.stable_area))
    print("mask IoU: {:.3f}".format(stats.iou))
    print("mask area ratio: {:.3f}".format(stats.area_ratio))
    print("mask current weight: {:.3f}".format(stats.current_weight))
    if stats.area_anomaly or stats.iou_anomaly:
        print(
            "mask anomaly: area={} iou={}".format(
                stats.area_anomaly, stats.iou_anomaly
            )
        )


def print_cloud_temporal_stats(stats: PointCloudTemporalStats) -> None:
    print("point cloud:")
    print("  raw: {}".format(stats.raw_points))
    print("  after temporal fusion: {}".format(stats.fused_points))
    print("  matched previous points: {}".format(stats.matched_points))
    print("  supplemented previous points: {}".format(stats.supplemented_points))
    if stats.icp_attempted:
        print(
            "  ICP: accepted={} fitness={:.3f} rmse={:.5f} m".format(
                stats.icp_accepted,
                stats.icp_fitness,
                stats.icp_inlier_rmse,
            )
        )
    if stats.history_retained:
        print("  previous history retained after rejected sparse ICP frame")
    if stats.warning_message:
        print("  WARNING: {}".format(stats.warning_message))


def print_grasp_comparison(
    raw_grasp: Dict[str, object],
    filtered_grasp: Dict[str, object],
    stats: GraspPoseFilterStats,
) -> None:
    raw_position = np.asarray(raw_grasp["position"])
    filtered_position = np.asarray(filtered_grasp["position"])
    print("grasp:")
    print("  raw position: {}".format(raw_position))
    print("  filtered position: {}".format(filtered_position))
    print("  raw rotation:\n{}".format(np.asarray(raw_grasp["rotation"])))
    print(
        "  filtered rotation:\n{}".format(
            np.asarray(filtered_grasp["rotation"])
        )
    )
    print("  score: {:.6f}".format(float(filtered_grasp["score"])))
    print(
        "  position step: raw={:.3f} cm filtered={:.3f} cm".format(
            stats.raw_position_delta_m * 100.0,
            stats.filtered_position_delta_m * 100.0,
        )
    )
    print(
        "  rotation step: raw={:.2f} deg filtered={:.2f} deg".format(
            stats.raw_rotation_delta_degrees,
            stats.filtered_rotation_delta_degrees,
        )
    )


def _point_count_variation(values) -> float:
    data = np.asarray(values, dtype=np.float64)
    if len(data) == 0:
        return 0.0
    return float((data.max() - data.min()) / max(data.mean(), 1.0))


def _mask_iou(current: np.ndarray, previous: Optional[np.ndarray]) -> float:
    if previous is None or previous.shape != current.shape:
        return 1.0
    union = int(np.count_nonzero(current | previous))
    if union == 0:
        return 1.0
    return float(np.count_nonzero(current & previous)) / float(union)


def _rotation_step_degrees(previous: np.ndarray, current: np.ndarray) -> float:
    relative = previous.T @ current
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def print_stability_evaluation(
    raw_points,
    stable_points,
    positions,
    rotations,
    target_window: int,
) -> None:
    sample_count = len(stable_points)
    print("{}-frame temporal stability evaluation:".format(sample_count))
    raw_variation = _point_count_variation(raw_points)
    stable_variation = _point_count_variation(stable_points)
    print("  raw point-count variation: {:.1f}%".format(raw_variation * 100.0))
    print(
        "  stable point-count variation: {:.1f}% ({})".format(
            stable_variation * 100.0,
            "PASS < 15%"
            if sample_count >= target_window and stable_variation < 0.15
            else "NOT PASSED/INCOMPLETE",
        )
    )

    if len(positions) >= 2:
        position_array = np.asarray(positions, dtype=np.float64)
        position_steps = np.linalg.norm(np.diff(position_array, axis=0), axis=1)
        maximum_position_step = float(position_steps.max())
        print(
            "  filtered position step: mean={:.2f} cm max={:.2f} cm ({})".format(
                float(position_steps.mean()) * 100.0,
                maximum_position_step * 100.0,
                "PASS < 5 cm"
                if len(positions) >= target_window
                and maximum_position_step < 0.05
                else "NOT PASSED/INCOMPLETE",
            )
        )
    if len(rotations) >= 2:
        rotation_steps = [
            _rotation_step_degrees(rotations[index - 1], rotations[index])
            for index in range(1, len(rotations))
        ]
        print(
            "  filtered rotation step: mean={:.2f} deg max={:.2f} deg".format(
                float(np.mean(rotation_steps)), float(np.max(rotation_steps))
            )
        )


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
    if args.stability_window < 2:
        raise ValueError("--stability-window must be at least 2")
    if args.temporal_target_min_points > args.temporal_target_max_points:
        raise ValueError(
            "--temporal-target-min-points must not exceed its maximum"
        )

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
    temporal_config = TemporalPointCloudConfig(
        mask_alpha=args.mask_alpha,
        area_ratio_min=args.mask_area_ratio_min,
        area_ratio_max=args.mask_area_ratio_max,
        area_anomaly_current_weight_scale=args.mask_area_anomaly_weight_scale,
        min_mask_iou=args.mask_min_iou,
        low_iou_current_weight_scale=args.mask_low_iou_weight_scale,
        point_beta=args.point_beta,
        low_point_threshold=args.temporal_low_point_threshold,
        target_points=args.num_points,
        target_min_points=min(args.temporal_target_min_points, args.num_points),
        target_max_points=max(args.temporal_target_max_points, args.num_points),
        enable_icp=not args.disable_temporal_icp,
        icp_voxel_size=args.temporal_icp_voxel_size,
        icp_max_correspondence_distance=(
            args.temporal_icp_max_correspondence_distance
        ),
        icp_min_fitness=args.temporal_icp_min_fitness,
        icp_max_inlier_rmse=args.temporal_icp_max_rmse,
        supplement_max_distance=args.temporal_supplement_max_distance,
        sampling_seed=args.sampling_seed,
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
    pose_filter = GraspPoseFilter(previous_weight=args.pose_previous_weight)
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
    print("Temporal filters: {}".format(temporal_config))
    depth_filter = (
        pointcloud_config.make_depth_filter()
        if pointcloud_config.enable_depth_preprocessing
        else None
    )

    captured_frames = 0
    successful_grasps = 0
    previous_unfiltered_mask = None
    raw_point_history = deque(maxlen=args.stability_window)
    stable_point_history = deque(maxlen=args.stability_window)
    filtered_position_history = deque(maxlen=args.stability_window)
    filtered_rotation_history = deque(maxlen=args.stability_window)
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

                raw_apple_mask = mask_detector.detect(rgb)
                if args.disable_temporal_filter:
                    raw_area = int(raw_apple_mask.sum())
                    previous_area = (
                        0
                        if previous_unfiltered_mask is None
                        else int(previous_unfiltered_mask.sum())
                    )
                    area_ratio = (
                        1.0
                        if previous_area == 0
                        else float(raw_area) / float(previous_area)
                    )
                    mask_stats = MaskTemporalStats(
                        raw_area=raw_area,
                        previous_area=previous_area,
                        stable_area=raw_area,
                        area_ratio=area_ratio,
                        iou=_mask_iou(
                            raw_apple_mask, previous_unfiltered_mask
                        ),
                    )
                    apple_mask = raw_apple_mask
                    previous_unfiltered_mask = raw_apple_mask.copy()
                else:
                    apple_mask = temporal_filter.filter_mask(raw_apple_mask)
                    mask_stats = temporal_filter.last_mask_stats
                    if mask_stats.reset_after_empty:
                        pose_filter.reset()

                print_mask_temporal_stats(captured_frames, mask_stats)
                if not np.any(apple_mask):
                    print("No stable apple mask available")
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
                    raw_apple_cloud = create_point_cloud(
                        depth=depth_mm,
                        intrinsic=intrinsic,
                        depth_scale=camera.depth_scale,
                        mask=apple_mask,
                        color=rgb,
                        config=pointcloud_config,
                        stats=cloud_stats,
                        depth_filter=depth_filter,
                    )
                    if args.disable_temporal_filter:
                        stable_apple_cloud = raw_apple_cloud
                        cloud_temporal_stats = PointCloudTemporalStats(
                            raw_points=len(raw_apple_cloud.points),
                            fused_points=len(raw_apple_cloud.points),
                        )
                    else:
                        stable_apple_cloud = temporal_filter.filter_point_cloud(
                            raw_apple_cloud
                        )
                        cloud_temporal_stats = (
                            temporal_filter.last_pointcloud_stats
                        )
                    model_cloud = sample_point_cloud(
                        stable_apple_cloud,
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
                print_cloud_temporal_stats(cloud_temporal_stats)
                raw_point_history.append(cloud_temporal_stats.raw_points)
                stable_point_history.append(cloud_temporal_stats.fused_points)

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
                        stable_apple_cloud,
                        None,
                        "No grasp candidates",
                    ):
                        break
                    continue

                raw_best_grasp = grasp_selector.get_best_grasp(grasp_group)
                if args.disable_pose_filter:
                    filtered_best_grasp = raw_best_grasp
                    pose_stats = GraspPoseFilterStats()
                else:
                    filtered_best_grasp = pose_filter.process(raw_best_grasp)
                    pose_stats = pose_filter.last_stats
                # GraspSelector initially persists the raw network result;
                # overwrite it with the pose actually exposed downstream.
                grasp_selector.save(filtered_best_grasp)
                filtered_position_history.append(
                    np.asarray(filtered_best_grasp["position"]).copy()
                )
                filtered_rotation_history.append(
                    np.asarray(filtered_best_grasp["rotation"]).copy()
                )
                successful_grasps += 1
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
                print_grasp_comparison(
                    raw_best_grasp, filtered_best_grasp, pose_stats
                )

                if successful_grasps % args.stability_window == 0:
                    print_stability_evaluation(
                        raw_point_history,
                        stable_point_history,
                        filtered_position_history,
                        filtered_rotation_history,
                        args.stability_window,
                    )

                status = "score={:.3f}  {:.0f} ms".format(
                    float(filtered_best_grasp["score"]), elapsed_ms
                )
                if not update_views(
                    visualizer,
                    not args.no_visualization,
                    rgb,
                    apple_mask,
                    stable_apple_cloud,
                    filtered_best_grasp,
                    status,
                ):
                    break
    except KeyboardInterrupt:
        print("Realtime grasp pipeline stopped by user")
    finally:
        if stable_point_history and (
            successful_grasps == 0
            or successful_grasps % args.stability_window != 0
        ):
            print_stability_evaluation(
                raw_point_history,
                stable_point_history,
                filtered_position_history,
                filtered_rotation_history,
                args.stability_window,
            )
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
        "--disable-temporal-filter",
        action="store_true",
        help="Disable mask/point-cloud temporal fusion for A/B comparison",
    )
    parser.add_argument("--mask-alpha", type=float, default=0.7)
    parser.add_argument("--mask-area-ratio-min", type=float, default=0.67)
    parser.add_argument("--mask-area-ratio-max", type=float, default=1.5)
    parser.add_argument(
        "--mask-area-anomaly-weight-scale", type=float, default=0.25
    )
    parser.add_argument("--mask-min-iou", type=float, default=0.3)
    parser.add_argument(
        "--mask-low-iou-weight-scale", type=float, default=0.5
    )
    parser.add_argument("--point-beta", type=float, default=0.5)
    parser.add_argument(
        "--temporal-low-point-threshold", type=int, default=15000
    )
    parser.add_argument(
        "--temporal-target-min-points", type=int, default=18000
    )
    parser.add_argument(
        "--temporal-target-max-points", type=int, default=22000
    )
    parser.add_argument(
        "--disable-temporal-icp",
        action="store_true",
        help="Use identity alignment instead of Open3D ICP",
    )
    parser.add_argument(
        "--temporal-icp-voxel-size", type=float, default=0.004
    )
    parser.add_argument(
        "--temporal-icp-max-correspondence-distance",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--temporal-icp-min-fitness", type=float, default=0.20
    )
    parser.add_argument(
        "--temporal-icp-max-rmse", type=float, default=0.02
    )
    parser.add_argument(
        "--temporal-supplement-max-distance", type=float, default=0.04
    )
    parser.add_argument(
        "--disable-pose-filter",
        action="store_true",
        help="Disable position EMA and rotation Slerp for A/B comparison",
    )
    parser.add_argument(
        "--pose-previous-weight",
        type=float,
        default=0.7,
        help="Previous-pose EMA/Slerp weight",
    )
    parser.add_argument(
        "--stability-window",
        type=int,
        default=100,
        help="Valid grasp frames used by the stability acceptance report",
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
