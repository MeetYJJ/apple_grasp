"""Inspect and benchmark the apple point-cloud pipeline.

The test can use the official offline RGB-D example or collect a sequence from
an aligned D435i stream. GraspNet benchmarking is optional so point-cloud
diagnostics can run without the CUDA extensions/checkpoint.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import warnings

import cv2
import numpy as np

from camera.realsense_camera import RealSenseCamera
from perception.apple_mask import AppleMaskDetector
from perception.mask_process import load_mask
from perception.pointcloud import (
    InsufficientPointCloudError,
    PointCloudConfig,
    PointCloudData,
    PointCloudStats,
    create_point_cloud,
    sample_point_cloud,
)
from perception.rgbd_loader import RGBDFrame, load_rgbd
from perception.yolo_apple_detector import DEFAULT_YOLO_MODEL, YOLOAppleDetector


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_DATA_DIR = WORKSPACE_ROOT / "graspnet-baseline" / "doc" / "example_data"
DEFAULT_BASELINE_DIR = WORKSPACE_ROOT / "graspnet-baseline"
DEFAULT_CHECKPOINT_PATH = DEFAULT_BASELINE_DIR / "checkpoint-rs.tar"
VOXEL_SWEEP_SIZES = (0.001, 0.002, 0.003, 0.005)


@dataclass
class QualityResult:
    raw_cloud: Any
    filtered_cloud: Any
    model_input: PointCloudData
    stats: PointCloudStats
    pointcloud_ms: float


class QualityVisualizer:
    """Maintain one OpenCV view and three live Open3D views."""

    def __init__(self, raw_cloud: Any, filtered_cloud: Any) -> None:
        try:
            import open3d as o3d
        except ImportError as exc:
            raise ImportError(
                "Open3D is required for visualization; use --no-display "
                "for a headless benchmark"
            ) from exc

        self._o3d = o3d
        self._closed = False
        self._raw_geometry = self._copy_cloud(raw_cloud)
        self._filtered_geometry = self._copy_cloud(filtered_cloud)
        self._normal_geometry = self._copy_cloud(filtered_cloud)
        self._windows = []

        specifications = (
            ("2 - Raw masked point cloud", self._raw_geometry, False, 0),
            ("3 - Filtered point cloud", self._filtered_geometry, False, 540),
            ("4 - Point-cloud normals", self._normal_geometry, True, 1080),
        )
        try:
            for title, geometry, show_normals, left in specifications:
                window = o3d.visualization.Visualizer()
                created = window.create_window(
                    window_name=title,
                    width=520,
                    height=390,
                    left=left,
                    top=40,
                )
                if created is False:
                    window.destroy_window()
                    raise RuntimeError(
                        "Could not create Open3D window: {}".format(title)
                    )
                self._windows.append((window, geometry))
                window.add_geometry(geometry, reset_bounding_box=True)
                render = window.get_render_option()
                render.background_color = np.asarray([0.04, 0.04, 0.04])
                render.point_size = 2.0
                if show_normals and hasattr(render, "point_show_normal"):
                    render.point_show_normal = True
        except Exception:
            self.close()
            raise

        try:
            cv2.namedWindow("1 - RGB + apple mask", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("1 - RGB + apple mask", 640, 480)
            cv2.moveWindow("1 - RGB + apple mask", 0, 470)
        except Exception:
            self.close()
            raise

    def _copy_cloud(self, source: Any) -> Any:
        target = self._o3d.geometry.PointCloud()
        self._assign_cloud(target, source)
        return target

    def _assign_cloud(self, target: Any, source: Any) -> None:
        o3d = self._o3d
        target.points = o3d.utility.Vector3dVector(
            np.asarray(source.points, dtype=np.float64).copy()
        )
        if source.has_colors():
            target.colors = o3d.utility.Vector3dVector(
                np.asarray(source.colors, dtype=np.float64).copy()
            )
        else:
            target.colors = o3d.utility.Vector3dVector(
                np.empty((0, 3), dtype=np.float64)
            )
        if source.has_normals():
            target.normals = o3d.utility.Vector3dVector(
                np.asarray(source.normals, dtype=np.float64).copy()
            )
        else:
            target.normals = o3d.utility.Vector3dVector(
                np.empty((0, 3), dtype=np.float64)
            )

    @staticmethod
    def _overlay(rgb: np.ndarray, mask: np.ndarray, text_value: str) -> np.ndarray:
        shown = np.asarray(rgb, dtype=np.uint8).copy()
        if np.any(mask):
            green = np.asarray([0, 255, 0], dtype=np.float32)
            shown[mask] = np.clip(
                shown[mask].astype(np.float32) * 0.65 + green * 0.35,
                0,
                255,
            ).astype(np.uint8)
        shown = cv2.cvtColor(shown, cv2.COLOR_RGB2BGR)
        cv2.putText(
            shown,
            text_value,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return shown

    def update(
        self,
        rgb: np.ndarray,
        mask: np.ndarray,
        raw_cloud: Any,
        filtered_cloud: Any,
        status: str,
    ) -> bool:
        if self._closed:
            return False
        cv2.imshow("1 - RGB + apple mask", self._overlay(rgb, mask, status))
        key = cv2.waitKey(1) & 0xFF

        self._assign_cloud(self._raw_geometry, raw_cloud)
        self._assign_cloud(self._filtered_geometry, filtered_cloud)
        self._assign_cloud(self._normal_geometry, filtered_cloud)
        alive = key not in (ord("q"), 27)
        for window, geometry in self._windows:
            window.update_geometry(geometry)
            alive = bool(window.poll_events()) and alive
            window.update_renderer()
        return alive

    def wait(self, rgb, mask, raw_cloud, filtered_cloud) -> None:
        while self.update(
            rgb,
            mask,
            raw_cloud,
            filtered_cloud,
            "Press q or Esc to close",
        ):
            time.sleep(0.01)

    def close(self) -> None:
        if self._closed:
            return
        for window, _ in getattr(self, "_windows", []):
            window.destroy_window()
        cv2.destroyAllWindows()
        self._closed = True


def show_rgb_only(rgb: np.ndarray, mask: np.ndarray, status: str) -> bool:
    """Keep the RGB/mask view responsive before a valid 3-D cloud exists."""

    cv2.namedWindow("1 - RGB + apple mask", cv2.WINDOW_NORMAL)
    cv2.imshow(
        "1 - RGB + apple mask", QualityVisualizer._overlay(rgb, mask, status)
    )
    return (cv2.waitKey(1) & 0xFF) not in (ord("q"), 27)


def make_config(args: argparse.Namespace) -> PointCloudConfig:
    return PointCloudConfig(
        workspace_roi=(
            None if args.workspace_roi is None else tuple(args.workspace_roi)
        ),
        min_depth_m=args.min_depth,
        max_depth_m=args.max_depth,
        enable_depth_preprocessing=not args.disable_depth_preprocessing,
        enable_median_filter=not args.disable_median_filter,
        median_kernel=args.depth_median_kernel,
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
        enable_normal_estimation=(
            not args.disable_normal_estimation and not args.no_display
        ),
        normal_radius=args.normal_radius,
        normal_max_nn=args.normal_max_nn,
        min_real_points_warning=args.min_real_points_warning,
        min_real_points_reject=args.min_real_points_reject,
    )


def process_frame(
    frame: RGBDFrame,
    mask: np.ndarray,
    config: PointCloudConfig,
    depth_filter: Optional[Any],
    num_points: int,
    sampling_seed: int,
) -> QualityResult:
    """Build raw/filtered views and the fixed-size GraspNet input."""

    stats = PointCloudStats()
    raw_config = replace(
        config,
        workspace_roi=None,
        min_depth_m=None,
        max_depth_m=None,
        enable_depth_preprocessing=False,
        enable_outlier_filter=False,
        enable_radius_outlier_filter=False,
        enable_voxel_downsample=False,
        enable_normal_estimation=False,
    )
    try:
        started = time.perf_counter()
        filtered_cloud = create_point_cloud(
            depth=frame.depth,
            intrinsic=frame.intrinsic,
            depth_scale=frame.depth_scale,
            mask=mask,
            color=frame.color,
            config=config,
            stats=stats,
            depth_filter=depth_filter,
        )
        model_input = sample_point_cloud(
            filtered_cloud,
            num_points=num_points,
            seed=sampling_seed,
            stats=stats,
            min_real_points_warning=config.min_real_points_warning,
            min_real_points_reject=config.min_real_points_reject,
        )
        pointcloud_ms = (time.perf_counter() - started) * 1000.0

        # The raw visualization is diagnostic-only and deliberately excluded
        # from production point-cloud timing.
        raw_cloud = create_point_cloud(
            depth=frame.depth,
            intrinsic=frame.intrinsic,
            depth_scale=frame.depth_scale,
            mask=mask,
            color=frame.color,
            config=raw_config,
        )
    except InsufficientPointCloudError as exc:
        # Preserve the diagnostic trace for callers that reject this frame.
        exc.pointcloud_stats = stats
        raise
    return QualityResult(
        raw_cloud=raw_cloud,
        filtered_cloud=filtered_cloud,
        model_input=model_input,
        stats=stats,
        pointcloud_ms=pointcloud_ms,
    )


def synchronize_runner(runner: Any) -> None:
    if runner is None or runner.device.type != "cuda":
        return
    import torch

    torch.cuda.synchronize(runner.device)


def benchmark_graspnet(runner: Any, points: np.ndarray) -> Tuple[Any, float]:
    synchronize_runner(runner)
    started = time.perf_counter()
    grasp_group = runner.predict(points)
    synchronize_runner(runner)
    return grasp_group, (time.perf_counter() - started) * 1000.0


def print_best_grasp(grasp_group: Any, selector: Any) -> None:
    if len(grasp_group) == 0:
        print("GraspNet returned no grasp candidates")
        return
    best = selector.get_best_grasp(grasp_group)
    print("position:\n{}".format(np.asarray(best["position"])))
    print("rotation:\n{}".format(np.asarray(best["rotation"])))
    print("score: {:.6f}".format(float(best["score"])))


def print_frame_result(
    frame_index: int,
    result: QualityResult,
    graspnet_ms: Optional[float],
    iteration_ms: float,
) -> None:
    result.stats.print(frame_index=frame_index)
    compute_ms = result.pointcloud_ms + (graspnet_ms or 0.0)
    print("point-cloud time: {:.2f} ms".format(result.pointcloud_ms))
    print(
        "GraspNet time: {}".format(
            "not measured"
            if graspnet_ms is None
            else "{:.2f} ms".format(graspnet_ms)
        )
    )
    print("compute FPS: {:.2f}".format(1000.0 / max(compute_ms, 1e-6)))
    print("end-to-end FPS: {:.2f}".format(1000.0 / max(iteration_ms, 1e-6)))


def print_stability(
    counts: Sequence[int], attempts: int, missed_masks: int, rejected: int
) -> None:
    print("\n10-frame point-count acceptance summary:")
    print("attempted frames: {}".format(attempts))
    print("valid point-cloud frames: {}".format(len(counts)))
    print("YOLO empty-mask frames: {}".format(missed_masks))
    print("rejected sparse frames: {}".format(rejected))
    if not counts:
        print("point-count stability: unavailable (no valid frames)")
        return
    values = np.asarray(counts, dtype=np.float64)
    mean = float(values.mean())
    variation = float((values.max() - values.min()) / max(mean, 1.0))
    cv = float(values.std() / max(mean, 1.0))
    print(
        "real points: min={} max={} mean={:.1f} std={:.1f} CV={:.1f}%".format(
            int(values.min()),
            int(values.max()),
            mean,
            float(values.std()),
            cv * 100.0,
        )
    )
    if len(counts) < 10:
        print(
            "peak-to-peak variation: {:.1f}% (NOT EVALUATED: need 10 valid frames)".format(
                variation * 100.0
            )
        )
    else:
        print(
            "peak-to-peak variation: {:.1f}% ({})".format(
                variation * 100.0,
                "PASS < 30%" if variation < 0.30 else "FAIL >= 30%",
            )
        )


def run_voxel_sweep(
    frame: RGBDFrame,
    mask: np.ndarray,
    base_config: PointCloudConfig,
    depth_filter: Optional[Any],
    args: argparse.Namespace,
    runner: Optional[Any],
) -> None:
    """Benchmark all requested voxel sizes on one frozen RGB-D frame."""

    print("\nVoxel sweep on one frozen frame:")
    print(
        "voxel(m)  real points  pointcloud median(ms)  "
        "GraspNet median(ms)  compute FPS"
    )
    results: List[Dict[str, float]] = []
    for voxel_size in args.voxel_sizes:
        config = replace(
            base_config,
            enable_voxel_downsample=True,
            voxel_size=float(voxel_size),
            enable_normal_estimation=False,
        )
        stats = PointCloudStats()
        try:
            # Warm up Open3D allocation/KD-tree paths without contaminating the
            # timing sample. Sparse-cloud warnings remain enabled for measured
            # repetitions below.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                _run_voxel_case(
                    frame, mask, config, depth_filter, args
                )

            pointcloud_samples = []
            graspnet_samples = []
            for _ in range(args.benchmark_repeats):
                stats, model_input, pointcloud_ms = _run_voxel_case(
                    frame, mask, config, depth_filter, args
                )
                pointcloud_samples.append(pointcloud_ms)
                if runner is not None:
                    _, graspnet_ms = benchmark_graspnet(
                        runner, model_input.points
                    )
                    graspnet_samples.append(graspnet_ms)

            pointcloud_ms = float(np.median(pointcloud_samples))
            graspnet_ms = (
                None
                if not graspnet_samples
                else float(np.median(graspnet_samples))
            )
            compute_ms = pointcloud_ms + (graspnet_ms or 0.0)
            compute_fps = 1000.0 / max(compute_ms, 1e-6)
            results.append(
                {
                    "voxel": float(voxel_size),
                    "points": float(stats.real_points),
                    "fps": compute_fps,
                }
            )
            grasp_text = (
                "n/a" if graspnet_ms is None else "{:.2f}".format(graspnet_ms)
            )
            print(
                "{:<9.3f} {:<12d} {:<22.2f} {:<20s} {:.2f}".format(
                    float(voxel_size),
                    stats.real_points,
                    pointcloud_ms,
                    grasp_text,
                    compute_fps,
                )
            )
        except InsufficientPointCloudError as exc:
            print(
                "{:<9.3f} {:<12d} {:<22s} {:<20s} rejected ({})".format(
                    float(voxel_size),
                    getattr(exc, "real_points", 0)
                    or stats.real_points
                    or stats.final_points,
                    "n/a",
                    "n/a",
                    exc,
                )
            )

    eligible = [item for item in results if item["points"] > 10000]
    print("Recommended production default: voxel downsampling disabled")
    if eligible:
        candidate = max(eligible, key=lambda item: item["fps"])
        print(
            "Fastest tested candidate retaining >10000 points: {:.3f} m".format(
                candidate["voxel"]
            )
        )
    else:
        print("No tested voxel size retained >10000 real points on this frame")
    if runner is None:
        print("GraspNet time is n/a; add --benchmark-graspnet for CUDA timing")


def _run_voxel_case(
    frame: RGBDFrame,
    mask: np.ndarray,
    config: PointCloudConfig,
    depth_filter: Optional[Any],
    args: argparse.Namespace,
) -> Tuple[PointCloudStats, PointCloudData, float]:
    """Build one measured voxel configuration on the frozen input frame."""

    stats = PointCloudStats()
    started = time.perf_counter()
    try:
        cloud = create_point_cloud(
            depth=frame.depth,
            intrinsic=frame.intrinsic,
            depth_scale=frame.depth_scale,
            mask=mask,
            color=frame.color,
            config=config,
            stats=stats,
            depth_filter=depth_filter,
        )
        model_input = sample_point_cloud(
            cloud,
            num_points=args.num_points,
            seed=args.sampling_seed,
            stats=stats,
            min_real_points_warning=config.min_real_points_warning,
            min_real_points_reject=config.min_real_points_reject,
        )
    except InsufficientPointCloudError as exc:
        exc.real_points = stats.real_points or stats.final_points
        raise
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return stats, model_input, elapsed_ms


def initialize_graspnet(
    args: argparse.Namespace,
) -> Tuple[Optional[Any], Optional[Any]]:
    if not args.benchmark_graspnet:
        return None, None
    from grasp.grasp_selector import GraspSelector
    from grasp.graspnet_runner import GraspNetRunner

    runner = GraspNetRunner(
        checkpoint_path=args.checkpoint_path,
        baseline_dir=args.baseline_dir,
    )
    print("GraspNet device: {}".format(runner.device))
    return runner, GraspSelector()


def run_file(
    args: argparse.Namespace,
    config: PointCloudConfig,
    runner: Optional[Any],
    selector: Optional[Any],
) -> None:
    frame = load_rgbd(args.data_dir)
    mask_path = (
        Path(args.mask_path)
        if args.mask_path
        else Path(args.data_dir) / "workspace_mask.png"
    )
    mask = load_mask(mask_path)
    depth_filter = (
        config.make_depth_filter() if config.enable_depth_preprocessing else None
    )
    try:
        result = process_frame(
            frame, mask, config, depth_filter, args.num_points, args.sampling_seed
        )
    except InsufficientPointCloudError as exc:
        print("Offline point cloud rejected: {}".format(exc))
        rejected_stats = getattr(exc, "pointcloud_stats", None)
        if rejected_stats is not None:
            rejected_stats.print(frame_index=1)
        return

    graspnet_ms = None
    if runner is not None:
        # One unmeasured warm-up keeps CUDA initialization out of the report.
        benchmark_graspnet(runner, result.model_input.points)
        grasp_group, graspnet_ms = benchmark_graspnet(
            runner, result.model_input.points
        )
        print_best_grasp(grasp_group, selector)
    # For an offline frame, report only the measured processing stages; file
    # I/O, model construction, and the unmeasured CUDA warm-up are excluded.
    iteration_ms = result.pointcloud_ms + (graspnet_ms or 0.0)
    print("Source: file ({})".format(Path(args.data_dir).resolve()))
    print("Raw masked point count: {}".format(len(result.raw_cloud.points)))
    print_frame_result(1, result, graspnet_ms, iteration_ms)

    if args.voxel_sweep:
        run_voxel_sweep(frame, mask, config, depth_filter, args, runner)

    if not args.no_display:
        visualizer = QualityVisualizer(result.raw_cloud, result.filtered_cloud)
        try:
            visualizer.wait(
                frame.color,
                mask,
                result.raw_cloud,
                result.filtered_cloud,
            )
        finally:
            visualizer.close()


def run_camera(
    args: argparse.Namespace,
    config: PointCloudConfig,
    runner: Optional[Any],
    selector: Optional[Any],
) -> None:
    yolo = YOLOAppleDetector(
        model_path=args.yolo_model,
        confidence=args.confidence,
        device=args.yolo_device,
    )
    detector = AppleMaskDetector(predictor=yolo)
    depth_filter = (
        config.make_depth_filter() if config.enable_depth_preprocessing else None
    )
    point_counts = deque(maxlen=args.frames)
    missed_masks = 0
    rejected = 0
    attempts = 0
    visualizer = None
    last_frame = None
    last_mask = None
    warmed_up = False
    try:
        with RealSenseCamera(
            width=args.width,
            height=args.height,
            fps=args.camera_fps,
            serial_number=args.serial,
        ) as camera:
            print(
                "RealSense: {} (serial={})".format(
                    camera.device_name, camera.serial_number
                )
            )
            while len(point_counts) < args.frames and attempts < args.max_attempts:
                iteration_started = time.perf_counter()
                rgb, depth = camera.get_frame()
                attempts += 1
                frame = RGBDFrame(
                    color=rgb,
                    depth=depth,
                    intrinsic=camera.intrinsic.copy(),
                    depth_scale=camera.depth_scale,
                )
                mask = detector.detect(rgb)
                if not np.any(mask):
                    missed_masks += 1
                    print("Frame {}: no apple detected".format(attempts))
                    if not args.no_display and not show_rgb_only(
                        rgb, mask, "No apple detected"
                    ):
                        break
                    continue
                try:
                    result = process_frame(
                        frame,
                        mask,
                        config,
                        depth_filter,
                        args.num_points,
                        args.sampling_seed,
                    )
                except InsufficientPointCloudError as exc:
                    rejected += 1
                    print("Frame {} rejected: {}".format(attempts, exc))
                    rejected_stats = getattr(exc, "pointcloud_stats", None)
                    if rejected_stats is not None:
                        rejected_stats.print(frame_index=attempts)
                    if not args.no_display and not show_rgb_only(
                        rgb, mask, "Sparse point cloud rejected"
                    ):
                        break
                    continue

                graspnet_ms = None
                if runner is not None:
                    if not warmed_up:
                        benchmark_graspnet(runner, result.model_input.points)
                        warmed_up = True
                    grasp_group, graspnet_ms = benchmark_graspnet(
                        runner, result.model_input.points
                    )
                    print_best_grasp(grasp_group, selector)

                point_counts.append(result.stats.real_points)
                last_frame, last_mask = frame, mask

                keep_running = True
                if not args.no_display:
                    if visualizer is None:
                        visualizer = QualityVisualizer(
                            result.raw_cloud, result.filtered_cloud
                        )
                    keep_running = visualizer.update(
                        rgb,
                        mask,
                        result.raw_cloud,
                        result.filtered_cloud,
                        "real={} sampled={}".format(
                            result.stats.real_points, result.stats.sampled_points
                        ),
                    )
                iteration_ms = (time.perf_counter() - iteration_started) * 1000.0
                print_frame_result(attempts, result, graspnet_ms, iteration_ms)
                if not keep_running:
                    break

            print_stability(point_counts, attempts, missed_masks, rejected)
            if args.voxel_sweep and last_frame is not None:
                run_voxel_sweep(
                    last_frame,
                    last_mask,
                    config,
                    depth_filter,
                    args,
                    runner,
                )
    finally:
        if visualizer is not None:
            visualizer.close()
        elif not args.no_display:
            cv2.destroyAllWindows()


def run(args: argparse.Namespace) -> None:
    if args.frames < 1:
        raise ValueError("--frames must be positive")
    if args.max_attempts < args.frames:
        raise ValueError("--max-attempts must be at least --frames")
    if args.num_points < 2048:
        raise ValueError("--num-points must be at least 2048")
    if args.benchmark_repeats < 1:
        raise ValueError("--benchmark-repeats must be positive")
    config = make_config(args)
    print("Point-cloud config: {}".format(config))
    runner, selector = initialize_graspnet(args)
    if args.source == "file":
        run_file(args, config, runner, selector)
    else:
        run_camera(args, config, runner, selector)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize and benchmark raw/filtered apple point clouds"
    )
    parser.add_argument("--source", choices=("file", "camera"), default="file")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument(
        "--mask-path",
        default=None,
        help="File mode mask; defaults to DATA_DIR/workspace_mask.png",
    )
    parser.add_argument("--yolo-model", default=DEFAULT_YOLO_MODEL)
    parser.add_argument("--yolo-device", default=None)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--serial", default=None)
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--max-attempts", type=int, default=30)
    parser.add_argument("--num-points", type=int, default=20000)
    parser.add_argument("--sampling-seed", type=int, default=0)
    parser.add_argument(
        "--workspace-roi",
        type=int,
        nargs=4,
        metavar=("X_MIN", "Y_MIN", "X_MAX", "Y_MAX"),
        default=None,
    )
    parser.add_argument("--min-depth", type=float, default=None)
    parser.add_argument("--max-depth", type=float, default=None)

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
    parser.add_argument("--hole-fill-max-depth-delta-mm", type=float, default=80.0)

    parser.add_argument("--enable-outlier-filter", action="store_true")
    parser.add_argument("--outlier-neighbors", type=int, default=20)
    parser.add_argument("--outlier-std-ratio", type=float, default=2.0)
    parser.add_argument("--enable-radius-outlier-filter", action="store_true")
    parser.add_argument("--radius-outlier-nb-points", type=int, default=8)
    parser.add_argument("--radius-outlier-radius", type=float, default=0.01)
    parser.add_argument("--enable-voxel-downsample", action="store_true")
    parser.add_argument("--voxel-size", type=float, default=0.001)
    parser.add_argument("--disable-normal-estimation", action="store_true")
    parser.add_argument("--normal-radius", type=float, default=0.01)
    parser.add_argument("--normal-max-nn", type=int, default=30)
    parser.add_argument("--min-real-points-warning", type=int, default=5000)
    parser.add_argument("--min-real-points-reject", type=int, default=1000)

    parser.add_argument("--voxel-sweep", action="store_true")
    parser.add_argument(
        "--voxel-sizes",
        nargs="+",
        type=float,
        default=list(VOXEL_SWEEP_SIZES),
        help="Voxel sizes in metres used by --voxel-sweep",
    )
    parser.add_argument("--benchmark-graspnet", action="store_true")
    parser.add_argument(
        "--benchmark-repeats",
        type=int,
        default=3,
        help="Measured repetitions per voxel size after one warm-up",
    )
    parser.add_argument("--checkpoint-path", default=str(DEFAULT_CHECKPOINT_PATH))
    parser.add_argument("--baseline-dir", default=str(DEFAULT_BASELINE_DIR))
    parser.add_argument("--no-display", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
