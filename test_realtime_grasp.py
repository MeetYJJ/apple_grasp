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


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_BASELINE_DIR = WORKSPACE_ROOT / "graspnet-baseline"
DEFAULT_CHECKPOINT_PATH = DEFAULT_BASELINE_DIR / "checkpoint-rs.tar"


def load_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError(
            "Open3D is required for realtime 3D visualization. "
            "Install open3d or run with --no-visualization."
        ) from exc
    return o3d


def create_grasp_coordinate_frame(
    best_grasp: Dict[str, object], size: float
):
    """Create an Open3D coordinate frame for a camera-frame grasp pose."""

    o3d = load_open3d()
    position = np.asarray(best_grasp["position"], dtype=np.float64)
    rotation = np.asarray(best_grasp["rotation"], dtype=np.float64)
    if position.shape != (3,) or rotation.shape != (3, 3):
        raise ValueError("Best grasp must contain position (3,) and rotation (3, 3)")

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = position
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    coordinate_frame.transform(transform)
    return coordinate_frame


class RealtimeVisualizer:
    """Keep OpenCV RGB and Open3D point-cloud windows responsive."""

    def __init__(self, enabled: bool = True, coordinate_size: float = 0.06) -> None:
        self.enabled = enabled
        self.coordinate_size = coordinate_size
        self._point_cloud = None
        self._grasp_frame = None
        self._visualizer = None
        self._o3d = None

        if self.enabled:
            self._o3d = load_open3d()
            self._visualizer = self._o3d.visualization.Visualizer()
            window_created = self._visualizer.create_window(
                window_name="Apple point cloud and best grasp",
                width=960,
                height=720,
            )
            if not window_created:
                raise RuntimeError("Failed to create the Open3D visualization window")
            render_option = self._visualizer.get_render_option()
            render_option.point_size = 2.0
            render_option.background_color = np.asarray([0.03, 0.03, 0.03])

    def update(
        self,
        rgb: np.ndarray,
        apple_mask: np.ndarray,
        cloud: Optional[PointCloudData],
        best_grasp: Optional[Dict[str, object]],
        status: str,
    ) -> bool:
        """Update both windows and return False when the user requests exit."""

        if not self.enabled:
            return True

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

        self._replace_geometry(cloud, best_grasp)
        window_alive = self._visualizer.poll_events()
        self._visualizer.update_renderer()
        key = cv2.waitKey(1) & 0xFF
        return bool(window_alive) and key not in (ord("q"), 27)

    def _replace_geometry(
        self,
        cloud: Optional[PointCloudData],
        best_grasp: Optional[Dict[str, object]],
    ) -> None:
        if self._grasp_frame is not None:
            self._visualizer.remove_geometry(
                self._grasp_frame, reset_bounding_box=False
            )
            self._grasp_frame = None

        if cloud is None:
            if self._point_cloud is not None:
                self._visualizer.remove_geometry(
                    self._point_cloud, reset_bounding_box=False
                )
                self._point_cloud = None
            return

        first_cloud = self._point_cloud is None
        if first_cloud:
            self._point_cloud = self._o3d.geometry.PointCloud()
        self._point_cloud.points = self._o3d.utility.Vector3dVector(cloud.points)
        if cloud.colors is not None:
            self._point_cloud.colors = self._o3d.utility.Vector3dVector(cloud.colors)
        else:
            self._point_cloud.colors = self._o3d.utility.Vector3dVector()
        if first_cloud:
            self._visualizer.add_geometry(self._point_cloud, reset_bounding_box=True)
        else:
            self._visualizer.update_geometry(self._point_cloud)

        if best_grasp is not None:
            self._grasp_frame = create_grasp_coordinate_frame(
                best_grasp, self.coordinate_size
            )
            self._visualizer.add_geometry(
                self._grasp_frame, reset_bounding_box=False
            )

    def close(self) -> None:
        if not self.enabled:
            return
        cv2.destroyAllWindows()
        if self._visualizer is not None:
            self._visualizer.destroy_window()


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
    visualizer = RealtimeVisualizer(
        enabled=not args.no_visualization,
        coordinate_size=args.coordinate_size,
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
                    if not visualizer.update(
                        rgb, apple_mask, None, None, "No apple detected"
                    ):
                        break
                    continue

                valid_mask = build_valid_mask(depth_mm, apple_mask)
                valid_depth_pixels = int(valid_mask.sum())
                if valid_depth_pixels == 0:
                    print("Frame {}: apple mask has no valid depth".format(
                        captured_frames
                    ))
                    if not visualizer.update(
                        rgb, apple_mask, None, None, "Apple has no valid depth"
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
                    if not visualizer.update(
                        rgb, apple_mask, apple_cloud, None, "No grasp candidates"
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
                if not visualizer.update(
                    rgb, apple_mask, apple_cloud, best_grasp, status
                ):
                    break
    except KeyboardInterrupt:
        print("Realtime grasp pipeline stopped by user")
    finally:
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
