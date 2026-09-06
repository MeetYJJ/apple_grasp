"""Visual comparison of raw and filtered RGB-D point clouds."""

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from camera.realsense_camera import RealSenseCamera
from perception.apple_mask import AppleMaskDetector
from perception.mask_process import load_mask
from perception.pointcloud import (
    PointCloudFilterConfig,
    PointCloudStats,
    create_point_cloud,
    sample_point_cloud,
)
from perception.rgbd_loader import RGBDFrame, load_rgbd
from perception.yolo_apple_detector import DEFAULT_YOLO_MODEL, YOLOAppleDetector


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT.parent / "graspnet-baseline" / "doc" / "example_data"


def load_source(
    source: str,
    yolo_model: str,
    yolo_device: Optional[str],
) -> Tuple[RGBDFrame, np.ndarray]:
    """Load one source-independent frame and its object/workspace mask."""

    if source == "file":
        frame = load_rgbd(DATA_DIR)
        mask_path = DATA_DIR / "workspace_mask.png"
        mask = load_mask(mask_path)
        return frame, mask

    if source == "camera":
        predictor = YOLOAppleDetector(yolo_model, device=yolo_device)
        detector = AppleMaskDetector(predictor=predictor)
        with RealSenseCamera() as camera:
            rgb, depth = camera.get_frame()
            frame = RGBDFrame(
                color=rgb,
                depth=depth,
                intrinsic=camera.intrinsic.copy(),
                depth_scale=camera.depth_scale,
            )
        mask = detector.detect(frame.color)
        if not np.any(mask):
            raise RuntimeError("No apple detected in the captured D435i frame")
        return frame, mask

    raise ValueError("Unsupported source: {}".format(source))


def display_clouds(raw_cloud, filtered_cloud) -> None:
    """Show raw cloud, filtered cloud, then the estimated normals."""

    import open3d as o3d

    o3d.visualization.draw_geometries(
        [raw_cloud], window_name="Raw masked point cloud"
    )
    o3d.visualization.draw_geometries(
        [filtered_cloud], window_name="Filtered point cloud"
    )
    o3d.visualization.draw_geometries(
        [filtered_cloud],
        window_name="Filtered point cloud normals",
        point_show_normal=True,
    )


def run(args: argparse.Namespace) -> None:
    frame, mask = load_source(args.source, args.yolo_model, args.yolo_device)
    config = PointCloudFilterConfig(
        median_kernel=args.depth_median_kernel,
        workspace_roi=(
            None if args.workspace_roi is None else tuple(args.workspace_roi)
        ),
        min_depth_m=args.min_depth,
        max_depth_m=args.max_depth,
        outlier_nb_neighbors=args.outlier_neighbors,
        outlier_std_ratio=args.outlier_std_ratio,
        voxel_size=args.voxel_size,
        normal_radius=args.normal_radius,
        normal_max_nn=args.normal_max_nn,
    )
    raw_config = replace(
        config,
        median_kernel=1,
        outlier_nb_neighbors=0,
        voxel_size=0.0,
        normal_radius=0.0,
    )

    raw_cloud = create_point_cloud(
        depth=frame.depth,
        intrinsic=frame.intrinsic,
        depth_scale=frame.depth_scale,
        mask=mask,
        color=frame.color,
        config=raw_config,
    )
    stats = PointCloudStats()
    filtered_cloud = create_point_cloud(
        depth=frame.depth,
        intrinsic=frame.intrinsic,
        depth_scale=frame.depth_scale,
        mask=mask,
        color=frame.color,
        config=config,
        stats=stats,
    )
    sample_point_cloud(
        filtered_cloud,
        num_points=args.num_points,
        seed=0,
        stats=stats,
    )

    if args.normal_radius > 0 and not filtered_cloud.has_normals():
        raise RuntimeError("Filtered point cloud does not contain estimated normals")

    print("Source: {}".format(args.source))
    print("原始点云点数: {}".format(len(raw_cloud.points)))
    stats.print()
    if not args.no_display:
        display_clouds(raw_cloud, filtered_cloud)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare raw and filtered point-cloud quality"
    )
    parser.add_argument("--source", choices=("file", "camera"), default="file")
    parser.add_argument("--yolo-model", default=DEFAULT_YOLO_MODEL)
    parser.add_argument("--yolo-device", default=None)
    parser.add_argument("--num-points", type=int, default=20000)
    parser.add_argument(
        "--workspace-roi",
        type=int,
        nargs=4,
        metavar=("X_MIN", "Y_MIN", "X_MAX", "Y_MAX"),
        default=None,
    )
    parser.add_argument("--min-depth", type=float, default=None)
    parser.add_argument("--max-depth", type=float, default=None)
    parser.add_argument("--depth-median-kernel", type=int, default=3)
    parser.add_argument("--outlier-neighbors", type=int, default=20)
    parser.add_argument("--outlier-std-ratio", type=float, default=2.0)
    parser.add_argument("--voxel-size", type=float, default=0.002)
    parser.add_argument("--normal-radius", type=float, default=0.01)
    parser.add_argument("--normal-max-nn", type=int, default=30)
    parser.add_argument("--no-display", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
