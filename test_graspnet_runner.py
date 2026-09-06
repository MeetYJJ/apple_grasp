"""Offline end-to-end smoke test using the official GraspNet example data."""

from pathlib import Path

import torch

from grasp.grasp_selector import GraspSelector
from grasp.graspnet_runner import GraspNetRunner
from perception.apple_mask import AppleMaskDetector
from perception.mask_process import build_valid_mask, load_mask
from perception.pointcloud import create_point_cloud, sample_point_cloud
from perception.rgbd_loader import load_rgbd


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
BASELINE_DIR = WORKSPACE_ROOT / "graspnet-baseline"
CHECKPOINT_PATH = BASELINE_DIR / "checkpoint-rs.tar"
DATA_DIR = BASELINE_DIR / "doc" / "example_data"
OUTPUT_PATH = PROJECT_ROOT / "output" / "best_grasp.npy"


def main() -> None:
    frame = load_rgbd(DATA_DIR)

    apple_mask = AppleMaskDetector().detect(frame.color)
    workspace_mask_path = DATA_DIR / "workspace_mask.png"
    if workspace_mask_path.is_file():
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
    main()
