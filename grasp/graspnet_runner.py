"""A narrow inference wrapper around the unmodified GraspNet baseline."""

import importlib
import sys
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch


PathLike = Union[str, Path]


class GraspNetRunner:
    """Load GraspNet and infer grasp candidates from a model-ready point cloud."""

    def __init__(
        self,
        checkpoint_path: PathLike,
        baseline_dir: Optional[PathLike] = None,
        num_view: int = 300,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                "GraspNet checkpoint not found: {}".format(self.checkpoint_path)
            )
        if num_view <= 0:
            raise ValueError("num_view must be positive")

        self.baseline_dir = self._resolve_baseline_dir(baseline_dir)
        self.device = self._select_device(device)
        GraspNet, self._pred_decode, self._grasp_group_type = self._load_dependencies()

        self.model = GraspNet(
            input_feature_dim=0,
            num_view=num_view,
            num_angle=12,
            num_depth=4,
            cylinder_radius=0.05,
            hmin=-0.02,
            hmax_list=[0.01, 0.02, 0.03, 0.04],
            is_training=False,
        ).to(self.device)

        checkpoint = self._load_checkpoint()
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise KeyError("Checkpoint does not contain model_state_dict")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.checkpoint_epoch = checkpoint.get("epoch")
        self.model.eval()

    @staticmethod
    def _select_device(device: Optional[Union[str, torch.device]]) -> torch.device:
        if device is None:
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        selected = torch.device(device)
        if selected.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("A CUDA device was requested, but CUDA is not available")
        return selected

    def _resolve_baseline_dir(self, baseline_dir: Optional[PathLike]) -> Path:
        candidates = []
        if baseline_dir is not None:
            candidates.append(Path(baseline_dir).expanduser())
        candidates.extend(
            [
                self.checkpoint_path.parent,
                Path(__file__).resolve().parents[2] / "graspnet-baseline",
            ]
        )

        for candidate in candidates:
            resolved = candidate.resolve()
            if (resolved / "models" / "graspnet.py").is_file():
                return resolved

        checked = ", ".join(str(candidate.resolve()) for candidate in candidates)
        raise FileNotFoundError("Cannot locate graspnet-baseline; checked: {}".format(checked))

    def _load_dependencies(self) -> Tuple[type, object, type]:
        # The upstream repository is script-oriented rather than an importable
        # package, so expose only its required source roots without changing it.
        import_roots = [
            self.baseline_dir / "models",
            self.baseline_dir / "dataset",
            self.baseline_dir / "utils",
        ]

        local_api_root = self.baseline_dir.parent / "graspnetAPI"
        if (local_api_root / "graspnetAPI" / "__init__.py").is_file():
            import_roots.append(local_api_root)

        for root in reversed(import_roots):
            root_string = str(root)
            if root_string not in sys.path:
                sys.path.insert(0, root_string)

        try:
            model_module = importlib.import_module("graspnet")
            api_module = importlib.import_module("graspnetAPI")
            return model_module.GraspNet, model_module.pred_decode, api_module.GraspGroup
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                "Failed to import GraspNet baseline dependencies. Build the pointnet2/knn "
                "extensions and install graspnetAPI in the active Python environment."
            ) from exc

    def _load_checkpoint(self) -> dict:
        # weights_only is safer for a trusted state-dict checkpoint. Older Torch
        # releases do not expose this argument, so retain a compatibility path.
        try:
            return torch.load(
                str(self.checkpoint_path), map_location=self.device, weights_only=True
            )
        except TypeError:
            return torch.load(str(self.checkpoint_path), map_location=self.device)

    def predict(self, point_cloud: np.ndarray):
        """Infer and return a ``graspnetAPI.GraspGroup`` from an ``(N, 3)`` cloud."""

        points = np.asarray(point_cloud, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("point_cloud must have shape (N, 3), got {}".format(points.shape))
        if len(points) < 2048:
            raise ValueError(
                "GraspNet requires at least 2048 points; sample the cloud in perception.pointcloud"
            )
        if not np.all(np.isfinite(points)):
            raise ValueError("point_cloud contains NaN or infinite coordinates")

        point_tensor = torch.from_numpy(np.ascontiguousarray(points))[None].to(self.device)
        end_points = {"point_clouds": point_tensor}

        with torch.no_grad():
            end_points = self.model(end_points)
            grasp_predictions = self._pred_decode(end_points)

        grasp_array = grasp_predictions[0].detach().cpu().numpy()
        return self._grasp_group_type(grasp_array)
