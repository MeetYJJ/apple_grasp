"""Select and persist grasp poses independently of model inference."""

from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np


PathLike = Union[str, Path]


class GraspSelector:
    """Select the highest-scoring grasp from a GraspGroup."""

    def __init__(self, output_path: Optional[PathLike] = None) -> None:
        if output_path is None:
            output_path = Path(__file__).resolve().parents[1] / "output" / "best_grasp.npy"
        self.output_path = Path(output_path).expanduser().resolve()

    def get_best_grasp(self, grasp_group) -> Dict[str, object]:
        """Sort by score, save the best pose, and return it as a dictionary.

        Position and rotation are in the same camera coordinate frame as the
        input point cloud. Robot-base conversion belongs in the robot layer.
        """

        if grasp_group is None or len(grasp_group) == 0:
            raise RuntimeError("No valid grasp candidates were produced")

        # Work on a copy because GraspGroup.sort_by_score() sorts in place.
        sorted_grasps = grasp_group.__class__(grasp_group.grasp_group_array.copy())
        sorted_grasps.sort_by_score()
        best = sorted_grasps[0]

        result = {
            "position": np.asarray(best.translation, dtype=np.float32).copy(),
            "rotation": np.asarray(best.rotation_matrix, dtype=np.float32).copy(),
            "score": float(best.score),
        }
        self.save(result)
        return result

    def save(self, best_grasp: Dict[str, object]) -> Path:
        """Save a best-grasp dictionary as a NumPy object file."""

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(self.output_path), best_grasp, allow_pickle=True)
        return self.output_path
