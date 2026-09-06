"""Temporal smoothing for camera-frame grasp poses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


@dataclass
class GraspPoseFilterStats:
    raw_position_delta_m: float = 0.0
    filtered_position_delta_m: float = 0.0
    raw_rotation_delta_degrees: float = 0.0
    filtered_rotation_delta_degrees: float = 0.0


class GraspPoseFilter:
    """Smooth translation with EMA and rotation on the SO(3) manifold.

    ``previous_weight=0.7`` implements ``p = 0.7*p_previous +
    0.3*p_current``. Rotation uses the same current-frame fraction through
    SciPy Slerp; rotation matrices are never averaged element by element.
    """

    def __init__(self, previous_weight: float = 0.7) -> None:
        if not 0.0 <= previous_weight < 1.0:
            raise ValueError("previous_weight must be in [0, 1)")
        self.previous_weight = float(previous_weight)
        self.previous_position: Optional[np.ndarray] = None
        self.previous_rotation: Optional[np.ndarray] = None
        self.previous_raw_position: Optional[np.ndarray] = None
        self.previous_raw_rotation: Optional[np.ndarray] = None
        self.last_stats = GraspPoseFilterStats()

    def reset(self) -> None:
        self.previous_position = None
        self.previous_rotation = None
        self.previous_raw_position = None
        self.previous_raw_rotation = None
        self.last_stats = GraspPoseFilterStats()

    def process(self, grasp: Dict[str, object]) -> Dict[str, object]:
        """Return a filtered copy of a ``position/rotation/score`` grasp."""

        if not isinstance(grasp, dict):
            raise TypeError("grasp must be a dictionary")
        if "position" not in grasp or "rotation" not in grasp:
            raise KeyError("grasp must contain position and rotation")

        raw_position = np.asarray(grasp["position"], dtype=np.float64).reshape(-1)
        raw_rotation = np.asarray(grasp["rotation"], dtype=np.float64)
        if raw_position.shape != (3,):
            raise ValueError("grasp position must have shape (3,)")
        if raw_rotation.shape != (3, 3):
            raise ValueError("grasp rotation must have shape (3, 3)")
        if not np.all(np.isfinite(raw_position)) or not np.all(
            np.isfinite(raw_rotation)
        ):
            raise ValueError("grasp pose contains NaN or infinite values")
        raw_rotation = self._project_to_so3(raw_rotation)

        if self.previous_position is None or self.previous_rotation is None:
            filtered_position = raw_position.copy()
            filtered_rotation = raw_rotation.copy()
            stats = GraspPoseFilterStats()
        else:
            current_weight = 1.0 - self.previous_weight
            filtered_position = (
                self.previous_weight * self.previous_position
                + current_weight * raw_position
            )
            key_rotations = Rotation.from_matrix(
                np.stack((self.previous_rotation, raw_rotation), axis=0)
            )
            filtered_rotation = Slerp(
                [0.0, 1.0], key_rotations
            )([current_weight]).as_matrix()[0]
            stats = GraspPoseFilterStats(
                raw_position_delta_m=self._position_delta(
                    self.previous_raw_position, raw_position
                ),
                filtered_position_delta_m=self._position_delta(
                    self.previous_position, filtered_position
                ),
                raw_rotation_delta_degrees=self._rotation_delta_degrees(
                    self.previous_raw_rotation, raw_rotation
                ),
                filtered_rotation_delta_degrees=self._rotation_delta_degrees(
                    self.previous_rotation, filtered_rotation
                ),
            )

        self.previous_raw_position = raw_position.copy()
        self.previous_raw_rotation = raw_rotation.copy()
        self.previous_position = filtered_position.copy()
        self.previous_rotation = filtered_rotation.copy()
        self.last_stats = stats

        result = dict(grasp)
        result["position"] = np.ascontiguousarray(
            filtered_position, dtype=np.float32
        )
        result["rotation"] = np.ascontiguousarray(
            filtered_rotation, dtype=np.float32
        )
        if "score" in result:
            result["score"] = float(result["score"])
        return result

    __call__ = process

    @staticmethod
    def _project_to_so3(matrix: np.ndarray) -> np.ndarray:
        u, _, vh = np.linalg.svd(matrix)
        rotation = u @ vh
        if np.linalg.det(rotation) < 0:
            u[:, -1] *= -1.0
            rotation = u @ vh
        return rotation

    @staticmethod
    def _position_delta(
        previous: Optional[np.ndarray], current: np.ndarray
    ) -> float:
        if previous is None:
            return 0.0
        return float(np.linalg.norm(current - previous))

    @staticmethod
    def _rotation_delta_degrees(
        previous: Optional[np.ndarray], current: np.ndarray
    ) -> float:
        if previous is None:
            return 0.0
        relative = previous.T @ current
        cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
        return float(np.degrees(np.arccos(cosine)))
