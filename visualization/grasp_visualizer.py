"""Realtime Open3D visualization of an apple cloud and selected grasp pose."""

from typing import Any, Optional, Tuple

import numpy as np


def build_grasp_transform(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Build the camera-from-grasp homogeneous transform ``[R t; 0 1]``."""

    translation = np.asarray(position, dtype=np.float64)
    rotation_matrix = np.asarray(rotation, dtype=np.float64)
    if translation.shape != (3,):
        raise ValueError("position must have shape (3,), got {}".format(
            translation.shape
        ))
    if rotation_matrix.shape != (3, 3):
        raise ValueError("rotation must have shape (3, 3), got {}".format(
            rotation_matrix.shape
        ))
    if not np.all(np.isfinite(translation)):
        raise ValueError("position contains NaN or infinite values")
    if not np.all(np.isfinite(rotation_matrix)):
        raise ValueError("rotation contains NaN or infinite values")

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = translation
    return transform


def get_approach_direction(rotation: np.ndarray) -> np.ndarray:
    """Return GraspNet's normalized approach direction in the camera frame.

    GraspNet stores the grasp approach axis in the first rotation-matrix
    column, i.e. the grasp frame's local positive X axis.
    """

    rotation_matrix = np.asarray(rotation, dtype=np.float64)
    if rotation_matrix.shape != (3, 3):
        raise ValueError("rotation must have shape (3, 3), got {}".format(
            rotation_matrix.shape
        ))
    direction = rotation_matrix[:, 0]
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("rotation has an invalid approach axis")
    return np.ascontiguousarray(direction / norm, dtype=np.float64)


class GraspVisualizer:
    """Display and update point cloud, camera frame, grasp frame, and approach.

    ``point_cloud`` passed to :meth:`update` may be the pipeline's
    ``PointCloudData``, an Open3D point cloud, or an ``(N, 3)`` NumPy array.
    All positions and point coordinates are expected in metres in the camera
    frame.
    """

    def __init__(
        self,
        enabled: bool = True,
        window_name: str = "Apple point cloud and best grasp",
        width: int = 960,
        height: int = 720,
        point_size: float = 2.0,
        camera_frame_size: float = 0.10,
        grasp_frame_size: float = 0.06,
        approach_length: float = 0.10,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("window width and height must be positive")
        if point_size <= 0:
            raise ValueError("point_size must be positive")
        if camera_frame_size <= 0 or grasp_frame_size <= 0:
            raise ValueError("coordinate-frame sizes must be positive")
        if approach_length <= 0:
            raise ValueError("approach_length must be positive")

        self.enabled = bool(enabled)
        self.grasp_frame_size = float(grasp_frame_size)
        self.approach_length = float(approach_length)
        self.last_grasp_transform: Optional[np.ndarray] = None
        self.last_approach_direction: Optional[np.ndarray] = None

        self._o3d = None
        self._visualizer = None
        self._camera_frame = None
        self._point_cloud = None
        self._grasp_frame = None
        self._approach_arrow = None

        if not self.enabled:
            return

        self._o3d = self._load_open3d()
        self._visualizer = self._o3d.visualization.Visualizer()
        window_created = self._visualizer.create_window(
            window_name=window_name,
            width=int(width),
            height=int(height),
        )
        if not window_created:
            raise RuntimeError("Failed to create the Open3D visualization window")

        render_option = self._visualizer.get_render_option()
        render_option.point_size = float(point_size)
        render_option.background_color = np.asarray([0.03, 0.03, 0.03])

        # This frame is fixed at the D435i color-camera origin.
        self._camera_frame = self._o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=float(camera_frame_size), origin=[0.0, 0.0, 0.0]
        )
        self._visualizer.add_geometry(self._camera_frame, reset_bounding_box=True)

    def update(
        self,
        point_cloud: Optional[Any],
        position: Optional[np.ndarray] = None,
        rotation: Optional[np.ndarray] = None,
    ) -> bool:
        """Update the scene and return ``False`` when its window is closed."""

        if (position is None) != (rotation is None):
            raise ValueError("position and rotation must be provided together")
        if not self.enabled:
            return True

        self._update_point_cloud(point_cloud)
        self._remove_dynamic_pose()

        if position is not None and rotation is not None:
            transform = build_grasp_transform(position, rotation)
            approach = get_approach_direction(rotation)
            self.last_grasp_transform = transform.copy()
            self.last_approach_direction = approach.copy()

            self._grasp_frame = (
                self._o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=self.grasp_frame_size
                )
            )
            self._grasp_frame.transform(transform)
            self._approach_arrow = self._create_approach_arrow(
                transform[:3, 3], approach
            )
            self._visualizer.add_geometry(
                self._grasp_frame, reset_bounding_box=False
            )
            self._visualizer.add_geometry(
                self._approach_arrow, reset_bounding_box=False
            )
        else:
            self.last_grasp_transform = None
            self.last_approach_direction = None

        window_alive = self._visualizer.poll_events()
        self._visualizer.update_renderer()
        return bool(window_alive)

    def close(self) -> None:
        """Close the Open3D window. Calling this more than once is safe."""

        if self._visualizer is not None:
            self._visualizer.destroy_window()
            self._visualizer = None

    def _update_point_cloud(self, point_cloud: Optional[Any]) -> None:
        if point_cloud is None:
            if self._point_cloud is not None:
                self._visualizer.remove_geometry(
                    self._point_cloud, reset_bounding_box=False
                )
                self._point_cloud = None
            return

        points, colors = self._extract_cloud_arrays(point_cloud)
        first_cloud = self._point_cloud is None
        if first_cloud:
            self._point_cloud = self._o3d.geometry.PointCloud()

        self._point_cloud.points = self._o3d.utility.Vector3dVector(points)
        if colors is None:
            self._point_cloud.colors = self._o3d.utility.Vector3dVector()
        else:
            self._point_cloud.colors = self._o3d.utility.Vector3dVector(colors)

        if first_cloud:
            self._visualizer.add_geometry(
                self._point_cloud, reset_bounding_box=True
            )
        else:
            self._visualizer.update_geometry(self._point_cloud)

    def _remove_dynamic_pose(self) -> None:
        for attribute in ("_grasp_frame", "_approach_arrow"):
            geometry = getattr(self, attribute)
            if geometry is not None:
                self._visualizer.remove_geometry(
                    geometry, reset_bounding_box=False
                )
                setattr(self, attribute, None)

    def _create_approach_arrow(
        self, position: np.ndarray, approach: np.ndarray
    ) -> Any:
        cone_height = self.approach_length * 0.25
        cylinder_height = self.approach_length - cone_height
        cylinder_radius = max(self.approach_length * 0.025, 0.001)
        cone_radius = cylinder_radius * 2.0
        arrow = self._o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=cylinder_radius,
            cone_radius=cone_radius,
            cylinder_height=cylinder_height,
            cone_height=cone_height,
        )
        arrow.compute_vertex_normals()
        arrow.paint_uniform_color([1.0, 0.75, 0.0])

        # Open3D arrows point along +Z. Place the tail behind the target so the
        # yellow arrow points along grasp +X and its tip ends at the grasp point.
        align_rotation = self._rotation_from_z_axis(approach)
        arrow.rotate(align_rotation, center=[0.0, 0.0, 0.0])
        arrow.translate(position - approach * self.approach_length)
        return arrow

    @staticmethod
    def _rotation_from_z_axis(direction: np.ndarray) -> np.ndarray:
        z_axis = np.asarray(direction, dtype=np.float64)
        z_axis = z_axis / np.linalg.norm(z_axis)
        reference = np.asarray([0.0, 1.0, 0.0])
        if abs(float(np.dot(reference, z_axis))) > 0.99:
            reference = np.asarray([1.0, 0.0, 0.0])
        x_axis = np.cross(reference, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        return np.column_stack((x_axis, y_axis, z_axis))

    @staticmethod
    def _extract_cloud_arrays(point_cloud: Any) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if isinstance(point_cloud, np.ndarray):
            points = np.asarray(point_cloud, dtype=np.float64)
            colors = None
        elif hasattr(point_cloud, "points"):
            points = np.asarray(point_cloud.points, dtype=np.float64)
            color_data = getattr(point_cloud, "colors", None)
            colors = None if color_data is None else np.asarray(
                color_data, dtype=np.float64
            )
            if colors.size == 0:
                colors = None
        else:
            raise TypeError(
                "point_cloud must be an (N, 3) array or expose points/colors"
            )

        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("point cloud must have shape (N, 3), got {}".format(
                points.shape
            ))
        if len(points) == 0:
            raise ValueError("point cloud must not be empty")
        if not np.all(np.isfinite(points)):
            raise ValueError("point cloud contains NaN or infinite values")

        if colors is not None:
            if colors.shape != points.shape:
                raise ValueError("point-cloud colors must have shape {}".format(
                    points.shape
                ))
            if not np.all(np.isfinite(colors)):
                raise ValueError("point-cloud colors contain NaN or infinite values")
            if colors.size and float(colors.max()) > 1.0:
                colors = colors / 255.0
            colors = np.ascontiguousarray(np.clip(colors, 0.0, 1.0))

        return np.ascontiguousarray(points), colors

    @staticmethod
    def _load_open3d() -> Any:
        try:
            import open3d as o3d
        except ImportError as exc:
            raise ImportError(
                "Open3D is required for grasp visualization. Install open3d "
                "or run the realtime pipeline with --no-visualization."
            ) from exc
        return o3d
