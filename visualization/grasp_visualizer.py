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
    frame. ``grasp_frame_size`` is retained for backwards compatibility; the
    displayed grasp frame is now scaled from the current apple bbox using
    ``grasp_frame_scale``.
    """

    def __init__(
        self,
        enabled: bool = True,
        window_name: str = "Apple point cloud and best grasp",
        width: int = 960,
        height: int = 720,
        point_size: float = 4.0,
        camera_frame_size: float = 0.10,
        grasp_frame_size: Optional[float] = None,
        approach_length: float = 0.20,
        auto_track: bool = True,
        view_padding: float = 2.50,
        target_apple_fraction: float = 0.65,
        grasp_frame_scale: float = 0.80,
        approach_apple_ratio: float = 1.20,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("window width and height must be positive")
        if point_size <= 0:
            raise ValueError("point_size must be positive")
        if camera_frame_size <= 0:
            raise ValueError("coordinate-frame sizes must be positive")
        if grasp_frame_size is not None and grasp_frame_size <= 0:
            raise ValueError("legacy grasp_frame_size must be positive")
        if approach_length <= 0:
            raise ValueError("approach_length must be positive")
        if view_padding < 1.0:
            raise ValueError("view_padding must be at least 1.0")
        if not 0.0 < target_apple_fraction <= 1.0:
            raise ValueError("target_apple_fraction must be in (0, 1]")
        if grasp_frame_scale <= 0:
            raise ValueError("grasp_frame_scale must be positive")
        if approach_apple_ratio <= 0:
            raise ValueError("approach_apple_ratio must be positive")

        self.enabled = bool(enabled)
        self.camera_frame_size = float(camera_frame_size)
        # Public compatibility attribute; dynamic sizing below deliberately
        # does not use this legacy fixed-size value.
        self.grasp_frame_size = (
            None if grasp_frame_size is None else float(grasp_frame_size)
        )
        self.approach_length = float(approach_length)
        self.auto_track = bool(auto_track)
        self.view_padding = float(view_padding)
        self.target_apple_fraction = float(target_apple_fraction)
        self.grasp_frame_scale = float(grasp_frame_scale)
        self.approach_apple_ratio = float(approach_apple_ratio)
        self.last_grasp_transform: Optional[np.ndarray] = None
        self.last_approach_direction: Optional[np.ndarray] = None
        self.last_apple_bbox_size: Optional[float] = None
        self.last_combined_bbox: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.last_view_zoom: Optional[float] = None
        self.last_padded_scene_size: Optional[float] = None
        self.last_camera_distance: Optional[float] = None
        self.last_grasp_frame_size: Optional[float] = None
        self.last_approach_length: Optional[float] = None

        self._o3d = None
        self._visualizer = None
        self._camera_frame = None
        self._point_cloud = None
        self._grasp_frame = None
        self._approach_arrow = None
        self._bounding_box = None
        self._point_cloud_added = False
        self._bounding_box_added = False
        self._grasp_frame_added = False
        self._approach_arrow_added = False
        self._grasp_vertices = None
        self._grasp_normals = None
        self._arrow_vertices = None
        self._arrow_normals = None

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
        self._visualizer.add_geometry(
            self._camera_frame, reset_bounding_box=False
        )
        self._point_cloud = self._o3d.geometry.PointCloud()
        self._bounding_box = self._o3d.geometry.LineSet()

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

        points = self._update_point_cloud(point_cloud)

        if position is not None and rotation is not None:
            transform = build_grasp_transform(position, rotation)
            approach = get_approach_direction(rotation)
            self.last_grasp_transform = transform.copy()
            self.last_approach_direction = approach.copy()

            apple_size = self.last_apple_bbox_size
            if points is not None:
                apple_size = self._apple_bbox_size(points)
            if apple_size is None:
                apple_size = self._minimum_display_size()
            self._update_grasp_frame(transform, apple_size)
            self._update_approach_arrow(transform[:3, 3], approach, apple_size)

        if points is not None and self.auto_track:
            self._update_view(points)

        window_alive = self._visualizer.poll_events()
        self._visualizer.update_renderer()
        return bool(window_alive)

    def close(self) -> None:
        """Close the Open3D window. Calling this more than once is safe."""

        if self._visualizer is not None:
            self._visualizer.destroy_window()
            self._visualizer = None

    def _update_point_cloud(self, point_cloud: Optional[Any]) -> Optional[np.ndarray]:
        if point_cloud is None:
            return None

        points, colors = self._extract_cloud_arrays(point_cloud)
        self._point_cloud.points = self._o3d.utility.Vector3dVector(points)
        if colors is None:
            self._point_cloud.colors = self._o3d.utility.Vector3dVector()
        else:
            self._point_cloud.colors = self._o3d.utility.Vector3dVector(colors)

        if not self._point_cloud_added:
            self._visualizer.add_geometry(
                self._point_cloud, reset_bounding_box=True
            )
            self._point_cloud_added = True
        else:
            self._visualizer.update_geometry(self._point_cloud)

        self._update_bounding_box(points)
        self.last_apple_bbox_size = self._apple_bbox_size(points)
        return points

    def _update_grasp_frame(
        self, transform: np.ndarray, apple_bbox_size: Optional[float] = None
    ) -> None:
        apple_bbox_size = self._resolve_apple_bbox_size(apple_bbox_size)
        frame_size = max(
            float(apple_bbox_size) * self.grasp_frame_scale,
            self._minimum_display_size(),
        )
        self.last_grasp_frame_size = frame_size
        if self._grasp_frame is None:
            self._grasp_frame = (
                self._o3d.geometry.TriangleMesh.create_coordinate_frame(
                    # Keep a unit template. Its vertices are scaled to the
                    # current apple bbox on every frame below.
                    size=1.0
                )
            )
            self._grasp_vertices = np.asarray(
                self._grasp_frame.vertices
            ).copy()
            self._grasp_normals = np.asarray(
                self._grasp_frame.vertex_normals
            ).copy()

        rotation = transform[:3, :3]
        translation = transform[:3, 3]
        vertices = (
            self._grasp_vertices * frame_size
        ) @ rotation.T + translation
        normals = self._grasp_normals @ rotation.T
        self._grasp_frame.vertices = self._o3d.utility.Vector3dVector(vertices)
        self._grasp_frame.vertex_normals = self._o3d.utility.Vector3dVector(
            normals
        )
        if not self._grasp_frame_added:
            self._visualizer.add_geometry(
                self._grasp_frame, reset_bounding_box=False
            )
            self._grasp_frame_added = True
        else:
            self._visualizer.update_geometry(self._grasp_frame)

    def _update_approach_arrow(
        self,
        position: np.ndarray,
        approach: np.ndarray,
        apple_bbox_size: Optional[float] = None,
    ) -> None:
        apple_bbox_size = self._resolve_apple_bbox_size(apple_bbox_size)
        # Treat the configured value as a maximum. A very small apple should
        # not be framed by an arrow several times larger than the target.
        display_length = min(
            self.approach_length,
            max(float(apple_bbox_size) * self.approach_apple_ratio,
                self._minimum_display_size()),
        )
        self.last_approach_length = display_length
        if self._approach_arrow is None:
            cone_height = 0.25
            cylinder_height = 0.75
            cylinder_radius = 0.025
            cone_radius = cylinder_radius * 2.0
            self._approach_arrow = self._o3d.geometry.TriangleMesh.create_arrow(
                cylinder_radius=cylinder_radius,
                cone_radius=cone_radius,
                cylinder_height=cylinder_height,
                cone_height=cone_height,
            )
            self._approach_arrow.compute_vertex_normals()
            self._approach_arrow.paint_uniform_color([1.0, 0.75, 0.0])
            self._arrow_vertices = np.asarray(
                self._approach_arrow.vertices
            ).copy()
            self._arrow_normals = np.asarray(
                self._approach_arrow.vertex_normals
            ).copy()

        # Open3D arrows point along +Z. Place the tail behind the target so the
        # yellow arrow points along grasp +X and its tip ends at the grasp point.
        align_rotation = self._rotation_from_z_axis(approach)
        tail = position - approach * display_length
        vertices = self._arrow_vertices * display_length
        vertices = vertices @ align_rotation.T + tail
        normals = self._arrow_normals @ align_rotation.T
        self._approach_arrow.vertices = self._o3d.utility.Vector3dVector(
            vertices
        )
        self._approach_arrow.vertex_normals = (
            self._o3d.utility.Vector3dVector(normals)
        )
        if not self._approach_arrow_added:
            self._visualizer.add_geometry(
                self._approach_arrow, reset_bounding_box=False
            )
            self._approach_arrow_added = True
        else:
            self._visualizer.update_geometry(self._approach_arrow)

    def _update_bounding_box(self, points: np.ndarray) -> None:
        minimum = points.min(axis=0)
        maximum = points.max(axis=0)
        x0, y0, z0 = minimum
        x1, y1, z1 = maximum
        corners = np.asarray(
            [
                [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
            ],
            dtype=np.float64,
        )
        lines = np.asarray(
            [
                [0, 1], [1, 2], [2, 3], [3, 0],
                [4, 5], [5, 6], [6, 7], [7, 4],
                [0, 4], [1, 5], [2, 6], [3, 7],
            ],
            dtype=np.int32,
        )
        self._bounding_box.points = self._o3d.utility.Vector3dVector(corners)
        self._bounding_box.lines = self._o3d.utility.Vector2iVector(lines)
        self._bounding_box.colors = self._o3d.utility.Vector3dVector(
            np.tile([0.0, 1.0, 1.0], (len(lines), 1))
        )
        if not self._bounding_box_added:
            self._visualizer.add_geometry(
                self._bounding_box, reset_bounding_box=False
            )
            self._bounding_box_added = True
        else:
            self._visualizer.update_geometry(self._bounding_box)

    def _update_view(self, points: np.ndarray) -> None:
        centroid = points.mean(axis=0)
        apple_min = points.min(axis=0)
        apple_max = points.max(axis=0)
        scene_parts = [points]
        if self._grasp_frame is not None:
            scene_parts.append(np.asarray(self._grasp_frame.vertices))
        if self._approach_arrow is not None:
            scene_parts.append(np.asarray(self._approach_arrow.vertices))

        combined = np.concatenate(scene_parts, axis=0)
        combined_min = combined.min(axis=0)
        combined_max = combined.max(axis=0)
        self.last_combined_bbox = (combined_min.copy(), combined_max.copy())

        apple_size = max(float(np.max(apple_max - apple_min)),
                         self._minimum_display_size())
        combined_size = max(float(np.max(combined_max - combined_min)),
                            self._minimum_display_size())
        padded_size = combined_size * self.view_padding
        self.last_padded_scene_size = padded_size

        # Recompute the Visualizer bounding box after the persistent geometry
        # has moved. This does not recreate or remove any geometry, but makes
        # Open3D's normalized zoom refer to the current scene rather than to a
        # stale first-frame box.
        reset_view_point = getattr(self._visualizer, "reset_view_point", None)
        if callable(reset_view_point):
            reset_view_point(True)

        # A larger/padded combined scene must move the camera farther away.
        # Open3D's zoom increases camera distance; keep it in the conservative
        # range requested for this application so the camera cannot enter the
        # apple. The reference padding is 2.5x, so the default target fraction
        # maps to a 0.40 baseline zoom and changing padding remains effective.
        padding_ratio = padded_size / (2.50 * combined_size)
        zoom = float(np.clip(
            0.40
            * (self.target_apple_fraction / 0.65)
            * np.sqrt(padding_ratio)
            * np.sqrt(combined_size / apple_size),
            0.30,
            0.50,
        ))
        # At the default 60 degree FOV, fitting half the padded extent uses
        # distance = padded_extent / (2*tan(FOV/2)). This is the desired
        # scene-space distance; Open3D's internal distance also includes the
        # fixed camera-frame geometry in its bounding box.
        self.last_camera_distance = padded_size / (
            2.0 * np.tan(np.deg2rad(30.0))
        )
        self.last_view_zoom = zoom
        view = self._visualizer.get_view_control()
        view.set_lookat(centroid.tolist())
        # The camera is placed on the sensor side (-Z) and looks toward +Z,
        # which keeps the default view outside an apple in the D435i frame.
        view.set_front([0.0, 0.0, -1.0])
        view.set_up([0.0, -1.0, 0.0])
        view.set_zoom(zoom)

    @staticmethod
    def _minimum_display_size() -> float:
        return 1e-4

    def _resolve_apple_bbox_size(self, apple_bbox_size: Optional[float]) -> float:
        if apple_bbox_size is not None:
            return max(float(apple_bbox_size), self._minimum_display_size())
        if self.last_apple_bbox_size is not None:
            return self.last_apple_bbox_size
        if self.grasp_frame_size is not None:
            return max(
                self.grasp_frame_size / self.grasp_frame_scale,
                self._minimum_display_size(),
            )
        return self._minimum_display_size()

    @staticmethod
    def _apple_bbox_size(points: np.ndarray) -> float:
        extent = np.max(points, axis=0) - np.min(points, axis=0)
        return max(float(np.max(extent)), GraspVisualizer._minimum_display_size())

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
    def _extract_cloud_arrays(
        point_cloud: Any,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
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
