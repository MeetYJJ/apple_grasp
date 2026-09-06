"""Intel RealSense D435i RGB-D input adapter."""

from typing import Optional, Tuple

import numpy as np


class RealSenseCamera:
    """Capture color-aligned RGB and millimetre depth frames from a D435i.

    Frame format:
      - RGB: uint8, shape (H, W, 3), RGB channel order.
      - Depth: uint16, shape (H, W), values in millimetres.

    ``intrinsic`` contains the aligned color-camera matrix. ``depth_scale`` is
    always 0.001 metres per millimetre for downstream point-cloud projection.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        serial_number: Optional[str] = None,
        timeout_ms: int = 5000,
    ) -> None:
        """Initialize and start the D435i color/depth pipeline."""

        if width <= 0 or height <= 0 or fps <= 0:
            raise ValueError("width, height and fps must be positive")
        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")

        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise ImportError(
                "pyrealsense2 is required for RealSenseCamera. "
                "Install the Intel RealSense Python bindings first."
            ) from exc

        self._rs = rs
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.timeout_ms = int(timeout_ms)
        self._pipeline = rs.pipeline()
        self._config = rs.config()
        if serial_number:
            self._config.enable_device(str(serial_number))
        self._config.enable_stream(
            rs.stream.depth, self.width, self.height, rs.format.z16, self.fps
        )
        self._config.enable_stream(
            rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps
        )

        self._started = False
        try:
            self._profile = self._pipeline.start(self._config)
            self._started = True
        except RuntimeError as exc:
            raise RuntimeError(
                "Failed to start RealSense D435i. Check the USB connection, "
                "stream configuration, and whether another process owns the camera."
            ) from exc

        try:
            device = self._profile.get_device()
            self.device_name = device.get_info(rs.camera_info.name)
            self.serial_number = device.get_info(rs.camera_info.serial_number)
            self._native_depth_scale = float(
                device.first_depth_sensor().get_depth_scale()
            )
            if self._native_depth_scale <= 0:
                raise RuntimeError("RealSense returned an invalid depth scale")

            color_profile = self._profile.get_stream(
                rs.stream.color
            ).as_video_stream_profile()
            color_intrinsics = color_profile.get_intrinsics()
            self.intrinsic = np.array(
                [
                    [color_intrinsics.fx, 0.0, color_intrinsics.ppx],
                    [0.0, color_intrinsics.fy, color_intrinsics.ppy],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            self.depth_scale = 0.001
            self._align = rs.align(rs.stream.color)
        except Exception:
            self.stop()
            raise

    def get_frame(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return one aligned ``(rgb, depth_mm)`` frame pair."""

        if not self._started:
            raise RuntimeError("RealSense camera is not running")

        for _ in range(30):
            frames = self._pipeline.wait_for_frames(self.timeout_ms)
            aligned_frames = self._align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            depth_raw = np.asanyarray(depth_frame.get_data())
            color_bgr = np.asanyarray(color_frame.get_data())
            if depth_raw.ndim != 2 or color_bgr.ndim != 3:
                continue
            if depth_raw.shape != color_bgr.shape[:2]:
                continue

            rgb = np.ascontiguousarray(color_bgr[..., ::-1], dtype=np.uint8)
            millimetres_per_unit = self._native_depth_scale * 1000.0
            depth_mm = np.rint(
                depth_raw.astype(np.float32) * np.float32(millimetres_per_unit)
            )
            depth_mm = np.ascontiguousarray(
                np.clip(depth_mm, 0, np.iinfo(np.uint16).max).astype(np.uint16)
            )
            return rgb, depth_mm

        raise RuntimeError("Failed to obtain a valid aligned RGB-D frame")

    def stop(self) -> None:
        """Stop streaming. Calling this method more than once is safe."""

        if getattr(self, "_started", False):
            self._pipeline.stop()
            self._started = False

    def __enter__(self) -> "RealSenseCamera":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()

    def __del__(self) -> None:
        try:
            self.stop()
        except Exception:
            pass
