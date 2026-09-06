"""Live RGB and depth viewer for Intel RealSense D435i."""

import argparse

import cv2
import numpy as np

from camera.realsense_camera import RealSenseCamera


def main(width: int = 640, height: int = 480, fps: int = 30) -> None:
    try:
        with RealSenseCamera(width=width, height=height, fps=fps) as camera:
            print("RealSense: {} (serial={})".format(
                camera.device_name, camera.serial_number
            ))
            print("Press q or Esc to quit")
            first_frame = True

            while True:
                rgb, depth_mm = camera.get_frame()
                if first_frame:
                    if rgb.dtype != np.uint8 or depth_mm.dtype != np.uint16:
                        raise TypeError("Unexpected RealSense RGB-D data types")
                    if rgb.shape[:2] != depth_mm.shape:
                        raise ValueError("Aligned RGB/depth resolutions do not match")
                    print("RGB: shape={}, dtype={}".format(rgb.shape, rgb.dtype))
                    print("Depth: shape={}, dtype={}, unit=mm".format(
                        depth_mm.shape, depth_mm.dtype
                    ))
                    print("Intrinsic:\n{}".format(camera.intrinsic))
                    first_frame = False

                color_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                depth_display = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_mm, alpha=0.03), cv2.COLORMAP_JET
                )
                cv2.imshow("RealSense RGB", color_bgr)
                cv2.imshow("RealSense Depth (mm)", depth_display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Display live D435i RGB-D frames")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    arguments = parser.parse_args()
    main(arguments.width, arguments.height, arguments.fps)
