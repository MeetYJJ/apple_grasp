# Apple Grasp

`apple_grasp` is a modular 6D grasp-pose pipeline built around the unmodified
GraspNet baseline. Perception, grasp inference, selection, and future robot
control are kept in separate packages.

## Current pipeline

```text
RGB-D
  ↓
Apple segmentation interface
  ↓
Point cloud extraction
  ↓
GraspNet
  ↓
Best grasp
```

The resulting position and rotation are expressed in the camera coordinate
frame. Camera-to-robot calibration and RealMan control are not implemented in
the current stage.

## Module boundaries

- `perception/rgbd_loader.py`: reads offline RGB-D data and camera intrinsics.
- `perception/apple_mask.py`: exposes `AppleMaskDetector.detect(rgb) -> mask`.
- `camera/realsense_camera.py`: captures aligned D435i RGB-D frames.
- `perception/mask_process.py`: combines segmentation and valid-depth masks.
- `perception/pointcloud.py`: projects masked depth and samples model input.
- `grasp/graspnet_runner.py`: loads GraspNet and returns a `GraspGroup`.
- `grasp/grasp_selector.py`: selects and saves the highest-scoring grasp.
- `visualization/grasp_visualizer.py`: updates the Open3D cloud, camera frame,
  grasp frame, and grasp-approach arrow.

`AppleMaskDetector` currently uses a simple red-region heuristic so that the
pipeline can be exercised without a model. It is not a trained apple detector
and the official GraspNet example image does not contain apples. To integrate a
YOLO/SAM implementation, inject a callable that accepts RGB and returns a
two-dimensional mask:

```python
detector = AppleMaskDetector(predictor=my_yolo_sam_predictor)
apple_mask = detector.detect(rgb)
```

The predictor output is normalized to a contiguous NumPy boolean array with
shape `(H, W)`. Downstream point-cloud and GraspNet modules remain unchanged.

## Tests

Generate and display the segmentation result:

```bash
python test_apple_mask.py
```

The mask is written to `output/apple_mask.png`. For a headless machine:

```bash
python test_apple_mask.py --no-display
```

Run the CUDA GraspNet pipeline with apple-mask filtering:

```bash
python test_graspnet_runner.py
```

The selected grasp is written to `output/best_grasp.npy`.

## YOLOv8-seg + GraspNet pipeline

The optional YOLO adapter changes the production pipeline to:

```text
RGB-D
  ↓
YOLOv8-seg apple instance segmentation
  ↓
Boolean union of detected apple instances
  ↓
Masked apple point cloud
  ↓
GraspNet
  ↓
Best grasp
```

`perception/yolo_apple_detector.py` loads an existing Ultralytics segmentation
checkpoint, filters detections whose class name is `apple`, and merges all apple
instance masks into one `(H, W)` boolean mask. No model training is included.

Install the optional inference dependency:

```bash
pip install ultralytics
```

For a public-model smoke test, run:

```bash
python test_yolo_apple.py
```

This loads the public COCO-pretrained `yolov8n-seg.pt`; on first use,
Ultralytics downloads the weights automatically. COCO class `apple` (class ID
47) is selected, all detected apple instances are merged, and the boolean
result is written to `output/apple_mask_yolo.png`. The official GraspNet
example image contains no apple, so an empty output is a valid result for that
image; use the D435i realtime command below to test with an apple in view.

A local or future custom checkpoint is still supported:

```bash
python test_yolo_apple.py --model-path /path/to/apple-seg.pt
```

Application integration uses the same segmentation interface as before:

```python
from perception.apple_mask import AppleMaskDetector
from perception.yolo_apple_detector import YOLOAppleDetector

yolo = YOLOAppleDetector("/path/to/apple-seg.pt")
detector = AppleMaskDetector(predictor=yolo)
apple_mask = detector.detect(rgb)
```

## Intel RealSense D435i

`camera/realsense_camera.py` supplies depth aligned to the color image in
the same format as the offline loader:

- RGB: NumPy `uint8`, `(H, W, 3)`, RGB channel order.
- Depth: NumPy `uint16`, `(H, W)`, millimetres.
- `camera.intrinsic`: aligned color-camera `3x3` intrinsic matrix.
- `camera.depth_scale`: `0.001` metres per millimetre for point-cloud creation.

Install the RealSense Python binding and the OpenCV viewer dependency:

```bash
pip install pyrealsense2 opencv-python
```

Connect the D435i over USB 3 and verify both live streams:

```bash
python test_realsense.py
```

Press `q` or `Esc` to close the RGB and depth windows.

The end-to-end GraspNet test supports both sources:

```bash
# Existing offline example
python test_graspnet_runner.py --source file

# Capture one aligned D435i frame and run grasp inference
python test_graspnet_runner.py --source camera

# D435i plus an existing YOLOv8-seg apple checkpoint
python test_graspnet_runner.py --source camera \
  --yolo-model /path/to/apple-seg.pt \
  --yolo-device 0
```

RealSense acquisition stops before grasp inference begins. Mechanical-arm
control and camera-to-robot calibration remain outside the current stage.

## Realtime grasp prediction

`test_realtime_grasp.py` keeps the camera and both neural networks initialized,
then repeatedly runs the complete perception and grasp pipeline:

```text
D435i aligned RGB-D
  ↓
YOLOv8-seg apple mask
  ↓
Masked metric apple point cloud
  ↓
GraspNet candidates
  ↓
Highest-scoring grasp in the camera frame
```

Run the complete chain with the public COCO model:

```bash
python test_realtime_grasp.py --yolo-device 0
```

`--yolo-model` defaults to `yolov8n-seg.pt`. It may also be set to a local
custom segmentation checkpoint. When no apple is detected, the frame is
skipped before point-cloud creation and GraspNet inference.

The OpenCV window shows RGB with the apple mask highlighted in green. The
Open3D window is updated in place and shows:

- the RGB-colored apple point cloud;
- the camera coordinate frame at the origin;
- the best-grasp coordinate frame transformed by `T_camera_grasp`;
- a yellow approach arrow pointing along GraspNet grasp +X (`R[:, 0]`) toward
  the grasp position.

The grasp transform is assembled directly from the selected camera-frame pose:

```text
T_camera_grasp = [ R  t ]
                 [ 0  1 ]
```

Coordinate-frame axes use the Open3D convention: x is red, y is green, and z
is blue. The terminal prints position, the `3x3` rotation matrix, score,
candidate count, and per-iteration elapsed time.

Press `q`, `Esc`, or `Ctrl+C` to stop. For one headless inference iteration:

```bash
python test_realtime_grasp.py \
  --yolo-device 0 \
  --max-frames 1 \
  --no-visualization
```

The latest best grasp is also saved to `output/best_grasp.npy`. All poses remain
in the D435i color-camera frame; no mechanical-arm commands are generated.

Visualization sizes can be adjusted without changing inference:

```bash
python test_realtime_grasp.py \
  --yolo-device 0 \
  --camera-coordinate-size 0.10 \
  --coordinate-size 0.06 \
  --approach-length 0.10
```
