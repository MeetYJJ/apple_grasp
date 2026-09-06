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
- `perception/realsense_camera.py`: captures aligned D435i RGB-D frames.
- `perception/mask_process.py`: combines segmentation and valid-depth masks.
- `perception/pointcloud.py`: projects masked depth and samples model input.
- `grasp/graspnet_runner.py`: loads GraspNet and returns a `GraspGroup`.
- `grasp/grasp_selector.py`: selects and saves the highest-scoring grasp.

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

Provide an existing YOLOv8-seg checkpoint containing an `apple` class, then run:

```bash
python test_yolo_apple.py --model-path /path/to/yolov8n-seg.pt
```

The official GraspNet example image contains no apple, so a compatible COCO
checkpoint may correctly produce an empty mask. The test still verifies model
loading, output shape/type, and writes `output/apple_mask_yolo.png`.

Application integration uses the same segmentation interface as before:

```python
from perception.apple_mask import AppleMaskDetector
from perception.yolo_apple_detector import YOLOAppleDetector

yolo = YOLOAppleDetector("/path/to/apple-seg.pt")
detector = AppleMaskDetector(predictor=yolo)
apple_mask = detector.detect(rgb)
```

## Intel RealSense D435i

`perception/realsense_camera.py` supplies depth aligned to the color image in
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
