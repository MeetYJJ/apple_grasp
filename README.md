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
