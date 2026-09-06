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
- `perception/depth_filter.py`: performs configurable, edge-aware D435i depth
  preprocessing without changing depth shape, dtype, or units.
- `perception/pointcloud.py`: projects the masked depth image, applies optional
  geometric filters, records per-stage statistics, and samples model input.
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

## Point-cloud quality filtering

`perception/pointcloud.py` keeps the existing aligned RGB/depth/mask call
contract and still returns an `open3d.geometry.PointCloud`. Depth preprocessing
is isolated in `perception/depth_filter.py`; the default processing order is:

```text
apple mask + optional pixel ROI -> masked depth
  ↓
3x3 median filter (measured pixels only)
  ↓
bilateral spatial smoothing (d=5, sigmaColor=30 mm, sigmaSpace=3)
  ↓
conservative 3x3 hole filling
  (at least 5 neighbours and local depth span <= 80 mm)
  ↓
camera-frame XYZ projection + optional metric depth range
  ↓
optional statistical/radius outlier removal
  ↓
optional voxel downsampling
  ↓
optional normal estimation
```

Statistical outlier removal, radius outlier removal, voxel downsampling, and
normal estimation are **disabled by default**. This conservative default keeps
the real depth samples instead of collapsing a dense apple surface before the
fixed-size GraspNet sampling step. Normals can be enabled for inspection and
are then oriented toward the camera; GraspNet still receives only XYZ (and
optional RGB) samples, so its network and runner interface are unchanged.

The previously observed `~37000 -> ~1700` collapse is consistent with a 2 mm
voxel being applied to a close, nearly two-dimensional apple surface: many
neighboring depth pixels fall into the same voxel. Since GraspNet is still fed
20,000 samples afterward, that voxel step removed unique geometry without
reducing the network input size.

The geometric stages remain available as experiment switches:

```bash
# Statistical outlier removal
python test_realtime_grasp.py --enable-outlier-filter

# Radius outlier removal
python test_realtime_grasp.py --enable-radius-outlier-filter

# 1 mm voxel downsampling
python test_realtime_grasp.py --enable-voxel-downsample --voxel-size 0.001

# Normals are useful for inspection but add realtime CPU work
python test_realtime_grasp.py --enable-normal-estimation
```

Depth preprocessing can be tuned or disabled independently with
`--disable-depth-preprocessing`, `--disable-median-filter`,
`--disable-spatial-smoothing`, and `--disable-hole-filling`. The corresponding
parameters include `--depth-median-kernel`, `--spatial-diameter`,
`--spatial-sigma-color`, `--spatial-sigma-space`, `--hole-fill-kernel`,
`--hole-fill-min-neighbors`, `--hole-fill-iterations`, and
`--hole-fill-max-depth-delta-mm`. Hole filling is intentionally edge-aware: it
does not fill across a local depth discontinuity wider than the configured
threshold.

Every processed frame reports the complete count trace:

```text
Frame N point cloud statistics:
mask pixels: ...
valid depth pixels: ...
after ROI: ...
hole-filled pixels: ...
xyz generated points: ...
after range filter: ...
after outlier removal: ...
after radius outlier removal: ...
after voxel downsample: ...
final points: ...
real_points: ...
sampled_points: ...
```

`valid depth pixels` counts original sensor measurements inside the apple mask
before ROI cropping, `after ROI` isolates workspace-crop loss, and
`hole-filled pixels` reports conservative interpolations separately.
`real_points` is the number of unique filtered cloud points before fixed-size
model sampling (and can therefore include those explicitly reported
interpolations). A frame below 5,000 real points emits a warning; a frame below
1,000 real points is rejected instead of silently duplicating an extremely
sparse cloud. When upsampling is required, all real points are retained before
the remainder is selected, and realtime sampling uses the fixed default seed
`0` to reduce frame-to-frame sampling jitter. Change it with
`--sampling-seed`.

### Point-cloud quality test

Run the offline RGB-D fixture:

```bash
python test_pointcloud_quality.py --source file
```

The official fixture uses `workspace_mask.png`, not an apple segmentation, so
it validates projection/filter plumbing but not the `>10000` apple-cloud target.

For a D435i apple sequence, collect ten valid frames for the point-count
stability check:

```bash
python test_pointcloud_quality.py \
  --source camera \
  --frames 10 \
  --yolo-device 0 \
  --workspace-roi 80 40 560 440 \
  --min-depth 0.20 \
  --max-depth 1.50
```

ROI coordinates are `(x_min, y_min, x_max, y_max)` pixels with exclusive
maximum bounds. Depth limits should match the physical grasp workspace; they
are intentionally disabled unless supplied. The quality test presents four
views: RGB with the apple mask, the raw mask-projected cloud, the filtered
cloud, and the filtered cloud with normals. Its ten-frame summary reports the
minimum, maximum, mean, standard deviation, and peak-to-peak variation; the
target is less than 30% variation across ten valid frames.

Normal estimation is enabled by default only when this quality viewer displays
its fourth window. Pass `--disable-normal-estimation`, or use `--no-display`,
for timing that matches the realtime default.

Sweep the requested voxel sizes and print real-point count, point-cloud
latency, total compute FPS, and (when enabled) GraspNet latency:

```bash
python test_pointcloud_quality.py \
  --source camera \
  --frames 10 \
  --voxel-sweep \
  --voxel-sizes 0.001 0.002 0.003 0.005 \
  --benchmark-repeats 3 \
  --benchmark-graspnet \
  --yolo-device 0
```

Each size is warmed once, then the median of three measured runs is reported.
The no-voxel configuration remains the default because voxelization can reduce
near-range D435i samples dramatically and does not reduce GraspNet's fixed
20,000-point input size. Treat the sweep as a measurement for the actual camera
distance and scene, not as a universal preset. Use `--no-display` for a
headless count/timing run.

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

Apply a physical workspace crop to reject distant masks and irrelevant depth:

```bash
python test_realtime_grasp.py \
  --yolo-device 0 \
  --workspace-roi 80 40 560 440 \
  --min-depth 0.20 \
  --max-depth 1.50
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
candidate count, every point-cloud stage count, real/sampled point counts,
point-cloud time, GraspNet time, total iteration time, and FPS. A rolling
ten-valid-frame point-count summary makes depth/mask instability visible during
normal realtime operation.

Press `q`, `Esc`, or `Ctrl+C` to stop. For one headless inference iteration:

```bash
python test_realtime_grasp.py \
  --yolo-device 0 \
  --max-frames 1 \
  --no-visualization
```

For a controlled comparison, keep the default `--sampling-seed 0` and change
only one filter at a time. For example:

```bash
python test_realtime_grasp.py \
  --yolo-device 0 \
  --enable-voxel-downsample \
  --voxel-size 0.002
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

## Hardware acceptance measurements

The design target for an apple mask of roughly 40,000 pixels is more than
10,000 real points before GraspNet sampling, less than 30% point-count variation
over ten valid frames, and continued `position`/`rotation`/`score` output. These
are **acceptance targets, not measurements claimed by this repository**. They
must be verified on Ubuntu with the intended D435i, scene geometry, YOLO
checkpoint, CUDA device, and GraspNet checkpoint. A bundled offline image or a
development machine without that hardware cannot establish the target point
count, stability, realtime FPS, or CUDA inference latency.
