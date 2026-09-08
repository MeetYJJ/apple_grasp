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
Temporal mask stabilization
  ↓
Mask-region depth temporal smoothing
  ↓
Point cloud extraction
  ↓
GraspNet
  ↓
moving-average/Slerp-filtered best grasp
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
- `perception/depth_temporal_filter.py`: applies depth EMA only in valid
  overlap between consecutive apple masks.
- `perception/pointcloud.py`: projects the masked depth image, applies optional
  geometric filters, records per-stage statistics, and samples model input.
- `perception/pointcloud_temporal_filter.py`: selects the mask EMA weight from
  adjacent-frame IoU so moving targets follow the current detection quickly.
- `grasp/graspnet_runner.py`: loads GraspNet and returns a `GraspGroup`.
- `grasp/grasp_selector.py`: selects and saves the highest-scoring grasp.
- `grasp/grasp_pose_filter.py`: filters translation with a moving average and
  rotation with SO(3) Slerp; it never averages rotation-matrix elements.
- `visualization/grasp_visualizer.py`: updates the Open3D cloud, camera frame,
  grasp frame, grasp-approach arrow, and combined-scene camera framing.

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
GraspNet sampling step. Normals can be enabled for inspection and
are then oriented toward the camera; GraspNet still receives only XYZ (and
optional RGB) samples, so its network and runner interface are unchanged.

The previously observed `~37000 -> ~1700` collapse is consistent with a 2 mm
voxel being applied to a close, nearly two-dimensional apple surface: many
neighboring depth pixels fall into the same voxel. That voxel step removed
unique geometry without improving the model input.

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

The realtime pipeline now relies on the persistent RealSense SDK filters and
disables this additional CPU preprocessing by default to avoid filtering the
same 640x480 depth frame twice. Use `--enable-depth-preprocessing` only for an
A/B quality measurement; the standalone point-cloud quality test keeps its
original configurable CPU path.

The point-cloud quality test reports the complete count trace:

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
`real_points` is the number of unique filtered cloud points before model
sampling and can include explicitly reported hole-filled measurements. A frame
below 2,048 real points is rejected with `cloud too sparse`; below 8,000 it
emits a quality warning. From 2,048 through 20,000 points every unique point is
passed directly to GraspNet; points are never copied to manufacture a
20,000-point tensor. Clouds above 20,000 points use Open3D farthest-point
sampling (FPS) to select 20,000 spatially distributed real samples.

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
near-range D435i samples dramatically. Treat the sweep as a measurement for
the actual camera distance and scene, not as a universal preset. Use
`--no-display` for a headless count/timing run.

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

The camera adapter creates one persistent RealSense post-processing chain:
decimation, spatial smoothing, temporal smoothing, and hole filling. The
default decimation magnitude is `1` to retain all scarce close-range depth
pixels; set `--decimation-magnitude 2` only when profiling shows that reduced
resolution is acceptable. Disable the complete SDK chain with
`--disable-realsense-depth-filters`.

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
YOLOv8-seg raw apple mask
  ↓
motion-adaptive mask EMA
  ↓
mask-overlap depth EMA
  ↓
current-frame masked metric apple point cloud
  ↓
GraspNet candidates
  ↓
highest-scoring grasp + position moving average/rotation Slerp
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
custom segmentation checkpoint. An empty current mask resets depth and grasp
history and skips point-cloud creation and GraspNet inference.

Depth is accepted only within 250--2000 mm. Before point-cloud construction the
pipeline computes `valid_depth_pixels / mask_pixels`; when it is below `0.20`,
GraspNet is skipped and the last valid cloud and grasp remain visible. The same
hold-last behavior is used below 2,048 real cloud points, which is the minimum
accepted by the unchanged `GraspNetRunner`. This prevents close-
range D435i holes from producing an expensive, meaningless inference.

The OpenCV window shows RGB with the apple mask highlighted in green. The
Open3D window is updated in place and shows:

- the RGB-colored apple point cloud;
- the camera coordinate frame at the origin;
- a cyan apple axis-aligned bounding box;
- the best-grasp coordinate frame transformed by `T_camera_grasp`;
- a yellow approach arrow pointing along GraspNet grasp +X (`R[:, 0]`) toward
  the grasp position.

The grasp transform is assembled directly from the filtered camera-frame pose:

```text
T_camera_grasp = [ R  t ]
                 [ 0  1 ]
```

Coordinate-frame axes use the Open3D convention: x is red, y is green, and z
is blue. The Open3D view looks at the current cloud centroid and derives zoom
from the combined point-cloud, grasp-frame, and approach-arrow bounds, so the
apple remains prominent as distance changes. Point cloud, bounding box, grasp
frame, and approach arrow objects are
allocated once and subsequently changed with `update_geometry()`; no geometry
is removed and recreated per frame. By default the terminal prints one compact
report every ten captured frames: frame number, FPS, mask pixels, valid-depth
ratio, real/model cloud points, grasp score, position, `3x3` rotation matrix,
and total latency. Change the interval with `--log-interval`.

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
  --approach-length 0.20
```

`--coordinate-size` remains accepted for scripts written against the previous
version, but is ignored; the displayed grasp frame is dynamic and is
controlled by `--grasp-frame-scale`.

The camera framing uses a 40% padding around the combined apple, grasp-frame,
and approach-arrow bounds. Its default target apple fraction is `0.65`; the
grasp frame is dynamically `0.8 * apple_bbox_size`, and the approach arrow is
limited to `1.2 * apple_bbox_size` (or the configured `--approach-length`,
whichever is smaller). These values can be tuned without changing GraspNet:

```bash
python test_realtime_grasp.py \
  --yolo-device 0 \
  --view-padding 1.40 \
  --target-apple-fraction 0.65 \
  --grasp-frame-scale 0.80 \
  --approach-apple-ratio 1.20
```

## Motion-adaptive realtime filtering

The mask filter compares the current raw YOLO mask with the previous raw mask,
then selects the historical EMA weight dynamically:

- `IoU > 0.7`: `alpha=0.5`;
- `0.3 < IoU <= 0.7`: `alpha=0.3`;
- `IoU <= 0.3`: `alpha=0.1`.

The update is `alpha * history + (1 - alpha) * current`, so large motion gives
the current detection 90% weight. Point-cloud ICP and historical-cloud fusion
have been removed: every GraspNet point comes from the current stable mask and
current aligned depth frame.

Depth smoothing uses `0.6 * previous + 0.4 * current` only where current and
previous masks overlap, both depths are valid, and their difference is within
80 mm. New pixels and depth discontinuities immediately use current depth. The
plus sign is intentional; subtraction would not be a valid temporal average.

Grasp output filtering remains independent: translation uses a five-valid-frame
moving average, while rotation uses incremental SO(3) Slerp.

Run a 100-frame hardware check without rendering cost:

```bash
python test_realtime_grasp.py \
  --yolo-device 0 \
  --max-frames 100 \
  --no-visualization
```

For a strict A/B comparison, run the same target motion once with adaptive
mask/depth filtering disabled:

```bash
python test_realtime_grasp.py \
  --yolo-device 0 \
  --max-frames 100 \
  --disable-temporal-filter \
  --no-visualization
```

Useful controls include `--mask-high-iou-alpha`,
`--mask-medium-iou-alpha`, `--mask-low-iou-alpha`,
`--depth-temporal-previous-weight`, `--depth-temporal-max-delta-mm`, and
`--pose-previous-weight`.

## Hardware acceptance measurements

The design target is at least 20% in-range depth coverage inside the mask and
at least 2,048 real points before inference, low tracking delay during motion,
smooth depth and pose changes, and continued
`position`/`rotation`/`score` output. These are **acceptance targets, not
measurements claimed by this repository**. They must be verified on Ubuntu
with the intended D435i, scene geometry, YOLO checkpoint, CUDA device, and
GraspNet checkpoint. A synthetic regression or a development machine without
that hardware cannot establish the physical point count, stability, realtime
FPS, or CUDA inference latency.
