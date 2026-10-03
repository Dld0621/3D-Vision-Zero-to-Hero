# 08 | Robot frames, cameras, and hand–eye calibration

[中文](../08_robot_frames_calibration.md) · [Home](../../README.en.md) · Next: [09 Perception to action](09_robot_perception_action.md)

> Documentation checked: 2026-10-03. OpenCV snippets are syntax/API examples only: not executed, with no dependencies installed. The standard-library exercises in `examples/` check coordinate calculations; they do not constitute a real calibration. See [VALIDATION.en.md](../../VALIDATION.en.md) for execution evidence.

## 1. Every point needs a frame, unit, and time

“Object at `(0.4, 0.1, 0.3)`” is incomplete. A useful record includes values, `frame_id`, metric units, acquisition time, estimated error, and calibration version. A frame consists of an origin and three directed axes; a physical point has different coordinates in different frames.

| Name | Meaning | Frequent mistake |
|---|---|---|
| `base` / `base_link` | Robot base or body reference | A mobile base moves over time |
| `gripper` / `flange` | Rigid end frame used during calibration | Controller pose might describe TCP instead |
| `tcp` | Tool center point, such as finger midpoint | Tool changes invalidate the offset |
| `camera_link` | Camera mounting frame | Its origin need not be the optical center |
| `camera_optical_frame` | Frame used by the camera projection | Its axes differ from robot link axes |
| `target` | Defined calibration-board origin and corners | Corner ordering and square size must agree |
| `object` | Object reference frame | A bounding-box center is not necessarily a CAD origin or center of mass |

Use right-handed coordinates, meters, radians, and seconds. ROS body axes normally point x forward, y left, z up; optical axes point x right, y down, z forward. Check the actual driver definitions. [Official REP-103 source](https://raw.githubusercontent.com/ros-infrastructure/rep/master/rep-0103.rst)

```mermaid
flowchart LR
    O[Point p_C in optical frame] -->|Fixed T_link_C| L[Camera link]
    L -->|Fixed T_G_link| G[End frame G]
    G -->|Time-varying T_B_G| B[Base B]
    B -->|T_W_B for mobile robots| W[World W]
```

These arrows describe coordinate transport. Read tf parent/child relationships through their definitions, rather than inferring transform direction from a drawing alone.

## 2. One transform convention throughout

Points are **column vectors**. `T_A_B` maps coordinates from B into A; equivalently, it represents B's pose in A:

```text
p_A = T_A_B p_B
T_A_B = [R_A_B  t_A_B]
        [0 0 0    1  ]
p_B = [x_B, y_B, z_B, 1]^T
```

The columns of R are B's axes expressed in A. Translation t is B's origin expressed in A. A point receives rotation and translation; a direction or surface normal receives rotation only, with homogeneous last coordinate zero.

```text
T_A_C = T_A_B T_B_C
R_A_C = R_A_B R_B_C
t_A_C = R_A_B t_B_C + t_A_B

T_B_A = inverse(T_A_B)
R_B_A = transpose(R_A_B)
t_B_A = -transpose(R_A_B) t_A_B
```

Adjacent intermediate frame labels cancel. Matrix multiplication is not commutative; inverse translation is generally not simply `-t`. Valid rigid rotations satisfy `R^T R = I` and `det(R)=+1`; determinant −1 is a reflection. Convert units before applying rigid transforms rather than inserting scale into R.

**Numerical exercise:** let B have no rotation and origin `(1,0,0)` m in A; let C have no rotation and origin `(0,2,0)` m in B. C's origin in A is `(1,2,0)`. If B instead rotates +90° about z, C's origin becomes `(-1,0,0)`: rotate the second translation, then add the first.

### Optical-to-link axes

Only for coincident origins with the optical axis aligned with the link's forward axis:

```text
[x_link]   [ 0  0  1 ] [x_optical]
[y_link] = [-1  0  0 ] [y_optical]
[z_link]   [ 0 -1  0 ] [z_optical]
```

Optical `(0,0,1)` m becomes link `(1,0,0)` m. Real mounting offsets and tilt require their measured extrinsics. If the driver already publishes optical-to-link, use that transform once; multiplying this axis conversion again duplicates it.

Run from the repository root:

```bash
python3 examples/rigid_transform_demo.py
```

Input is synthetic transforms embedded in the script; output checks composition, inversion, and optical axes. Unit-axis and round-trip checks come before real measurements.

## 3. Intrinsics, extrinsics, and depth

For an ideal distortion-free pinhole camera:

```text
K = [fx 0 cx; 0 fy cy; 0 0 1]
u = fx X_C / Z_C + cx
v = fy Y_C / Z_C + cy

Given axial depth Z:
X_C = (u-cx) Z / fx
Y_C = (v-cy) Z / fy
Z_C = Z
```

The intrinsic values use pixels; Z uses meters. Intrinsics select a ray, depth selects a point on that ray, and extrinsics transport the point into a robot frame. Raw distorted pixels need the matching lens model and undistortion; rectified pixels need rectified projection parameters. Resize, crop, and digital zoom require updated intrinsics. [Official OpenCV camera model](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)

Axial depth differs from Euclidean distance to the optical center. For a sensor reporting ray range r, use `Z = r / sqrt(1+x_n²+y_n²)`, where `x_n=(u-cx)/fx`, `y_n=(v-cy)/fy`. Check the device contract first.

**Sanity check:** with `fx=fy=500`, `cx=320,cy=240`, pixel `(370,240)` at Z=1 m gives `(0.1,0,1)` m. Its Euclidean range is approximately 1.005 m. A raw value of 1000 meaning millimeters needs a single factor of 0.001.

RGB masks and depth do not automatically share a pixel grid, even when their sizes match. Use an explicitly aligned product, or unproject depth, apply depth-to-RGB extrinsics, reproject, and resolve occlusions.

### Minimum offline interface

| Input | Checks | Output |
|---|---|---|
| Aligned depth and mask | Valid Z, scaling, `(u,v)` versus array `[v,u]` | Object points in optical frame |
| Matching K | Positive focal length, image size, ROI, rectification state | Points that reproject consistently |
| `T_base_camera` at acquisition | Rigid transform, calibration version, matching time | Base-frame points and spatial statistics |

```bash
python3 examples/depth_to_robot_demo.py
```

This uses synthetic depth and masks, reads no camera, and estimates no real object pose. A centroid or axis-aligned bounding box describes position or extent; a full 6D pose requires additional geometry or pose estimation. Continue with [09](09_robot_perception_action.md).

## 4. Time is part of the extrinsic contract

For a camera on the wrist:

```text
p_base(t_capture) = T_base_gripper(t_capture) T_gripper_camera p_camera(t_capture)
```

Combining an image captured at 10:00:00.000 with a wrist pose at 10:00:00.100 mixes instants. At 0.5 m/s, a 0.1 s offset contributes roughly 5 cm of translation error. Rotation also moves points, increasingly with distance. This estimate is an error-budget intuition, not a full uncertainty model.

Define exposure/sample timestamp, device-to-robot clock mapping, synchronization tolerance, maximum age, and drop policy. Hardware triggering reduces exposure mismatch; clock synchronization aligns time bases. Callback arrival includes transport and queuing delay and is not acquisition time. Rolling shutter can require a different pose for each row during fast motion.

Interpolate translation and valid SO(3) orientation separately; do not interpolate 4×4 matrix entries. Interpolation requires bracketing data. Extrapolation needs a separate model and error bound. Online tf2 queries use acquisition stamps, and missing transforms or stale input should reject that action. See [10 ROS2 and MuJoCo](10_ros2_mujoco.md).

## 5. Eye-in-hand: camera on the moving end frame

Let B=base, G=robot end frame, C=camera optical, Q=target. The target stays fixed in the environment; the camera is rigidly attached to G. Each paired observation supplies:

- `T_B_G_i` from forward kinematics/controller pose, with flange/TCP identified.
- `T_C_Q_i` from known metric target corners and their image locations.
- Acquisition times, corner quality, intrinsics, and distortion model.

Solve fixed `X=T_G_C`:

```text
T_B_Q = T_B_G_i X T_C_Q_i   (constant during acquisition)
A X = X D
A = inverse(T_B_G_j) T_B_G_i
D = T_C_Q_j inverse(T_C_Q_i)
```

This relative-motion constraint eliminates the unknown fixed board pose. Writing it down, or checking it using synthetic matrices, does not solve a physical calibration.

`solvePnP` returns target/object-to-camera `rvec,tvec`, so its result is `T_C_Q`. Translation uses the units of `objectPoints`; convert millimeters to meters before combining robot poses. For ambiguous planar solutions, check positive depths, reprojection, and consistency across observations. [Official solvePnP documentation](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html)

**OpenCV + NumPy syntax/documentation only; not executed.** `T_B_G` and `T_C_Q` are equal-length lists of time-paired 4×4 NumPy arrays; all translations use meters:

```python
import cv2
import numpy as np

def split_rt(poses):
    return ([T[:3, :3] for T in poses],
            [T[:3, 3:4] for T in poses])

R_G2B, t_G2B = split_rt(T_B_G)
R_Q2C, t_Q2C = split_rt(T_C_Q)
R_C2G, t_C2G = cv2.calibrateHandEye(
    R_G2B, t_G2B, R_Q2C, t_Q2C,
    method=cv2.CALIB_HAND_EYE_PARK)
T_G_C = np.eye(4)
T_G_C[:3, :3] = R_C2G
T_G_C[:3, 3] = t_C2G.reshape(3)
```

API directions are `gripper2base = T_B_G`, `target2cam = T_C_Q`, and returned `cam2gripper = T_G_C`. Choosing another method does not repair reversed inputs, poor timestamps, or wrong square size. [Official calibrateHandEye definition](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)

## 6. Eye-to-hand: fixed camera, target on the end frame

The camera is fixed relative to B, and Q is rigidly attached to G. Robot motion moves the target. Unknown constants are `X=T_B_C` and `Y=T_G_Q`:

```text
T_B_G_i Y = X T_C_Q_i
inverse(T_B_G_i) X T_C_Q_i = Y
```

The second equation has the same constant-product form as the previous section. Relabeling the API's frames therefore gives: **pass `T_G_B_i` as the first pose list, keep `T_C_Q_i` as the second, and interpret the return as `T_B_C`**. Invert each complete robot transform, including translation; merely swapping argument groups is incorrect.

**OpenCV + NumPy syntax/documentation only; not executed.** Reuse `split_rt` above:

```python
def inv_rigid(T):
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out

T_G_B = [inv_rigid(T) for T in T_B_G]
R_B2G, t_B2G = split_rt(T_G_B)
R_Q2C, t_Q2C = split_rt(T_C_Q)
R_C2B, t_C2B = cv2.calibrateHandEye(
    R_B2G, t_B2G, R_Q2C, t_Q2C,
    method=cv2.CALIB_HAND_EYE_PARK)
T_B_C = np.eye(4)
T_B_C[:3, :3] = R_C2B
T_B_C[:3, 3] = t_C2B.reshape(3)
```

Variable names describe actual physical directions; the API's parameter names do not detect mounting configuration. Check that `Y_i=inverse(T_B_G_i) T_B_C T_C_Q_i` remains constant. If both camera and target are fixed in the environment, robot motion does not change their observations and these data cannot estimate hand–eye this way. With independently known `T_B_Q`, directly compute `T_B_C=T_B_Q inverse(T_C_Q)` and validate it. If B moves, reformulate using a fixed world reference.

## 7. Observability, acquisition, and independent validation

OpenCV states a geometric minimum of two motions with nonparallel rotation axes, hence at least three different poses. Real noisy acquisition requires more. Pure translation, rotation about a single axis, or tiny pose changes can leave the estimate degenerate or unstable. [Official hand–eye requirements](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)

For a teaching exercise, start with 15–25 clear poses; this is a practice suggestion, not a success threshold. Rotate about multiple axes and vary position/distance. Cover image center and edges while keeping corners visible and sharp. Capturing after settling reduces synchronization uncertainty. Board deformation, autofocus/zoom changes, and loose mounts violate assumptions.

Split complete poses into fitting and held-out sets; nearly duplicate adjacent frames provide weak independence. Keep evidence for:

| Validation | Procedure | Detects |
|---|---|---|
| Held-out intrinsic reprojection | Corner pixel residuals by image region/view | Camera model and distortion errors |
| Held-out hand–eye closure | Compute `T_B_Q_i` for eye-in-hand or `T_G_Q_i` for eye-to-hand | Direction, pose, mounting, and timing inconsistency |
| Independent spatial measurement | Known points/distances at new locations, compared with independent measurement or fixtures | Bias, scale error, workspace extrapolation |
| Repeat acquisition/mounting | Re-estimate X across sessions | Repeatability and mechanical stability |

For closure, take a reference `T_ref` established **only from fitting data**, and compute `E_i=inverse(T_ref) T_i`. Translation error is `||t_E||`; rotation error is `acos(clamp((trace(R_E)-1)/2,-1,1))`. Report median, high percentile, maximum, sample count, and workspace coverage. Predict held-out corners using fitting-derived installation quantities and robot poses. A low error after fitting fresh PnP to every held-out image alone does not independently establish hand–eye accuracy.

Acceptance thresholds come from the task budget: grasp clearance, TCP error, depth noise, calibration error, and latency all contribute. A universal “less than one pixel” criterion cannot guarantee “less than one millimeter.” Robot execution has additional checks in [09](09_robot_perception_action.md).

## 8. Troubleshooting and practice

| Symptom | First check |
|---|---|
| Mirrored points or wrong axes | Handedness, optical/link, duplicate conversion |
| Roughly 1000× error | Mixed mm/m in target, robot, or depth |
| Works stationary, drifts in motion | Acquisition stamps, clock mapping, exposure/arrival delay |
| Large errors in some image/workspace regions | Distortion, crop-adjusted K, RGB-depth registration, coverage |
| Solver returns a matrix with poor results | Rotation diversity, corner ordering, rigid mount, input direction |
| Error resembles tool length | Flange/TCP definition and tool offset |

1. Check optical-to-link unit axes; add a 0.02 m mounting offset and verify inverse round trips. Record directions and error.
2. Reproject `(370,240,1m)` to its original pixel; deliberately treat the depth as 1000 m and identify which checks catch it.
3. Write both full calibration loops and cancel intermediate labels. Explain why eye-to-hand requires inverted robot poses.
4. Compare z-only rotation with a multi-axis acquisition plan; propose independent held-out and ruler/fixture measurements.
5. Estimate translation error for 20 ms at 0.2 m/s: about 4 mm. Explain why angular motion and target distance need additional consideration.

Completion means you can attach frame, unit, and time to every output point, explain both calibration input/output directions, and produce independent error evidence. Continue with [09 | Perception to robot action](09_robot_perception_action.md).
