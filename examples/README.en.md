# Runnable CPU examples

[中文版本](README.md)

These teaching scripts use only the Python standard library for small geometry calculations on the CPU. They require Python 3.9 or newer. No NumPy, GPU, weights, training, network connection, or dependency installation is needed. Run the following commands from the repository root. Substitute `python` for `python3` if that is your interpreter command.

```bash
python3 examples/rigid_transform_demo.py
python3 examples/depth_to_robot_demo.py
python3 examples/grasp_pose_demo.py
python3 -m unittest discover -s examples -p 'test_*.py' -v
```

## 1. Rigid transforms: state the coordinate convention first

`T_A_B` maps **column vectors in frame B** into A: `p_A = T_A_B @ p_B`. Homogeneous points have a final element of 1. Translations and point coordinates use metres. Composition is `T_A_C = T_A_B @ T_B_C`; inversion uses `Rᵀ` and `-Rᵀ t`. The two input rotations in the script are `Rz(+90°)` and `Rz(-90°)`, so the example exercises rotation as well as translation.

Expected key output:

```text
point_A = (1.100000, 2.100000, 1.100000)
inverse recovers point_C = (0.400000, -0.100000, 0.200000)
optical (1, 2, 3) -> body = (3.000000, -1.000000, -2.000000)
```

Camera optical axes are `+x` right, `+y` down, `+z` forward. Body axes are `+x` forward, `+y` left, `+z` up. The conversion is `(x,y,z) → (z,-x,-y)`. This changes the axis convention; an actual camera's mounting rotation and translation need separate calibration. The helpers check matrix dimensions, finite values, rotation orthogonality, determinant `+1`, and the homogeneous matrix's final row. Callers remain responsible for matching frame names correctly.

## 2. Depth → camera points → robot base points

Input is a synthetic 3×3 depth image in millimetres:

```text
   0  1000     0
2000   NaN  2000
   0  1000    -1
```

Intrinsics are `fx=fy=2 px, cx=cy=1 px`. Pixel centres use integer `(u,v)`, with `u` right and `v` down. The input is assumed to be undistorted and aligned to these intrinsics. Its dimensions must match the intrinsics' width and height. `None`, NaN, either infinity, zero, and negative depth samples are masked. Strings and other unsupported sample types are rejected. `depth_unit_m=0.001` converts millimetres to metres once during back-projection.

The default `--depth-kind axial` means camera Z depth:

```text
x = (u-cx) * Z / fx
y = (v-cy) * Z / fy
z = Z
```

The robot transform is `T_base_camera = T_base_mount @ T_mount_camera`. `T_mount_camera` uses the optical→body conversion above; `T_base_mount` uses `Rz(+90°)` and translation `(0.5,-0.2,1.0) m`. There are 4 valid points and 5 masked samples:

| Pixel `(u,v)` | Camera point / m | Base point / m |
|---|---|---|
| `(1,0)` | `(0,-0.5,1)` | `(0.5,0.8,1.5)` |
| `(0,1)` | `(-1,0,2)` | `(-0.5,1.8,1)` |
| `(2,1)` | `(1,0,2)` | `(1.5,1.8,1)` |
| `(1,2)` | `(0,0.5,1)` | `(0.5,0.8,0.5)` |

For Euclidean distance along the viewing ray, run the command below. The script normalizes `(x_ray,y_ray,1)` before multiplying by distance. A depth value cannot be interpreted as both camera Z and ray distance at the same time.

```bash
python3 examples/depth_to_robot_demo.py --depth-kind ray
```

This still produces 4 valid points. For example, pixel `(1,0)` becomes camera point `(0,-0.447214,0.894427)` and base point `(0.5,0.694427,1.447214)`. An independent test also uses `fx=1, cx=0, u=1, distance=2`: axial depth gives `(2,0,2)`; ray distance gives `(√2,0,√2)`, whose vector length is 2.

You can optionally create a vertex-only ASCII PLY at a path you choose. Its parent directory must exist; existing files are not overwritten. Vertices are in the robot base frame and use metres; a header comment records these conventions. Replace the example path as needed:

```bash
mkdir -p output
python3 examples/depth_to_robot_demo.py --ply output/synthetic_base.ply
```

The expected PLY has 4 vertices and the script prints `Wrote 4 base-frame vertices to output/synthetic_base.ply`. PLY generally does not enforce a unit convention, so communicate the units and frame when passing it to another tool.

## 3. Object pose → TCP target → flange target

Given an object pose `T_base_object`, an object-relative TCP target `T_object_tcp`, and tool calibration `T_flange_tcp`:

```text
T_base_tcp = T_base_object @ T_object_tcp
T_base_flange = T_base_tcp @ inverse(T_flange_tcp)
```

The example inputs are respectively: `Rz(+90°), t=(0.4,0.2,0.1)`; `Rx(+90°), t=(0.02,0,0.06)`; `Ry(+90°), t=(0,0,0.10)`. Translations use metres. Invert the tool offset together with its rotation; subtracting a tool-frame offset directly from a base-frame TCP position is insufficient.

Expected key output:

```text
TCP position in base = (0.400000, 0.220000, 0.160000)
flange position in base = (0.400000, 0.320000, 0.160000)
flange rotation in base (rows):
  (1.000000, 0.000000, 0.000000)
  (0.000000, 0.000000, -1.000000)
  (0.000000, 1.000000, 0.000000)
```

This target is a pose arithmetic result. It does not establish an IK solution, reachability, collision clearance, grasp stability, correct contact force, or executable robot motion. The script does not connect to a robot.

## Verification and boundaries

`test_examples.py` contains 12 tests. Independent fixed numbers check nonidentity rotation, inversion, composition order, optical axes, millimetre conversion, invalid-depth masking, the difference between Z depth and ray distance, the TCP/flange chain, PLY contents, and refusal to overwrite. Tests also reject invalid matrices and dimensions. Expected result: `Ran 12 tests ... OK`. These tests and all three default script commands were run on Python 3.14.7 for this update; the repository's [VALIDATION.en.md](../VALIDATION.en.md) summarizes verification records.

`mujoco_pose_demo.xml` is an illustrative model for further learning. It includes gravity, a floor, and a mocap body named `tracked_object`. Appearance uses an ellipsoid excluded from collision; a separate box provides a collision primitive. Mocap prescribes the object's pose, so the object does not fall under gravity as a free rigid body. The gravity setting does not change this. Tests only parse XML syntax using the standard library and check names/attributes; **MuJoCo was not loaded or run**. The model has no robot arm, controller, camera tracking integration, dynamic object parameter identification, or verified physics results.

These examples verify geometry arithmetic on small synthetic inputs. Real sensor calibration, time synchronization, scale recovery, distortion, noise, object recognition, robot control, and simulation physics require independent verification.
