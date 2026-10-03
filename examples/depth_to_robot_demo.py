"""Back-project synthetic depth and transform points into a robot base frame."""

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
from numbers import Real

from transform_utils import (apply_transform, compose, format_point, make_transform,
                             optical_to_body_transform, rotation_z, _finite_number)


@dataclass(frozen=True)
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def validate(self):
        for name in ("width", "height"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("fx", "fy", "cx", "cy"):
            _finite_number(getattr(self, name), name)
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("focal lengths must be positive")


def depth_to_camera(depth, intrinsics, depth_kind="axial", depth_unit_m=1.0):
    """Return [(u, v, point_camera_metres), ...], masking missing/invalid depths.

    axial: the depth is camera Z. ray: the depth is Euclidean distance along
    the viewing ray. Pixels use u right, v down, integer pixel centres.
    Input must already be undistorted and aligned to these intrinsics.
    """
    intrinsics.validate()
    depth_unit_m = _finite_number(depth_unit_m, "depth_unit_m")
    if depth_unit_m <= 0:
        raise ValueError("depth_unit_m must be positive")
    if depth_kind not in ("axial", "ray"):
        raise ValueError("depth_kind must be axial or ray")
    try:
        if len(depth) != intrinsics.height or any(len(row) != intrinsics.width for row in depth):
            raise ValueError("depth dimensions must match intrinsics width and height")
    except TypeError as exc:
        raise ValueError("depth must be a rectangular sequence of rows") from exc
    points = []
    for v, row in enumerate(depth):
        for u, sample in enumerate(row):
            if sample is None:
                continue
            if isinstance(sample, bool) or not isinstance(sample, Real):
                raise ValueError("depth samples must be real numbers or None")
            distance = float(sample) * depth_unit_m
            if not math.isfinite(distance) or distance <= 0:
                continue
            x_ray = (u - intrinsics.cx) / intrinsics.fx
            y_ray = (v - intrinsics.cy) / intrinsics.fy
            z = distance if depth_kind == "axial" else distance / math.sqrt(x_ray*x_ray + y_ray*y_ray + 1)
            point = (x_ray * z, y_ray * z, z)
            if not all(math.isfinite(value) for value in point):
                raise ValueError("back-projected coordinates are not finite")
            points.append((u, v, point))
    return points


def synthetic_example(depth_kind="axial"):
    # Millimetres are converted to metres once, before geometry operations.
    depth_mm = ((0., 1000., 0.), (2000., float("nan"), 2000.), (0., 1000., -1.))
    intrinsics = Intrinsics(width=3, height=3, fx=2., fy=2., cx=1., cy=1.)
    camera_points = depth_to_camera(depth_mm, intrinsics, depth_kind, depth_unit_m=.001)
    t_base_mount = make_transform(rotation_z(math.pi / 2), (.5, -.2, 1.))
    t_base_camera = compose(t_base_mount, optical_to_body_transform())
    base_points = [(u, v, apply_transform(t_base_camera, point)) for u, v, point in camera_points]
    return camera_points, base_points


def write_ascii_ply(output_path, points):
    """Create a vertex-only PLY in metres at a caller-chosen path; refuse overwrite."""
    coordinates = [tuple(_finite_number(v, "PLY coordinate") for v in point) for point in points]
    if any(len(point) != 3 for point in coordinates):
        raise ValueError("PLY points must have 3 coordinates")
    with Path(output_path).open("x", encoding="ascii", newline="\n") as stream:
        stream.write("ply\nformat ascii 1.0\ncomment coordinates in metres; frame robot_base\n")
        stream.write(f"element vertex {len(coordinates)}\n")
        stream.write("property double x\nproperty double y\nproperty double z\nend_header\n")
        for point in coordinates:
            stream.write(" ".join(f"{v:.9f}" for v in point) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth-kind", choices=("axial", "ray"), default="axial")
    parser.add_argument("--ply", type=Path, help="optional new ASCII PLY path; parent directory must exist")
    args = parser.parse_args()
    camera_points, base_points = synthetic_example(args.depth_kind)
    print(f"Synthetic input: 3x3 depth in mm; kind={args.depth_kind}; output positions in metres.")
    print("Intrinsics: fx=fy=2 px, cx=cy=1 px; no distortion; integer pixel centres.")
    print(f"valid samples = {len(camera_points)}; masked samples = {9 - len(camera_points)}")
    for (u, v, point_camera), (_, _, point_base) in zip(camera_points, base_points):
        print(f"pixel ({u}, {v}): camera {format_point(point_camera)} -> base {format_point(point_base)}")
    if args.ply:
        write_ascii_ply(args.ply, [point for _, _, point in base_points])
        print(f"Wrote {len(base_points)} base-frame vertices to {args.ply}")
    print("Synthetic geometry only: no sensor calibration, localization, IK or robot command validation.")


if __name__ == "__main__":
    main()
