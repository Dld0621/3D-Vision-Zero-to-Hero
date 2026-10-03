"""Small rigid-transform helpers: T_A_B maps column vectors from B into A."""

import math
from numbers import Real


def _finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return value


def _vector(values, size, name):
    try:
        if len(values) != size:
            raise ValueError(f"{name} must have {size} elements")
        return tuple(_finite_number(v, name) for v in values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence") from exc


def validate_rotation(rotation):
    """Require a finite 3x3 proper rotation (orthonormal, determinant +1)."""
    try:
        if len(rotation) != 3:
            raise ValueError("rotation must have 3 rows")
        rows = tuple(_vector(row, 3, "rotation row") for row in rotation)
    except TypeError as exc:
        raise ValueError("rotation must be a 3x3 sequence") from exc
    for i in range(3):
        for j in range(3):
            dot = sum(rows[i][k] * rows[j][k] for k in range(3))
            if not math.isclose(dot, float(i == j), abs_tol=1e-9):
                raise ValueError("rotation must be orthonormal")
    a, b, c = rows
    determinant = (a[0] * (b[1] * c[2] - b[2] * c[1])
                   - a[1] * (b[0] * c[2] - b[2] * c[0])
                   + a[2] * (b[0] * c[1] - b[1] * c[0]))
    if not math.isclose(determinant, 1.0, abs_tol=1e-9):
        raise ValueError("rotation must have determinant +1; reflections are invalid")
    return rows


def make_transform(rotation, translation):
    """Build T_A_B; translations and point coordinates must use the same units."""
    rotation = validate_rotation(rotation)
    translation = _vector(translation, 3, "translation")
    return tuple(rotation[i] + (translation[i],) for i in range(3)) + ((0., 0., 0., 1.),)


def validate_transform(transform):
    try:
        if len(transform) != 4:
            raise ValueError("transform must have 4 rows")
        rows = tuple(_vector(row, 4, "transform row") for row in transform)
    except TypeError as exc:
        raise ValueError("transform must be a 4x4 sequence") from exc
    if any(not math.isclose(rows[3][i], v, abs_tol=1e-9)
           for i, v in enumerate((0., 0., 0., 1.))):
        raise ValueError("homogeneous last row must be [0, 0, 0, 1]")
    validate_rotation(tuple(row[:3] for row in rows[:3]))
    return rows


def compose(t_a_b, t_b_c):
    """Return T_A_C = T_A_B @ T_B_C. Frame labels are the caller's responsibility."""
    left, right = validate_transform(t_a_b), validate_transform(t_b_c)
    return tuple(tuple(sum(left[i][k] * right[k][j] for k in range(4))
                       for j in range(4)) for i in range(4))


def apply_transform(transform, point):
    """Apply a rigid transform to one 3D point, using homogeneous w=1."""
    transform, point = validate_transform(transform), _vector(point, 3, "point")
    return tuple(sum(transform[i][j] * point[j] for j in range(3)) + transform[i][3]
                 for i in range(3))


def invert_transform(transform):
    """Return T_B_A using R transpose and -R transpose times t."""
    transform = validate_transform(transform)
    inverse_rotation = tuple(tuple(transform[j][i] for j in range(3)) for i in range(3))
    inverse_translation = tuple(-sum(inverse_rotation[i][j] * transform[j][3]
                                     for j in range(3)) for i in range(3))
    return make_transform(inverse_rotation, inverse_translation)


def rotation_x(angle_rad):
    angle_rad = _finite_number(angle_rad, "angle_rad")
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return ((1., 0., 0.), (0., c, -s), (0., s, c))


def rotation_z(angle_rad):
    angle_rad = _finite_number(angle_rad, "angle_rad")
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return ((c, -s, 0.), (s, c, 0.), (0., 0., 1.))


def rotation_y(angle_rad):
    angle_rad = _finite_number(angle_rad, "angle_rad")
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return ((c, 0., s), (0., 1., 0.), (-s, 0., c))


def optical_to_body_transform():
    """Optical (+x right,+y down,+z forward) to body (+x forward,+y left,+z up)."""
    return make_transform(((0., 0., 1.), (-1., 0., 0.), (0., -1., 0.)), (0., 0., 0.))


def format_point(point):
    # Suppress signed zero in display only; do not modify stored coordinates.
    return "(" + ", ".join(f"{(0. if abs(v) < .5e-6 else v):.6f}" for v in point) + ")"
