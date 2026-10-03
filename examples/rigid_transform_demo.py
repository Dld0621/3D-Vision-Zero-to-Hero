"""Compose and invert rigid transforms without third-party packages."""

import math

from transform_utils import (apply_transform, compose, format_point, invert_transform,
                             make_transform, optical_to_body_transform, rotation_z)


def demo_values():
    t_a_b = make_transform(rotation_z(math.pi / 2), (1., 2., .5))
    t_b_c = make_transform(rotation_z(-math.pi / 2), (.2, .3, .4))
    point_c = (.4, -.1, .2)
    t_a_c = compose(t_a_b, t_b_c)
    point_a = apply_transform(t_a_c, point_c)
    recovered_c = apply_transform(invert_transform(t_a_c), point_a)
    body_point = apply_transform(optical_to_body_transform(), (1., 2., 3.))
    return point_a, recovered_c, body_point


def main():
    point_a, recovered_c, body_point = demo_values()
    print("Convention: p_A = T_A_B @ p_B, column vectors; all positions in metres.")
    print("T_A_C = T_A_B @ T_B_C; R_A_B = Rz(+90 deg), R_B_C = Rz(-90 deg).")
    print("point_A =", format_point(point_a))
    print("inverse recovers point_C =", format_point(recovered_c))
    print("optical (1, 2, 3) -> body =", format_point(body_point))
    print("Optical axes: right, down, forward. Body axes: forward, left, up.")


if __name__ == "__main__":
    main()
