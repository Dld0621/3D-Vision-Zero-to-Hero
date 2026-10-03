"""Convert an object-relative TCP target into a flange pose using a known tool offset."""

import math

from transform_utils import (compose, format_point, invert_transform, make_transform,
                             rotation_x, rotation_y, rotation_z)


def grasp_target(t_base_object, t_object_tcp, t_flange_tcp):
    """T_base_flange = T_base_object @ T_object_tcp @ inverse(T_flange_tcp)."""
    t_base_tcp = compose(t_base_object, t_object_tcp)
    return t_base_tcp, compose(t_base_tcp, invert_transform(t_flange_tcp))


def synthetic_example():
    t_base_object = make_transform(rotation_z(math.pi / 2), (.4, .2, .1))
    t_object_tcp = make_transform(rotation_x(math.pi / 2), (.02, 0., .06))
    t_flange_tcp = make_transform(rotation_y(math.pi / 2), (0., 0., .10))
    return grasp_target(t_base_object, t_object_tcp, t_flange_tcp)


def main():
    t_base_tcp, t_base_flange = synthetic_example()
    print("Convention: T_A_B maps column vectors from B to A; translations in metres.")
    print("T_base_flange = T_base_object @ T_object_tcp @ inverse(T_flange_tcp)")
    print("TCP position in base =", format_point(tuple(row[3] for row in t_base_tcp[:3])))
    print("flange position in base =", format_point(tuple(row[3] for row in t_base_flange[:3])))
    print("flange rotation in base (rows):")
    for row in t_base_flange[:3]:
        print(" ", format_point(row[:3]))
    print("Known tool offset: R_flange_tcp = Ry(+90 deg), t_flange_tcp = (0, 0, 0.10) m.")
    print("Pose arithmetic only: no IK, reachability, collision, force, grasp stability or execution validation.")


if __name__ == "__main__":
    main()
