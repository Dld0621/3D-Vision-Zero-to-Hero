"""Independent numeric checks for teaching examples; standard library only."""

import math
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from depth_to_robot_demo import Intrinsics, depth_to_camera, synthetic_example as depth_example, write_ascii_ply
from grasp_pose_demo import synthetic_example as grasp_example
from rigid_transform_demo import demo_values
from transform_utils import (apply_transform, compose, invert_transform, make_transform,
                             optical_to_body_transform, rotation_z)


class ExamplesTest(unittest.TestCase):
    def assertVector(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for value, target in zip(actual, expected):
            self.assertAlmostEqual(value, target, places=9)

    def assertMatrix(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for row, target in zip(actual, expected):
            self.assertVector(row, target)

    def test_nonidentity_rotation_and_inverse_numeric(self):
        transform = make_transform(rotation_z(math.pi / 2), (1., 2., 3.))
        self.assertVector(apply_transform(transform, (2., 0., 1.)), (1., 4., 4.))
        self.assertMatrix(invert_transform(transform),
                          ((0., 1., 0., -2.), (-1., 0., 0., 1.),
                           (0., 0., 1., -3.), (0., 0., 0., 1.)))

    def test_composition_order_changes_result(self):
        rotated = make_transform(rotation_z(math.pi / 2), (1., 2., 3.))
        translated = make_transform(((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)), (1., 0., 0.))
        self.assertVector(apply_transform(compose(rotated, translated), (0., 0., 0.)), (1., 3., 3.))
        self.assertVector(apply_transform(compose(translated, rotated), (0., 0., 0.)), (2., 2., 3.))

    def test_rigid_demo_expected_values(self):
        point_a, recovered, body = demo_values()
        self.assertVector(point_a, (1.1, 2.1, 1.1))
        self.assertVector(recovered, (.4, -.1, .2))
        self.assertVector(body, (3., -1., -2.))

    def test_optical_axis_convention(self):
        transform = optical_to_body_transform()
        self.assertVector(apply_transform(transform, (1., 0., 0.)), (0., -1., 0.))
        self.assertVector(apply_transform(transform, (0., 1., 0.)), (0., 0., -1.))
        self.assertVector(apply_transform(transform, (0., 0., 1.)), (1., 0., 0.))

    def test_reject_bad_rotations_transforms_and_points(self):
        bad_rotations = [((1., 0., 0.), (0., 1., 0.), (0., 0., -1.)),
                         ((2., 0., 0.), (0., 1., 0.), (0., 0., 1.)),
                         ((float("nan"), 0., 0.), (0., 1., 0.), (0., 0., 1.)),
                         ((1., 0.), (0., 1.))]
        for rotation in bad_rotations:
            with self.subTest(rotation=rotation), self.assertRaises(ValueError):
                make_transform(rotation, (0., 0., 0.))
        identity = make_transform(rotation_z(0.), (0., 0., 0.))
        with self.assertRaises(ValueError):
            make_transform(rotation_z(0.), (0., float("inf"), 0.))
        with self.assertRaises(ValueError):
            apply_transform(identity, (1., 2.))
        with self.assertRaises(ValueError):
            invert_transform(identity[:3])
        with self.assertRaises(ValueError):
            invert_transform(identity[:3] + ((0., 0., 1., 1.),))

    def test_axial_depth_and_base_numeric(self):
        camera_points, base_points = depth_example()
        self.assertEqual([(u, v) for u, v, _ in camera_points], [(1, 0), (0, 1), (2, 1), (1, 2)])
        expected_camera = [(0., -.5, 1.), (-1., 0., 2.), (1., 0., 2.), (0., .5, 1.)]
        expected_base = [(.5, .8, 1.5), (-.5, 1.8, 1.), (1.5, 1.8, 1.), (.5, .8, .5)]
        for (_, _, point), expected in zip(camera_points, expected_camera):
            self.assertVector(point, expected)
        for (_, _, point), expected in zip(base_points, expected_base):
            self.assertVector(point, expected)

    def test_ray_distance_differs_from_axial_depth(self):
        intrinsics = Intrinsics(2, 1, 1., 1., 0., 0.)
        axial = depth_to_camera(((0., 2.),), intrinsics, "axial")
        ray = depth_to_camera(((0., 2.),), intrinsics, "ray")
        self.assertVector(axial[0][2], (2., 0., 2.))
        self.assertVector(ray[0][2], (math.sqrt(2.), 0., math.sqrt(2.)))
        self.assertAlmostEqual(math.sqrt(sum(v*v for v in ray[0][2])), 2.)

    def test_mask_invalid_depth_samples(self):
        intrinsics = Intrinsics(7, 1, 1., 1., 0., 0.)
        points = depth_to_camera(((None, float("nan"), float("inf"), -float("inf"), 0., -1., 1.),), intrinsics)
        self.assertEqual(len(points), 1)
        self.assertVector(points[0][2], (6., 0., 1.))

    def test_reject_bad_depth_dimensions_and_parameters(self):
        intrinsics = Intrinsics(2, 2, 1., 1., 0., 0.)
        for depth in (((1.,), (1.,)), ((1., 2.),), ((1., 2.), (1.,))):
            with self.subTest(depth=depth), self.assertRaises(ValueError):
                depth_to_camera(depth, intrinsics)
        for intrinsics in (Intrinsics(1, 1, 0., 1., 0., 0.),
                           Intrinsics(1, 1, 1., 1., float("nan"), 0.),
                           Intrinsics(0, 1, 1., 1., 0., 0.)):
            with self.subTest(intrinsics=intrinsics), self.assertRaises(ValueError):
                depth_to_camera(((1.,),), intrinsics)
        valid = Intrinsics(1, 1, 1., 1., 0., 0.)
        with self.assertRaises(ValueError):
            depth_to_camera(((1.,),), valid, "disparity")
        with self.assertRaises(ValueError):
            depth_to_camera(((1.,),), valid, depth_unit_m=0.)
        with self.assertRaises(ValueError):
            depth_to_camera((("unknown",),), valid)

    def test_grasp_tcp_and_flange_numeric(self):
        tcp, flange = grasp_example()
        self.assertMatrix(tcp, ((0., 0., 1., .4), (1., 0., 0., .22),
                                (0., 1., 0., .16), (0., 0., 0., 1.)))
        self.assertMatrix(flange, ((1., 0., 0., .4), (0., 0., -1., .32),
                                   (0., 1., 0., .16), (0., 0., 0., 1.)))

    def test_ply_artifact_coordinates_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "points.ply"
            write_ascii_ply(path, ((1., 2., 3.), (-.5, 0., 1.)))
            lines = path.read_text(encoding="ascii").splitlines()
            self.assertIn("element vertex 2", lines)
            self.assertIn("comment coordinates in metres; frame robot_base", lines)
            self.assertEqual(lines[-2:], ["1.000000000 2.000000000 3.000000000", "-0.500000000 0.000000000 1.000000000"])
            with self.assertRaises(FileExistsError):
                write_ascii_ply(path, ((0., 0., 0.),))
            with self.assertRaises(ValueError):
                write_ascii_ply(Path(directory) / "bad.ply", ((float("nan"), 0., 0.),))

    def test_mujoco_xml_syntax_and_named_body_only(self):
        root = ET.parse(Path(__file__).with_name("mujoco_pose_demo.xml")).getroot()
        self.assertEqual(root.tag, "mujoco")
        body = root.find("./worldbody/body[@name='tracked_object']")
        self.assertIsNotNone(body)
        self.assertEqual(body.attrib["mocap"], "true")
        self.assertEqual(body.find("./geom[@name='object_visual']").attrib["contype"], "0")
        self.assertEqual(body.find("./geom[@name='object_collision']").attrib["contype"], "1")


if __name__ == "__main__":
    unittest.main()
