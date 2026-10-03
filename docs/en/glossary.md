# Glossary: Find Concepts by Question

[中文](../glossary.md) | English

[Back to the guide](../../README.en.md) · [Foundations](01_foundations.md) · [Point Clouds and Meshes](02_pointcloud_mesh.md) · [MANO](03_mano_hand.md) · [3DGS](04_nerf_3dgs.md) · [4DGS](05_4dgs.md)

**How to use it**: First look at what a term answers, then do an exercise in the corresponding chapter. These explanations follow this tutorial's conventions and do not replace a dataset or software package's own field definitions. Reference verification date: **2026-10-03**.

## 1 What Data Do I Have?

| Term | Intuitive meaning | Common confusion |
|---|---|---|
| Point cloud | A set of samples with coordinates | No default face connectivity; does not automatically define a solid interior |
| Vertex | A position record in a mesh | Surface vertices are not skeleton joints |
| Face / Triangle | Connectivity defined by vertex indices | The existence of three points does not automatically tell software to connect them as a face |
| Mesh | Vertices, face connectivity, and other attributes | Can be open; not necessarily a usable solid or precise CAD |
| Topology | How points and faces connect | Equal coordinates do not mean equal connectivity |
| Normal | Orientation of a local surface | Point-cloud normals are often estimated from neighborhoods; their signs may be inconsistent |
| UV / Texture | Mapping a surface to a 2D image / texture | Texture detail can be high while geometry remains coarse |
| Voxel | A small grid cell in space | Voxels can store different values such as occupancy, distance, or color |
| Occupancy | Whether a position is inside an object or occupied | Probabilistic and binary occupancy differ; neither directly gives distance to the surface |
| SDF | Signed distance field; positive outside and negative inside in this tutorial | Sign conventions vary; open meshes may lack reliable inside/outside definitions |
| TSDF | Truncated signed distance field, mainly preserving distances near surfaces | Useful for multi-frame depth fusion; does not automatically correct wrong camera poses |
| PLY / OBJ / STL | Common 3D file containers/formats | An extension alone cannot establish units, faces, materials, or custom attributes |
| NPZ | Archive of multiple NumPy arrays | Not a MANO-specific format; check keys, shapes, and meanings |

More: [3D Representations and File Formats](01_foundations.md). Official references: [Open3D geometry tutorials](https://www.open3d.org/docs/0.19.0/tutorial/geometry/index.html), [NumPy savez](https://numpy.org/doc/stable/reference/generated/numpy.savez.html).

## 2 Position, Time, and Cameras

| Term | What it answers | Required conventions |
|---|---|---|
| Coordinate frame | Relative to which origin/axes is a position described? | World, camera, object, robot base, etc. |
| Pose | Position and orientation of a rigid body | Reference frame and transform direction |
| Extrinsics | How do coordinates transform between frames? | World-to-camera or camera-to-world |
| Intrinsics | How does a camera map rays to pixels internally? | Resolution, focal lengths, principal point, specific camera model |
| Distortion | How does a real lens deviate from ideal projection? | Model type, coefficient order, whether already undistorted |
| Depth | Camera-axis Z coordinate in this tutorial's pinhole formulas | Do not equate it directly with ray range from optical center to point |
| Range | How far is the point from the optical center/measurement origin? | Measurement origin, direction, units |
| RGB-D | Color image plus depth image | They may use different pixel grids; require spatial and temporal alignment |
| Back-projection | Recover a 3D point from a pixel and depth | Depth meaning, K, distortion, units |
| Reprojection | Project an estimated 3D point back into an image | Small pixel error does not mean small millimeter error |
| Axis-angle | Direction gives rotation axis; magnitude gives angle | Usually radians; not three Euler-angle components |
| Quaternion | A rotation representation | `wxyz/xyzw`, conventions, normalization |
| Timestamp | At what time was the data captured? | Units, clock source, synchronization error; more than a frame number |
| Scale ambiguity | Projection alone cannot determine overall physical size | Ordinary monocular SfM needs an additional scale reference |

More: [Coordinates and Cameras](01_foundations.md). Official references: [OpenCV calibration](https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html), [COLMAP pose conventions](https://colmap.github.io/format.html).

## 3 What Processing Step Am I Performing?

| Term | Input to output | What it does not also guarantee |
|---|---|---|
| Calibration | Observations of known patterns and other information → camera/sensor parameters | One calibration does not cover later changes to lenses or capture settings |
| SfM | Overlapping images → cameras and sparse 3D structure | Absolute scale, dense surfaces, or a complete object |
| MVS | Multi-view images and cameras → denser depth/surface observations | Transparent, reflective, and textureless regions remain difficult |
| Registration | Multiple geometries → alignment in a common frame | Low alignment residuals do not establish correct correspondences |
| ICP | Iterative Closest Point, a local fine-registration method | Usually needs a reasonable initialization; does not guarantee global correctness |
| RANSAC | Repeatedly sample outlier-containing data and fit candidates | Still depends on the model, thresholds, and enough correct correspondences |
| Downsampling | Reduce point count/density | An overly large scale loses thin walls and detail |
| Surface reconstruction | Points/normals/volumetric data → continuous surface approximation | Filled regions do not equal actual observations |
| Poisson reconstruction | Estimate a smooth surface using oriented normals | Can create inferred surfaces in unobserved regions |
| Watertight | Surface meets closure-related conditions | Watertightness does not establish shape accuracy or a reliable mechanics model |
| Manifold | Local connectivity resembles a normal continuous surface | Non-manifold edges/vertices cause problems for some algorithms |
| Self-intersection | Surface patches pass through other surface patches | Attractive rendering may hide self-intersections |
| Rendering | Compute images from a model and camera | Realistic images do not replace geometry checks |
| Fitting | Adjust model parameters to explain current observations | Fitting training observations well does not establish a unique true state |
| Training | Adjust learnable parameters to reduce an objective loss | 3DGS training often means single-scene optimization, differing from large-scale general-model training |

More: [Point-Cloud Processing Workflow](02_pointcloud_mesh.md). Official references: [COLMAP tutorial](https://colmap.github.io/tutorial.html), [Open3D ICP](https://www.open3d.org/docs/0.19.0/tutorial/pipelines/icp_registration.html), [surface reconstruction](https://www.open3d.org/docs/0.19.0/tutorial/geometry/surface_reconstruction.html).

## 4 MANO Hands and Dynamic Representations

| Term | Meaning in this tutorial | Next question to ask |
|---|---|---|
| MANO | Parametric model of human hand shape and pose | Is this forward generation, or is another network estimating its parameters? |
| Beta / Shape parameters | Control statistical hand-shape variation | What model version and number of shape bases? |
| Theta / Pose parameters | Control skeleton/surface deformation | PCA coefficients, full axis-angle, or rotation matrices? |
| PCA | Principal Component Analysis; fewer coefficients combine original high-dimensional variations | Do dimensions, bases, and means match? |
| LBS | Linear Blend Skinning; deform a surface by weighted bone influences | Which skeleton, weights, and root? |
| Joint | A skeletal joint node | Which of the 16 MANO nodes is it? |
| Landmark | Designated geometric/semantic point, such as a fingertip | Skeleton point, surface vertex, or regressed point? |
| 21 hand landmarks | Common protocol of 16 skeleton points plus five fingertips | What order? Have fingertips already been appended? |
| Retargeting | Map source motion to another kinematic structure | Preserve fingertip targets, motion appearance, or task contact? |
| NeRF | Neural Radiance Field; a continuous field of spatial density and view-dependent appearance | Which implementation, cameras, and training observations? |
| 3DGS | Represent a scene with 3D Gaussians carrying position, shape, opacity, and appearance parameters | What file fields and renderer protocol? |
| Splatting | Blend projected contributions of spatial primitives into an image | It is not simply drawing a center point as an ordinary pixel |
| 4D | Commonly, 3D space plus time | Does geometry, appearance, or a field vary over time? |
| 4DGS | General term/related method names for time-varying Gaussian scenes | 3D Gaussians plus a deformation field, or explicit spatiotemporal Gaussians? |
| Deformation field | Position/time → deformation | Are there correspondence, rigidity, or temporal-consistency constraints? |
| Canonical space | Organize deformation around a common reference state | What mapping direction and reference time? |

More: [MANO](03_mano_hand.md), [NeRF and 3DGS](04_nerf_3dgs.md), [4DGS](05_4dgs.md). See the corresponding chapters for distinctions among methods, papers, and implementations; “4D” alone is not an algorithm description. [MANO official project](https://mano.is.tue.mpg.de/)

## 5 Language for Assessing Results

| Term | What to check when reading a report |
|---|---|
| Ground truth | Direct measurement, manual annotation, optimized fitting, or simulation? What errors and scope of applicability? |
| Confidence | Probability, model score, visibility, point-support density, or a manual threshold? These quantities are not interchangeable. |
| Accuracy | How far from an independent reference? How much error is in the reference itself? |
| Repeatability | Are repeated measurements under the same conditions consistent? Good repeatability does not rule out systematic bias. |
| Resolution | How fine a structure can be distinguished? More vertices do not raise sensor resolution. |
| RMSE | Root mean square error, relatively sensitive to large residuals; specify units, direction, samples, and alignment. |
| Held-out view | Independent view excluded from optimization, used to check whether new observations can be explained |
| Ablation | Change one component and check whether it actually improves results |
| Contact candidate | A location where geometry suggests possible contact; affected by scale, noise, and thresholds |
| Contact force | Requires additional sensing, material/dynamics models, or statistical assumptions; geometry cannot uniquely determine it |
| Reproducibility | Can the result be repeated and checked using data versions, environment, parameters, and steps? |

### Four Self-Checks

1. “There is a model” is insufficient: which representation, units, and coordinates?
2. “Reconstruction succeeded” is insufficient: were scale, geometry, and failure regions checked independently?
3. “The error is small” is insufficient: with which alignment, units, and reference?
4. “It is publicly downloadable” is insufficient: which licenses govern code, data, models, and outputs separately?

---

[Back to home](../../README.en.md) · [Coordinates and Calibration](08_robot_frames_calibration.md) · [Robotics Applications](09_robot_perception_action.md) · [ROS 2 / MuJoCo](10_ros2_mujoco.md)
