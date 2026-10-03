# Validation record

[简体中文](VALIDATION.md) | **English**

Checked on 2026-10-03. This record distinguishes execution, syntax/static checks, and official-documentation review. Successful synthetic examples do not establish that a real sensor or robot has passed acceptance tests.

## Scope of this version

| Item | Status | Boundary |
|---|---|---|
| Complete Chinese/English homepages, 10 chapters, glossary, and example guide | Cross-checked | Original Chinese paths preserved; bidirectional language navigation; equations, commands, sources, and execution status aligned |
| Official papers, software documentation, and relevant source code | Documentation reviewed | Official sources appear in each chapter; recheck changing versions/APIs in the actual environment |
| Relative Markdown links, heading anchors, fences, and bilingual page coverage | Static checks | Repeat with `scripts/check_docs.py`; external software is not executed |
| Python / Bash tutorial code | Syntax checks | Python AST / `bash -n`; does not establish dependency, API, or GPU availability |
| Nine-point depth-unprojection exercise | Executed and verified | Chapter 01 standard-library code: 9 points, Z=1m, width/height=0.02m; synthetic numbers |
| Three CPU geometry scripts and ray-distance mode | Executed and verified | Rigid transforms, depth to base, and TCP/flange chain; no sensor or robot connection |
| Unit tests | 12/12 passed | Independent expected numbers, invalid inputs, axial/ray depth, transform order, and PLY overwrite refusal; the XML test checks syntax/names only |
| PLY command-line export | Executed and verified | Four expected base-frame vertices generated in a temporary directory, then cleaned up; no models or real data |
| Original workflow and representation illustrations | Checked | SVGs parse; the new workflow was rendered and visually inspected locally; conceptual illustrations are not scans |
| Open3D synthetic-sphere reconstruction exercise | Not executed | Requires a compatible Open3D environment |
| MANO forward and retargeting workflows | Not executed | Require licensed models, compatible dependencies, and robot kinematics; models are not distributed |
| Hand-eye calibration, ROS 2, MoveIt 2 | Documentation/syntax checked; no integration run | No acceptance test with real calibration images, robot poses, or communication links |
| MuJoCo XML and Python snippets | XML/Python syntax checked; MuJoCo not run | No model compilation, simulation stepping, collision, or contact validation |
| NeRF / 3DGS / 4DGS training and data capture | Not executed | Based on official implementations; CUDA/drivers/dependencies, performance, and quality require separate validation |
| Real-robot grasping, tracking, force control, and safety | Not executed | No hardware success-rate or ground-truth force report |

## Actual CPU environment and commands

Execution environment: **macOS 27.0.1, arm64, CPython 3.14.7**. Examples declare Python 3.9 as their minimum version; this run did not separately test every Python version, Windows, or Linux. No third-party dependencies were installed for these examples. All inputs come from synthetic numbers in the code.

Run from the repository root:

```bash
python3 examples/rigid_transform_demo.py
python3 examples/depth_to_robot_demo.py
python3 examples/depth_to_robot_demo.py --depth-kind ray
python3 examples/grasp_pose_demo.py
python3 -m unittest discover -s examples -p 'test_*.py' -v
python3 scripts/check_docs.py
```

Representative verified results:

- The rigid chain produces `point_A=(1.1,2.1,1.1)m`; its inverse recovers `(0.4,-0.1,0.2)m`. The axis-convention transform maps optical `(1,2,3)` to body `(3,-1,-2)`.
- The 3×3 depth image in millimetres gives four valid points and masks five samples. The first point maps from camera `(0,-0.5,1)m` to base `(0.5,0.8,1.5)m`.
- Interpreting the same numbers as ray distances changes the first camera point to `(0,-0.447214,0.894427)m`, demonstrating the distinction between Z depth and range.
- The grasp chain produces TCP position `(0.4,0.22,0.16)m` and flange position `(0.4,0.32,0.16)m`. This does not establish IK, collision, contact, or execution feasibility.
- Unit tests report `Ran 12 tests` / `OK`. PLY export was also run separately and compared with four analytical coordinates.

See the [example guide](examples/README.en.md) and source code for complete inputs, per-point outputs, and matrices. The Chapter 01 nine-point code was separately run in a temporary directory; its PLY point count, Z values, and bounds were checked.

## Reproducing static and bilingual checks

`python3 scripts/check_docs.py` checks Markdown file links, heading anchors, paired fences, Python/Bash/JSON/XML snippet syntax, SVG XML syntax, Chinese/English page coverage, language navigation, and code-block types/counts. It also flags source-link differences for manual review. **It does not contact external websites or establish translation semantics or external API correctness.**

This update also compares translated sections, tables, equations, code behavior, and sources. External-link availability depends on networks and site policies. Access denials or bot checks do not by themselves establish that a paper/API has disappeared; use the same official project's index or source entry point. Linked official sources support the technical review; recheck them when changing versions.

The external-link check covered 107 addresses with heading fragments removed: 102 responded successfully; four OpenCV pages returned HTTP 403 to automated requests but were readable through web retrieval. The old CloudCompare platform page also returned 403, so both languages now link to readable platform information in its official GitHub repository. The final documentation contains 106 such addresses. Reachability checks do not constitute technical review of every page in full.

## Submitting new execution records

Record the OS, CPU/GPU, driver, Python/library versions, code commit, input source and license, commands, random seed where relevant, output files, and acceptance results. Distinguish installation success, program startup, training completion, and correct results.

Remove personal paths, tokens, private images, and other nonpublic information from raw logs. Minimal reproducible failures are welcome; do not remove failed frames or mix training views into evaluation to improve scores.

[Back to the homepage](README.en.md) · [Examples](examples/README.en.md)
