[中文](../06_workflows.md) | English

# 06 | From Input to Acceptance: Five Software Workflows

> **Validation boundary: official documentation and source were checked on 2026-10-03; the following installation, training, and GPU commands have not been run on the user's machine. This repository has no measured training times, peak GPU memory usage, or quality benchmarks.**
>
> All paths are fictional examples for a public tutorial. Work in a new, empty working directory and preserve original images; do not commit source material, model weights, or runtime environments to GitHub. Code branches change: record the commit, environment, and local `--help` when running commands.

## 0. Choose a Route before Installing Every Tool

| Your goal/device | Starting route | Boundary |
|---|---|---|
| Understand geometry on Windows/Linux/Mac | A: photos → COLMAP sparse model | CPU can run SfM; large datasets may still be slow |
| Reproduce classic 3DGS with a compatible NVIDIA GPU | A → B: official graphdeco | Historical CUDA environment; compatibility with new GPUs is not guaranteed |
| Build a browsable Gaussian scene first, on Linux CUDA | C: Nerfstudio Splatfacto | More complete engineering workflow, but not equivalent to original 3DGS |
| Study dynamic reconstruction with Linux CUDA | D: start with a hustvl synthetic sequence | Then use synchronized multiple cameras; do not start with arbitrary phone video |
| Only have a Mac | E: capture, organize, CPU SfM, view | This tutorial does not promise Metal/MPS training for these official CUDA trainers |

For **appearance assets**, use B/C; for **measurement and collision surfaces**, focus on A's geometry branch and the [mesh chapter](02_pointcloud_mesh.md). Visual, measurement, and physical models can be combined, but their acceptance criteria differ.

### GPU Check: Distinguish Three “CUDA Versions”

The CUDA version supported by the driver, the runtime bundled with a PyTorch wheel, and the toolkit used to compile extensions are three different things.

```bash
nvidia-smi
nvcc --version
python -c "import torch; print('torch=',torch.__version__,'runtime=',torch.version.cuda,'available=',torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

A successful `nvidia-smi` does not prove `nvcc` is installed; `torch.cuda.is_available()` being true does not prove a particular custom CUDA extension can compile for your GPU. First check the driver/compiler compatibility requirements for the corresponding toolkit in the [NVIDIA Linux installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) or [Windows installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/).

---

## A. Static Photos → COLMAP → Sparse Model, with Optional Dense/Mesh

### Input and Capture

- Input: clear, overlapping photos of the same static object/scene from multiple views
- For a first exercise, a few dozen to one or two hundred photos at moderate resolution is a teaching-scale suggestion, not a success threshold
- Keep the object, background, and lighting as unchanged as possible; do not rotate the same object while walking around it and retaining a stationary background
- Fix zoom and resolution; photograph a circuit around the object, then add higher/lower views; avoid only rotating the camera in place
- Remove blurry and nearly duplicate frames first, working on copies and preserving original data

```text
scene_demo/
  input/                 # Working copies of original photos
    frame_00001.jpg
    frame_00002.jpg
```

If the input is video of a static scene and FFmpeg is installed, extract frames first. This example takes 2 frames per second, only as a starting point; select frames according to actual baseline and blur:

```bash
mkdir -p scene_demo/input
ffmpeg -n -i capture.mp4 -vf "fps=2" -q:v 2 scene_demo/input/frame_%05d.jpg
```

`-n` avoids overwriting existing output; the output directory should be empty. This step does not solve cameras or reconstruct a scene, and is unsuitable as a synchronization procedure for dynamic multiple-camera data. [FFmpeg fps documentation](https://ffmpeg.org/ffmpeg-filters.html#fps-1)

### Installation and Platforms

1. Choose a build/distribution from the [official COLMAP installation page](https://colmap.github.io/install.html)
2. Windows: use `COLMAP.bat` from the official distribution; if CUDA dense reconstruction is needed, confirm the package supports CUDA. Add its directory to the current terminal's PATH or use its full path
3. Linux: install COLMAP/FFmpeg in a separate Conda environment; the commands below only prepare data processing tools and do not guarantee CUDA dense support in the package
4. Mac: install through Homebrew for CPU SfM; see E

```bash
# Linux, when Conda is already installed
conda create -n vision-prep -c conda-forge python colmap ffmpeg
conda activate vision-prep
colmap -h
colmap feature_extractor -h
colmap exhaustive_matcher -h
```

On the check date, the online CLI uses `FeatureExtraction.use_gpu` / `FeatureMatching.use_gpu`; older releases use `SiftExtraction.use_gpu` / `SiftMatching.use_gpu`. **Replace parameter names only with ones explicitly supported by local help**; do not mix versions. [Current CLI](https://colmap.github.io/cli.html)

### Execution: Start with SfM Only

The following is Bash. On Windows, put each command on one line, replace `colmap` with `COLMAP.bat`, and replace paths with, for example, `D:/vision/scene_demo`; create directories in File Explorer or with PowerShell's `New-Item -ItemType Directory`. Do not paste Bash's `$SCENE` and backslash line continuations directly into cmd.

```bash
SCENE="$PWD/scene_demo"
mkdir -p "$SCENE/raw_sparse" "$SCENE/undistorted"

colmap feature_extractor \
  --database_path "$SCENE/database.db" \
  --image_path "$SCENE/input" \
  --ImageReader.single_camera 1

colmap exhaustive_matcher --database_path "$SCENE/database.db"

colmap mapper \
  --database_path "$SCENE/database.db" \
  --image_path "$SCENE/input" \
  --output_path "$SCENE/raw_sparse"
```

`single_camera 1` assumes the photos share one intrinsic configuration; do not copy it mechanically for multiple cameras, zoom changes, or different resolutions. Without NVIDIA/when not using the GPU, append the CPU options supported by your local version to feature/matcher commands respectively; see E.

**Intermediate acceptance: do not train further if this fails.** Import `raw_sparse/0` into the COLMAP GUI and check that the camera arrangement matches the capture route, the subject takes shape, and the reconstruction has not split into disconnected models. `0` is only a model number, not a guarantee that it is your desired complete result; if there are multiple models, select and record the correct directory first.

### Undistort and Arrange Official 3DGS Input

```bash
colmap image_undistorter \
  --image_path "$SCENE/input" \
  --input_path "$SCENE/raw_sparse/0" \
  --output_path "$SCENE/undistorted" \
  --output_type COLMAP
```

The undistorted workspace usually writes the model under `undistorted/sparse`. Official 3DGS requires `sparse/0`; use this short Python snippet to **copy** the necessary files without moving or deleting originals. If local output already has `sparse/0`, simply retain it.

```bash
python - <<'PY'
from pathlib import Path
import shutil
root = Path('scene_demo/undistorted/sparse')
out = root / '0'
out.mkdir(exist_ok=True)
for name in ('cameras.bin', 'images.bin', 'points3D.bin'):
    src, dst = root / name, out / name
    if not dst.exists():
        if not src.exists():
            raise FileNotFoundError(f'Check COLMAP output format and directory: {src}')
        shutil.copy2(src, dst)
print('Model files ready; cameras and sparse points still need manual inspection')
PY
```

Run this Python snippet from the parent directory of `scene_demo`; it does not “check that the geometry is correct.” On Windows, save the same Python content as `prepare_sparse.py` and run it.

The final input root for B is `scene_demo/undistorted`:

```text
undistorted/
  images/                # Undistorted images, not original input
  sparse/0/
    cameras.bin
    images.bin
    points3D.bin
```

### Optional Geometry Branch: Dense and Mesh

This requires a COLMAP build/GPU with the corresponding CUDA support. CPU SfM does not imply that this implementation's PatchMatch dense reconstruction also runs on Mac/CPU.

```bash
colmap patch_match_stereo --workspace_path "$SCENE/undistorted" --workspace_format COLMAP --PatchMatchStereo.geom_consistency true
colmap stereo_fusion --workspace_path "$SCENE/undistorted" --workspace_format COLMAP --input_type geometric --output_path "$SCENE/undistorted/fused.ply"
colmap poisson_mesher --input_path "$SCENE/undistorted/fused.ply" --output_path "$SCENE/undistorted/mesh_poisson.ply"
```

**Outputs and acceptance:** The sparse model contains cameras and points; `fused.ply` is a dense point cloud; `mesh_poisson.ply` is a triangle mesh. Inspect holes, floating patches, normals, closures, and scale. A back surface filled by Poisson is not a new observation; do not directly treat it as measurable/collision ground truth. The main COLMAP project has a BSD-style license; check dependencies and data separately through the [official license page](https://colmap.github.io/license.html).

---

## B. COLMAP → Official graphdeco 3DGS

### Input, Dependencies, and Boundaries

Input is A's `images + sparse/0`, with an undistorted `PINHOLE` or `SIMPLE_PINHOLE` camera model. Dense/mesh reconstruction is not required first.

The official README specifies CUDA compute capability 7.0+ and recommends 24 GB GPU memory for paper evaluation scale. This is neither the minimum memory for every small scene nor a promise that every 24 GB GPU is automatically compatible. On the check date, the official environment file still uses a historical Python/PyTorch/CUDA combination; it is not a “latest universal environment.” New GPU architectures may require toolchain updates and dependency adaptation.

- Linux CUDA: Conda, Git, compatible g++, CUDA toolkit, and NVIDIA driver
- Windows NVIDIA: install Visual Studio C++ tools first, then a matching CUDA toolkit; use an x64 developer command prompt and confirm Conda runs there
- Mac: this official CUDA training route does not apply

[Official installation instructions](https://github.com/graphdeco-inria/gaussian-splatting#optimizer), [environment file](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/environment.yml)

### Installation: Separate Environment from Nerfstudio/4DGS

```bash
git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting.git
cd gaussian-splatting
git rev-parse HEAD
git submodule status
```

First run these in the Windows cmd developer terminal:

```bat
set DISTUTILS_USE_SDK=1
where cl
where nvcc
```

Then use the following on either Windows or Linux:

```bash
conda env create --file environment.yml
conda activate gaussian_splatting
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import diff_gaussian_rasterization, simple_knn; print('extensions import OK')"
python train.py --help
```

Successful imports only check the environment; they are not a GPU training test. If the historical environment fails to resolve, consult [Troubleshooting](07_troubleshooting.md) first. Do not blindly upgrade every package and still call it a reproduction of the original version.

The official repository also offers `convert.py -s <scene>`, which requires original images in `input/`. However, on the check date [convert.py](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/convert.py) still uses old COLMAP GPU option names, so this tutorial primarily uses A's explicit steps. Once valid `images/sparse/0` is available, conversion is unnecessary.

### Training, Rendering, and Evaluation

Run from the `gaussian-splatting` repository root; replace the path below with A's absolute path:

```bash
python train.py -s /absolute/path/scene_demo/undistorted -m output/scene_demo --eval -r 2 --data_device cpu
python render.py -m output/scene_demo
python metrics.py -m output/scene_demo
```

Windows example:

```bat
python train.py -s D:/vision/scene_demo/undistorted -m output/scene_demo --eval -r 2 --data_device cpu
python render.py -m output/scene_demo
python metrics.py -m output/scene_demo
```

`-r 2` halves width and height as a practice choice; `--data_device cpu` keeps input images on the CPU, but **the trainer still requires CUDA**. `--eval` holds out evaluation views; do not train on every image and label training error as test error.

Main outputs:

```text
output/scene_demo/
  cfg_args
  cameras.json
  point_cloud/iteration_*/point_cloud.ply
  train/ours_*/renders/   # After running render
  test/ours_*/renders/
  test/ours_*/gt/
  results.json           # After running metrics
```

If resumable training state is needed, additionally set `--checkpoint_iterations` when starting training; a rendering PLY is not a complete optimizer checkpoint.

**Acceptance:** Compare at least several test ground-truth/rendered image pairs; inspect side views, thin structures, reflective surfaces, and insufficiently covered regions. Record split, resolution, background handling, iterations, and commit alongside PSNR/SSIM/LPIPS; these numbers do not establish millimeter-level geometric accuracy.

### Interactive Viewing and Licensing

Windows users can obtain precompiled SIBR from the [official README's viewer entry](https://github.com/graphdeco-inria/gaussian-splatting#interactive-viewers); Linux build steps are on the same page. After installation, run the corresponding `SIBR_gaussianViewer_app` (add `.exe` on Windows), with `-m` pointing to the entire model directory. It needs CUDA/OpenGL support; do not describe it as a universal Mac viewer.

The official code uses a restrictive research/evaluation license; verify authorization separately for commercial use. **“Source code is downloadable” does not mean unrestricted commercial use.** See [LICENSE.md](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md).

---

## C. Nerfstudio Splatfacto: A More Coherent Beginner Workflow

### Input and Choice

Input is a static photo directory or video. Nerfstudio can invoke COLMAP and organize data. Splatfacto uses the gsplat backend and is an evolving Gaussian method implementation; do not label it as “an exact reproduction of the classic paper.” The roughly 6 GB/12 GB values on the official page are example budgets for different presets, not guarantees for every scene. [Official Splatfacto page](https://docs.nerf.studio/nerfology/methods/splat.html)

### Linux NVIDIA: Official Pixi Route

Prepare a compatible NVIDIA driver and Git; after installing Pixi from the [official Pixi entry](https://pixi.sh/), follow the [Nerfstudio installation page](https://docs.nerf.studio/quickstart/installation.html). On the check date, that page states this Pixi workflow supports Linux. This tutorial does not pipe a remote script directly into execution.

```bash
git clone https://github.com/nerfstudio-project/nerfstudio.git
cd nerfstudio
git rev-parse HEAD
pixi run post-install
pixi shell
colmap -h
ffmpeg -version
ns-train splatfacto --help
```

Save the checkout's commit and lockfile; when opening a new terminal, enter the same repository and run `pixi shell`. Do not install additional packages in a different Conda environment and mix them.

### Windows NVIDIA: Conda Route

Use a CUDA-compatible Visual Studio C++ developer terminal, Conda, and Git; first prepare COLMAP and FFmpeg using A. The following combination listed on the official installation page on the check date is a **historical recommendation in the documentation**, not a claim by this tutorial that these are the latest versions; use it only if the combination supports your GPU:

```bat
conda create -n nerfstudio python=3.8
conda activate nerfstudio
python -m pip install --upgrade pip
python -m pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
conda install -c nvidia/label/cuda-11.8.0 cuda-toolkit
python -m pip install ninja
python -m pip install "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
python -m pip install nerfstudio
python -m pip check
ns-train splatfacto --help
```

tiny-cuda-nn is part of the official overall installation process; Splatfacto's Gaussian rasterization backend is gsplat. Do not confuse them. Future PyPI releases may no longer support this historical Python combination. If resolution fails, rebuild the environment according to the declared dependencies of the chosen Nerfstudio version and record actual installed versions. Do not repeatedly install “the latest torch” in the wrong environment.

### Processing, Training, and Export: Same CLI on Both Platforms

Prepare photos in `raw_images/` in a new working directory, with the output directory not yet present. The following uses relative paths; these single-line commands also work on Windows:

```bash
ns-process-data images --data raw_images --output-dir processed/scene_demo
ns-train splatfacto --data processed/scene_demo
```

For video input, replace only the processing step with:

```bash
ns-process-data video --data capture.mp4 --output-dir processed/scene_demo
```

Outputs usually include `images/`, COLMAP data, `transforms.json`, and a sparse point file for initialization; consult processing logs and the current parser for specifics. [Official custom-data workflow](https://docs.nerf.studio/quickstart/custom_dataset.html)

The training terminal provides the viewer address and save location. Find this run's `config.yml` and replace `PATH_TO_CONFIG` below with its **real path**; it is a placeholder, not a filename:

```bash
ns-viewer --load-config PATH_TO_CONFIG
ns-eval --load-config PATH_TO_CONFIG --output-path evaluation_scene_demo.json
ns-export gaussian-splat --load-config PATH_TO_CONFIG --output-dir exports/scene_demo
```

Training outputs under `outputs/` are organized by experiment/method/time and contain configuration and checkpoints; by default, the exporter generates `exports/scene_demo/splat.ply`. [Export code](https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/scripts/exporter.py), [evaluation code](https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/scripts/eval.py)

**Acceptance:** Inspect COLMAP registration first, then the viewer; compare held-out views separately. Choose a tool that supports Gaussian properties to open the exported PLY, or you will see only centers. Camera paths created in the viewer can also render videos; consult the [viewer guide](https://docs.nerf.studio/quickstart/viewer_quickstart.html) and `ns-render --help` for the exact exported path format.

The main Nerfstudio and gsplat projects use Apache-2.0; check dependencies, third-party plugins, data, and models separately. [Nerfstudio license](https://github.com/nerfstudio-project/nerfstudio/blob/main/LICENSE), [gsplat license](https://github.com/nerfstudio-project/gsplat/blob/main/LICENSE)

### C's NeRF Branch: Train Nerfacto on the Same Photos

**Input and environment:** Reuse C's configured Nerfstudio environment and the same `processed/scene_demo`. Nerfacto is a modern NeRF method combining techniques such as hash encoding, sampling strategies, and appearance conditioning; it is not an item-by-item reproduction of the original NeRF paper. This CUDA/tiny-cuda-nn route still requires a compatible NVIDIA GPU; a Mac can prepare data or view remote results, but this section does not promise local MPS training. Official documentation gives a roughly 6 GB reference budget for default Nerfacto; actual peaks vary with configuration, data, and viewing/evaluation stages. [Nerfacto method documentation](https://docs.nerf.studio/nerfology/methods/nerfacto.html)

1. **Process data.** Skip this step if C already generated and checked `processed/scene_demo`; otherwise run it once, using C's video branch for video input:

```bash
ns-process-data images --data raw_images --output-dir processed/scene_demo
```

2. **Train.** Check local parameters in the same environment, then launch; the training terminal prints the viewer address and this run's configuration path:

```bash
ns-train nerfacto --help
ns-train nerfacto --data processed/scene_demo
```

3. **View and evaluate.** Replace `PATH_TO_NERFACTO_CONFIG` with the `config.yml` path from this **nerfacto** run; do not accidentally use the earlier splatfacto configuration:

```bash
ns-viewer --load-config PATH_TO_NERFACTO_CONFIG
ns-eval --load-config PATH_TO_NERFACTO_CONFIG --output-path evaluation_nerfacto_scene_demo.json
```

These are independent commands: the viewer is a continuously running process. Close it before evaluation, or use another terminal in the same environment if resources permit. [Official first training, reopening the viewer, and evaluation workflow](https://docs.nerf.studio/quickstart/first_nerf.html)

**Outputs and acceptance:** This experiment's `nerfacto` run directory under `outputs/` contains `config.yml`, model checkpoints, and other files; `ns-eval` writes the specified JSON. Inspect occlusion edges, fine structures, and reflective areas in novel views, and verify that evaluation views were excluded from training. When comparing with Splatfacto, fix the data split, image resolution, and evaluation settings, and record the different representations and parameters of both methods; do not directly compare their numerical training losses.

Nerfacto saves a radiance field model; `ns-export gaussian-splat` cannot directly turn it into a Gaussian PLY. To present a video, set a camera path in the viewer's render panel and use the `ns-render` command it generates. Extracting a point cloud/mesh is a separate derived process, not the recovery of exact CAD or a physical model. The Nerfstudio/dependency licensing notes above also apply. **This branch's official documentation was checked on 2026-10-03; no installation, training, or performance tests were run.**

---

## D. hustvl 4DGaussians: Start by Reproducing a Synthetic Sequence

### Input, Installation, and Boundaries

Linux NVIDIA in a separate environment is recommended; native Windows and Mac are outside the runnable paths promised by this tutorial. Windows users choosing WSL2 with a configured GPU must separately verify driver, CUDA, compiler, and display compatibility; WSL2 is not a guaranteed one-click replacement.

Obtain the `bouncingballs` scene from the official data entry linked by the [D-NeRF authors' repository](https://github.com/albertpumarola/D-NeRF), check data terms first, and extract it manually. Only paths are provided here; large datasets are not downloaded automatically. Expected contents:

```text
4DGaussians/data/dnerf/bouncingballs/
  transforms_train.json
  transforms_test.json
  train/                 # Follow file_path in the JSON
  test/
```

Each frame's JSON needs an image path, camera transform, and `time`; renaming a folder does not turn ordinary static NeRF data into dynamic data.

The official README uses a historical Python 3.7, PyTorch 1.13.1+cu116 environment. First configure a CUDA toolkit/compiler that supports this environment and your GPU; **do not share the environments from B or C**.

```bash
git clone https://github.com/hustvl/4DGaussians.git
cd 4DGaussians
git submodule update --init --recursive
git rev-parse HEAD
conda create -n Gaussians4D python=3.7
conda activate Gaussians4D
python -m pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
python -m pip install -r requirements.txt
python -m pip install -e submodules/depth-diff-gaussian-rasterization
python -m pip install -e submodules/simple-knn
python -m pip check
python train.py --help
```

The explicit cu116 installation comes from the [PyTorch historical installation table](https://pytorch.org/get-started/previous-versions/); the rest comes from the [official requirements](https://github.com/hustvl/4DGaussians#environmental-setups). This does not guarantee all unpinned transitive dependencies still resolve years later. On failure, record logs and use a host that supports the historical environment/pin dependencies, rather than claiming every new GPU works.

### Training, Rendering, and Evaluation

```bash
python train.py -s data/dnerf/bouncingballs --port 6017 --expname dnerf/bouncingballs --configs arguments/dnerf/bouncingballs.py
python render.py --model_path output/dnerf/bouncingballs --skip_train --configs arguments/dnerf/bouncingballs.py
python metrics.py --model_path output/dnerf/bouncingballs
```

Outputs in `output/dnerf/bouncingballs/` include configuration and Gaussian/deformation weights under `point_cloud/iteration_*/`; rendering produces images/videos in `test/` and `video/`; metrics produces evaluation JSON. Preserve the entire experiment directory, not just a PLY. [Save logic](https://github.com/hustvl/4DGaussians/blob/master/scene/__init__.py), [render logic](https://github.com/hustvl/4DGaussians/blob/master/render.py)

**Acceptance:** Check continuity while playing time from a fixed view, and plausibility when changing views at frozen time; confirm test data is truly independent. Complete this calibrated dataset workflow before troubleshooting captured data; the first success need not chase paper metrics.

### Advanced: Your Own Fixed, Synchronized Multi-Camera Sequence

> **This captured-data adaptation procedure requires real calibration and has not been run; it is not a recipe for training after renaming arbitrary multi-camera videos. If any camera prerequisite below cannot be verified, stop at data preparation, first modify the loader/projection or improve calibration, and do not execute the subsequent copy or training commands.**

On the check date, [multipleview_dataset.py](https://github.com/hustvl/4DGaussians/blob/master/scene/multipleview_dataset.py) hardcodes reading the first focal length from `camera_id=1` and sets `fx=fy`; [getProjectionMatrix](https://github.com/hustvl/4DGaussians/blob/master/utils/graphics_utils.py) uses a symmetric frustum and accepts neither per-camera `cx/cy` nor distortion. **Merely having “same-sized images” or “individually undistorted images” does not satisfy these assumptions.**

Without modifying the official loader, first use trustworthy original calibration for each camera to remap **all cameras at all times** to one shared set of virtual pinhole camera intrinsics:

- Same output width/height `W,H` and same pixel focal length `fx=fy=f0`
- No distortion, zero skew, and principal point at the image center: `cx=W/2, cy=H/2` under COLMAP's pixel convention
- Each camera's mapping remains fixed throughout the sequence; camera, zoom, and focus configurations must also remain unchanged
- No subsequent independent automatic cropping/scaling; if cropping is needed, incorporate it consistently into the common virtual camera design and revalidate intrinsics

`f0,W,H` must come from the recorded remapping design; do not guess values from an example. When using tools such as OpenCV, first unify pixel-center conventions: COLMAP and OpenCV principal-point values differ by a half-pixel convention and cannot be mixed unchanged. Validate effective field of view, boundaries, and reprojection errors after remapping; this tutorial does not provide an automatic remapping script lacking calibration validation. [COLMAP known intrinsics and pixel conventions](https://colmap.github.io/faq.html#using-calibration-from-opencv-kalibr-or-other-tools)

Keep original videos; the directories below must contain only **working copies that passed the remapping above**, with equally long, continuous, synchronized frame sequences:

```text
data/multipleview/demo_motion/
  cam01/frame_00001.jpg
  cam01/frame_00002.jpg
  cam02/frame_00001.jpg
  cam02/frame_00002.jpg
  ...
```

The official `multipleviewprogress.sh` takes only each camera's first frame to solve fixed extrinsics and includes temporary-directory cleanup. This tutorial **does not directly execute that script**; consult the [script](https://github.com/hustvl/4DGaussians/blob/master/multipleviewprogress.sh) and [image extraction logic](https://github.com/hustvl/4DGaussians/blob/master/scripts/extractimages.py). After satisfying the shared virtual-camera prerequisites, proceed in this order:

1. **Start afresh with processed first frames.** Create a dedicated, empty `prep_demo/images`; copy `cam01/frame_00001.jpg` above as `image1.jpg`, `cam02/...` as `image2.jpg`, and so on. Record the mapping. Do not reuse original distorted images or a database/camera model from an old `prep_demo`
2. **Solve extrinsics with the same intrinsics.** Use A's feature → matcher → mapper workflow, but select `--ImageReader.single_camera 1` and `--ImageReader.camera_model SIMPLE_PINHOLE` during extraction, and supply the verified numeric list `f0,cx,cy` through `--ImageReader.camera_params`. Replace all three symbols with actual numbers; `cx/cy` are the centered principal point above. In mapper, turn off all of `--Mapper.ba_refine_focal_length 0`, `--Mapper.ba_refine_principal_point 0`, and `--Mapper.ba_refine_extra_params 0` to prevent refitting K. Check local `colmap mapper -h` before running; in the GUI, equivalently disable the corresponding intrinsic refine options in Bundle Adjustment. Do not later perform extra BA that releases the intrinsics. [Fixing intrinsics](https://colmap.github.io/faq.html#fix-intrinsics), [checked Mapper option source](https://github.com/colmap/colmap/blob/main/src/colmap/controllers/option_manager.cc)
3. **Continue only after passing mandatory acceptance.** Export `prep_demo/sparse/0` to text in COLMAP and inspect it: actual `camera_id=1` exists; all images refer to the shared camera; `SIMPLE_PINHOLE` dimensions and `f,cx,cy` exactly match the remapping record; camera names map correctly to `camXX`; every camera position registered successfully; independent reprojection checks pass. If IDs, parameters, dimensions, or images do not match, stop and correct actual data correspondence. Do not merely hand-edit parameters to make the checklist “pass”
4. **Build the initialization point cloud.** Using these validated processed first frames and the fixed-intrinsics model, follow A's undistorter → PatchMatch → fusion workflow to create `prep_demo/dense/fused.ply`. Here the undistorter generates a dense workspace; it does not repair distortion left untreated throughout the sequence. Subsequent training still pairs “shared virtual-camera images + their matching fixed-intrinsics sparse model”; do not mix these with camera files generated by another scaling/cropping operation
5. **Only after all of 1–4 pass**, organize the outputs below in the same 4DGS environment. The copied `prep_demo/sparse/0` must be the model **rebuilt and validated from the processed first frames above**, not an old model from before remapping the entire sequence. The destination should be a new directory; inspect existing files first and do not overwrite other experiments

```bash
python scripts/downsample_point.py prep_demo/dense/fused.ply data/multipleview/demo_motion/points3D_multipleview.ply
git clone https://github.com/Fyusion/LLFF.git external_LLFF
python -m pip install scikit-image
python external_LLFF/imgs2poses.py prep_demo
mkdir -p data/multipleview/demo_motion/sparse_
cp prep_demo/sparse/0/cameras.bin data/multipleview/demo_motion/sparse_/
cp prep_demo/sparse/0/images.bin data/multipleview/demo_motion/sparse_/
cp prep_demo/sparse/0/points3D.bin data/multipleview/demo_motion/sparse_/
cp prep_demo/poses_bounds.npy data/multipleview/demo_motion/poses_bounds_multipleview.npy
cp arguments/multipleview/default.py arguments/multipleview/demo_motion.py
```

Run LLFF conversion according to the [authors' script](https://github.com/Fyusion/LLFF/blob/master/imgs2poses.py); it also assumes particular camera models/sorting and is not a universal calibration-format converter. Ensure the model is complete before conversion; keep intermediate directories for cross-checking.

6. Read the copied configuration and inspect spatial bounds, temporal grid, iteration count, and batch size. The default template is only a starting point, not settings calibrated for your motion
7. Train and render only after all previous calibration, intrinsic-pairing, and reprojection acceptance checks pass:

```bash
python train.py -s data/multipleview/demo_motion --port 6017 --expname multipleview/demo_motion --configs arguments/multipleview/demo_motion.py
python render.py --model_path output/multipleview/demo_motion --skip_train --configs arguments/multipleview/demo_motion.py
```

**Critical acceptance limitation:** The original loader's default test and train sets overlap; its default metrics cannot report independent generalization performance. If the project needs held-out camera/time evaluation, change the split and inspect sample lists. Also keep thumbnails and hidden files out of camera folders: the loader infers sequence length from directory entry counts.

Hardware budgets vary with resolution, camera count, frame count, initial points, spatiotemporal grids, and densification; this tutorial promises neither fixed GPU memory usage nor a fixed number of minutes. Verify the process on a shorter sequence at lower resolution before scaling up.

**Licensing:** The top-level [LICENSE](https://github.com/hustvl/4DGaussians/blob/master/LICENSE.md) is Apache-2.0, but some files retain graphdeco research-use notices and submodules have their own licenses. Commercial use of the whole system cannot be assessed from GitHub's top-level label alone; check sources and terms individually. Handle data licensing separately.

---

## E. Mac: Capture, Data Preparation, CPU SfM, and Viewing

### Input and Installation

Input is your own static photos/video; if Homebrew is already installed:

```bash
brew install colmap ffmpeg python
colmap -h
ffmpeg -version
```

Use A's frame extraction, directory creation, and SfM steps, changing extraction/matching to CPU. **Below are parameter names in current documentation on the check date; if local help lists only the old SIFT options, use the old names**:

```bash
colmap feature_extractor --database_path scene_demo/database.db --image_path scene_demo/input --ImageReader.single_camera 1 --FeatureExtraction.use_gpu 0
colmap exhaustive_matcher --database_path scene_demo/database.db --FeatureMatching.use_gpu 0
```

Then run A's mapper/undistorter; run A's Python snippet with `python3` on Mac. CPU handles only this part of data preparation: it does not turn official GS CUDA extensions into Metal extensions or provide a CPU implementation of COLMAP CUDA dense reconstruction.

### Outputs and Acceptance

- Inspect cameras and sparse points in the COLMAP GUI; transfer the complete undistorted data directory to a trusted NVIDIA training environment
- After training, a Mac can play rendered videos; ordinary point clouds/meshes can be viewed with the tools in [Chapter 02](02_pointcloud_mesh.md)
- Gaussian PLY requires a compatible browser/desktop Gaussian viewer; actual frame rate depends on GPU, browser, resolution, and Gaussian count, and smooth playback is not guaranteed on every Mac
- For any online viewer, first check whether it uploads data and review its terms; do not casually publish private scenes. Prefer images/videos rendered by the remote trainer if appropriate

**Completion conditions:** Images are organized, SfM cameras are plausible, the target training environment can read the data, and exported/rendered results can be viewed on Mac. This does not mean “local Mac training succeeded.”

---

## Save an Experiment Record for Every Route

```text
Data source and authorization:
Static/dynamic, camera count, frame count, image dimensions:
Camera calibration/synchronization/scale reference:
Training and test split lists:
OS, GPU, driver, toolkit:
Code repository commit, submodule commits:
Python/PyTorch/key dependencies and configuration:
Actual commands and log paths:
Generated files and local opening checks:
Image metrics, geometry validation, and unresolved issues:
```

Actual records are more useful for reproduction than “latest version + defaults.” Troubleshoot from the earliest failing step: [07 Troubleshooting](07_troubleshooting.md).

[Back to Home](../../README.en.md) · [Coordinates and Calibration](08_robot_frames_calibration.md) · [Robot Applications](09_robot_perception_action.md) · [ROS 2 / MuJoCo](10_ros2_mujoco.md)
