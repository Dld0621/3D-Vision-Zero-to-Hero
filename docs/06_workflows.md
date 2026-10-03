# 06｜从输入到验收：五条软件工作流

**简体中文** | [English](en/06_workflows.md)

> **验证边界：2026-10-03 核查官方文档与源码；以下安装、训练和 GPU 命令未在用户机器运行。没有本仓库实测的训练时间、显存峰值或质量 benchmark。**
>
> 路径均为公开教程的虚构示例。先在新的空工作目录操作，保留原图；不要把原始素材、模型权重和运行环境提交到 GitHub。代码分支会变化，运行时记录 commit、环境和本地 `--help`。

## 0. 先选路线，不要一次装齐全部软件

| 你的目标/设备 | 起步路线 | 边界 |
|---|---|---|
| 想理解几何、Windows/Linux/Mac | A：照片 → COLMAP 稀疏模型 | CPU 可做 SfM；大数据仍可能很慢 |
| 想复现经典 3DGS、有兼容 NVIDIA GPU | A → B：官方 graphdeco | 历史 CUDA 环境；新 GPU 不保证兼容 |
| 想先做一个可浏览的高斯场景、Linux CUDA | C：Nerfstudio Splatfacto | 工程化更完整，但不等同原始 3DGS |
| 想研究动态重建、有 Linux CUDA | D：hustvl 合成序列起步 | 再做同步多机数据，别从任意手机视频开始 |
| 只有 Mac | E：采集、整理、CPU SfM、查看 | 本教程不承诺这几套官方 CUDA 训练器可在 Metal/MPS 上训练 |

**外观资产**走 B/C，**测量与碰撞表面**关注 A 的几何支线与 [mesh 章节](02_pointcloud_mesh.md)。视觉模型、测量模型和物理模型可以结合，但验收标准不同。

### GPU 检查：三个“CUDA 版本”不要混为一谈

驱动能支持的 CUDA 版本、PyTorch wheel 携带的 runtime、编译扩展使用的 toolkit 是三回事。

```bash
nvidia-smi
nvcc --version
python -c "import torch; print('torch=',torch.__version__,'runtime=',torch.version.cuda,'available=',torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

`nvidia-smi` 成功不证明 `nvcc` 已安装；`torch.cuda.is_available()` 为真也不证明某个自定义 CUDA 扩展能为你的 GPU 编译。先核对 [NVIDIA Linux 安装指南](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) 或 [Windows 安装指南](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/) 中相应 toolkit 的驱动/编译器兼容要求。

---

## A. 静态照片 → COLMAP → 稀疏模型，可选 dense/mesh

### 输入与采集

- 输入：同一静态物体/场景的清晰、多视角、有重叠照片
- 初次练习可用几十到一两百张中等分辨率照片，这是教学规模建议，不是成功阈值
- 物体、背景、光照尽量不变；不要边绕拍边转动同一物体而保留静止背景
- 固定变焦和分辨率；绕物体拍一圈，再增加较高/较低视角；避免只原地旋转相机
- 先去掉模糊与几乎重复的帧，但在工作副本中操作，保留原始数据

```text
scene_demo/
  input/                 # 原始照片的工作副本
    frame_00001.jpg
    frame_00002.jpg
```

若输入是静态场景的视频，已安装 FFmpeg 时可先抽帧。下例每秒取 2 帧，只是起点，按实际基线和模糊情况筛选：

```bash
mkdir -p scene_demo/input
ffmpeg -n -i capture.mp4 -vf "fps=2" -q:v 2 scene_demo/input/frame_%05d.jpg
```

`-n` 避免覆盖现有输出；输出目录应为空。这一步不求相机、不重建，也不适合作为动态多机同步操作。[FFmpeg fps 文档](https://ffmpeg.org/ffmpeg-filters.html#fps-1)

### 安装与平台

1. 按 [COLMAP 官方安装页](https://colmap.github.io/install.html) 选择构建/发行包
2. Windows：使用官方发行包的 `COLMAP.bat`；需要 dense CUDA 时确认包支持 CUDA。将其目录加入当前终端 PATH，或写完整路径
3. Linux：可在独立 Conda 环境安装 COLMAP/FFmpeg，下面命令仅准备数据处理工具，不保证包具备 CUDA dense 能力
4. Mac：Homebrew 安装，做 CPU SfM；见 E

```bash
# Linux，已有 Conda 时
conda create -n vision-prep -c conda-forge python colmap ffmpeg
conda activate vision-prep
colmap -h
colmap feature_extractor -h
colmap exhaustive_matcher -h
```

核查日的在线 CLI 使用 `FeatureExtraction.use_gpu` / `FeatureMatching.use_gpu`；较旧发行版使用 `SiftExtraction.use_gpu` / `SiftMatching.use_gpu`。**只替换本机 help 明确支持的参数名**，不要混用版本。[当前 CLI](https://colmap.github.io/cli.html)

### 执行：先只做 SfM

以下是 Bash。Windows 可把每条命令合成一行执行，用 `COLMAP.bat` 替换 `colmap`，路径换成如 `D:/vision/scene_demo`；建目录用资源管理器或 PowerShell 的 `New-Item -ItemType Directory`。不要把 Bash 的 `$SCENE` 和反斜杠续行直接贴进 cmd。

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

`single_camera 1` 假设照片来自同一内参配置；多相机、变焦、不同分辨率不能机械照抄。无 NVIDIA/不使用 GPU 时，在 feature/matcher 命令分别追加本机版本支持的 CPU 参数，见 E。

**中间验收，没通过别往下训：** 在 COLMAP GUI 中导入 `raw_sparse/0`，检查相机排列是否符合拍摄路线、主体是否成形、是否分裂成多个互不相连的模型。`0` 只是模型编号，不保证是你想要的完整结果；有多个模型时先选择并记录正确目录。

### 去畸变并整理为官方 3DGS 输入

```bash
colmap image_undistorter \
  --image_path "$SCENE/input" \
  --input_path "$SCENE/raw_sparse/0" \
  --output_path "$SCENE/undistorted" \
  --output_type COLMAP
```

去畸变工作区一般将模型写在 `undistorted/sparse`。官方 3DGS 需要 `sparse/0`，用下面的小段 Python **复制**必要文件，不移动或删除源文件。若本机输出已经有 `sparse/0`，直接保留即可。

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
            raise FileNotFoundError(f'检查 COLMAP 输出格式与目录：{src}')
        shutil.copy2(src, dst)
print('模型文件就绪；仍须人工检查相机和稀疏点')
PY
```

这段 Python 在 `scene_demo` 的父目录执行；它不是“检查几何正确”的程序。Windows 可将同一 Python 内容存成 `prepare_sparse.py` 后执行。

最终给 B 的输入根目录是 `scene_demo/undistorted`：

```text
undistorted/
  images/                # 去畸变图像，不是原始 input
  sparse/0/
    cameras.bin
    images.bin
    points3D.bin
```

### 可选几何支线：dense 与 mesh

需要具有对应 CUDA 支持的 COLMAP 构建/GPU。SfM 能在 CPU 上跑，不代表本实现的 PatchMatch dense 可在 Mac/CPU 上同样运行。

```bash
colmap patch_match_stereo --workspace_path "$SCENE/undistorted" --workspace_format COLMAP --PatchMatchStereo.geom_consistency true
colmap stereo_fusion --workspace_path "$SCENE/undistorted" --workspace_format COLMAP --input_type geometric --output_path "$SCENE/undistorted/fused.ply"
colmap poisson_mesher --input_path "$SCENE/undistorted/fused.ply" --output_path "$SCENE/undistorted/mesh_poisson.ply"
```

**产物与验收：** 稀疏模型有相机与点；`fused.ply` 是稠密点云；`mesh_poisson.ply` 是三角网格。检查孔洞、浮片、法线、封口与尺度。Poisson 补出来的背面不是新增观测，不要直接当可测量/可碰撞的真值。COLMAP 主项目为 BSD 类许可，依赖与数据另查 [官方许可页](https://colmap.github.io/license.html)。

---

## B. COLMAP → 官方 graphdeco 3DGS

### 输入、依赖与边界

输入是 A 的 `images + sparse/0`，相机模型应为去畸变后的 `PINHOLE` 或 `SIMPLE_PINHOLE`。不需要先做 dense/mesh。

官方 README 列出 CUDA compute capability 7.0+，论文评价规模建议 24 GB 显存；这不是所有小场景的最低显存，也不是任何 24 GB GPU 都自动兼容的承诺。核查日官方环境文件仍是历史 Python/PyTorch/CUDA 组合，不能当成“最新通用环境”。新架构 GPU 可能需要更新工具链和依赖适配。

- Linux CUDA：Conda、Git、兼容的 g++、CUDA toolkit 和 NVIDIA 驱动
- Windows NVIDIA：先装 Visual Studio 的 C++ 工具，再装匹配的 CUDA toolkit；使用 x64 开发者命令提示符，并确认其中能运行 Conda
- Mac：这条官方 CUDA 训练流程不适用

[官方安装说明](https://github.com/graphdeco-inria/gaussian-splatting#optimizer)、[环境文件](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/environment.yml)

### 安装：独立环境，不和 Nerfstudio/4DGS 混装

```bash
git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting.git
cd gaussian-splatting
git rev-parse HEAD
git submodule status
```

Windows 的 cmd 开发者终端先执行：

```bat
set DISTUTILS_USE_SDK=1
where cl
where nvcc
```

然后 Windows/Linux 共用：

```bash
conda env create --file environment.yml
conda activate gaussian_splatting
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import diff_gaussian_rasterization, simple_knn; print('extensions import OK')"
python train.py --help
```

导入成功只是环境检查，不是 GPU 训练实测。历史环境解析失败时先看 [排错章节](07_troubleshooting.md)，不要把所有包盲目升级后还称为原版复现。

官方另有 `convert.py -s <scene>` 入口，要求原图在 `input/`。但核查日 [convert.py](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/convert.py) 仍使用旧 COLMAP GPU 参数名；所以本教程以 A 的显式步骤为主。已有合格 `images/sparse/0` 后无需再 convert。

### 训练、渲染、评价

在 `gaussian-splatting` 仓库根目录执行；将下面路径换为 A 的绝对路径：

```bash
python train.py -s /absolute/path/scene_demo/undistorted -m output/scene_demo --eval -r 2 --data_device cpu
python render.py -m output/scene_demo
python metrics.py -m output/scene_demo
```

Windows 例：

```bat
python train.py -s D:/vision/scene_demo/undistorted -m output/scene_demo --eval -r 2 --data_device cpu
python render.py -m output/scene_demo
python metrics.py -m output/scene_demo
```

`-r 2` 是宽高各减半的练习选择；`--data_device cpu` 把输入图像放在 CPU，**训练器仍需要 CUDA**。`--eval` 用于留出评价视角；别在用全部图片训练后，把训练误差标成测试误差。

主要产物：

```text
output/scene_demo/
  cfg_args
  cameras.json
  point_cloud/iteration_*/point_cloud.ply
  train/ours_*/renders/   # 执行 render 后
  test/ours_*/renders/
  test/ours_*/gt/
  results.json           # 执行 metrics 后
```

需要可续训状态时，在开始训练时另设 `--checkpoint_iterations`；渲染用的 PLY 不等于完整优化器检查点。

**验收：** 至少比较若干 test 真图/渲染图；观察侧面、薄结构、反光面、未充分覆盖位置。记录 PSNR/SSIM/LPIPS 时带上划分、分辨率、背景处理、迭代数与 commit；数值不代表毫米级几何准确度。

### 交互查看与许可

Windows 可以从 [官方 README 的 viewer 入口](https://github.com/graphdeco-inria/gaussian-splatting#interactive-viewers) 获取预编译 SIBR；Linux 构建步骤也在该页。安装后运行对应的 `SIBR_gaussianViewer_app`（Windows 加 `.exe`），参数为 `-m` 指向整个模型目录。它需要 CUDA/OpenGL 条件；不要把它写成 Mac 通用查看器。

官方代码使用限制性研究/评估许可，商用需另核实授权。**“能下载源码”不等于可任意商用。** 见 [LICENSE.md](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md)。

---

## C. Nerfstudio Splatfacto：更连贯的初学者流程

### 输入与选择

输入为静态照片目录或视频。Nerfstudio 能调用 COLMAP 并整理数据。Splatfacto 使用 gsplat 后端，是持续演进的 Gaussian 方法实现，不应标注为“完全复现经典论文”。官方页给出的约 6 GB/12 GB 是不同预设的示例预算，不是所有场景保证。[Splatfacto 官方页](https://docs.nerf.studio/nerfology/methods/splat.html)

### Linux NVIDIA：官方 Pixi 路线

准备兼容 NVIDIA 驱动和 Git；从 [Pixi 官方入口](https://pixi.sh/) 安装 Pixi 后，按 [Nerfstudio 安装页](https://docs.nerf.studio/quickstart/installation.html) 操作。该页核查日说明此 Pixi 流程支持 Linux。此处不使用远程脚本管道直接执行。

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

保存该 checkout 的 commit 与 lockfile；重新开终端时进入同一仓库运行 `pixi shell`。不要在另一套 Conda 环境里继续装包导致混用。

### Windows NVIDIA：Conda 路线

使用与 CUDA 兼容的 Visual Studio C++ 开发者终端、Conda 与 Git，先按 A 准备 COLMAP 和 FFmpeg。核查日官方安装页列出的下面组合是**文档中的历史推荐组合**，不是本教程宣称的最新版本；只在该组合支持你的 GPU 时采用：

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

tiny-cuda-nn 是官方整体安装流程的一部分，Splatfacto 的 Gaussian 光栅化后端是 gsplat；不要将二者混为一谈。PyPI 未来发布可能不再支持该历史 Python 组合；若解析失败，以选定 Nerfstudio 版本的声明依赖为准重新建立环境，并记录实际安装版本。不要在错误环境里反复装“最新 torch”。

### 处理、训练与导出：两平台相同的 CLI

在新工作目录准备照片 `raw_images/`，输出目录尚不存在。下列采用相对路径；Windows 同样可用这些单行命令：

```bash
ns-process-data images --data raw_images --output-dir processed/scene_demo
ns-train splatfacto --data processed/scene_demo
```

视频输入时，只把处理步骤替换为：

```bash
ns-process-data video --data capture.mp4 --output-dir processed/scene_demo
```

产物通常包括 `images/`、COLMAP 数据、`transforms.json` 和用于初始化的稀疏点文件；具体以处理日志与当前 parser 为准。[自采数据官方流程](https://docs.nerf.studio/quickstart/custom_dataset.html)

训练终端会给出 viewer 地址和保存位置。找到本次运行的 `config.yml`，用其**真实路径**替换下面的 `PATH_TO_CONFIG`，它是占位符，不是文件名：

```bash
ns-viewer --load-config PATH_TO_CONFIG
ns-eval --load-config PATH_TO_CONFIG --output-path evaluation_scene_demo.json
ns-export gaussian-splat --load-config PATH_TO_CONFIG --output-dir exports/scene_demo
```

训练输出在 `outputs/` 下按实验/方法/时间组织，含配置和检查点；导出器默认生成 `exports/scene_demo/splat.ply`。[导出代码](https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/scripts/exporter.py)、[评价代码](https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/scripts/eval.py)

**验收：** 先检查 COLMAP 注册情况，再看 viewer；单独比较留出视角。打开导出 PLY 时要选支持 Gaussian 属性的工具，否则只能看见中心点。viewer 创建的相机路径也可以用于视频渲染；具体导出路径格式以 [viewer 指南](https://docs.nerf.studio/quickstart/viewer_quickstart.html) 和 `ns-render --help` 为准。

Nerfstudio 与 gsplat 主项目采用 Apache-2.0；依赖、第三方插件、数据与模型仍分别核查。[Nerfstudio 许可](https://github.com/nerfstudio-project/nerfstudio/blob/main/LICENSE)、[gsplat 许可](https://github.com/nerfstudio-project/gsplat/blob/main/LICENSE)


### C 的 NeRF 支线：同一套照片训练 Nerfacto

**输入与环境：** 复用 C 已配置的 Nerfstudio 环境，以及同一个 `processed/scene_demo`。Nerfacto 是融合哈希编码、采样策略、外观条件等技术的现代 NeRF 方法，不等于原始 NeRF 论文逐项复现。采用这里的 CUDA/tiny-cuda-nn 路线时仍需兼容 NVIDIA GPU；Mac 可准备数据或查看远端结果，本节不承诺本地 MPS 训练。官方对默认 Nerfacto 给出约 6 GB 的参考预算，实际峰值随配置、数据和查看/评价阶段变化。[Nerfacto 方法文档](https://docs.nerf.studio/nerfology/methods/nerfacto.html)

1. **处理数据。** 若 C 已成功生成并检查 `processed/scene_demo`，跳过本步骤；否则执行一次，视频输入沿用 C 的 video 分支：

```bash
ns-process-data images --data raw_images --output-dir processed/scene_demo
```

2. **训练。** 在同一环境检查本机参数，然后启动；训练终端会输出查看器地址和这次运行的配置路径：

```bash
ns-train nerfacto --help
ns-train nerfacto --data processed/scene_demo
```

3. **查看与评价。** `PATH_TO_NERFACTO_CONFIG` 必须替换为这次 **nerfacto** 运行的 `config.yml` 路径，不能误用前面 splatfacto 的配置：

```bash
ns-viewer --load-config PATH_TO_NERFACTO_CONFIG
ns-eval --load-config PATH_TO_NERFACTO_CONFIG --output-path evaluation_nerfacto_scene_demo.json
```

这些是独立命令：viewer 为持续运行进程，可关闭后再评价，或在资源足够时使用另一个相同环境的终端。[官方首次训练、重开 viewer 与评价流程](https://docs.nerf.studio/quickstart/first_nerf.html)

**产物与验收：** `outputs/` 下该实验的 `nerfacto` 运行目录保存 `config.yml`、模型检查点等；`ns-eval` 写出指定的 JSON。检查新视角遮挡边缘、细结构与反光处，核实评价视角未进入训练。与 Splatfacto 比较时固定数据划分、图像分辨率和评价设置，并记录两种方法不同的表示与参数；不要把训练 loss 的数值直接横向比较。

Nerfacto 保存的是辐射场模型，不能用 `ns-export gaussian-splat` 直接导成高斯 PLY。需要展示视频时，可在 viewer 的渲染面板设相机路径并使用它生成的 `ns-render` 命令；提取点云/mesh 是另外的派生过程，不等于得到精确 CAD 或物理模型。许可沿用上面的 Nerfstudio/依赖说明。**此支线于 2026-10-03 核查官方文档，未执行安装、训练或性能测试。**

---

## D. hustvl 4DGaussians：先复现合成序列

### 输入、安装与边界

建议用 Linux NVIDIA，单独环境；Windows 原生及 Mac 不在本教程承诺可运行的路径内。Windows 用户若选用已配置 GPU 的 WSL2，需要另外核实驱动、CUDA、编译器及显示兼容性，不能把 WSL2 当成必然成功的一键替代。

从 [D-NeRF 作者仓库](https://github.com/albertpumarola/D-NeRF) 指向的正式数据入口取得 `bouncingballs` 场景，先核查数据条款并手动解压。这里只给路径，不自动下载大数据。预期包含：

```text
4DGaussians/data/dnerf/bouncingballs/
  transforms_train.json
  transforms_test.json
  train/                 # 以 JSON 的 file_path 为准
  test/
```

每个 frame 的 JSON 需要图像路径、相机变换与 `time`；普通静态 NeRF 数据不因改文件夹名就成为动态数据。

官方 README 使用 Python 3.7、PyTorch 1.13.1+cu116 的历史环境。先配置能支持该环境与 GPU 的 CUDA toolkit/编译器；**不要与 B、C 共用环境**。

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

显式 cu116 安装来自 [PyTorch 历史安装表](https://pytorch.org/get-started/previous-versions/)，其余来自 [官方要求](https://github.com/hustvl/4DGaussians#environmental-setups)。这并不保证多年后所有未锁定传递依赖仍可解析；失败时记录日志，使用能支持历史环境的主机/固定依赖，而不是宣称任意新 GPU 可用。

### 训练、渲染、评价

```bash
python train.py -s data/dnerf/bouncingballs --port 6017 --expname dnerf/bouncingballs --configs arguments/dnerf/bouncingballs.py
python render.py --model_path output/dnerf/bouncingballs --skip_train --configs arguments/dnerf/bouncingballs.py
python metrics.py --model_path output/dnerf/bouncingballs
```

产物在 `output/dnerf/bouncingballs/`：配置、`point_cloud/iteration_*/` 下的高斯与变形权重；render 后有 `test/` 和 `video/` 的图像/视频；metrics 后有评价 JSON。完整保留整个实验目录，不要只拷一个 PLY。[保存逻辑](https://github.com/hustvl/4DGaussians/blob/master/scene/__init__.py)、[渲染逻辑](https://github.com/hustvl/4DGaussians/blob/master/render.py)

**验收：** 查看固定视角随时间播放是否连续、冻结时间换视角是否合理；确认 test 数据真的独立。先完整跑通这套已标定数据，再排查自采数据；首次成功不必追逐论文指标。

### 进阶：自己的固定、同步多相机序列

> **这是一段需要真实标定的自采数据适配流程，未实跑；不是把任意多机视频改名后即可训练的配方。下面的相机前提有一项无法验证，就停在数据准备阶段，先修改 loader/投影或完善标定，不执行后续复制与训练命令。**

核查日的 [multipleview_dataset.py](https://github.com/hustvl/4DGaussians/blob/master/scene/multipleview_dataset.py) 硬编码读取 `camera_id=1` 的第一个焦距，并令 `fx=fy`；[getProjectionMatrix](https://github.com/hustvl/4DGaussians/blob/master/utils/graphics_utils.py) 使用对称视锥，不接收每相机的 `cx/cy` 或畸变。**仅做到“图像同尺寸”或“分别去畸变”，不足以满足这些假设。**

若不改官方 loader，必须先用每台相机可信的原始标定，将**全部相机、全部时刻**重映射为同一个虚拟针孔相机内参：

- 相同输出宽高 `W,H`，相同像素焦距 `fx=fy=f0`
- 无畸变、零 skew、主点位于图像中心；按 COLMAP 像素约定为 `cx=W/2, cy=H/2`
- 每台相机的映射在整段序列中固定；相机、变焦与对焦配置也不能改变
- 后续不再各自自动裁剪/缩放；如需裁剪，必须在共同虚拟相机设计中统一处理并重新校验内参

`f0,W,H` 应来自已记录的重映射设计，不能从示例猜一个数。使用 OpenCV 等工具时先统一像素中心约定；COLMAP 与 OpenCV 的主点数值存在半像素约定差异，不能原样混用。重映射需验证有效视场、边界与重投影误差；本教程不提供未经标定验证的自动重映射脚本。[COLMAP 已知内参与像素约定](https://colmap.github.io/faq.html#using-calibration-from-opencv-kalibr-or-other-tools)

保存原始视频；以下目录只能放**通过上述重映射的工作副本**，帧序列要等长、连续且已同步：

```text
data/multipleview/demo_motion/
  cam01/frame_00001.jpg
  cam01/frame_00002.jpg
  cam02/frame_00001.jpg
  cam02/frame_00002.jpg
  ...
```

官方 `multipleviewprogress.sh` 只取各相机第一帧求固定外参，并包含临时目录清理步骤。本教程**不直接执行该脚本**；参阅 [脚本](https://github.com/hustvl/4DGaussians/blob/master/multipleviewprogress.sh) 与 [抽图逻辑](https://github.com/hustvl/4DGaussians/blob/master/scripts/extractimages.py)。满足共同虚拟相机前提后，按以下顺序操作：

1. **用处理后的首帧重新开始。** 新建独占、空的 `prep_demo/images`，将上述 `cam01/frame_00001.jpg` 复制为 `image1.jpg`，`cam02/...` 为 `image2.jpg`，依此类推。记录映射。不要复用原始畸变图或旧 `prep_demo` 的数据库/相机模型
2. **在相同内参下求外参。** 使用 A 的 feature → matcher → mapper 流程，但提取时选择 `--ImageReader.single_camera 1`、`--ImageReader.camera_model SIMPLE_PINHOLE`，用 `--ImageReader.camera_params` 输入已验证的数值列表 `f0,cx,cy`。这里三个符号必须换成实际数值；`cx/cy` 是上述中心主点。mapper 阶段把 `--Mapper.ba_refine_focal_length 0`、`--Mapper.ba_refine_principal_point 0`、`--Mapper.ba_refine_extra_params 0` 全部设为关闭，避免重新拟合 K。运行前检查本机 `colmap mapper -h`；若使用 GUI，等价地关闭 Bundle Adjustment 对应的内参 refine 选项。后续也不要额外执行会放开内参的 BA。[固定内参说明](https://colmap.github.io/faq.html#fix-intrinsics)、[已核对的 Mapper 选项源码](https://github.com/colmap/colmap/blob/main/src/colmap/controllers/option_manager.cc)
3. **通过硬性验收后才继续。** 将 `prep_demo/sparse/0` 在 COLMAP 中导出文本检查：实际 `camera_id=1` 存在，所有图像都关联共同相机，`SIMPLE_PINHOLE` 的尺寸及 `f,cx,cy` 与重映射记录完全对应；相机名与 `camXX` 映射正确，所有机位注册成功，独立重投影检查通过。若 ID、参数、尺寸或图像不符，停止并修正真实数据关联，不可只手改参数使检查表“通过”
4. **建立初始化点云。** 使用这套已验证的处理后首帧和固定内参模型，按 A 的 undistorter → PatchMatch → fusion 建立 `prep_demo/dense/fused.ply`。这里 undistorter 负责生成 dense 工作区，不用于补救整段序列尚未处理的畸变。后续训练仍配对“共同虚拟相机图像 + 与之匹配的固定内参 sparse 模型”；不要将另一次缩放/裁剪产生的相机文件与它们混合
5. **仅在 1–4 全部通过后**，在同一 4DGS 环境整理下面的产物。此处复制的 `prep_demo/sparse/0` 必须是上述**处理后首帧重新重建并验证过**的模型，不是全序列重映射前的旧模型。目标目录应是新目录；已有文件时先检查，不覆盖其他实验

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

LLFF 转换按 [作者脚本](https://github.com/Fyusion/LLFF/blob/master/imgs2poses.py) 执行；它也有相机模型/排序假设，不是通用标定格式转换器。先确保模型完整，再转；保留中间目录方便核对。

6. 阅读复制出来的配置，检查空间包围范围、时间网格、迭代数与批大小。默认模板只是起点，不代表为你的运动校准好的设置
7. 只有前述标定、内参配对及重投影验收均通过后，才训练与渲染：

```bash
python train.py -s data/multipleview/demo_motion --port 6017 --expname multipleview/demo_motion --configs arguments/multipleview/demo_motion.py
python render.py --model_path output/multipleview/demo_motion --skip_train --configs arguments/multipleview/demo_motion.py
```

**关键验收限制：** 原 loader 的默认 test 和 train 存在重叠，不能拿其默认 metrics 报告独立泛化性能。若项目需要留出相机/时间评价，必须改划分并检查样本清单。camera 文件夹中也不要混入缩略图或隐藏文件：loader 用目录条目数推断序列长度。

硬件预算依分辨率、相机数、帧数、初始点、时空网格和增密而变；本教程不承诺固定显存或分钟数。先缩短序列、降分辨率验证流程，再扩展规模。

**许可：** 顶层 [LICENSE](https://github.com/hustvl/4DGaussians/blob/master/LICENSE.md) 是 Apache-2.0，但部分文件保留 graphdeco 研究用途声明，子模块也有自己的许可；对整套系统的商业使用不能只看 GitHub 顶层标签，需要逐项核查来源与条款。数据许可另外处理。

---

## E. Mac：采集、数据准备、CPU SfM 与查看

### 输入与安装

输入为自己的静态照片/视频；已有 Homebrew 时：

```bash
brew install colmap ffmpeg python
colmap -h
ffmpeg -version
```

使用 A 的抽帧、建目录和 SfM 步骤，把提取与匹配改为 CPU。**下面使用核查日当前文档参数名；若本机 help 只列旧 SIFT 参数，应使用旧名字**：

```bash
colmap feature_extractor --database_path scene_demo/database.db --image_path scene_demo/input --ImageReader.single_camera 1 --FeatureExtraction.use_gpu 0
colmap exhaustive_matcher --database_path scene_demo/database.db --FeatureMatching.use_gpu 0
```

之后运行 A 的 mapper/undistorter；A 的 Python 小段在 Mac 使用 `python3` 执行。CPU 只解决这部分数据准备，不把官方 GS CUDA 扩展变成 Metal 扩展，也不使 COLMAP CUDA dense 获得 CPU 实现。

### 产物与验收

- 在 COLMAP GUI 查看相机与稀疏点；把完整去畸变数据目录交给受信任的 NVIDIA 训练环境
- 训练完成后，Mac 可以播放渲染视频；普通点云/mesh 可用 [第 02 章](02_pointcloud_mesh.md) 的工具查看
- 高斯 PLY 需要兼容的浏览器/桌面 Gaussian viewer，具体帧率取决于 GPU、浏览器、分辨率和高斯数量；不保证所有 Mac 流畅
- 任何线上 viewer 都先确认数据是否上传及其条款；私有场景不应随手公开。可优先查看远端训练器渲染出的图片/视频

**完成条件：** 图像已整理，SfM 相机合理，数据在目标训练环境可读取，导出/渲染结果在 Mac 可查看。不是“Mac 已本地训练成功”。

---

## 每条路线都保存一张实验记录

```text
数据来源与授权：
静态/动态、相机数、帧数、图像尺寸：
相机标定/同步/尺度基准：
训练与测试划分清单：
OS、GPU、驱动、toolkit：
代码仓库 commit、子模块 commit：
Python/PyTorch/关键依赖与配置：
实际执行命令与日志路径：
生成文件及本地打开检查：
图像指标、几何验证与尚未解决的问题：
```

真实记录比“最新版 + 默认参数”更有复现价值。排错时从最早失败的一步查起：[07 排错](07_troubleshooting.md)。

---

[返回首页](../README.md) · [坐标与标定](08_robot_frames_calibration.md) · [机器人应用](09_robot_perception_action.md) · [ROS 2 / MuJoCo](10_ros2_mujoco.md)
