# 可运行的 CPU 小例子

[English version](README.en.md)

这些教学脚本只使用 Python 标准库，在 CPU 上完成小规模几何计算。不需要 NumPy、GPU、模型权重、训练、网络连接或依赖安装。要求 Python 3.9 或更新版本；下列命令从仓库根目录执行。若系统使用 `python` 命令，请对应替换 `python3`。

```bash
python3 examples/rigid_transform_demo.py
python3 examples/depth_to_robot_demo.py
python3 examples/grasp_pose_demo.py
python3 -m unittest discover -s examples -p 'test_*.py' -v
```

## 1. 刚体变换：先写清坐标约定

`T_A_B` 把 **B 坐标系中的列向量**转换到 A：`p_A = T_A_B @ p_B`。齐次点的最后一个元素为 1，平移和点位置均以米为单位。组合顺序是 `T_A_C = T_A_B @ T_B_C`，逆变换使用 `Rᵀ` 与 `-Rᵀ t`。脚本中的两个输入旋转分别为 `Rz(+90°)` 和 `Rz(-90°)`，不是仅做平移的特例。

预期关键输出：

```text
point_A = (1.100000, 2.100000, 1.100000)
inverse recovers point_C = (0.400000, -0.100000, 0.200000)
optical (1, 2, 3) -> body = (3.000000, -1.000000, -2.000000)
```

相机 optical 坐标为 `+x` 右、`+y` 下、`+z` 前；body 坐标为 `+x` 前、`+y` 左、`+z` 上。对应变换为 `(x, y, z) → (z, -x, -y)`。它只改变轴约定；真实相机安装的旋转和平移还需另行标定。帮助函数检查矩阵尺寸、有限数值、旋转正交性、行列式 `+1` 及齐次矩阵末行。坐标系名称是否接对仍由调用者负责。

## 2. 深度 → 相机点 → 机器人基座点

输入是 3×3 合成深度图，单位毫米：

```text
   0  1000     0
2000   NaN  2000
   0  1000    -1
```

内参为 `fx=fy=2 px, cx=cy=1 px`。像素中心使用整数 `(u,v)`，`u` 向右、`v` 向下；深度图已假定去畸变并与这些内参对齐。输入尺寸必须与内参的宽高一致。`None`、NaN、正负无穷、零和负深度被掩掉；不支持把字符串等数据当成有效深度。`depth_unit_m=0.001` 只在反投影时将毫米转为米。

默认 `--depth-kind axial` 表示沿光轴的 Z 深度，公式为：

```text
x = (u-cx) * Z / fx
y = (v-cy) * Z / fy
z = Z
```

机器人变换为 `T_base_camera = T_base_mount @ T_mount_camera`。`T_mount_camera` 使用上述 optical→body 轴转换；`T_base_mount` 为 `Rz(+90°)` 和平移 `(0.5, -0.2, 1.0) m`。得到 4 个有效点、5 个被掩掉的样本：

| 像素 `(u,v)` | 相机点 / m | 基座点 / m |
|---|---|---|
| `(1,0)` | `(0,-0.5,1)` | `(0.5,0.8,1.5)` |
| `(0,1)` | `(-1,0,2)` | `(-0.5,1.8,1)` |
| `(2,1)` | `(1,0,2)` | `(1.5,1.8,1)` |
| `(1,2)` | `(0,0.5,1)` | `(0.5,0.8,0.5)` |

若输入是沿视线的欧氏距离，用下面的命令。脚本将 `(x_ray,y_ray,1)` 归一化后乘以距离；同一数值不能同时被解释为 Z 和视线距离。

```bash
python3 examples/depth_to_robot_demo.py --depth-kind ray
```

仍得到 4 个有效点。例如像素 `(1,0)` 的相机点变成 `(0,-0.447214,0.894427)`，基座点变成 `(0.5,0.694427,1.447214)`。独立测试还使用 `fx=1, cx=0, u=1, distance=2`：轴向深度得到 `(2,0,2)`；视线距离得到 `(√2,0,√2)`，向量长度为 2。

可选择生成顶点型 ASCII PLY。输出路径由使用者指定，父目录需存在；已有文件不会被覆盖。文件中的点位于机器人基座坐标系，单位米，头部注释写明这些约定。以下是一个可替换路径的例子：

```bash
mkdir -p output
python3 examples/depth_to_robot_demo.py --ply output/synthetic_base.ply
```

预期 PLY 含 4 个顶点，最后打印 `Wrote 4 base-frame vertices to output/synthetic_base.ply`。PLY 通常不强制约定单位，交给其他工具时仍需传递单位和坐标说明。

## 3. 物体位姿 → TCP 目标 → 法兰目标

已知物体位姿 `T_base_object`、相对物体的 TCP 目标 `T_object_tcp` 和工具标定 `T_flange_tcp`：

```text
T_base_tcp = T_base_object @ T_object_tcp
T_base_flange = T_base_tcp @ inverse(T_flange_tcp)
```

示例输入分别为：`Rz(+90°), t=(0.4,0.2,0.1)`；`Rx(+90°), t=(0.02,0,0.06)`；`Ry(+90°), t=(0,0,0.10)`。平移单位米。工具偏移需连同旋转一起求逆，不能只从 TCP 的基座位置减去工具坐标系中的偏移。

预期关键输出：

```text
TCP position in base = (0.400000, 0.220000, 0.160000)
flange position in base = (0.400000, 0.320000, 0.160000)
flange rotation in base (rows):
  (1.000000, 0.000000, 0.000000)
  (0.000000, 0.000000, -1.000000)
  (0.000000, 1.000000, 0.000000)
```

该目标只是位姿运算结果。它没有证明逆运动学有解、机械臂可达、无碰撞、夹持稳定、接触力正确或可以执行；脚本也不会连接机器人。

## 验证与边界

`test_examples.py` 包含 12 项测试。测试以独立固定数值核对非单位旋转、逆变换、组合顺序、光学轴方向、毫米转米、深度掩码、Z 深度与视线距离的差异、TCP/法兰链、PLY 内容及拒绝覆盖，并检查非法矩阵与尺寸。预期结果为 `Ran 12 tests ... OK`。本次已在 Python 3.14.7 上运行这些测试和三条默认脚本命令；根目录 [VALIDATION.md](../VALIDATION.md) 汇总仓库验收记录。

`mujoco_pose_demo.xml` 是可供后续学习引用的模型草稿：有重力、地面、名为 `tracked_object` 的 mocap 物体；外观使用不参与碰撞的椭球，碰撞使用独立的盒子。mocap 指定物体位姿，物体不会作为自由刚体因重力落下；重力设置不改变这个事实。代码测试只用标准库解析 XML 语法并检查名称/属性，**未加载或运行 MuJoCo**。它没有机械臂、控制器、相机跟踪接入、动态物体参数辨识或物理验证结果。

这些例子验收的是小规模合成数据的几何算术。真实传感器标定、时间同步、尺度恢复、畸变、噪声、物体识别、机器人控制及仿真物理均需独立验证。
