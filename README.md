# RadarPcd

`RadarPcd` 用于从雷达点云中交互式提取棋盘格角点，并结合图像角点与相机内参，完成相机-雷达联合标定。

项目当前包含两部分核心流程：

1. 用 Open3D 图形界面在雷达点云中选取标定板并导出雷达角点。
2. 用 `joint_calibration.py` 将雷达角点、图像角点和相机内参联合求解外参，并输出重投影结果。

## 目录结构

```text
RadarPcd/
├── docs/                        # README 使用的截图资源
├── output/                      # 运行后生成的角点、参数和标定结果
├── res/                         # 示例点云、图像等输入资源
├── scripts/
│   └── startup.sh               # 环境初始化脚本
├── src/                         # 交互式点云选点程序
├── joint_calibration.py         # 联合标定脚本
├── rgb_chessboard.py            # 图像棋盘格角点提取脚本
└── run.sh                       # 启动雷达角点选取界面
```

## 环境准备

首次使用建议执行：

```bash
chmod +x scripts/startup.sh run.sh
./scripts/startup.sh
```

`scripts/startup.sh` 当前会创建 `RadarPcd` conda 环境并安装 `open3d==0.19.0`。

进入环境：

```bash
conda activate RadarPcd
```

完成工作后退出环境：

```bash
conda deactivate
```

## 标准使用流程

推荐按下面顺序完成一整套标定。

### 1. 准备输入数据

默认输入文件如下：

1. 雷达点云：`res/lidar.csv` 或 `res/map.csv`
2. 棋盘图像：`res/chess.jpg`
3. 相机内参：`output/camera_params.npz` 或 `output/camera_params.yaml`

如果你使用自己的数据，请先准备与这些文件等价的输入。

### 2. 在雷达点云中选取标定板角点

运行：

```bash
./run.sh
```

`run.sh` 默认执行：

```bash
python3 -u src/main.py --input res/lidar.csv
```

程序会在 `output/lidar_corner.csv` 中输出雷达角点。

雷达标点程序的详细操作、输入输出格式、模块职责和注意事项，见 [src/README.md](file:///Users/kerry-mac/RadarPcd/src/README.md)。

### 3. 提取图像棋盘格角点

运行：

```bash
python rgb_chessboard.py
```

脚本默认读取 `res/chess.jpg`，并将角点输出到 `output/camera_corner.csv`。

如果检测失败，请优先检查：

1. `PATTERN_SIZE` 是否填写为内角点数量，而不是棋盘格数量。
2. 图像是否清晰，棋盘格是否完整可见。
3. 是否存在明显反光、过曝或遮挡。

### 4. 运行联合标定

在完成雷达角点和图像角点准备后，运行：

```bash
python joint_calibration.py
```

默认输入：

1. 图像角点：`output/camera_corner.csv`
2. 雷达角点：`output/lidar_corner.csv`
3. 相机参数：`output/camera_params.npz`
4. 原始图像：`res/chess.jpg`
5. 雷达点云：`res/lidar.csv`

如果你使用 YAML 相机参数文件，也可以显式指定：

```bash
python joint_calibration.py --camera-params output/camera_params.yaml
```

### 5. 查看标定结果

运行完成后，`output/` 中常见产物包括：

1. `cam_lidar_extrinsic.npz`：保存外参矩阵。
2. `cam_lidar_extrinsic.mat`：便于其他工具链读取的外参文件。
3. `calibration_report.json`：包含 RMSE、内点比例、排序配置等评估信息。
4. `reprojection_detail.csv`：每个角点的重投影结果及是否为内点。
5. `lidar_projection_overlay.jpg`：将雷达点投影到图像后的叠加效果图。
6. `lidar_projection_canvas.jpg`：仅保留投影点的空白画布图。

## 输出文件职责

本项目常见文件的作用如下：

1. `output/lidar_corner.csv`：从点云界面导出的雷达角点。
2. `output/camera_corner.csv`：从图像中提取的棋盘格角点。
3. `output/camera_params.npz` 或 `output/camera_params.yaml`：相机内参与畸变参数。
4. `output/calibration_report.json`：标定质量报告，适合先看数值再看图。
5. `output/lidar_projection_overlay.jpg`：最直观的结果检查图。

## 结果检查建议

建议同时看数值和可视化，不要只看单张图是否“看起来能对上”。

优先检查以下指标：

1. `rmse_px` 是否稳定偏小。
2. `num_inliers` 是否足够高。
3. 棋盘格各区域是否整体贴合，而不是只在局部贴合。
4. 投影到其他图像时是否仍然保持一致。

如果出现“用于标定的那一张图基本准确，但映射到另外两张图明显偏移”的情况，通常优先考虑以下原因：

1. 雷达角点和图像角点顺序不一致。
2. 某一组雷达角点存在翻转、转置或起始点定义不一致。
3. 手工微调造成整行或整列角点系统性偏移。
4. 单张数据标定本身不够稳定，外参只拟合了当前这一张图。

## 常见问题

### 1. 为什么本张图能对上，换一张图就不准？

这通常不表示投影公式一定写错，更常见的原因是：

1. 单张数据对外参约束不足。
2. 角点顺序在不同数据组之间不一致。
3. 某些角点存在手工标注误差。

对于固定安装的相机和雷达，理论上应只有一套稳定外参。如果不同样本各自求出来的矩阵差异很大，优先排查数据一致性和标注质量。

### 2. 需要把多组矩阵直接平均吗？

不建议直接对多个外参矩阵做普通平均。

更合理的做法是：

1. 采集多组姿态不同的数据。
2. 保证每组角点顺序定义完全一致。
3. 用多组数据联合优化一套统一外参。

### 3. 第三组数据为什么更容易出问题？

如果某一组数据需要额外翻转或特殊顺序处理，通常说明这组数据的角点方向定义和其他组不一致。遇到这种情况，应先统一角点顺序，再讨论外参是否稳定。

## 主要模块说明

1. `src/`：雷达标点程序，详细说明见 [src/README.md](file:///Users/kerry-mac/RadarPcd/src/README.md)。
2. `joint_calibration.py`：联合标定、误差统计和投影可视化。
3. `rgb_chessboard.py`：图像棋盘格角点提取。

## 声明

1. 本项目仅供学习交流使用，不涉及商业用途。
2. 如有建议，可联系作者：13670196946@163.com
