# RealSense D435 四圆点双目测量

按《双目测量.docx》实现：采集 IR1 / IR2，关闭红外散斑，检测四个黑圆点，利用双目几何重建圆心，输出上、右、下、左四条空间边长。**不开启深度流、不使用深度图，RGB 仅作参考。**

## 安装与运行

建议 Python 3.10+，D435 通过 USB 3 连接，使用有图形界面的终端：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python stereo_measure.py
```

Linux 如遇权限问题，按 librealsense 官方安装说明配置设备 udev 规则。若 pip 没有当前 Python / 平台对应的 `pyrealsense2` wheel，需要使用受支持的 Python 版本或安装 librealsense Python 绑定。

默认左右红外均为 **1280×720 @ 30 fps**，先丢弃 40 帧。程序从设备读取当前流内参、畸变及左右外参，打印设备序列号和实际基线。若启动提示流配置不支持，可尝试 `--width 848 --height 480`；此时自动使用新分辨率对应内参，不能人为缩放测量图像。

```bash
# 彩色参考窗；RGB 不参与测量
python stereo_measure.py --color

# 静态靶标，8 对同步帧平均降噪（会降低更新率）
python stereo_measure.py --average 8

# 默认同时打开图表窗口（边长柱状图、XY 投影图、原始/滤波时序曲线、三维圆心表），S 保存时生成 charts.png
# 滤波：对四个三维圆心做 15 帧滑动截尾均值，跳变超过 5 mm 视为离群；连续 3 帧一致的跳变视为靶标移动并重新开始
python stereo_measure.py --calibration calibration.json --filter-window 15 --filter-jump-mm 5
python stereo_measure.py --calibration calibration.json --filter-window 1 --no-charts  # 关闭滤波和图表

# 靶标极性默认自适应（白底黑圆 / 黑底白圆自动切换），也可强制指定
python stereo_measure.py --calibration calibration.json --polarity light

# 每帧实时输出到终端（窗口模式同时显示画面；--headless 下持续运行，Ctrl+C 退出）
python stereo_measure.py --calibration calibration.json --live
python stereo_measure.py --calibration calibration.json --headless --live jsonl > live.jsonl

# 无显示器：测量一次并自动保存
python stereo_measure.py --headless --average 8

# 指定相机、手动曝光和增益
python stereo_measure.py --serial YOUR_SERIAL --exposure 6000 --gain 16

# 限定纸面区域（同一矩形用于左右图，要包含右图的视差位移）
python stereo_measure.py --roi 300 180 650 400 --min-area 30 --max-area 1500
```

窗口按键：**S** 保存当前数据，**B** 切换二值图，**Q / Esc** 退出。窗口可缩放，显示缩放不影响原始坐标。`--frames N` 测量 N 次后保存最后一组并退出；无窗口默认测量一次。

## 靶标与检测

- 平整白纸上放置四个大小相近的实心黑圆，确保圆点不贴边、不被遮挡。
- 尽量接近相机正下方放置，使四点按画面左上、右上、右下、左下组成凸四边形。标签为 TL、TR、BR、BL，是图像方位标签，不是可追踪的物理 ID；大角度旋转后标签可能变化。
- 关闭散斑后需有充足环境光，避免反光、过曝或运动模糊。并非所有黑色墨水在红外下都有足够对比度，应检查 IR 图。
- 高斯滤波 + Otsu 反相阈值；用连通域质心作为亚像素数值圆心，再按面积、长宽比、轮廓圆度筛选。默认面积 15～800 px，圆度至少 0.65，可通过命令行调整。
- 检测候选最多保留 16 个，最多保留 12 组凸四边形假设，再联合左右极线误差、正深度、重投影误差、半径相似度和深度一致性匹配。背景圆斑多时建议使用 `--roi`，避免错误靶标。
- 默认极线误差上限 2 px，重投影 RMS 上限 1.5 px，深度范围 150～3000 mm，可分别通过 `--epipolar-px`、`--reprojection-px`、`--min-z-mm`、`--max-z-mm` 修改。

## 几何与结果

采用 SDK **左 IR → 右 IR** 外参，正确读取列主序旋转矩阵。对圆心去畸变后，用 `[I|0]`、`[R|t]` 做一般双目三角化，支持左右内参及主点不同，不直接套用文档示例参数或简单 `fx*B/d`。SDK 平移单位米，输出统一转换为毫米；坐标系为左 IR 的 X 向右、Y 向下、Z 向前。

保存目录为 `output/时间戳/`：

| 文件 | 内容 |
|---|---|
| `left.png` / `right.png` | 用于测量的原始灰度图；开启平均时为平均图 |
| `left_binary.png` / `right_binary.png` | 二值中间结果 |
| `annotated.png` | 圆心标签、匹配连线、四条边长和质量指标 |
| `calibration.json` | 实际分辨率、内参、畸变、外参，平移单位米 |
| `result.json` | 有效状态、圆心像素坐标、原始水平视差、三维坐标、边长、平面和误差指标 |
| `edges.csv` | 四条边长，有效测量才生成 |
| `color.png` | 可选 RGB 参考图 |

边长由**原始三角化三维圆心**计算，不通过投影到拟合平面修改结果。SVD 平面拟合仅作共面诊断，输出 RMS、法向和中心。四点共面残差小不代表绝对尺寸准确，不能当作测量精度证明。画面匹配线连接的是原始图坐标，一般几何下无需严格水平。JSON 中水平视差仅作诊断，实际三角化使用完整内外参。

支持零畸变或标准 Brown-Conrady 红外标定；对非零的其他畸变模型明确拒绝，避免静默使用错误相机模型。默认使用出厂标定，也可通过 `--calibration` 使用下述棋盘格自标定参数。

## 棋盘格双目标定（12×9，格长 2 cm）

标定窗口中的状态、角度分区、进度和操作提示为中文，通过 Pillow 绘制（OpenCV 原生 `putText` 不支持中文）。默认查找系统中文字体；Ubuntu/Debian 缺少字体时安装 `sudo apt install fonts-noto-cjk`，或使用 `--font /路径/中文字体.ttf` 指定字体。为兼容不同 OpenCV 窗口后端，窗口标题保留英文。

默认将 **12×9 理解为方格数量**，因此检测 **11×8 个内角点**；每格边长 **20 mm**，方格区域为 240×180 mm（不含白边）。若实际是 **12×9 个内角点**，请增加 `--cols 12 --rows 9`，对应 13×10 个方格。打印选择实际尺寸、关闭缩放，并用尺检查格长；贴在平整硬板上，保留白边。

```bash
# 开启同步左右红外预览，关闭红外散斑
python stereo_calibrate.py

# 如果 12×9 指的是内角点数
python stereo_calibrate.py --cols 12 --rows 9 --square-mm 20

# 测量时使用自标定结果（分辨率、设备必须与标定一致）
python stereo_measure.py --calibration calibration.json
```

默认**实时自动采集**：两路均检测到完整棋盘、角点连续稳定 0.6 秒、姿态与已存样本有足够差异时，自动保存一对原始图。窗口实时显示偏航角 yaw、俯仰角 pitch、距离、姿态重投影误差，以及 **9 个角度分区的独立进度条和总进度**。分区是正面、上/下/左/右倾斜及四个对角倾斜；正面范围为 yaw、pitch 均在 ±15° 内，其他分区需对应方向超出 15°。方向表示**左 IR 坐标系中的棋盘法向**，不表示手应该向哪个方向移动；面内旋转不算新的倾斜方向。

每区默认采集 **3 对，共至少 27 对**。按提示补齐缺失角度，轻微改变位置、距离或倾角后静止，避免重复采集。**所有分区完成后，自动结束采集、计算标定并保存 `calibration.json` 与 `calibration_report.json`，无需再按键确认。** 保存前仍需完成数值求解；异常会在终端明确报错，原始图保留，不能把采集完成当作标定成功。

快捷键：**A** 开关自动采集；**S** 强制保存当前有效图对（允许重复，不受自动去重限制）；**C** 满 10 对后在后台计算一次标定预览，窗口显示双目 RMS，不覆盖最终文件；**Q / Esc** 提前结束并尝试保存最终标定，少于 10 对会报错。`--manual` 启动手动模式。实时角度使用出厂内参估算，最终报告使用新标定内参重新分析；因此最终分区进度可能与采集进度不同，进度不是精度保证。

```bash
# 每个角度采集 4 对，静止 0.8 秒后自动采集
python3 stereo_calibrate.py --target-per-angle 4 --hold-seconds 0.8
```

实时采集仅使用**本次会话**的图对和进度，避免自动混入旧设备/旧参数的数据；历史目录可通过 `--offline` 显式重算。每次采样立即保存图像、姿态和设备信息，并更新 `calibration_images/capture_progress.json`，该文件反映最近会话已采集的进度。自动去重要求位置移动至少约画幅的 8%、法向角度变化至少 8°，或距离变化至少 12%（满足其中之一）；自动采集间隔默认不少于 1.5 秒，可用 `--capture-interval` 修改。稳定阈值 `--stable-px` 默认 2 像素，是相邻检测间的最大角点位移，不能替代模糊检查。

还应覆盖画面中心、四角和边缘，改变距离，避免全部正对相机。采集时棋盘保持静止、完整可见，避免模糊、反光；不要让棋盘在画面中倒置或接近 90°/180° 旋转，普通对称棋盘没有物理角点 ID，左右角点排序歧义可能使标定失败或 RMS 异常。不要混用不同设备、分辨率或裁剪后的图像。

采集保存至 `calibration_images/时间戳/left.png`、`right.png` 和 `capture.json`。程序先独立标定左右内参及标准五参数畸变，再固定内参计算左 IR → 右 IR 的旋转和平移，**不写入相机固件**。结果为 `calibration.json`（与测量程序格式一致，平移单位米）和 `calibration_report.json`（左右单目 RMS、双目 RMS、基线、每对图的单目重投影误差以及参与/跳过的目录，数组顺序与 `used_pairs` 一致）。默认输出会覆盖同名文件，需要保留旧结果时请指定不同的 `--output`。

```bash
# 用已采集图离线重算，无需设备、GUI 或 pyrealsense2
python stereo_calibrate.py --offline --images calibration_images --output calibration_new.json

# 对已保存的测量图，使用新标定重新测量
python stereo_measure.py --offline output/某次采集目录 --calibration calibration_new.json --headless
```

离线输入目录的每个直接子目录包含一对 `left.png` / `right.png`；未检测到完整棋盘的图对会跳过，分辨率不一致会报错。标定报告中双目 RMS 大于 1 px 时会警告，需检查异常图对和角点顺序并重新采集；RMS 小也不保证绝对精度，应检查基线是否符合设备实际，并用未参与标定的已知尺寸实物验证。程序不会自动剔除高误差图对或保证标定质量。

无匹配时会显示 `NO VALID MATCH`，保存 `valid: false`，不伪造边长。退出码：0 最后一次测量有效，2 无有效匹配，1 启动/输入异常，130 用户中断。多组相似圆点可能造成几何歧义，本程序不能保证自动识别真实靶标，建议在纯净背景或明确 ROI 下使用，并用已知尺寸实物验证。

## 离线复测

不连接相机、不安装 `pyrealsense2` 也可处理已保存的数据（仍需 NumPy 和 OpenCV）：

```bash
python stereo_measure.py --offline output/某次采集目录 --headless
```

也可自行准备包含 `left.png`、`right.png`、`calibration.json` 的目录。左右顺序必须正确，标定必须对应同一图像分辨率。所有参数请见 `python stereo_measure.py --help`。

## 自动测试

```bash
pip install pytest
python -m pytest -q
```

测试覆盖非一致主点三角化、乱序匹配、错误极线/负深度拒绝、合成图圆点检测及离线命令行保存。合成测试不能替代 D435 真机验证及实物精度评估。
