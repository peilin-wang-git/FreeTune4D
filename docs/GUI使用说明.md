# FreeTune4D 图形界面使用说明

## 启动

本界面使用 Python 标准库 Tkinter，不引入新的深度学习或 GUI 包版本。Python 发行版必须包含 Tk 8.6；部分 Linux 发行版需要通过系统包安装 `python3-tk`。请先按项目原有方式准备 TensorFlow、PyTorch、VoxelMorph、CUDA、Octave 和 Elastix 环境，然后在仓库根目录运行：

```bash
python run_gui.py
```

也可以运行：

```bash
python -m freetune4d_gui
```

## 后端能力

- **T2-weighted**：界面调用仓库现有的 `STEP_02_UTSW_ImageTest_YP_T2_Clinic_Amp_v2.py` 和 `4DMRI Synthesis_UTSW_DVFsmooth_YP_T2_Steps.py`，没有复制或修改重建数学逻辑。
- **T1-weighted**：当前仓库没有完整 T1 入口。界面保留明确的 T1 选项，但选择后会显示后端不可用并禁止启动，不会用 T2 冒充 T1。

## 操作流程

1. 选择动态 LQ 4D-MRI DICOM 目录；当前后端要求文件名匹配 `IM-*`。
2. 选择静态 UQ 3D-MRI DICOM 目录；当前后端同样要求 `IM-*`。
3. 明确选择 MRI modality。当前可运行后端为 T2。
4. 选择输出根目录。
5. 展开 **Advanced Settings**，选择真实的 `coarse.h5`、`fine.h5`；可选指定参考 DICOM，未指定时使用静态目录中排序后的第一个文件。
6. 输入和资源全部有效后，**Preprocessing** 才会启用；运行成功后 **Motion Reconstruction** 才会启用。
7. 重建完成后，界面验证 UQ DICOM、LQ DICOM 和 QC 文件并显示实际文件数和相位数。

界面默认窗口为 1280×840，最小尺寸为 1100×720。上半部使用约 65:35 的“输入与配置／工作流与状态”双栏布局；下半部 Runtime Log 通过分隔条调整高度。标题、章节、正文、按钮和日志分别使用约 24、16、13、14 和 12 的字体尺寸，并在 Windows 上启用系统 DPI 感知。

## 输出结构

```text
<output_root>/
├── preprocessing/
│   ├── phase_T2.mat
│   ├── Full_data.mat（后端生成时）
│   ├── case_manifest.json
│   └── _backend_case/（保持旧脚本所需目录结构）
├── reconstructed/
│   ├── UQ_T2_<frame>.mat
│   ├── T2w_frame<frame>_<slice>.dcm
│   ├── LQ_T2w_frame<frame>_<slice>.dcm
│   └── case_manifest.json
└── QC/
    ├── preprocessing/
    └── reconstruction/
```

`case_manifest.json` 保存不含患者姓名的配置签名。只要输入路径、输出路径、模态、模型、参考 DICOM 或相位数变化，旧预处理状态立即失效。

## 执行与错误处理

两个原后端脚本均由工作线程启动的独立子进程执行。GUI 线程只接收日志和完成/失败事件，因此长任务不会阻塞界面；TensorFlow/PyTorch/CUDA 行为仍由原脚本决定。界面不显示虚构百分比，只使用不定进度条，并流式显示后端真实输出。

后端非零退出、缺少权重、无 DICOM、T1 不支持、预处理清单不匹配或输出缺失都会形成明确错误。详细 traceback 保留在 Runtime Log。已有受管输出不会被静默删除，重新预处理或重建前会要求确认。

### CUDA 诊断模式

如果 Runtime Log 中出现 CUDA 异步错误，可在 **Advanced Settings** 中临时启用：

```text
CUDA diagnostic mode (CUDA_LAUNCH_BLOCKING=1; slower)
```

该选项只向本次后端子进程注入 `CUDA_LAUNCH_BLOCKING=1`，不会修改主 GUI 进程或永久环境。同步执行会降低速度，因此正常运行时应关闭。失败时 Runtime Log 会分别保留 stdout、stderr、准确命令、工作目录、退出码和末尾 traceback；“Copy Error”可复制结构化诊断信息。

## 当前部署前提

仓库本身不包含 `coarse.h5`、`fine.h5`、匿名化测试 DICOM 或 Linux Elastix，因此这些资源必须由部署者提供。原重建脚本还会初始化 Octave，并调用 PATH 中的 `elastix`；界面不会改变这些既有要求，也不会自动切换到 CPU。
