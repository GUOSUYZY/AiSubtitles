# 🤖 本地Whisper语音识别工具 - 完全离线版

这是一个完全离线的语音识别工具，基于OpenAI的Whisper模型，无需任何网络连接即可运行。

## ✨ 特性

- 🔒 **完全离线** - 无需网络连接，保护隐私
- 🚀 **快速处理** - 支持多种模型大小，平衡速度和精度
- 🎬 **视频支持** - 自动提取视频中的音频进行识别
- 📁 **批量处理** - 支持文件夹批量处理
- 🎯 **多语言** - 自动检测语言，支持多种语言识别
- 📝 **SRT字幕** - 自动生成SRT格式字幕文件
- 💻 **GPU加速** - 支持CUDA GPU加速，自动检测NVIDIA显卡

## 📁 目录结构

```
app/
├── whisper_app.py           # 主程序入口
├── main_window.py           # 主窗口模块
├── ui_components.py         # UI组件模块
├── transcription_worker.py  # 转录工作线程
├── app_config.py            # 应用程序配置
├── 启动本地Whisper.bat      # 启动脚本
├── requirements.txt         # 依赖包列表
├── README.md               # 说明文档
├── py/                     # 内置Python环境
├── external/               # 外部工具
│   ├── ffmpeg/            # 音视频处理工具
│   └── handler_local.py   # 离线处理模块
├── models/                 # 语音识别模型
├── temp/                   # 临时文件目录
└── logs/                   # 日志文件
```

## 🚀 快速开始

### 1. 启动程序

**推荐方式**：双击 `启动程序.bat` 文件（更稳定）

**备用方式**：双击 `启动本地Whisper.bat` 文件

**命令行方式**：运行 `py\python.exe whisper_app.py`

**注意**：程序使用PyQt5界面，已内置PyQt5库。

### 2. 选择文件

- **单个文件**：选择音频或视频文件
- **批量处理**：选择包含视频文件的文件夹

### 3. 选择模型

根据需求选择合适的模型：
- **tiny** (39MB) - 速度最快，精度较低
- **base** (74MB) - 平衡速度和精度
- **small** (244MB) - 好精度
- **medium** (769MB) - 很好精度
- **large-v3-turbo** (1.5GB) - 最佳精度（推荐）

### 4. 开始处理

点击"开始处理"按钮，程序将自动：
1. 提取音频（如果是视频文件）
2. 进行语音识别
3. 生成SRT字幕文件

## 📋 支持的格式

### 音频格式
- WAV, MP3, M4A, FLAC, OGG

### 视频格式
- MP4, AVI, MKV, MOV, WMV, FLV, WebM

## 🔧 系统要求

### 最低配置
- Windows 10/11
- 4GB RAM
- 2GB 可用磁盘空间

### 推荐配置
- Windows 10/11
- 8GB+ RAM
- NVIDIA GPU (支持CUDA，已预装PyTorch CUDA版本)
- 5GB+ 可用磁盘空间

## 📦 模型说明

程序会自动检测 `models/` 目录中的可用模型。确保模型文件结构如下：

```
models/
├── models--Systran--faster-whisper-tiny/
│   └── snapshots/
│       └── [hash]/
│           ├── config.json
│           ├── model.bin
│           └── tokenizer.json
├── models--Systran--faster-whisper-small/
│   └── ...
└── ...
```

## 🛠️ 故障排除

### 常见问题

1. **找不到Python环境**
   - 确保 `py/python.exe` 文件存在
   - 重新下载完整程序包

2. **找不到模型文件**
   - 确保模型文件已下载到 `models/` 目录
   - 检查模型文件是否完整

3. **GPU不可用**
   - 程序会自动检测NVIDIA GPU
   - 已预装PyTorch CUDA版本，支持RTX 3070 Ti等显卡
   - 如果没有GPU，会自动使用CPU模式

4. **处理速度慢**
   - 尝试使用更小的模型（如tiny或base）
   - 确保有足够的内存

### 日志查看

程序运行日志保存在 `logs/tools.txt` 文件中，可以查看详细的运行信息。

## 🔒 隐私说明

- 所有处理都在本地完成，不会上传任何数据
- 临时文件会在处理完成后自动清理
- 生成的SRT文件保存在与输入文件相同的位置

## 📄 许可证

本项目基于MIT许可证开源。

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个工具。

---

**注意**：此版本完全离线运行，无需任何网络连接。所有模型文件需要预先下载到 `models/` 目录中。 