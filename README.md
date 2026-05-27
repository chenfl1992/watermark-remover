# 🧽 批量图片去水印工具

**本地运行，无需联网，零额外依赖。** OpenCV inpainting + 智能文本检测 + 多区域框选 + Web UI。

## 快速开始

```bash
# 安装依赖
pip install opencv-python-headless numpy Pillow tqdm

# 启动 Web 界面
python main.py --web
# 访问 http://localhost:7860
```

## Web 界面功能

| 功能 | 说明 |
|------|------|
| 🎨 **自动颜色（智能）** | 基于边缘检测+自适应亮度，精准定位文字水印 |
| 🔤 **指定文字** | 针对「豆包AI生成」等固定文字，合成参考图+模板匹配 |
| ✂️ **手动区域（多框选）** | 在原图上拖拽框选，支持多个区域同时去除 |
| 📍 **预设位置** | 四角/中心 logo，按位置+大小比例 |
| 📦 **批量处理** | 指定输入输出目录，多线程并发处理 |

## 命令行

```bash
# 自动检测（最常用，适合半透明文字水印）
python main.py ./images -o ./output

# 指定文字水印（如 豆包AI生成）
python main.py ./images -o ./output -m text --target-text "豆包AI生成"

# 手动区域（单区域）
python main.py ./images -o ./output -m manual --roi 100 50 300 150

# 多区域（批量处理文件中用 Web UI 配置）
python main.py ./images --position bottom-right -m position

# 深色水印（浅色背景上的黑字）
python main.py ./images -o ./output --dark

# 调参
python main.py ./images -o ./output --radius 7 --workers 8
```

## 一键部署（公网访问）

### 方式一：Render（推荐，免费）

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/chenfl1992/watermark-remover)

1. 点击上方按钮
2. 登录 GitHub 并授权 Render
3. 选择 Free 套餐，区域选 Singapore（亚洲访问最快）
4. 等待 3-5 分钟构建部署完成
5. 访问 `https://watermark-remover.onrender.com`

### 方式二：GitHub Container Registry → 任意云平台

工作流已配置好，每次推送 `main` 分支会自动构建 Docker 镜像至 GHCR。

```bash
docker pull ghcr.io/chenfl1992/watermark-remover:latest
docker run -p 7860:7860 ghcr.io/chenfl1992/watermark-remover:latest
# 访问 http://localhost:7860
```

### 方式三：Docker 手动构建

```bash
docker build -t watermark-remover .
docker run -p 7860:7860 watermark-remover
# 访问 http://localhost:7860
```

### 方式四：Hugging Face Spaces

1. 创建新 Space → 选择 **Docker** 运行时
2. 导入本仓库
3. 自动构建后即可访问

## 检测模式详解

| 模式 | 效果 | 适用场景 |
|------|------|----------|
| 🎨 `color`（智能） | ⭐⭐⭐⭐⭐ | 半透明文字水印、图片上的文字 |
| 🔤 `text` | ⭐⭐⭐⭐ | 「豆包AI生成」等固定文字水印 |
| ✂️ `manual`（多框选） | 精确 | 知道水印位置（可框多个） |
| 📍 `position` | ⭐⭐⭐ | 四角固定 logo/水印 |

## Python API

```python
from water_remover import WatermarkConfig, process_batch, process_image

# 批量处理
config = WatermarkConfig(
    mode="color",
    inpaint_radius=5,
    workers=4,
    output_dir="./output",
)
results = process_batch("./input_images", config)

# 单张处理
result_path, success, msg = process_image(
    "input.jpg",
    WatermarkConfig(mode="text", target_text="豆包AI生成"),
    output_path="output.jpg",
)
```

## 效果对比

| 原图 | 旧版（HSV范围） | 新版（智能检测） |
|------|-----------------|------------------|
| 遮罩覆盖 100% | 遮罩覆盖 **86.7%**（全图变色） | 遮罩覆盖 **1.7%**（仅水印） |
| — | 非水印区色差 **92** | 非水印区色差 **0.00** ✅ |
