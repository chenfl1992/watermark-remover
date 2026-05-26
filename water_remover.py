#!/usr/bin/env python3
"""
water_remover.py — 批量图片去水印核心引擎

支持功能：
- 手动区域遮罩（输入 ROI 坐标）
- 自动颜色检测（半透明白/黑水印）
- 基于 HSV 的精细颜色过滤
- OpenCV Telea / Navier-Stokes 图像修复
- 多线程批量处理
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional, Literal
import json
import shutil


# ── 配置数据结构 ────────────────────────────────────────────────

@dataclass
class WatermarkConfig:
    """去水印配置"""
    # 检测方式: manual / color / position / text
    mode: Literal["manual", "color", "position", "text"] = "color"

    # ── manual 模式 ──
    #  ROI 矩形 [x, y, w, h] 相对于原图坐标（单选框，兼容旧版本）
    roi: Optional[List[int]] = None  # [x, y, w, h]
    #  多区域框选（手动模式）：[[x, y, w, h], ...]
    roi_list: List[List[int]] = field(default_factory=list)

    # ── color 模式 ──
    #  颜色范围，多个范围用列表叠加（OR 逻辑）
    #  每个范围: {lower: [H,S,V], upper: [H,S,V]}
    #  留空则使用默认预设
    color_ranges: List[dict] = field(default_factory=list)

    # ── position 模式 ──
    #  水印位置: top-left, top-right, bottom-left, bottom-right, center
    position: str = "bottom-right"
    #  水印预估大小，占图片宽高的比例
    size_ratio: float = 0.15

    # ── text 模式 ──
    #  要查找的文字（用于精确匹配）
    target_text: str = "豆包AI生成"
    #  模板匹配阈值（0-1，越高越严格）
    text_match_threshold: float = 0.35

    # ── 通用设置 ──
    #  修复算法: telea / ns
    inpaint_method: Literal["telea", "ns"] = "telea"
    #  修复半径（越大越模糊，适合大块水印）
    inpaint_radius: int = 5
    #  处理前缩放（加快处理速度）
    scale_factor: float = 1.0
    #  输出质量
    output_quality: int = 95
    #  覆盖原图（不另存为副本）
    overwrite: bool = False
    #  输出目录（overwrite=False 时使用）
    output_dir: Optional[str] = None
    #  保留目录结构
    preserve_structure: bool = True
    #  线程数
    workers: int = 4
    #  自动二值化阈值（0-255，0=自动计算）
    threshold: int = 0

    # ── 形态学操作 ──
    #  膨胀核大小（扩大水印边缘，覆盖更干净）
    dilate_kernel: int = 2
    #  腐蚀核大小（去除噪点）
    erode_kernel: int = 0

    # ── 高级 ──
    #  水印是白底上的深色文字（反色检测）
    dark_on_light: bool = False


# ── 默认颜色预设 ──────────────────────────────────────────────

# 旧版宽泛 HSV 范围已废弃（会遮盖 80%+ 图片区域）
# 新版使用 _detect_text_watermark() 基于边缘+自适应亮度
DEFAULT_COLOR_RANGES = []  # 空 = 使用智能文本检测
DEFAULT_DARK_RANGES = []  # 空 = 使用智能文本检测（dark_on_light 反转亮度判断）


# ── 核心函数 ──────────────────────────────────────────────────

def create_mask(image: np.ndarray, config: WatermarkConfig) -> np.ndarray:
    """
    根据配置生成水印遮罩（二值图）
    返回: mask (uint8, 0/255)
    """
    h, w = image.shape[:2]

    if config.mode == "manual":
        return _mask_manual(image, config)

    elif config.mode == "color":
        return _mask_color(image, config)

    elif config.mode == "position":
        return _mask_position(image, config)

    elif config.mode == "text":
        return _mask_text(image, config)

    else:
        raise ValueError(f"Unknown mode: {config.mode}")


def _mask_manual(image: np.ndarray, config: WatermarkConfig) -> np.ndarray:
    """手动区域遮罩（支持单个 ROI 和多区域 roi_list）"""
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # 收集所有 ROI
    rects = []
    if config.roi_list:
        rects.extend(config.roi_list)
    if config.roi:
        rects.append(config.roi)
    
    for rect in rects:
        x, y, rw, rh = rect
        x = max(0, min(x, image.shape[1]))
        y = max(0, min(y, image.shape[0]))
        rw = min(rw, image.shape[1] - x)
        rh = min(rh, image.shape[0] - y)
        mask[y:y+rh, x:x+rw] = 255
    
    return mask


def _mask_color(image: np.ndarray, config: WatermarkConfig) -> np.ndarray:
    """
    水印检测遮罩（智能模式）
    
    策略：
    1. 用户提供了 color_ranges → 使用传统 HSV 范围检测（精确控制）
    2. 否则 → 使用 _detect_text_watermark() 基于边缘+自适应亮度
    """
    if config.color_ranges:
        # 用户自定义 HSV 范围 → 精确颜色检测
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        combined = np.zeros(image.shape[:2], dtype=np.uint8)
        for cr in config.color_ranges:
            lower = np.array(cr["lower"], dtype=np.uint8)
            upper = np.array(cr["upper"], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
            combined = cv2.bitwise_or(combined, mask)
        
        # 形态学后处理
        if config.erode_kernel > 0:
            kernel = np.ones((config.erode_kernel, config.erode_kernel), np.uint8)
            combined = cv2.erode(combined, kernel, iterations=1)
        if config.dilate_kernel > 0:
            kernel = np.ones((config.dilate_kernel, config.dilate_kernel), np.uint8)
            combined = cv2.dilate(combined, kernel, iterations=1)
        return combined
    
    # 智能文本水印检测（默认）
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return _detect_text_watermark(gray, config)


def _detect_text_watermark(gray: np.ndarray, config: WatermarkConfig) -> np.ndarray:
    """
    基于边缘检测 + 自适应亮度阈值的文本水印检测。
    专为半透明文字水印设计，比 HSV 范围精确得多。
    """
    h, w = gray.shape
    
    # 1. 自适应亮度阈值
    # 用亮度直方图找到阈值：取 top-8% 最亮像素
    flat = gray.flatten()
    sorted_vals = np.sort(flat)
    pct = 0.08  # 取最亮的 8%
    idx = int(len(flat) * (1 - pct))
    brightness_thresh = max(sorted_vals[idx], 180)
    
    if config.dark_on_light:
        # 深色水印：取最暗的 8%
        brightness_thresh = min(sorted_vals[int(len(flat) * pct)], 80)
        _, bright = cv2.threshold(gray, brightness_thresh, 255, cv2.THRESH_BINARY_INV)
    else:
        # 白色水印：取最亮的 8%
        _, bright = cv2.threshold(gray, brightness_thresh, 255, cv2.THRESH_BINARY)
    
    # 2. Canny 边缘检测 — 水印文字有清晰的轮廓
    edges = cv2.Canny(gray, 30, 100)
    
    # 3. 膨胀边缘，连接字符笔画
    edge_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=3)
    
    # 4. 取交集：只有在「最亮区域」且「附近有边缘」的像素才保留
    text_mask = cv2.bitwise_and(bright, edge_dilated)
    
    # 5. 形态学闭合，填充文字内部间隙
    text_mask = cv2.morphologyEx(
        text_mask, cv2.MORPH_CLOSE,
        np.ones((5, 5), np.uint8), iterations=2
    )
    
    # 6. 连通域过滤：去掉极小噪点和极大区域
    max_area = int(h * w * 0.15)  # 水印不超过 15% 图片面积
    min_area = 20
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(text_mask, 8)
    result = np.zeros_like(text_mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area < area < max_area:
            result[labels == i] = 255
    
    # 7. 膨胀覆盖半透明边界
    dk = max(config.dilate_kernel, 2)
    result = cv2.dilate(result, np.ones((dk * 2 + 1, dk * 2 + 1), np.uint8), iterations=1)
    
    return result


def _mask_position(image: np.ndarray, config: WatermarkConfig) -> np.ndarray:
    """位置预设遮罩"""
    h, w = image.shape[:2]
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mw = int(w * config.size_ratio)
    mh = int(h * config.size_ratio)

    positions = {
        "top-left": (0, 0),
        "top-right": (w - mw, 0),
        "bottom-left": (0, h - mh),
        "bottom-right": (w - mw, h - mh),
        "center": (w // 2 - mw // 2, h // 2 - mh // 2),
    }

    if config.position in positions:
        x, y = positions[config.position]
        mask[y:y+mh, x:x+mw] = 255

    return mask


# ── 字体查找 ──────────────────────────────────────────────────

_CACHED_FONT_PATH = None

def _find_chinese_font() -> str:
    """查找系统中支持中文的字体"""
    global _CACHED_FONT_PATH
    if _CACHED_FONT_PATH:
        return _CACHED_FONT_PATH
    
    candidates = [
        "/System/Library/Fonts/STHeiti Medium.ttc",       # macOS 黑体
        "/System/Library/Fonts/Supplemental/Songti.ttc",   # macOS 宋体
        "/System/Library/Fonts/PingFang.ttc",              # macOS 苹方
        "/Library/Fonts/Arial Unicode.ttf",                # Arial Unicode
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",  # Linux 文泉驿
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Noto
    ]
    for fp in candidates:
        if os.path.exists(fp):
            _CACHED_FONT_PATH = fp
            return fp
    
    # Fallback: try fc-list
    try:
        import subprocess
        r = subprocess.run(
            ["fc-list", ":lang=zh", "file"],
            capture_output=True, text=True, timeout=3
        )
        for line in r.stdout.strip().split("\n"):
            fp = line.split(":")[0].strip()
            if fp and os.path.exists(fp):
                _CACHED_FONT_PATH = fp
                return fp
    except Exception:
        pass
    
    return None


# ── 文本水印检测（模板匹配） ──────────────────────────────────

_CACHED_REF_CACHE = {}  # (text, font_size, font_path) → np.ndarray

def _render_text_ref(text: str, font_size: int, font_path: str) -> np.ndarray:
    """渲染指定文字为灰度图片（白字黑底）"""
    cache_key = (text, font_size, font_path)
    if cache_key in _CACHED_REF_CACHE:
        return _CACHED_REF_CACHE[cache_key]
    
    from PIL import ImageDraw, ImageFont
    
    # 用大画布保证文字完整渲染
    canvas_w = font_size * len(text) * 2
    canvas_h = font_size * 3
    img = Image.new("L", (canvas_w, canvas_h), 0)
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()
    
    draw.text((10, canvas_h // 4), text, fill=255, font=font)
    
    # 裁剪到实际内容区域
    arr = np.array(img, dtype=np.uint8)
    rows = np.any(arr > 0, axis=1)
    cols = np.any(arr > 0, axis=0)
    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        arr = arr[rmin:rmax+1, cmin:cmax+1]
    
    # 加白边（模板匹配需要外边框）
    arr = cv2.copyMakeBorder(arr, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=255)
    
    _CACHED_REF_CACHE[cache_key] = arr
    return arr


def _mask_text(image: np.ndarray, config: WatermarkConfig) -> np.ndarray:
    """
    精确文本水印检测：用合成参考图 + 多尺度模板匹配查找指定文字。
    
    专为 "豆包AI生成" 等固定文字水印设计。
    不需要 OCR，纯 OpenCV 模板匹配。
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    text = config.target_text or "豆包AI生成"
    threshold = config.text_match_threshold
    
    # 找字体
    font_path = _find_chinese_font()
    if font_path is None:
        # 无中文字体，回退到位置模式
        cfg = WatermarkConfig(mode="position", position="bottom-right", size_ratio=0.12)
        return _mask_position(image, cfg)
    
    best_score = 0.0
    best_box = None
    
    # 多尺度匹配：从图片尺寸估出大致文字大小
    # 水印文字通常占图片宽度的 10%~50%
    base_size = int(min(w, h) * 0.04)  # 基础字号（按图尺寸缩放）
    
    # 尝试 10 个尺度，覆盖小字到大字
    scales = [base_size * s for s in [0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.2, 2.8, 3.5]]
    scales = [int(s) for s in scales if 12 <= s <= min(w, h) * 0.6]
    
    for font_size in sorted(set(scales)):
        ref = _render_text_ref(text, font_size, font_path)
        ref_h, ref_w = ref.shape
        
        # 参考图不能比目标图大
        if ref_h > h or ref_w > w:
            continue
        
        # 模板匹配
        result = cv2.matchTemplate(gray, ref, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        
        if max_val > best_score:
            best_score = max_val
            best_box = (max_loc[0], max_loc[1], ref_w, ref_h)
    
    # 检查是否匹配到
    if best_score < threshold or best_box is None:
        return np.zeros((h, w), dtype=np.uint8)
    
    # 创建遮罩（适当膨胀覆盖半透明边界）
    x, y, rw, rh = best_box
    pad = int(max(5, min(rw, rh) * 0.15))  # 15% 边距
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w, x + rw + pad)
    y2 = min(h, y + rh + pad)
    
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    
    # 膨胀覆盖半透明边缘
    dk = max(config.dilate_kernel, 2)
    mask = cv2.dilate(mask, np.ones((dk * 2 + 1, dk * 2 + 1), np.uint8), iterations=2)
    
    return mask


def inpaint(image: np.ndarray, mask: np.ndarray, config: WatermarkConfig) -> np.ndarray:
    """执行图像修复"""
    method = cv2.INPAINT_TELEA if config.inpaint_method == "telea" else cv2.INPAINT_NS
    radius = max(1, config.inpaint_radius)
    return cv2.inpaint(image, mask, radius, method)


def process_image(
    input_path: str,
    config: WatermarkConfig,
    output_path: Optional[str] = None,
) -> Tuple[str, bool, str]:
    """
    处理单张图片
    返回: (路径, 是否成功, 消息)
    """
    try:
        # 读取
        img = cv2.imread(input_path)
        if img is None:
            return (input_path, False, "无法读取图片")

        # 缩放
        if config.scale_factor != 1.0:
            h, w = img.shape[:2]
            new_size = (int(w * config.scale_factor), int(h * config.scale_factor))
            img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
            scale_used = True
        else:
            scale_used = False

        # 生成遮罩
        mask = create_mask(img, config)

        # 检查遮罩是否为空
        if cv2.countNonZero(mask) == 0:
            return (input_path, False, "未检测到水印区域")

        # 修复
        result = inpaint(img, mask, config)

        # 如果有缩放，恢复原尺寸
        if scale_used:
            h, w = cv2.imread(input_path).shape[:2]
            result = cv2.resize(result, (w, h), interpolation=cv2.INTER_LINEAR)

        # 保存
        out_path = output_path or input_path
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # 用 PIL 保存（支持质量参数）
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(result_rgb)
        ext = os.path.splitext(out_path)[1].lower()

        save_kwargs = {"quality": config.output_quality}
        if ext == ".webp":
            # WebP 需要额外参数
            pil_img.save(out_path, "WEBP", **save_kwargs)
        elif ext in (".jpg", ".jpeg"):
            pil_img.save(out_path, "JPEG", **save_kwargs)
        else:
            pil_img.save(out_path)

        return (out_path, True, "成功")

    except Exception as e:
        return (input_path, False, str(e))


def find_images(input_path: str, recursive: bool = True) -> List[str]:
    """递归查找图片文件"""
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
    path = Path(input_path)

    if path.is_file():
        if path.suffix.lower() in exts:
            return [str(path)]
        return []

    if path.is_dir():
        pattern = "**/*" if recursive else "*"
        results = []
        for f in path.rglob("*"):
            if f.is_file() and f.suffix.lower() in exts:
                results.append(str(f))
        return sorted(results)

    return []


def process_batch(
    input_path: str,
    config: WatermarkConfig,
    progress_callback=None,
) -> List[Tuple[str, bool, str]]:
    """
    批量处理图片
    返回: [(路径, 是否成功, 消息)]
    """
    # 统一为绝对路径
    input_path_abs = os.path.abspath(input_path)
    images = find_images(input_path_abs)
    if not images:
        return [(input_path_abs, False, "未找到图片文件")]

    results = []
    total = len(images)
    base_path = input_path_abs if os.path.isdir(input_path_abs) else None

    def get_output(img_path):
        if config.overwrite:
            return img_path
        if config.output_dir:
            abs_output = os.path.abspath(config.output_dir)
            if config.preserve_structure and base_path:
                try:
                    rel = os.path.relpath(img_path, base_path)
                    # 确保不跳出输出目录
                    return os.path.join(abs_output, rel)
                except ValueError:
                    pass
            return os.path.join(abs_output, os.path.basename(img_path))
        return img_path

    with ThreadPoolExecutor(max_workers=config.workers) as executor:
        futures = {}
        for img_path in images:
            out_path = get_output(img_path)
            future = executor.submit(process_image, img_path, config, out_path)
            futures[future] = img_path

        completed = 0
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1
            if progress_callback:
                progress_callback(completed, total, result[0], result[1])

    return results


def batch_to_dir(
    input_dir: str,
    output_dir: str,
    config: Optional[WatermarkConfig] = None,
    **kwargs,
) -> List[Tuple[str, bool, str]]:
    """便捷函数：批量处理目录到输出目录"""
    if config is None:
        config = WatermarkConfig(**kwargs)
    config.output_dir = output_dir
    return process_batch(input_dir, config)


# ── 保存/加载配置 ─────────────────────────────────────────────

def save_config(config: WatermarkConfig, path: str):
    """保存配置到 JSON"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, ensure_ascii=False, indent=2)


def load_config(path: str) -> WatermarkConfig:
    """从 JSON 加载配置"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return WatermarkConfig(**data)


# ── 命令行入口（理论上用 cli.py 但也可直接运行） ──────────────

def main():
    """简单 CLI 入口"""
    import argparse

    parser = argparse.ArgumentParser(description="批量图片去水印工具")
    parser.add_argument("input", help="输入文件或目录")
    parser.add_argument("-o", "--output", help="输出目录（默认覆盖原图）")
    parser.add_argument("-m", "--mode", default="color",
                        choices=["manual", "color", "position", "text"])
    parser.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"),
                        help="手动 ROI: x y w h")
    parser.add_argument("--position", default="bottom-right",
                        help="水印位置 (color 模式)")
    parser.add_argument("--method", default="telea", choices=["telea", "ns"])
    parser.add_argument("--radius", type=int, default=5,
                        help="修复半径 (默认 5)")
    parser.add_argument("--workers", type=int, default=4,
                        help="并发线程数")
    parser.add_argument("--dark", action="store_true",
                        help="深色水印（浅色背景）")
    parser.add_argument("--target-text", default="豆包AI生成",
                        help="text 模式：要查找的文字（默认: 豆包AI生成）")

    args = parser.parse_args()

    config = WatermarkConfig(
        mode=args.mode,
        inpaint_method=args.method,
        inpaint_radius=args.radius,
        workers=args.workers,
        dark_on_light=args.dark,
        target_text=args.target_text,
        overwrite=(args.output is None),
        output_dir=args.output,
    )

    if args.mode == "manual" and args.roi:
        config.roi = list(args.roi)
    if args.mode == "position":
        config.position = args.position

    print(f"🚀 开始处理: {args.input}")
    print(f"   模式: {config.mode}, 算法: {config.inpaint_method}")
    print(f"   输出: {'覆盖原图' if config.overwrite else args.output}")

    def on_progress(completed, total, path, success):
        icon = "✓" if success else "✗"
        print(f"  [{completed}/{total}] {icon} {os.path.basename(path)}")

    results = process_batch(args.input, config, progress_callback=on_progress)

    successes = sum(1 for _, s, _ in results if s)
    failures = sum(1 for _, s, _ in results if not s)
    print(f"\n✅ 完成: {successes} 成功, {failures} 失败 (共 {len(results)} 张)")


if __name__ == "__main__":
    main()