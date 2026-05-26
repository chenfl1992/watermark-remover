#!/usr/bin/env python3
"""
webui.py — 批量图片去水印 Gradio Web 界面

启动: python webui.py
访问: http://localhost:7860
"""

import os
import sys
import tempfile
import json
from pathlib import Path

import gradio as gr
import cv2
import numpy as np
from PIL import Image

# 添加上级目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from water_remover import (
    WatermarkConfig,
    create_mask,
    inpaint,
    process_image,
    process_batch,
    find_images,
)


# ── 全局状态 ──────────────────────────────────────────────────

TEMP_DIR = os.path.join(tempfile.gettempdir(), "watermark_remover_web")
os.makedirs(TEMP_DIR, exist_ok=True)

# 当前配置（在 UI 调整时更新）
current_config = WatermarkConfig()


# ── 辅助函数 ──────────────────────────────────────────────────

def cv2_to_pil(img: np.ndarray) -> Image.Image:
    """OpenCV BGR → PIL RGB"""
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def pil_to_cv2(img: Image.Image) -> np.ndarray:
    """PIL → OpenCV BGR"""
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def preview_mask_and_result(
    input_img: np.ndarray,
    mode: str,
    color_ranges_json: str,
    threshold: int,
    inpaint_method: str,
    inpaint_radius: int,
    dark_on_light: bool,
    dilate_kernel: int,
    erode_kernel: int,
    roi_x: int,
    roi_y: int,
    roi_w: int,
    roi_h: int,
    position: str,
    size_ratio: float,
):
    """
    预览遮罩和修复效果
    返回: (遮罩图, 修复结果图)
    """
    if input_img is None:
        return None, None

    img = pil_to_cv2(input_img)

    # 解析颜色范围 JSON
    try:
        color_ranges = json.loads(color_ranges_json) if color_ranges_json.strip() else []
    except json.JSONDecodeError:
        color_ranges = []

    config = WatermarkConfig(
        mode=mode,
        color_ranges=color_ranges,
        threshold=threshold,
        inpaint_method=inpaint_method,
        inpaint_radius=inpaint_radius,
        dark_on_light=dark_on_light,
        dilate_kernel=dilate_kernel,
        erode_kernel=erode_kernel,
    )

    if mode == "manual":
        config.roi = [roi_x, roi_y, roi_w, roi_h]
    elif mode == "position":
        config.position = position
        config.size_ratio = size_ratio

    # 生成遮罩
    mask = create_mask(img, config)

    # 修复
    result = inpaint(img, mask, config)

    # 遮罩可视化（转为彩色显示）
    mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_vis = np.where(mask_vis > 0, (0, 255, 0), (0, 0, 0)).astype(np.uint8)
    overlay = cv2.addWeighted(img, 0.7, mask_vis, 0.3, 0)

    return (
        cv2_to_pil(overlay),  # 遮罩叠加图
        cv2_to_pil(result),   # 修复结果
    )


def process_single_file(
    input_img: np.ndarray,
    mode: str,
    color_ranges_json: str,
    threshold: int,
    inpaint_method: str,
    inpaint_radius: int,
    dark_on_light: bool,
    dilate_kernel: int,
    erode_kernel: int,
    roi_x: int,
    roi_y: int,
    roi_w: int,
    roi_h: int,
    position: str,
    size_ratio: float,
):
    """处理单张测试图片"""
    if input_img is None:
        return None, None, "请上传图片"

    img = pil_to_cv2(input_img)
    try:
        color_ranges = json.loads(color_ranges_json) if color_ranges_json.strip() else []
    except json.JSONDecodeError:
        color_ranges = []

    config = WatermarkConfig(
        mode=mode,
        color_ranges=color_ranges,
        threshold=threshold,
        inpaint_method=inpaint_method,
        inpaint_radius=inpaint_radius,
        dark_on_light=dark_on_light,
        dilate_kernel=dilate_kernel,
        erode_kernel=erode_kernel,
    )

    if mode == "manual":
        config.roi = [roi_x, roi_y, roi_w, roi_h]
    elif mode == "position":
        config.position = position
        config.size_ratio = size_ratio

    # 保存到临时文件再处理（复用 process_image）
    temp_in = os.path.join(TEMP_DIR, "preview_input.png")
    cv2.imwrite(temp_in, img)
    temp_out = os.path.join(TEMP_DIR, "preview_output.png")
    _, success, msg = process_image(temp_in, config, temp_out)

    if success and os.path.exists(temp_out):
        result_img = cv2.imread(temp_out)
        return (
            cv2_to_pil(result_img),
            None,  # 不需要预览遮罩
            "✅ 处理成功！调整参数后可用批量处理。",
        )
    else:
        return (
            cv2_to_pil(img) if input_img is not None else None,
            None,
            f"❌ {msg}",
        )


def batch_process(
    input_dir: str,
    output_dir: str,
    mode: str,
    color_ranges_json: str,
    threshold: int,
    inpaint_method: str,
    inpaint_radius: int,
    dark_on_light: bool,
    dilate_kernel: int,
    erode_kernel: int,
    erode_kernel_2: int,
    workers: int,
    overwrite: bool,
    position: str,
    size_ratio: float,
):
    """批量处理目录"""
    if not input_dir or not os.path.isdir(input_dir):
        return "❌ 输入目录无效"

    if not overwrite and (not output_dir or not os.path.isdir(output_dir)):
        return "❌ 输出目录无效"

    try:
        color_ranges = json.loads(color_ranges_json) if color_ranges_json.strip() else []
    except json.JSONDecodeError:
        color_ranges = []

    config = WatermarkConfig(
        mode=mode,
        color_ranges=color_ranges,
        threshold=threshold,
        inpaint_method=inpaint_method,
        inpaint_radius=inpaint_radius,
        dark_on_light=dark_on_light,
        dilate_kernel=dilate_kernel,
        erode_kernel=erode_kernel_2,
        workers=workers,
        overwrite=overwrite,
        output_dir=output_dir,
    )

    if mode == "position":
        config.position = position
        config.size_ratio = size_ratio

    results = process_batch(input_dir, config)
    successes = sum(1 for _, s, _ in results if s)
    failures = sum(1 for _, s, _ in results if not s)

    detail_lines = []
    for path, ok, msg in results:
        status = "✓" if ok else "✗"
        detail_lines.append(f"  {status} {os.path.basename(path)}: {msg}")

    summary = f"✅ 完成: {successes} 成功, {failures} 失败 (共 {len(results)} 张)"
    details = "\n".join(detail_lines)

    return f"{summary}\n\n{details}"


# ── 创建界面 ──────────────────────────────────────────────────

def create_ui():
    with gr.Blocks(title="批量图片去水印工具", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🧽 批量图片去水印工具
        支持自动颜色检测、手动区域、预设位置三种模式，批量处理图片。
        """)

        with gr.Tabs():
            # ── Tab 1: 参数调试 ──
            with gr.TabItem("🔧 参数调试"):
                with gr.Row():
                    with gr.Column(scale=1):
                        input_image = gr.Image(
                            label="上传测试图片",
                            type="pil",
                            height=400,
                        )

                    with gr.Column(scale=2):
                        with gr.Tabs():
                            with gr.TabItem("🎯 检测"):
                                mode = gr.Radio(
                                    label="检测模式",
                                    choices=[
                                        ("自动颜色检测", "color"),
                                        ("手动区域", "manual"),
                                        ("预设位置", "position"),
                                    ],
                                    value="color",
                                )

                                with gr.Group(visible=True) as color_group:
                                    color_ranges_json = gr.Textbox(
                                        label="颜色范围 (JSON)",
                                        value=json.dumps([
                                            {"lower": [0, 0, 180], "upper": [180, 30, 255]},
                                            {"lower": [0, 0, 100], "upper": [180, 30, 180]},
                                        ], ensure_ascii=False),
                                        lines=4,
                                    )

                                with gr.Group(visible=False) as manual_group:
                                    gr.Markdown("ROI 坐标 (x, y, w, h):")
                                    with gr.Row():
                                        roi_x = gr.Number(label="X", value=0, minimum=0)
                                        roi_y = gr.Number(label="Y", value=0, minimum=0)
                                    with gr.Row():
                                        roi_w = gr.Number(label="宽度", value=200, minimum=1)
                                        roi_h = gr.Number(label="高度", value=100, minimum=1)

                                with gr.Group(visible=False) as position_group:
                                    position = gr.Dropdown(
                                        label="水印位置",
                                        choices=[
                                            ("右下角", "bottom-right"),
                                            ("左下角", "bottom-left"),
                                            ("右上角", "top-right"),
                                            ("左上角", "top-left"),
                                            ("中心", "center"),
                                        ],
                                        value="bottom-right",
                                    )
                                    size_ratio = gr.Slider(
                                        label="预估大小比例",
                                        minimum=0.05,
                                        maximum=0.5,
                                        value=0.15,
                                        step=0.01,
                                    )

                            with gr.TabItem("⚙️ 修复"):
                                inpaint_method = gr.Radio(
                                    label="修复算法",
                                    choices=[
                                        ("Telea（较快，边缘平滑）", "telea"),
                                        ("Navier-Stokes（更平滑）", "ns"),
                                    ],
                                    value="telea",
                                )
                                inpaint_radius = gr.Slider(
                                    label="修复半径（大块水印用大值）",
                                    minimum=1,
                                    maximum=20,
                                    value=3,
                                    step=1,
                                )
                                threshold = gr.Slider(
                                    label="二值化阈值（0=自动）",
                                    minimum=0,
                                    maximum=255,
                                    value=0,
                                    step=1,
                                )
                                dilate_kernel = gr.Slider(
                                    label="膨胀核（扩大覆盖区域）",
                                    minimum=0,
                                    maximum=10,
                                    value=1,
                                    step=1,
                                )
                                erode_kernel = gr.Slider(
                                    label="腐蚀核（去除噪点）",
                                    minimum=0,
                                    maximum=10,
                                    value=0,
                                    step=1,
                                )
                                dark_on_light = gr.Checkbox(
                                    label="深色水印（白底黑字）",
                                    value=False,
                                )

            # ── Tab 2: 批量处理 ──
            with gr.TabItem("📦 批量处理"):
                with gr.Row():
                    with gr.Column():
                        input_dir = gr.Textbox(
                            label="输入目录",
                            placeholder="/path/to/images/",
                        )
                        output_dir = gr.Textbox(
                            label="输出目录（覆盖模式不需要）",
                            placeholder="/path/to/output/",
                        )
                        with gr.Row():
                            overwrite_mode = gr.Checkbox(
                                label="覆盖原图",
                                value=False,
                            )
                            batch_workers = gr.Slider(
                                label="线程数",
                                minimum=1,
                                maximum=16,
                                value=4,
                                step=1,
                            )
                        batch_btn = gr.Button("🚀 开始批量处理", variant="primary", size="lg")

                    with gr.Column():
                        batch_output = gr.Textbox(
                            label="批量处理结果",
                            lines=15,
                        )

        # ── 预览区域 ──
        with gr.Row():
            with gr.Column():
                preview_mask_btn = gr.Button("👁️ 预览遮罩", variant="secondary")
                preview_mask_img = gr.Image(label="遮罩叠加预览", type="pil", height=350)

            with gr.Column():
                process_single_btn = gr.Button("▶️ 测试处理单张", variant="secondary")
                preview_result_img = gr.Image(label="修复结果预览", type="pil", height=350)
                single_status = gr.Textbox(label="状态", max_lines=2)

        # ── 动态显示分组 ──
        def update_groups(mode_val):
            return {
                color_group: gr.update(visible=mode_val == "color"),
                manual_group: gr.update(visible=mode_val == "manual"),
                position_group: gr.update(visible=mode_val == "position"),
            }

        mode.change(
            fn=update_groups,
            inputs=[mode],
            outputs=[color_group, manual_group, position_group],
        )

        # ── 绑定事件 ──
        preview_mask_btn.click(
            fn=preview_mask_and_result,
            inputs=[
                input_image, mode, color_ranges_json, threshold,
                inpaint_method, inpaint_radius, dark_on_light,
                dilate_kernel, erode_kernel,
                roi_x, roi_y, roi_w, roi_h,
                position, size_ratio,
            ],
            outputs=[preview_mask_img],
        )

        process_single_btn.click(
            fn=process_single_file,
            inputs=[
                input_image, mode, color_ranges_json, threshold,
                inpaint_method, inpaint_radius, dark_on_light,
                dilate_kernel, erode_kernel,
                roi_x, roi_y, roi_w, roi_h,
                position, size_ratio,
            ],
            outputs=[preview_result_img, None, single_status],
        )

        batch_btn.click(
            fn=batch_process,
            inputs=[
                input_dir, output_dir,
                mode, color_ranges_json, threshold,
                inpaint_method, inpaint_radius, dark_on_light,
                dilate_kernel, erode_kernel,
                batch_workers, overwrite_mode,
                position, size_ratio,
            ],
            outputs=[batch_output],
        )

    return demo


# ── 启动 ──────────────────────────────────────────────────────

if __name__ == "__main__":
    demo = create_ui()
    print("🧽 批量图片去水印工具启动中...")
    print(f"   临时目录: {TEMP_DIR}")
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
    )