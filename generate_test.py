#!/usr/bin/env python3
"""生成测试图片：一张带半透明水印的图片"""
import numpy as np
import cv2
import os

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_images")
os.makedirs(output_dir, exist_ok=True)

for i, (bg_color, text_color) in enumerate([
    ((200, 200, 200), (255, 255, 255)),  # 灰色背景 + 白色水印
    ((100, 100, 100), (255, 255, 255)),  # 深灰背景 + 白色水印
    ((240, 240, 240), (0, 0, 0)),        # 浅灰背景 + 黑色水印
]):
    img = np.full((400, 600, 3), bg_color, dtype=np.uint8)
    # 加一些内容
    cv2.putText(img, "This is a sample image", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 2)
    cv2.putText(img, "Some content here...", (50, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 1)
    cv2.putText(img, "And more content here...", (50, 220),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 1)
    # 画一些图形
    cv2.rectangle(img, (50, 280), (200, 350), (100, 100, 200), -1)
    cv2.rectangle(img, (250, 280), (400, 350), (200, 100, 100), -1)
    cv2.rectangle(img, (450, 280), (550, 350), (100, 200, 100), -1)

    # 添加水印（半透明文字，右下角）
    overlay = img.copy()
    cv2.putText(overlay, "WATERMARK", (350, 350),
                cv2.FONT_HERSHEY_DUPLEX, 1.5, text_color, 4)
    cv2.putText(overlay, "SAMPLE", (150, 180),
                cv2.FONT_HERSHEY_DUPLEX, 2, text_color, 3)
    # 半透明混合
    alpha = 0.3
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    output_path = os.path.join(output_dir, f"test_{i}.png")
    cv2.imwrite(output_path, img)
    print(f"  ✓ 生成: {output_path}")

print(f"\n✅ 测试图片已生成到 {output_dir}/")
