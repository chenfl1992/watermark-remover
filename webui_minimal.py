#!/usr/bin/env python3
"""
webui_minimal.py — 零依赖 Web 界面（仅用 Python 内置 http.server）
Python 3.13+ compatible (no cgi module, uses base64 JSON POST)

Usage:
    python webui_minimal.py
    # Visit http://localhost:7860
"""

import os
import sys
import json
import base64
import io
import html
import mimetypes
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# ── Add parent to path ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from water_remover import (
    WatermarkConfig,
    create_mask,
    inpaint,
    process_image,
    process_batch,
    find_images,
)
import cv2
import numpy as np
from PIL import Image

PORT = 7860
TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".web_tmp")
os.makedirs(TEMP_DIR, exist_ok=True)


def cv2_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def pil_to_cv2(img):
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def img_to_base64(pil_img, fmt="PNG"):
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


# ── HTML Template ──

HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>🧽 批量图片去水印工具</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#f5f5f5;color:#333;min-height:100vh}
.container{max-width:1200px;margin:0 auto;padding:20px}
h1{font-size:1.5rem;margin-bottom:8px;display:flex;align-items:center;gap:8px}
.subtitle{color:#666;margin-bottom:20px;font-size:.9rem}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}
@media(max-width:768px){.grid{grid-template-columns:1fr}}
.card{background:#fff;border-radius:12px;padding:20px;box-shadow:0 1px 3px rgba(0,0,0,.1)}
.card h2{font-size:1rem;margin-bottom:12px;color:#555}
label{display:block;margin:8px 0 4px;font-size:.85rem;color:#555}
select,input[type=text],input[type=number]{width:100%;padding:8px 10px;border:1px solid #ddd;border-radius:6px;font-size:.9rem}
input[type=file]{width:100%;padding:8px;border:1px dashed #ddd;border-radius:6px;font-size:.85rem;cursor:pointer}
input[type=range]{width:100%;margin:4px 0}
.row{display:flex;gap:8px;flex-wrap:wrap}
.half{flex:1;min-width:60px}
.btn{display:inline-flex;align-items:center;gap:6px;padding:10px 20px;border:none;border-radius:8px;font-size:.9rem;cursor:pointer;transition:opacity .2s}
.btn:hover{opacity:.85}
.btn-primary{background:#4f46e5;color:#fff}
.btn-secondary{background:#e0e0e0;color:#333}
.btn:disabled{opacity:.5;cursor:not-allowed}
.btn-group{display:flex;gap:8px;margin-top:16px;flex-wrap:wrap}
.img-box{background:#fafafa;border:1px solid #eee;border-radius:8px;min-height:200px;display:flex;align-items:center;justify-content:center;overflow:hidden;margin-top:8px;position:relative}
.img-box img{max-width:100%;max-height:350px;display:block;touch-action:none;user-select:none;-webkit-user-select:none}
.img-box .placeholder{color:#aaa;font-size:.85rem}
.img-box canvas{position:absolute;top:0;left:0;width:100%;height:100%;cursor:crosshair;touch-action:none}
.roi-canvas-wrapper{position:relative;display:inline-block;max-width:100%}
.roi-coords{padding:4px 0;font-size:.8rem;color:#4f46e5;font-weight:500}
.roi-hint{display:none;color:#888;font-size:.75rem;margin-bottom:4px;text-align:center}
.mode-manual-active .roi-hint{display:block}
.touch-preview{position:absolute;border:2px solid #4f46e5;background:rgba(79,70,229,.15);pointer-events:none;display:none}
.touch-preview .roi-label{position:absolute;bottom:-18px;left:2px;font-size:.7rem;color:#4f46e5;white-space:nowrap;font-weight:600}
.status{margin-top:8px;padding:8px 12px;border-radius:6px;font-size:.85rem}
.status-ok{background:#dcfce7;color:#166534}
.status-err{background:#fee2e2;color:#991b1b}
.status-info{background:#e0e7ff;color:#3730a3}
.hidden{display:none}
.mode-group{padding:8px;margin:4px 0;border-radius:6px;background:#fafafa}
progress{width:100%;height:8px;border-radius:4px;margin:8px 0}
progress::-webkit-progress-bar{background:#eee;border-radius:4px}
progress::-webkit-progress-value{background:#4f46e5;border-radius:4px}
.tab-bar{display:flex;gap:4px;margin-bottom:16px;border-bottom:2px solid #eee}
.tab{padding:8px 16px;cursor:pointer;border-bottom:2px solid transparent;margin-bottom:-2px;font-size:.9rem;color:#666}
.tab.active{color:#4f46e5;border-bottom-color:#4f46e5}
.tab-content{display:none}
.tab-content.active{display:block}
.textarea{width:100%;min-height:60px;padding:8px;border:1px solid #ddd;border-radius:6px;font-family:monospace;font-size:.8rem}
</style>
</head>
<body>
<div class="container">
<h1>🧽 批量图片去水印</h1>
<p class="subtitle">本地运行 · OpenCV 智能检测 · 批量处理</p>

<div class="tab-bar">
<div class="tab active" onclick="switchTab('single')">🎯 单张调试</div>
<div class="tab" onclick="switchTab('batch')">📦 批量处理</div>
</div>

<!-- Single Image Tab -->
<div id="tab-single" class="tab-content active">
<div class="grid">
<div class="card">
<h2>📷 上传图片</h2>
<input type="file" id="inputImage" accept="image/*" onchange="loadImage(event)">
<div class="img-box" id="inputBox">
  <span class="placeholder">选择图片后，切换到<span style="font-weight:600">手动区域</span>模式可在图上拖拽框选</span>
  <div class="touch-preview" id="touchPreview">
    <span class="roi-label" id="roiLabel"></span>
  </div>
  <canvas id="roiCanvas"></canvas>
</div>
</div>
<div class="card">
<h2>⚙️ 参数</h2>
<label>检测模式</label>
<select id="mode" onchange="updateMode()">
  <option value="color">🎨 自动颜色检测</option>
  <option value="manual">✂️ 手动区域</option>
  <option value="position">📍 预设位置</option>
</select>
<div class="mode-group" id="colorGroup">
<label>二值化阈值 (0=自动)</label>
<input type="range" id="threshold" min="0" max="255" value="0" oninput="document.getElementById('thresholdVal').textContent=this.value">
<span id="thresholdVal" style="font-size:.8rem;color:#666">0</span>
<label>膨胀核</label>
<input type="range" id="dilate" min="0" max="10" value="1" step="1">
<label>腐蚀核</label>
<input type="range" id="erode" min="0" max="10" value="0" step="1">
<label><input type="checkbox" id="darkMode"> 深色水印（白底黑字）</label>
</div>
<div class="mode-group hidden" id="manualGroup">
<div class="roi-hint">👆 在图片上拖拽框选水印区域</div>
<label>ROI (x, y, width, height)</label>
<div class="row"><div class="half"><label>X</label><input type="number" id="roiX" value="0" oninput="showROI()"></div>
<div class="half"><label>Y</label><input type="number" id="roiY" value="0" oninput="showROI()"></div></div>
<div class="row"><div class="half"><label>W</label><input type="number" id="roiW" value="200" oninput="showROI()"></div>
<div class="half"><label>H</label><input type="number" id="roiH" value="100" oninput="showROI()"></div></div>
</div>
<div class="mode-group hidden" id="positionGroup">
<label>位置</label>
<select id="position">
  <option value="bottom-right">右下角</option>
  <option value="bottom-left">左下角</option>
  <option value="top-right">右上角</option>
  <option value="top-left">左上角</option>
  <option value="center">中心</option>
</select>
<label>大小比例</label>
<input type="range" id="sizeRatio" min="0.05" max="0.5" step="0.01" value="0.15">
</div>
<label>修复半径</label>
<input type="range" id="radius" min="1" max="20" value="3" step="1">
<div class="btn-group">
<button class="btn btn-secondary" onclick="previewMask()">👁️ 预览遮罩</button>
<button class="btn btn-primary" onclick="processSingle()">▶️ 处理</button>
</div>
<div class="status hidden" id="singleStatus"></div>
</div>
</div>
<div class="grid" style="margin-top:20px">
<div class="card">
<h2>🔍 遮罩预览</h2>
<div class="img-box" id="maskBox"><span class="placeholder">点击"预览遮罩"</span></div>
</div>
<div class="card">
<h2>✅ 修复结果</h2>
<div class="img-box" id="resultBox"><span class="placeholder">点击"处理"</span></div>
</div>
</div>
</div>

<!-- Batch Tab -->
<div id="tab-batch" class="tab-content">
<div class="grid">
<div class="card">
<h2>📁 批量处理</h2>
<label>输入目录</label>
<input type="text" id="batchInput" placeholder="/path/to/images/">
<label>输出目录</label>
<input type="text" id="batchOutput" placeholder="/path/to/output/">
<label><input type="checkbox" id="overwrite"> 覆盖原图</label>
<label>线程数</label>
<input type="range" id="workers" min="1" max="16" value="4" step="1">
<button class="btn btn-primary" onclick="startBatch()" style="margin-top:16px">🚀 开始批量处理</button>
<progress id="batchProgress" class="hidden" max="100" value="0"></progress>
<div class="status hidden" id="batchStatus"></div>
</div>
<div class="card">
<h2>📋 结果</h2>
<textarea class="textarea" id="batchResult" readonly placeholder="处理结果将显示在这里..."></textarea>
</div>
</div>
</div>
</div>

<script>
let currentImage = null;
let roiStart = null;
let roiEnd = null;
let isDragging = false;

function switchTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.querySelector(`.tab[onclick*="'${name}'"]`).classList.add('active');
  document.getElementById(`tab-${name}`).classList.add('active');
}

function updateMode() {
  const mode = document.getElementById('mode').value;
  document.getElementById('colorGroup').className = 'mode-group' + (mode === 'color' ? '' : ' hidden');
  document.getElementById('manualGroup').className = 'mode-group' + (mode === 'manual' ? '' : ' hidden');
  document.getElementById('positionGroup').className = 'mode-group' + (mode === 'position' ? '' : ' hidden');
  document.body.classList.toggle('mode-manual-active', mode === 'manual');
  if (mode === 'manual' && currentImage) showROI();
  else hideROI();
}

function loadImage(event) {
  const file = event.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = function(e) {
    currentImage = e.target.result.split(',')[1];
    const box = document.getElementById('inputBox');
    box.innerHTML = `<img src="${e.target.result}" id="previewImg">
      <div class="touch-preview" id="touchPreview"><span class="roi-label" id="roiLabel"></span></div>
      <canvas id="roiCanvas"></canvas>`;
    const img = document.getElementById('previewImg');
    img.onload = function() {
      setupCanvas();
      if (document.getElementById('mode').value === 'manual') showROI();
    };
  };
  reader.readAsDataURL(file);
}

function setupCanvas() {
  const img = document.getElementById('previewImg');
  const canvas = document.getElementById('roiCanvas');
  if (!img || !canvas) return;
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  canvas.style.width = img.offsetWidth + 'px';
  canvas.style.height = img.offsetHeight + 'px';

  // Mouse events
  canvas.onmousedown = function(e) { startDrag(e.offsetX, e.offsetY); };
  canvas.onmousemove = function(e) { if (isDragging) moveDrag(e.offsetX, e.offsetY); };
  canvas.onmouseup = function() { endDrag(); };
  canvas.onmouseleave = function() { if (isDragging) endDrag(); };

  // Touch events
  canvas.addEventListener('touchstart', function(e) {
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const t = e.touches[0];
    const x = (t.clientX - rect.left) * (canvas.width / rect.width);
    const y = (t.clientY - rect.top) * (canvas.height / rect.height);
    startDrag(x, y);
  }, {passive: false});

  canvas.addEventListener('touchmove', function(e) {
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const t = e.touches[0];
    const x = (t.clientX - rect.left) * (canvas.width / rect.width);
    const y = (t.clientY - rect.top) * (canvas.height / rect.height);
    moveDrag(x, y);
  }, {passive: false});

  canvas.addEventListener('touchend', function(e) {
    e.preventDefault();
    endDrag();
  }, {passive: false});
}

function startDrag(x, y) {
  if (document.getElementById('mode').value !== 'manual') return;
  isDragging = true;
  roiStart = {x, y};
  roiEnd = null;
}

function moveDrag(x, y) {
  if (!isDragging) return;
  roiEnd = {x, y};
  drawSelection();
}

function endDrag() {
  if (!isDragging || !roiStart || !roiEnd) {
    isDragging = false;
    return;
  }
  isDragging = false;
  const x1 = Math.round(Math.min(roiStart.x, roiEnd.x));
  const y1 = Math.round(Math.min(roiStart.y, roiEnd.y));
  const x2 = Math.round(Math.max(roiStart.x, roiEnd.x));
  const y2 = Math.round(Math.max(roiStart.y, roiEnd.y));
  const w = x2 - x1;
  const h = y2 - y1;
  if (w < 5 || h < 5) return;

  document.getElementById('roiX').value = x1;
  document.getElementById('roiY').value = y1;
  document.getElementById('roiW').value = w;
  document.getElementById('roiH').value = h;
  updateROILabel(x1, y1, w, h);
}

function drawSelection() {
  const canvas = document.getElementById('roiCanvas');
  const img = document.getElementById('previewImg');
  if (!canvas || !img) return;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const x1 = Math.min(roiStart.x, roiEnd.x);
  const y1 = Math.min(roiStart.y, roiEnd.y);
  const w = Math.abs(roiEnd.x - roiStart.x);
  const h = Math.abs(roiEnd.y - roiStart.y);

  ctx.fillStyle = 'rgba(79, 70, 229, 0.15)';
  ctx.fillRect(x1, y1, w, h);
  ctx.strokeStyle = '#4f46e5';
  ctx.lineWidth = 2;
  ctx.strokeRect(x1, y1, w, h);

  // Corner handles for visual feedback
  const s = 6;
  ctx.fillStyle = '#4f46e5';
  ctx.fillRect(x1 - s/2, y1 - s/2, s, s);
  ctx.fillRect(x1 + w - s/2, y1 - s/2, s, s);
  ctx.fillRect(x1 - s/2, y1 + h - s/2, s, s);
  ctx.fillRect(x1 + w - s/2, y1 + h - s/2, s, s);

  // Size label
  ctx.fillStyle = '#4f46e5';
  ctx.font = 'bold 13px sans-serif';
  ctx.fillText(`${Math.round(w)}×${Math.round(h)}`, x1 + 4, y1 - 6);
  updateROILabel(Math.round(x1), Math.round(y1), Math.round(w), Math.round(h));
}

function showROI() {
  const canvas = document.getElementById('roiCanvas');
  const img = document.getElementById('previewImg');
  if (!canvas || !img) return;
  const x = parseInt(document.getElementById('roiX').value);
  const y = parseInt(document.getElementById('roiY').value);
  const w = parseInt(document.getElementById('roiW').value);
  const h = parseInt(document.getElementById('roiH').value);
  if (w < 1 || h < 1) return;

  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = 'rgba(79, 70, 229, 0.15)';
  ctx.fillRect(x, y, w, h);
  ctx.strokeStyle = '#4f46e5';
  ctx.lineWidth = 2;
  ctx.strokeRect(x, y, w, h);
  updateROILabel(x, y, w, h);
}

function hideROI() {
  const canvas = document.getElementById('roiCanvas');
  if (canvas) {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }
  const label = document.getElementById('roiLabel');
  if (label) label.textContent = '';
}

function updateROILabel(x, y, w, h) {
  const label = document.getElementById('roiLabel');
  if (label) label.textContent = `ROI: ${x},${y}  ${w}×${h}`;
}

function showStatus(id, msg, type) {
  const el = document.getElementById(id);
  el.className = `status status-${type}`;
  el.textContent = msg;
}

async function previewMask() {
  const status = document.getElementById('singleStatus');
  status.className = 'status hidden';
  if (!currentImage) { showStatus('singleStatus','请先上传图片','err'); return; }
  const r = await fetch('/api/preview', {
    method:'POST',
    body: JSON.stringify(getParams()),
    headers:{'Content-Type':'application/json'}
  });
  const data = await r.json();
  if (data.error) { showStatus('singleStatus', data.error, 'err'); return; }
  if (data.mask) { document.getElementById('maskBox').innerHTML = `<img src="data:image/png;base64,${data.mask}">`; }
  if (data.result) { document.getElementById('resultBox').innerHTML = `<img src="data:image/png;base64,${data.result}">`; }
}

async function processSingle() {
  const status = document.getElementById('singleStatus');
  status.className = 'status hidden';
  if (!currentImage) { showStatus('singleStatus','请先上传图片','err'); return; }
  const params = getParams();
  const r = await fetch('/api/process', {
    method:'POST',
    body: JSON.stringify(params),
    headers:{'Content-Type':'application/json'}
  });
  const data = await r.json();
  if (data.result) { document.getElementById('resultBox').innerHTML = `<img src="data:image/png;base64,${data.result}">`; }
  if (data.mask) { document.getElementById('maskBox').innerHTML = `<img src="data:image/png;base64,${data.mask}">`; }
  showStatus('singleStatus', data.message || '✅ 处理完成', data.error ? 'err' : 'ok');
}

function getParams() {
  return {
    image: currentImage,
    mode: document.getElementById('mode').value,
    threshold: parseInt(document.getElementById('threshold').value),
    dilate: parseInt(document.getElementById('dilate').value),
    erode: parseInt(document.getElementById('erode').value),
    dark: document.getElementById('darkMode').checked,
    radius: parseInt(document.getElementById('radius').value),
    roi_x: parseInt(document.getElementById('roiX').value),
    roi_y: parseInt(document.getElementById('roiY').value),
    roi_w: parseInt(document.getElementById('roiW').value),
    roi_h: parseInt(document.getElementById('roiH').value),
    position: document.getElementById('position').value,
    size_ratio: parseFloat(document.getElementById('sizeRatio').value),
  };
}

async function startBatch() {
  const btn = document.querySelector('#tab-batch .btn-primary');
  const progress = document.getElementById('batchProgress');
  btn.disabled = true;
  progress.classList.remove('hidden');
  const r = await fetch('/api/batch', {
    method:'POST',
    body: JSON.stringify({
      input_dir: document.getElementById('batchInput').value,
      output_dir: document.getElementById('batchOutput').value,
      overwrite: document.getElementById('overwrite').checked,
      workers: parseInt(document.getElementById('workers').value),
      ...getParams()
    }),
    headers:{'Content-Type':'application/json'}
  });
  const data = await r.json();
  progress.classList.add('hidden');
  document.getElementById('batchResult').value = data.message || '❌ 处理失败';
  showStatus('batchStatus', data.error || '✅ 完成', data.error ? 'err' : 'ok');
  btn.disabled = false;
}
</script>
</body>
</html>"""


class WatermarkHandler(BaseHTTPRequestHandler):
    """HTTP handler for watermark removal tool"""

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == '/' or path == '/index.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML.encode('utf-8'))
        elif path.startswith('/static/'):
            self.serve_static(path)
        else:
            self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        data = json.loads(body.decode('utf-8'))

        if path == '/api/preview':
            self.handle_preview(data)
        elif path == '/api/process':
            self.handle_process(data)
        elif path == '/api/batch':
            self.handle_batch(data)
        else:
            self.send_error(404)

    def serve_static(self, path):
        self.send_error(404)

    def send_json(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))

    def get_config_from_data(self, data):
        mode = data.get('mode', 'color')
        config = WatermarkConfig(
            mode=mode,
            threshold=int(data.get('threshold', 0)),
            inpaint_method='telea',
            inpaint_radius=int(data.get('radius', 3)),
            dark_on_light=bool(data.get('dark', False)),
            dilate_kernel=int(data.get('dilate', 1)),
            erode_kernel=int(data.get('erode', 0)),
        )
        if mode == 'manual':
            config.roi = [
                int(data.get('roi_x', 0)),
                int(data.get('roi_y', 0)),
                int(data.get('roi_w', 200)),
                int(data.get('roi_h', 100)),
            ]
        elif mode == 'position':
            config.position = data.get('position', 'bottom-right')
            config.size_ratio = float(data.get('size_ratio', 0.15))
        return config

    def decode_image(self, data):
        img_b64 = data.get('image', '')
        if not img_b64:
            return None
        img_bytes = base64.b64decode(img_b64)
        img_pil = Image.open(io.BytesIO(img_bytes))
        return pil_to_cv2(img_pil)

    def handle_preview(self, data):
        img = self.decode_image(data)
        if img is None:
            self.send_json({'error': '请上传图片'})
            return
        config = self.get_config_from_data(data)
        try:
            mask = create_mask(img, config)
            result = inpaint(img, mask, config)

            # Create overlay
            mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mask_vis = np.where(mask_vis > 0, (0, 255, 0), (0, 0, 0)).astype(np.uint8)
            overlay = cv2.addWeighted(img, 0.7, mask_vis, 0.3, 0)

            overlay_pil = cv2_to_pil(overlay)
            result_pil = cv2_to_pil(result)

            self.send_json({
                'mask': img_to_base64(overlay_pil),
                'result': img_to_base64(result_pil),
            })
        except Exception as e:
            self.send_json({'error': str(e)})

    def handle_process(self, data):
        img = self.decode_image(data)
        if img is None:
            self.send_json({'error': '请上传图片'})
            return
        config = self.get_config_from_data(data)
        try:
            mask = create_mask(img, config)
            result = inpaint(img, mask, config)
            result_pil = cv2_to_pil(result)
            mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mask_vis = np.where(mask_vis > 0, (0, 255, 0), (0, 0, 0)).astype(np.uint8)
            overlay = cv2.addWeighted(img, 0.7, mask_vis, 0.3, 0)
            overlay_pil = cv2_to_pil(overlay)

            self.send_json({
                'result': img_to_base64(result_pil),
                'mask': img_to_base64(overlay_pil),
                'message': '✅ 处理成功！',
            })
        except Exception as e:
            self.send_json({'error': str(e)})

    def handle_batch(self, data):
        input_dir = data.get('input_dir', '')
        output_dir = data.get('output_dir', '')
        overwrite = bool(data.get('overwrite', False))
        workers = int(data.get('workers', 4))

        if not input_dir or not os.path.isdir(input_dir):
            self.send_json({'error': '❌ 输入目录无效'})
            return

        config = self.get_config_from_data(data)
        config.workers = workers
        config.overwrite = overwrite
        config.output_dir = output_dir

        try:
            results = process_batch(input_dir, config)
            successes = sum(1 for _, s, _ in results if s)
            failures = sum(1 for _, s, _ in results if not s)
            lines = [f"  {'✓' if ok else '✗'} {os.path.basename(p)}: {msg}" for p, ok, msg in results]
            summary = f"✅ 完成: {successes} 成功, {failures} 失败 (共 {len(results)} 张)"
            details = "\n".join(lines)
            self.send_json({'message': f"{summary}\n\n{details}"})
        except Exception as e:
            self.send_json({'error': str(e)})

    def log_message(self, format, *args):
        pass  # Suppress HTTP log spam


def run_server(host="127.0.0.1"):
    """Start the HTTP server"""
    server = HTTPServer((host, PORT), WatermarkHandler)
    print(f"🧽 批量图片去水印工具 — Web 界面")
    print(f"   访问: http://{host}:{PORT}")
    print(f"   按 Ctrl+C 停止")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  停止服务")


if __name__ == "__main__":
    host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    run_server(host)
