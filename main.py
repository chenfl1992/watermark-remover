#!/usr/bin/env python3
"""
main.py — 批量图片去水印工具 统一入口

用法:
  # 启动 Web 界面（零依赖，Python 内置 http.server）
  python main.py --web                    # 本机访问 http://127.0.0.1:7860
  python main.py --web --host 0.0.0.0     # 局域网/公网访问

  # 命令行批量处理
  python main.py ./input_dir -o ./output_dir --mode color
"""

import sys
import os

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--web":
        from webui_minimal import run_server
        
        host = "127.0.0.1"
        if "--host" in sys.argv:
            idx = sys.argv.index("--host")
            if idx + 1 < len(sys.argv):
                host = sys.argv[idx + 1]
        
        print(f"🌐 Web UI 启动中 → http://{host}:7860")
        run_server(host=host)
    else:
        from water_remover import main
        main()