def run_server(host="127.0.0.1"):
    """启动 HTTP 服务器"""
    server = HTTPServer((host, PORT), WatermarkHandler)
    print(f"🧽 批量图片去水印工具 — Web 界面")
    print(f"   访问: http://{host}:{PORT}")
    print(f"   按 Ctrl+C 停止")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  停止服务")