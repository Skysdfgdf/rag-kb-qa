"""PyInstaller 打包入口（exe 版的 run.py）。

双击 rag-kb-qa.exe 即启动服务并自动打开浏览器；关闭窗口即停止服务。
"""
import multiprocessing
import os
import sys
import threading
import webbrowser

import faulthandler

faulthandler.enable()  # 原生崩溃时打印 Python 调用栈，便于定位


def _open_browser(url: str) -> None:
    try:
        webbrowser.open(url)
    except Exception:
        pass


def main() -> None:
    multiprocessing.freeze_support()
    from app.main import app  # 确保静态收集全部依赖
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    url = f"http://127.0.0.1:{port}"
    threading.Timer(2.0, _open_browser, args=(url,)).start()
    print(f"\n  知识库问答服务已启动: {url}")
    print("  关闭本窗口即停止服务。数据保存在本目录 data/ 文件夹。\n")
    uvicorn.run(app, host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
