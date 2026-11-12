"""PyInstaller 打包入口（exe 版的 run.py）。

双击 rag-kb-qa.exe：后台启动服务 → 自动打开浏览器 → 右下角托盘常驻。
退出方式：右键托盘图标 → 退出。关闭浏览器不影响服务。

无窗口模式下 stdout/stderr 为 None，日志改写到 logs/app.log。
"""
import multiprocessing
import os
import socket
import sys
import threading
import time
import traceback
import webbrowser
from pathlib import Path


def _base_dir() -> Path:
    """打包后按 exe 所在目录算，开发时按源码目录算。"""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


BASE_DIR = _base_dir()
LOG_FILE = BASE_DIR / "logs" / "app.log"


def _setup_logging() -> None:
    """打包后把 stdout/stderr 统一指向日志文件。

    无控制台时 sys.stdout 是 None，print 会直接报错；有控制台时中文会按 GBK 乱码。
    两种情况下写文件都更可靠，也方便用户排查问题。
    """
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    handle = open(LOG_FILE, "a", encoding="utf-8", buffering=1)
    sys.stdout = handle
    sys.stderr = handle


FROZEN = bool(getattr(sys, "frozen", False))
# 打包后：后台运行 + 托盘退出；源码运行：保留控制台，方便调试
WINDOWED = FROZEN
if WINDOWED:
    _setup_logging()

import faulthandler  # noqa: E402

faulthandler.enable()  # 原生崩溃时打印调用栈到日志，便于排查


def _port() -> int:
    return int(os.environ.get("PORT", "8000"))


def _already_running(port: int) -> bool:
    """端口已被占用 => 说明已经有一个实例在跑。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.4)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _open_url(url: str) -> None:
    """打开默认浏览器。

    Windows 用 os.startfile 走 ShellExecute，立即返回、不会阻塞进程；
    webbrowser 在打包环境下可能等待浏览器响应，导致进程卡住。
    """
    try:
        if sys.platform == "win32":
            os.startfile(url)  # noqa: S606
            return
        webbrowser.open(url, new=2)
    except Exception:
        try:
            webbrowser.open(url)
        except Exception:
            print(f"[warn] 打开浏览器失败，请手动访问 {url}")


def _run_server(app, port: int, ready: threading.Event) -> None:
    import uvicorn

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="info")
    server = uvicorn.Server(config)

    def _watch_ready() -> None:
        while not server.started:
            time.sleep(0.1)
        ready.set()

    threading.Thread(target=_watch_ready, daemon=True).start()
    server.run()


def _tray(app, port: int, url: str) -> None:
    """托盘图标。右键菜单：打开网页 / 数据文件夹 / 日志 / 退出。"""
    import pystray

    from app_icon import make_image

    def on_open(icon, item):
        _open_url(url)

    def on_data(icon, item):
        target = BASE_DIR / "data"
        target.mkdir(exist_ok=True)
        os.startfile(target)  # noqa: S606 - Windows 专用，打开资源管理器

    def on_log(icon, item):
        if LOG_FILE.exists():
            os.startfile(LOG_FILE)  # noqa: S606
        else:
            print("[info] 暂无日志文件")

    def on_quit(icon, item):
        print("[info] 用户从托盘退出，服务停止")
        icon.stop()
        os._exit(0)

    icon = pystray.Icon(
        "rag-kb-qa",
        make_image(64),
        f"知识库问答（运行中 · 端口 {port}）",
        menu=pystray.Menu(
            pystray.MenuItem("打开问答页面", on_open, default=True),
            pystray.MenuItem("打开数据文件夹", on_data),
            pystray.MenuItem("查看日志", on_log),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("退出", on_quit),
        ),
    )
    print("[info] 托盘图标已创建，服务进入后台运行")
    icon.run()
    print("[info] 托盘循环结束")


def _notify(title: str, message: str) -> None:
    """无窗口模式下给用户一个提示（托盘气泡），失败就写日志。"""
    try:
        import pystray

        from app_icon import make_image

        icon = pystray.Icon("rag-kb-qa-tip", make_image(64), title)
        icon.notify(message, title)
        time.sleep(3)
        icon.stop()
    except Exception:
        print(f"[info] {title}: {message}")


def main() -> None:
    multiprocessing.freeze_support()

    from app.main import app  # 提前导入，确保依赖被收集
    import uvicorn  # noqa: F401 - 校验可用

    port = _port()
    url = f"http://127.0.0.1:{port}"

    # 已经有一个实例在跑：直接打开浏览器，不重复启动（避免端口冲突报错）
    if _already_running(port):
        print(f"[info] 服务已在运行（端口 {port}），直接打开页面")
        _open_url(url)
        if not WINDOWED:
            time.sleep(1.0)
        return

    ready = threading.Event()
    threading.Thread(target=_run_server, args=(app, port, ready), daemon=True).start()

    if not ready.wait(timeout=30):
        msg = f"服务启动超时，请查看日志：{LOG_FILE}"
        print(f"[error] {msg}")
        if WINDOWED:
            _notify("知识库问答", msg)
        return

    print(f"\n  知识库问答服务已启动: {url}")
    print(f"  日志文件: {LOG_FILE}")
    print("  退出方式: 右键任务栏右下角的托盘图标 → 退出\n")

    threading.Timer(0.8, _open_url, args=(url,)).start()

    if WINDOWED:
        try:
            _tray(app, port, url)
        except Exception:
            print("[error] 托盘初始化失败：\n" + traceback.format_exc())
            _notify("知识库问答", f"托盘图标创建失败，服务仍在后台运行：{url}")
            while True:  # 退化成纯后台进程
                time.sleep(3600)
    else:
        # 有控制台（开发调试用）：关掉窗口/Ctrl+C 即停止
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            print("\n[info] 已停止")


if __name__ == "__main__":
    main()
