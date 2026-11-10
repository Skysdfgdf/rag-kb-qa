@echo off
chcp 936 >nul
title RAG 知识库问答
cd /d "%~dp0"

rem 口令统一在 .env 里配置（ADMIN_PASSWORD / USER_PASSWORD），这里不再覆盖。
rem 本地与线上共用同一份 .env，sync.py 也用这个口令写主库。

where python >nul 2>&1
if errorlevel 1 goto NOPY

netstat -ano | findstr ":8000" | findstr "LISTENING" >nul 2>&1
if not errorlevel 1 goto RUNNING

echo ==================================================
echo   RAG 知识库问答  正在启动...
echo.
echo   地址 http://localhost:8000
echo   本机是主设备：点右上角身份徽章，输入 ADMIN_PASSWORD 进入主库
echo   其他设备打开页面自动以游客进入，各自独立资料库
echo.
echo   浏览器会在几秒后自动打开
echo   关闭本窗口或按 Ctrl+C 可停止服务
echo ==================================================
echo.

start "" cmd /c "timeout /t 4 >nul & start http://localhost:8000"
python run.py

echo.
echo 服务已停止。
pause
exit /b 0

:NOPY
echo [错误] 没有找到 python 命令。
echo 请先安装 Python 并勾选 Add Python to PATH，然后重试。
pause
exit /b 1

:RUNNING
echo [提示] 服务已经在 8000 端口运行，正在打开浏览器...
start "" http://localhost:8000
timeout /t 3 >nul
exit /b 0
