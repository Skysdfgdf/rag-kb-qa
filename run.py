"""启动入口：python run.py  （等价于 uvicorn app.main:app）

监听端口：优先使用环境变量 PORT（部署平台注入），默认 8000。
绑定 0.0.0.0 以允许容器/远端访问。
"""
import os

import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)
