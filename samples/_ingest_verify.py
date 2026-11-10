"""临时客户端：把教程文档上传到本地运行的服务，并做一次真实问答验证。

口令从 app.config（读取 .env）获取，不出现在命令行；
连接 127.0.0.1 时禁用代理（通过 requests proxies=None + 环境变量 NO_PROXY）。
"""
import os
import sys
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import app.config as cfg  # noqa: E402  (读取 .env 自动加载)

ADMIN = cfg.settings.effective_admin_password
assert ADMIN, "admin_password 未设置，无法鉴权上传"

BASE = "http://127.0.0.1:8000"
H = {"Authorization": f"Bearer {ADMIN}"}
PROXIES = {"http": None, "https": None}

DOC = os.path.join(os.path.dirname(__file__), "python_zh_tutorial.md")


def main():
    # 1) 上传
    with open(DOC, "rb") as f:
        r = requests.post(
            BASE + "/api/documents",
            files={"file": ("python_zh_tutorial.md", f, "text/markdown")},
            headers=H, timeout=180, proxies=PROXIES,
        )
    print(">> 上传状态:", r.status_code)
    j = r.json()
    print("   文档名:", j.get("name"), "| 切片数:", j.get("chunk_count"), "| doc_id:", j.get("doc_id"))

    # 2) 真实问答（走 .env 里的 DeepSeek）
    questions = [
        "Python 中如何用 open 函数读写文件？请说明常见用法",
        "Python 中如何定义类？类与实例、方法是什么关系？",
        "Python 列表和元组有什么区别？",
    ]
    for q in questions:
        print("\n" + "=" * 60)
        print(">> 提问:", q)
        r2 = requests.post(
            BASE + "/api/query",
            json={"question": q, "stream": False},
            headers=H, timeout=180, proxies=PROXIES,
        )
        print(">> 问答状态:", r2.status_code)
        ans = r2.json()
        print("\n--- 回答 ---\n" + ans.get("answer", ""))
        print("\n--- 引用溯源（含召回片段）---")
        for c in ans.get("citations", []):
            snip = (c.get("text") or "")[:120].replace("\n", " ")
            print(f"  [{c['index']}] {c['doc_name']} (页 {c['page']})：{snip}")


if __name__ == "__main__":
    main()
