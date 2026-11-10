"""一次性脚本：抓取 Python 3 官方中文教程并清洗为单一 markdown 文件。

来源：https://docs.python.org/zh-cn/3/tutorial/
输出：samples/python_zh_tutorial.md

仅依赖 requests + beautifulsoup4 + lxml（pip install beautifulsoup4 lxml）。
请求会自动走环境 HTTPS_PROXY。
"""
import os
import sys
import time

import requests
from bs4 import BeautifulSoup, NavigableString
from urllib.parse import urljoin

BASE = "https://docs.python.org/zh-cn/3/tutorial/"
OUT = os.path.join(os.path.dirname(__file__), "python_zh_tutorial.md")
UA = {"User-Agent": "Mozilla/5.0 (compatible; RAG-KB-Builder/1.0)"}


def fetch(url: str) -> str:
    last = None
    for attempt in range(3):
        try:
            r = requests.get(url, headers=UA, timeout=45)
            r.raise_for_status()
            r.encoding = r.apparent_encoding or "utf-8"
            return r.text
        except Exception as e:  # noqa
            last = e
            time.sleep(1.5)
    raise last


def body_soup(html: str) -> BeautifulSoup:
    soup = BeautifulSoup(html, "lxml")
    # 去掉侧边栏 / 导航 / 页脚等无关块
    for sel in ["div.sphinxsidebar", "div.related", "div.footer", "div.navbar",
                "div.navigation", "script", "style"]:
        for el in soup.select(sel):
            el.decompose()
    return soup


def to_md(elem) -> str:
    """把 Sphinx 正文递归转成轻量 markdown（保留标题层级与代码块）。"""
    out = []
    for child in elem.children:
        if isinstance(child, NavigableString):
            continue
        tag = child.name
        if tag in ("h1", "h2", "h3", "h4"):
            level = int(tag[1])
            txt = child.get_text(" ", strip=True)
            if txt:
                out.append("\n" + "#" * level + " " + txt + "\n")
        elif tag == "p":
            txt = child.get_text(" ", strip=True)
            if txt:
                out.append(txt + "\n")
        elif tag == "pre":
            code = child.get_text("\n").rstrip()
            lang = "python" if "python" in (child.get("class") or []) else ""
            out.append("\n```" + lang + "\n" + code + "\n```\n")
        elif tag in ("ul", "ol"):
            for li in child.find_all("li", recursive=False):
                out.append("- " + li.get_text(" ", strip=True) + "\n")
            out.append("\n")
        elif tag == "dl":  # 术语表 / 函数签名等定义列表
            for dt in child.find_all("dt"):
                out.append("**" + dt.get_text(" ", strip=True) + "**  ")
            for dd in child.find_all("dd"):
                out.append(dd.get_text(" ", strip=True) + "\n")
            out.append("\n")
        elif tag in ("div", "section", "blockquote"):
            out.append(to_md(child))
        elif tag == "table":
            out.append(child.get_text(" | ", strip=True) + "\n")
    return "".join(out)


def main():
    print(">> 抓取教程目录页 ...")
    soup = body_soup(fetch(urljoin(BASE, "index.html")))
    chapters = []
    seen = set()
    for a in soup.select("div.body a"):
        href = a.get("href") or ""
        if href.endswith(".html") and not href.startswith("#") and href != "index.html":
            url = urljoin(BASE, href)
            if url not in seen:
                seen.add(url)
                chapters.append(url)
    if not chapters:
        print("!! 未解析到章节链接，退出", file=sys.stderr)
        sys.exit(1)
    print(f">> 找到 {len(chapters)} 个章节")

    parts = [
        "# Python 3 官方教程（中文版）\n",
        "> 来源：Python 官方文档 https://docs.python.org/zh-cn/3/tutorial/\n",
        "> 本文件由脚本自动抓取并清洗，保留章节标题层级与代码示例，用于知识库检索。\n",
        "---\n",
    ]
    for i, url in enumerate(chapters, 1):
        print(f"  [{i}/{len(chapters)}] {url}")
        try:
            s = body_soup(fetch(url))
            body = s.select_one("div.body") or s
            md = to_md(body)
            parts.append("\n\n" + md.strip() + "\n")
        except Exception as e:  # noqa
            print(f"  !! 章节失败 {url}: {e}", file=sys.stderr)
        time.sleep(0.3)

    text = "\n".join(parts)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text)
    size = os.path.getsize(OUT)
    print(f">> 完成，写入 {OUT}（{size/1024:.1f} KB，约 {len(text)} 字符）")


if __name__ == "__main__":
    main()
