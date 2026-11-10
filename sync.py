"""本地知识库 -> 线上知识库 的单向增量同步。

背景：本地和线上是两套独立实例，各存一份 data/，彼此不会自动同步。
本脚本按“文件名”比对两边清单，把「本地有、线上没有」的文档推上去，
已存在的直接跳过，不会重复入库、也不会删除线上已有文档。

用法：
    python sync.py                # 执行同步
    python sync.py --dry-run      # 只看差异，不真传
    python sync.py --force        # 同名也重传（会得到一份新的线上副本）

配置（.env 或环境变量）：
    REMOTE_BASE_URL=https://xxx.app.workbuddy.host
    REMOTE_SHARECODE=xxxxxx

注意：线上冷启动（沙箱休眠唤醒）偶尔会返回 504，脚本会自动退避重试。
"""
import argparse
import mimetypes
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests

from app import store, tenants
from app.config import settings

# 同步的是「主设备」那个库；线上对应 tenant 也是同一个名字
TENANT = settings.default_tenant

# 网关侧限制：单次请求 60s 左右，这里给足余量；重试时逐步退避
TIMEOUT = 180
MAX_RETRIES = 3


def _fail(msg: str) -> None:
    print(f"[错误] {msg}")
    sys.exit(1)


def _check_config() -> tuple:
    base = (settings.remote_base_url or "").rstrip("/")
    sc = settings.remote_sharecode or ""
    if not base:
        _fail("未配置 REMOTE_BASE_URL，请在 .env 里填写线上域名，例如：\n"
              "  REMOTE_BASE_URL=https://rag-kb-qa-04404.app.workbuddy.host")
    if not sc:
        _fail("未配置 REMOTE_SHARECODE，请在 .env 里填写线上访问码（链接里 ?sharecode= 后面那串）")
    if not settings.effective_admin_password:
        _fail("未配置 ADMIN_PASSWORD：线上主库需要管理员口令才能写入，\n"
              "  请在 .env 里设置 ADMIN_PASSWORD，并确保与线上实例一致")
    return base, sc


def _tok() -> str:
    """主库口令：线上主库受 ADMIN_PASSWORD 保护，口令随 body 发送。"""
    return settings.effective_admin_password or ""


def remote_list(base: str, sc: str) -> List[Dict]:
    """拉取线上文档清单，带重试（冷启动唤醒可能很慢）。

    用 POST + body 传 tenant/token：部署平台网关会剥离查询参数与 Authorization 头，
    只有请求体能稳定透传。
    """
    last_err: Optional[Exception] = None
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(
                f"{base}/api/documents/list",
                params={"sharecode": sc},
                json={"tenant": TENANT, "token": _tok()},
                timeout=TIMEOUT,
            )
            if r.status_code == 200:
                return r.json()
            last_err = RuntimeError(f"HTTP {r.status_code}: {r.text[:160]}")
        except requests.RequestException as e:
            last_err = e
        wait = 10 * (attempt + 1)
        print(f"  拉取线上清单失败（{type(last_err).__name__}），{wait}s 后重试…")
        time.sleep(wait)
    raise last_err  # type: ignore[misc]


def remote_upload(base: str, sc: str, path: Path, name: str) -> Dict:
    """上传单个文件，返回线上登记的文档信息。"""
    mime = mimetypes.guess_type(name)[0] or "application/octet-stream"
    last_err: Optional[Exception] = None
    for attempt in range(MAX_RETRIES):
        try:
            with open(path, "rb") as f:
                r = requests.post(
                    f"{base}/api/documents",
                    params={"sharecode": sc},
                    data={"tenant": TENANT, "token": _tok()},
                    files={"file": (name, f, mime)},
                    timeout=TIMEOUT,
                )
            if r.status_code == 200:
                return r.json()
            # 502/504 多为网关在冷启动时被占满，退避重试往往就过了
            last_err = RuntimeError(f"HTTP {r.status_code}: {r.text[:160]}")
        except requests.RequestException as e:
            last_err = e
        wait = 10 * (attempt + 1)
        print(f"    上传失败（{type(last_err).__name__}），{wait}s 后重试…")
        time.sleep(wait)
    raise last_err  # type: ignore[misc]


def local_source_file(doc: Dict) -> Optional[Path]:
    """按 doc_id 找回当初上传的原始文件（文件名形如 <doc_id>.<ext>）。"""
    uploads = Path(tenants.uploads_dir(TENANT))
    for p in uploads.glob(f"{doc['doc_id']}.*"):
        return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="本地知识库 -> 线上知识库 单向增量同步")
    ap.add_argument("--dry-run", action="store_true", help="只打印差异，不上传")
    ap.add_argument("--force", action="store_true", help="同名文档也重新上传")
    args = ap.parse_args()

    base, sc = _check_config()

    print(f"本地库：{settings.chroma_path}")
    print(f"线上库：{base}  (sharecode {sc[:4]}***)")

    local_docs = store.list_documents(TENANT)
    print(f"\n本地文档 {len(local_docs)} 篇，正在读取线上清单…")
    try:
        remote_docs = remote_list(base, sc)
    except Exception as e:  # noqa: BLE001
        _fail(f"连不上线上服务：{e}")
    print(f"线上文档 {len(remote_docs)} 篇")

    remote_names = {d.get("name") for d in remote_docs}
    todo = [d for d in local_docs if args.force or d.get("name") not in remote_names]
    skipped = len(local_docs) - len(todo)

    print()
    for d in local_docs:
        if d in todo:
            print(f"  [待同步] {d['name']}  ({d.get('chunk_count', 0)} 切片)")
        else:
            print(f"  [已存在] {d['name']}  ({d.get('chunk_count', 0)} 切片)")

    if skipped:
        print(f"\n线上已有 {skipped} 篇同名文档，跳过（加 --force 可强制重传）")

    if not todo:
        print("\n无需同步，两边一致。")
        return 0

    if args.dry_run:
        print(f"\n[dry-run] 本次会上传 {len(todo)} 篇，未实际执行。")
        return 0

    print(f"\n开始上传 {len(todo)} 篇…")
    ok, failed = 0, []
    for d in todo:
        src = local_source_file(d)
        if src is None:
            print(f"  ✗ {d['name']}：本地找不到原始文件（data/uploads/{d['doc_id']}.*），跳过")
            failed.append(d["name"])
            continue
        size_kb = src.stat().st_size / 1024
        print(f"  → {d['name']}  ({size_kb:.1f} KB)")
        t0 = time.time()
        try:
            info = remote_upload(base, sc, src, d["name"])
            print(f"    ✓ 完成，{time.time() - t0:.1f}s，线上切片 {info.get('chunk_count', '?')}")
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"    ✗ 失败：{type(e).__name__} {e}")
            failed.append(d["name"])

    print(f"\n同步结束：成功 {ok} 篇，失败 {len(failed)} 篇")
    if failed:
        print("失败清单：" + "、".join(failed))
        print("提示：线上沙箱冷启动较慢时容易 504，隔一两分钟重跑本脚本即可（已同步的会自动跳过）。")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
